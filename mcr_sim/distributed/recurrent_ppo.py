"""SB3-Contrib RecurrentPPO with the project's synchronized gradients."""

from __future__ import annotations

import math
import time
from typing import Optional, Tuple

import torch as th
from gymnasium import spaces
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.utils import explained_variance

from .context import DistributedContext
from .npu_performance import zero_optimizer_grad
from ..training_config import PPO_MAX_ACTION_STD, PPO_MIN_ACTION_STD


class DistributedRecurrentPPO(RecurrentPPO):
    """Official recurrent rollout/buffer plus the existing HCCL update path.

    SB3-Contrib owns recurrent state collection, per-environment episode-start
    resets, sequence construction, padding, masks, and recurrent prediction.
    This subclass changes only the optimizer step: after the official masked
    PPO loss is backpropagated, every policy gradient is averaged through the
    same ``DistributedContext`` used by the MLP-PPO baseline.
    """

    def __init__(
        self,
        *args,
        distributed_context: Optional[DistributedContext] = None,
        **kwargs,
    ):
        self.distributed_context = distributed_context
        self.min_action_std = float(
            kwargs.pop("min_action_std", PPO_MIN_ACTION_STD)
        )
        self.max_action_std = float(
            kwargs.pop("max_action_std", PPO_MAX_ACTION_STD)
        )
        if not (0.0 < self.min_action_std <= self.max_action_std):
            raise ValueError("PPO action std bounds must satisfy 0 < min <= max.")
        super().__init__(*args, **kwargs)
        if distributed_context is not None:
            self.set_distributed_context(distributed_context)

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["distributed_context"]

    def set_distributed_context(self, context: DistributedContext) -> None:
        self.distributed_context = context

    def recurrent_parameter_names(self) -> Tuple[str, ...]:
        """Return registered actor/critic LSTM parameters for diagnostics."""

        return tuple(
            name
            for name, _ in self.policy.named_parameters()
            if name.startswith("lstm_actor.") or name.startswith("lstm_critic.")
        )

    def assert_recurrent_parameters_registered(self) -> Tuple[str, ...]:
        """Fail before training if actor/critic LSTM weights are not registered."""

        names = self.recurrent_parameter_names()
        required_suffixes = (
            "weight_ih_l0",
            "weight_hh_l0",
            "bias_ih_l0",
            "bias_hh_l0",
        )
        missing = [
            f"{module}.{suffix}"
            for module in ("lstm_actor", "lstm_critic")
            for suffix in required_suffixes
            if f"{module}.{suffix}" not in names
        ]
        if missing:
            raise RuntimeError(
                "Recurrent policy is missing registered LSTM parameters: "
                + ", ".join(missing)
            )
        return names

    def train(self) -> None:
        started = time.perf_counter()
        try:
            self._distributed_train()
        finally:
            if self.distributed_context is not None:
                self.distributed_context.record_update_time(
                    time.perf_counter() - started
                )

    def _distributed_train(self) -> None:
        """SB3-Contrib 2.4 masked recurrent PPO with gradient averaging."""

        context = self.distributed_context
        if context is None:
            raise RuntimeError(
                "DistributedRecurrentPPO requires a DistributedContext before training."
            )

        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        clip_range_vf = None
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses, policy_losses, value_losses = [], [], []
        clip_fractions, approx_kl_divs = [], []
        continue_training = True
        last_loss = None

        for epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()

                mask = rollout_data.mask > 1e-8
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    rollout_data.lstm_states,
                    rollout_data.episode_starts,
                )
                values = values.flatten()
                advantages = rollout_data.advantages
                valid_advantages = advantages[mask]
                if self.normalize_advantage and valid_advantages.numel() > 1:
                    advantages = (advantages - valid_advantages.mean()) / (
                        valid_advantages.std() + 1e-8
                    )

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(
                    ratio, 1 - clip_range, 1 + clip_range
                )
                policy_loss = -th.min(policy_loss_1, policy_loss_2)[mask].mean()
                policy_losses.append(policy_loss.detach())
                clip_fractions.append(
                    (th.abs(ratio - 1) > clip_range).float()[mask].mean().detach()
                )

                if clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )
                value_loss = ((rollout_data.returns - values_pred) ** 2)[
                    mask
                ].mean()
                value_losses.append(value_loss.detach())

                if entropy is None:
                    entropy_loss = -(-log_prob[mask]).mean()
                else:
                    entropy_loss = -entropy[mask].mean()
                entropy_losses.append(entropy_loss.detach())

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                )
                last_loss = loss.detach()

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl = ((th.exp(log_ratio) - 1) - log_ratio)[mask].mean()
                approx_kl_divs.append(approx_kl.detach())

                if self.target_kl is not None:
                    global_kl = context.average_metric_tensor(
                        approx_kl.detach().clone()
                    )
                    global_kl_value = float(global_kl.cpu().item())
                    if global_kl_value > 1.5 * self.target_kl:
                        continue_training = False
                        if self.verbose >= 1 and context.is_main:
                            print(
                                f"Early stopping at step {epoch} due to reaching "
                                f"max kl: {global_kl_value:.2f}"
                            )
                        break

                zero_optimizer_grad(self.policy.optimizer)
                loss.backward()
                # Includes features, actor/critic LSTMs, MLP heads, value head,
                # action head, and log_std in one deterministic parameter order.
                context.average_gradients(self.policy.parameters())
                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()
                self.clamp_action_std()

            self._n_updates += 1
            if not continue_training:
                break

        if last_loss is None:
            return

        explained_var = float(
            explained_variance(
                self.rollout_buffer.values.flatten(),
                self.rollout_buffer.returns.flatten(),
            )
        )
        metric_tensors = [
            th.stack(entropy_losses).mean(),
            th.stack(policy_losses).mean(),
            th.stack(value_losses).mean(),
            th.stack(approx_kl_divs).mean(),
            th.stack(clip_fractions).mean(),
            last_loss,
            th.as_tensor(
                explained_var,
                dtype=last_loss.dtype,
                device=last_loss.device,
            ),
        ]
        has_log_std = hasattr(self.policy, "log_std")
        if has_log_std:
            metric_tensors.append(th.exp(self.policy.log_std).mean().detach())
        metric_values = context.reduce_metrics_periodically(
            "recurrent_ppo_train",
            th.stack(metric_tensors),
            interval=1,
        )

        self.logger.record("train/entropy_loss", metric_values[0])
        self.logger.record("train/policy_gradient_loss", metric_values[1])
        self.logger.record("train/value_loss", metric_values[2])
        self.logger.record("train/approx_kl", metric_values[3])
        self.logger.record("train/clip_fraction", metric_values[4])
        self.logger.record("train/loss", metric_values[5])
        self.logger.record("train/explained_variance", metric_values[6])
        if has_log_std:
            self.logger.record("train/std", metric_values[7])
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def clamp_action_std(self) -> None:
        """Use the exact same Gaussian exploration bounds as MLP-PPO."""

        log_std = getattr(self.policy, "log_std", None)
        if log_std is None:
            return
        with th.no_grad():
            log_std.clamp_(
                min=math.log(self.min_action_std),
                max=math.log(self.max_action_std),
            )

    def synchronize_parameters(self) -> None:
        """Broadcast the complete recurrent policy, including both LSTMs."""

        context = self.distributed_context
        if context is None:
            raise RuntimeError("DistributedRecurrentPPO requires a DistributedContext.")
        self.assert_recurrent_parameters_registered()
        context.broadcast_module(self.policy)
        context.barrier()
