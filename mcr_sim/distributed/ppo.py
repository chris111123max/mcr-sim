"""Stable-Baselines3 PPO with synchronized cross-rank policy gradients."""

from __future__ import annotations

import time
from typing import Optional

import torch as th
from gymnasium import spaces
from torch.nn import functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance

from .context import DistributedContext
from .npu_performance import zero_optimizer_grad


class DistributedPPO(PPO):
    """One on-policy PPO learner over rank-local rollout buffers.

    Every rank owns an equal share of environments and an independent on-policy
    rollout buffer.  Gradients are flattened and averaged after ``backward()``
    and before SB3's global gradient clipping and optimizer step.
    """

    def __init__(self, *args, distributed_context: Optional[DistributedContext] = None, **kwargs):
        self.distributed_context = distributed_context
        super().__init__(*args, **kwargs)
        if distributed_context is not None:
            self.set_distributed_context(distributed_context)

    def _excluded_save_params(self):
        return super()._excluded_save_params() + [
            "distributed_context",
        ]

    def set_distributed_context(self, context: DistributedContext) -> None:
        self.distributed_context = context

    def train(self) -> None:
        started = time.perf_counter()
        try:
            self._distributed_train()
        finally:
            if self.distributed_context is not None:
                self.distributed_context.record_update_time(time.perf_counter() - started)

    def _distributed_train(self) -> None:
        """SB3 2.4 PPO update with bucketed distributed synchronization."""

        context = self.distributed_context
        if context is None:
            raise RuntimeError("DistributedPPO requires a DistributedContext before training.")

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

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(
                    ratio, 1 - clip_range, 1 + clip_range
                )
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                policy_losses.append(policy_loss.detach())
                clip_fractions.append(
                    th.mean((th.abs(ratio - 1) > clip_range).float()).detach()
                )

                if clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.detach())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.detach())
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                last_loss = loss.detach()

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl = th.mean((th.exp(log_ratio) - 1) - log_ratio)
                approx_kl_divs.append(approx_kl.detach())

                # target_kl is normally disabled.  When enabled, decide from a
                # rank-averaged value so every rank enters the same collectives.
                if self.target_kl is not None:
                    global_kl = context.average_metric_tensor(approx_kl.detach().clone())
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
                context.average_gradients(self.policy.parameters())
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

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
            th.as_tensor(explained_var, dtype=last_loss.dtype, device=last_loss.device),
        ]
        has_log_std = hasattr(self.policy, "log_std")
        if has_log_std:
            metric_tensors.append(th.exp(self.policy.log_std).mean().detach())
        metric_values = context.reduce_metrics_periodically(
            "ppo_train",
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

    def synchronize_parameters(self) -> None:
        context = self.distributed_context
        if context is None:
            raise RuntimeError("DistributedPPO requires a DistributedContext.")
        context.broadcast_module(self.policy)
        context.barrier()
