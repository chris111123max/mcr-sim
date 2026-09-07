"""Isolated twin-Q Goal-SAC with Safe HER and optional gradient averaging."""

from __future__ import annotations

import math
from typing import Optional

import torch as th
from torch import nn
from torch.nn import functional as F

from stable_baselines3.sac.sac import SAC
from stable_baselines3.common.utils import polyak_update

from .context import DistributedContext
from .npu_performance import zero_optimizer_grad


class GoalConditionedSAC(SAC):
    """Twin-Q baseline. Both actor and bootstrap use min(Q1, Q2)."""

    def __init__(
        self,
        *args,
        distributed_context: Optional[DistributedContext] = None,
        critic_ensemble_size: int = 2,
        target_critic_subset_size: int = 2,
        utd_ratio: int = 2,
        utd_warmup_steps: int = 50_000,
        actor_update_interval: int = 1,
        critic_layer_norm: bool = False,
        metric_log_interval: int = 64,
        min_ent_coef: float = 0.02,
        **kwargs,
    ):
        self.distributed_context = distributed_context
        self.critic_ensemble_size = max(2, int(critic_ensemble_size))
        self.target_critic_subset_size = max(
            1, min(int(target_critic_subset_size), self.critic_ensemble_size)
        )
        self.utd_ratio = max(1, int(utd_ratio))
        self.utd_warmup_steps = max(0, int(utd_warmup_steps))
        self.actor_update_interval = max(1, int(actor_update_interval))
        self.critic_layer_norm = bool(critic_layer_norm)
        self.metric_log_interval = max(1, int(metric_log_interval))
        self.min_ent_coef = max(1e-6, float(min_ent_coef))
        if critic_ensemble_size != 2 or target_critic_subset_size != 2 or critic_layer_norm:
            raise ValueError("Goal-SAC v3 uses twin Q, both targets, and no input LayerNorm")
        super().__init__(*args, **kwargs)

    def _setup_model(self):
        super()._setup_model()
        contract = getattr(self.replay_buffer, "reward_contract", None)
        if contract is not None and not math.isclose(contract.gamma, self.gamma):
            raise ValueError("HER reward gamma must equal SAC gamma")
        env_contract = getattr(self.env, "reward_contract", None)
        if env_contract is not None and contract != env_contract:
            raise ValueError("Rollout and HER must use identical reward contracts")
        self._build_critic_ensemble()

    def set_distributed_context(self, context: DistributedContext) -> None:
        self.distributed_context = context

    def synchronize_parameters(self) -> None:
        """Broadcast the complete ensemble after construction in torchrun."""
        context = self.distributed_context
        if context is None or not context.enabled:
            return
        context.broadcast_module(self.actor)
        for critic in self.critic_ensemble:
            context.broadcast_module(critic)
        for critic in self.critic_target_ensemble:
            context.broadcast_module(critic)
        for norm in self.critic_input_norms:
            context.broadcast_module(norm)
        context.broadcast_tensor(getattr(self, "log_ent_coef", None))
        context.broadcast_tensor(getattr(self, "ent_coef_tensor", None))
        context.barrier()

    def _build_critic_ensemble(self) -> None:
        base = self.critic
        targets = self.critic_target
        self._q_heads_per_module = len(base.q_networks)
        if self._q_heads_per_module != 2:
            raise ValueError("Goal-SAC requires policy_kwargs n_critics=2")
        self._critic_module_count = 1
        self.critic_ensemble = nn.ModuleList([base])
        self.critic_target_ensemble = nn.ModuleList([targets])
        self.critic_input_norms = nn.ModuleList()
        # Keep SB3 aliases pointing at the first member for compatibility with
        # callbacks and model metadata.
        self.critic = self.critic_ensemble[0]
        self.critic_target = self.critic_target_ensemble[0]
        # Retain SB3's optimizer and policy optimizer kwargs for the twin Q.
        for critic in self.critic_target_ensemble:
            critic.set_training_mode(False)
            for parameter in critic.parameters():
                parameter.requires_grad_(False)
        self._goal_sac_n_updates = 0
        self._goal_sac_actor_updates = 0

    def _critic_obs(self, index: int, observations):
        return observations

    def _reduce_gradients(self, parameters) -> None:
        context = self.distributed_context
        if context is not None and context.enabled:
            context.average_gradients(parameters)

    def _utd_for_replay(self) -> int:
        replay_slots = int(getattr(self.replay_buffer, "pos", 0))
        if getattr(self.replay_buffer, "full", False):
            replay_slots = int(self.replay_buffer.buffer_size)
        # ReplayBuffer.pos counts vector slots, not transitions.  With 32
        # environments one slot represents 32 transitions.
        replay_transitions = replay_slots * int(getattr(self.replay_buffer, "n_envs", 1))
        if self.utd_ratio <= 1 or self.utd_warmup_steps <= 0:
            return self.utd_ratio
        since_learning_start = max(0, replay_transitions - int(self.learning_starts))
        if since_learning_start < self.utd_warmup_steps // 2:
            return 1
        if since_learning_start < self.utd_warmup_steps:
            return min(5, self.utd_ratio)
        return self.utd_ratio

    def _flat_q_values(self, modules, observations, actions):
        """Return exactly ``critic_ensemble_size`` individual Q tensors."""
        values = []
        for index, critic in enumerate(modules):
            values.extend(critic(self._critic_obs(index, observations), actions))
        return values[: self.critic_ensemble_size]

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        started = __import__("time").perf_counter()
        try:
            self._distributed_train(
                gradient_steps=max(1, int(gradient_steps)) * self._utd_for_replay(),
                batch_size=batch_size,
            )
        finally:
            context = self.distributed_context
            if context is not None:
                context.record_update_time(__import__("time").perf_counter() - started)

    def _distributed_train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers.append(self.ent_coef_optimizer)
        self._update_learning_rate(optimizers)
        metric_sums = th.zeros(9, dtype=th.float32, device=self.device)
        metric_counts = th.zeros(4, dtype=th.float32, device=self.device)
        actor_update_count = 0
        for update_index in range(int(gradient_steps)):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            if self.use_sde:
                self.actor.reset_noise()
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)
            metric_sums[8].add_(-log_prob.detach().mean())
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef_loss = -(
                    self.log_ent_coef * (log_prob + self.target_entropy).detach()
                ).mean()
                zero_optimizer_grad(self.ent_coef_optimizer)
                ent_coef_loss.backward()
                self._reduce_gradients([self.log_ent_coef])
                self.ent_coef_optimizer.step()
                with th.no_grad():
                    self.log_ent_coef.clamp_(min=math.log(self.min_ent_coef))
                metric_sums[3].add_(ent_coef_loss.detach().reshape(()))
                metric_counts[3].add_(1.0)
                ent_coef = th.exp(self.log_ent_coef.detach())
            else:
                ent_coef = self.ent_coef_tensor
            # torch-npu does not allow inplace broadcasting from [1] into a
            # scalar tensor view.  Normalize all diagnostic values to true
            # zero-dimensional tensors before accumulating them.
            metric_sums[2].add_(ent_coef.detach().reshape(()))
            metric_counts[2].add_(1.0)

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(
                    replay_data.next_observations
                )
                target_values = self.critic_target(replay_data.next_observations, next_actions)
                next_q = th.cat(target_values, dim=1).min(dim=1, keepdim=True).values
                target_q = replay_data.rewards + (1.0 - replay_data.dones) * self.gamma * (
                    next_q - ent_coef * next_log_prob.reshape(-1, 1)
                )

            zero_optimizer_grad(self.critic.optimizer)
            q_values = self._flat_q_values(
                self.critic_ensemble, replay_data.observations, replay_data.actions
            )
            q_stack = th.cat(q_values, dim=1)
            critic_loss = sum(F.mse_loss(q_value, target_q) for q_value in q_values)
            critic_loss = critic_loss / float(self.critic_ensemble_size)
            critic_loss.backward()
            self._reduce_gradients(self.critic.optimizer.param_groups[0]["params"])
            self.critic.optimizer.step()
            metric_sums[1].add_(critic_loss.detach().reshape(()))
            metric_sums[4].add_(q_stack.detach().mean())
            metric_sums[5].add_(q_stack.detach().std(dim=1, unbiased=False).mean())
            metric_sums[6].add_(target_q.detach().mean())
            metric_sums[7].add_((q_stack.detach().mean(dim=1, keepdim=True) - target_q).abs().mean())
            metric_counts[1].add_(1.0)

            if (self._n_updates + update_index) % self.actor_update_interval == 0:
                critic_states = [p.requires_grad for p in self.critic_ensemble.parameters()]
                norm_states = [p.requires_grad for p in self.critic_input_norms.parameters()]
                for parameter in self.critic_ensemble.parameters():
                    parameter.requires_grad_(False)
                for parameter in self.critic_input_norms.parameters():
                    parameter.requires_grad_(False)
                try:
                    q_values = self._flat_q_values(
                        self.critic_ensemble, replay_data.observations, actions_pi
                    )
                    min_q = th.cat(q_values, dim=1).min(dim=1, keepdim=True).values
                    actor_loss = (ent_coef * log_prob - min_q).mean()
                    zero_optimizer_grad(self.actor.optimizer)
                    actor_loss.backward()
                    self._reduce_gradients(self.actor.parameters())
                    self.actor.optimizer.step()
                    metric_sums[0].add_(actor_loss.detach().reshape(()))
                    metric_counts[0].add_(1.0)
                    actor_update_count += 1
                finally:
                    for parameter, state in zip(self.critic_ensemble.parameters(), critic_states):
                        parameter.requires_grad_(state)
                    for parameter, state in zip(self.critic_input_norms.parameters(), norm_states):
                        parameter.requires_grad_(state)

            if (self._n_updates + update_index) % self.target_update_interval == 0:
                for critic, target in zip(self.critic_ensemble, self.critic_target_ensemble):
                    polyak_update(critic.parameters(), target.parameters(), self.tau)
        self._goal_sac_n_updates += int(gradient_steps)
        self._goal_sac_actor_updates += int(actor_update_count)
        self._n_updates += int(gradient_steps)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/goal_sac_actual_utd", float(gradient_steps))
        self.logger.record("train/goal_sac_actor_updates", float(actor_update_count))
        denominators = th.stack(
            [metric_counts[0], metric_counts[1], metric_counts[2], metric_counts[3],
             metric_counts[1], metric_counts[1], metric_counts[1], metric_counts[1], metric_counts[1]]
        ).clamp_min_(1.0)
        local_metrics = metric_sums / denominators
        context = self.distributed_context
        values = context.reduce_metrics_periodically(
            "goal_sac_train", local_metrics, self.metric_log_interval
        ) if context is not None else local_metrics.detach().cpu().tolist()
        if values is not None:
            names = (
                "actor_loss", "critic_loss", "ent_coef", "ent_coef_loss",
                "q_mean", "q_ensemble_std", "target_q_mean", "td_error_abs", "policy_entropy",
            )
            for name, value in zip(names, values):
                self.logger.record(f"train/{name}", float(value))
            stats = self.replay_buffer.pop_window_stats() if hasattr(
                self.replay_buffer, "pop_window_stats") else {}
            if stats:
                samples = max(1, int(stats.get("samples", 0)))
                her_samples = max(1, int(stats.get("her_samples", 0)))
                self.logger.record("replay/her_fraction", float(stats.get("her_samples", 0)) / samples)
                self.logger.record(
                    "replay/her_success_fraction",
                    float(stats.get("her_successes", 0)) / her_samples,
                )
                self.logger.record("replay/safe_candidates", float(stats.get("safe_candidates", 0)))
                self.logger.record("replay/rejected_candidates", float(stats.get("rejected_candidates", 0)))
                self.logger.record(
                    "replay/insufficient_progress_candidates",
                    float(stats.get("insufficient_progress_candidates", 0)),
                )

    def _get_torch_save_params(self):
        names, state_dicts = super()._get_torch_save_params()
        return (
            list(dict.fromkeys(names + ["critic_ensemble", "critic_target_ensemble", "critic_input_norms"])),
            state_dicts,
        )

    def _excluded_save_params(self):
        return super()._excluded_save_params() + [
            "distributed_context", "critic_ensemble", "critic_target_ensemble", "critic_input_norms"]
