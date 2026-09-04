"""Goal-conditioned SAC with Safe-HER replay and a critic ensemble.

This module is deliberately separate from ``distributed.sac.DistributedSAC``.
It keeps the baseline SAC implementation untouched while adding randomized
target subsets, LayerNorm inputs, critic-heavy UTD, and optional HCCL gradient
averaging for the new goal-conditioned experiment.
"""

from __future__ import annotations

import copy
import math
import random
from typing import Optional

import torch as th
from torch import nn
from torch.nn import functional as F

from stable_baselines3.sac.sac import SAC
from stable_baselines3.common.utils import polyak_update

from .context import DistributedContext
from .npu_performance import zero_optimizer_grad


class GoalConditionedSAC(SAC):
    """SAC variant for flat ``state + desired_goal`` observations.

    ``critic_ensemble_size`` is the number of independent twin-critic
    modules.  The target uses the minimum of a fresh random subset, while the
    actor uses the mean over all modules.  This is intentionally implemented
    without changing the baseline SAC class.
    """

    def __init__(
        self,
        *args,
        distributed_context: Optional[DistributedContext] = None,
        critic_ensemble_size: int = 10,
        target_critic_subset_size: int = 2,
        utd_ratio: int = 10,
        utd_warmup_steps: int = 50_000,
        actor_update_interval: int = 2,
        critic_layer_norm: bool = True,
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
        super().__init__(*args, **kwargs)
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
        self.critic_ensemble = nn.ModuleList([base])
        self.critic_target_ensemble = nn.ModuleList([targets])
        for _ in range(1, self.critic_ensemble_size):
            self.critic_ensemble.append(copy.deepcopy(base))
            self.critic_target_ensemble.append(copy.deepcopy(targets))
        obs_dim = int(self.observation_space.shape[0])
        self.critic_input_norms = nn.ModuleList(
            [nn.LayerNorm(obs_dim).to(self.device) for _ in range(self.critic_ensemble_size)]
        ) if self.critic_layer_norm else nn.ModuleList()
        # Keep SB3 aliases pointing at the first member for compatibility with
        # callbacks and model metadata.
        self.critic = self.critic_ensemble[0]
        self.critic_target = self.critic_target_ensemble[0]
        lr = float(self.lr_schedule(1.0))
        critic_parameters = list(self.critic_ensemble.parameters()) + list(
            self.critic_input_norms.parameters()
        )
        self.critic.optimizer = th.optim.Adam(critic_parameters, lr=lr)
        for critic in self.critic_target_ensemble:
            critic.set_training_mode(False)
            for parameter in critic.parameters():
                parameter.requires_grad_(False)
        self._goal_sac_n_updates = 0
        self._goal_sac_actor_updates = 0

    def _critic_obs(self, index: int, observations):
        if self.critic_layer_norm:
            return self.critic_input_norms[index](observations)
        return observations

    def _reduce_gradients(self, parameters) -> None:
        context = self.distributed_context
        if context is not None and context.enabled:
            context.average_gradients(parameters)

    def _utd_for_replay(self) -> int:
        replay_size = int(getattr(self.replay_buffer, "pos", 0))
        if getattr(self.replay_buffer, "full", False):
            replay_size = int(self.replay_buffer.buffer_size)
        if replay_size < self.utd_warmup_steps:
            return max(1, min(self.utd_ratio, max(1, replay_size // 10_000)))
        return self.utd_ratio

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
        ent_coef_losses, ent_coefs, actor_losses, critic_losses = [], [], [], []
        q_means, q_stds = [], []
        actor_update_count = 0
        for update_index in range(int(gradient_steps)):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            if self.use_sde:
                self.actor.reset_noise()
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef_loss = -(
                    self.log_ent_coef * (log_prob + self.target_entropy).detach()
                ).mean()
                zero_optimizer_grad(self.ent_coef_optimizer)
                ent_coef_loss.backward()
                self._reduce_gradients([self.log_ent_coef])
                self.ent_coef_optimizer.step()
                with th.no_grad():
                    self.log_ent_coef.clamp_(min=math.log(1e-4))
                ent_coef_losses.append(ent_coef_loss.detach())
                ent_coef = th.exp(self.log_ent_coef.detach())
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(ent_coef.detach())

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(
                    replay_data.next_observations
                )
                subset = random.sample(
                    range(self.critic_ensemble_size), self.target_critic_subset_size
                )
                target_values = []
                for index in subset:
                    q_pair = self.critic_target_ensemble[index](
                        self._critic_obs(index, replay_data.next_observations),
                        next_actions,
                    )
                    target_values.append(th.cat(q_pair, dim=1).min(dim=1, keepdim=True).values)
                next_q = th.cat(target_values, dim=1).min(dim=1, keepdim=True).values
                target_q = replay_data.rewards + (1.0 - replay_data.dones) * self.gamma * (
                    next_q - ent_coef * next_log_prob.reshape(-1, 1)
                )

            zero_optimizer_grad(self.critic.optimizer)
            critic_loss = th.zeros((), device=replay_data.observations.device)
            for index, critic in enumerate(self.critic_ensemble):
                q_pair = critic(self._critic_obs(index, replay_data.observations), replay_data.actions)
                q_stack = th.cat(q_pair, dim=1)
                q_means.append(q_stack.mean().detach())
                q_stds.append(q_stack.std(unbiased=False).detach())
                critic_loss = critic_loss + 0.5 * sum(
                    F.mse_loss(q_value, target_q) for q_value in q_pair
                )
            critic_loss = critic_loss / float(self.critic_ensemble_size)
            critic_loss.backward()
            self._reduce_gradients(self.critic.optimizer.param_groups[0]["params"])
            self.critic.optimizer.step()
            critic_losses.append(critic_loss.detach())

            if update_index % self.actor_update_interval == 0:
                critic_states = [p.requires_grad for p in self.critic_ensemble.parameters()]
                norm_states = [p.requires_grad for p in self.critic_input_norms.parameters()]
                for parameter in self.critic_ensemble.parameters():
                    parameter.requires_grad_(False)
                for parameter in self.critic_input_norms.parameters():
                    parameter.requires_grad_(False)
                try:
                    q_values = []
                    for index, critic in enumerate(self.critic_ensemble):
                        pair = critic(self._critic_obs(index, replay_data.observations), actions_pi)
                        q_values.append(th.cat(pair, dim=1).min(dim=1, keepdim=True).values)
                    mean_q = th.cat(q_values, dim=1).mean(dim=1, keepdim=True)
                    actor_loss = (ent_coef * log_prob - mean_q).mean()
                    zero_optimizer_grad(self.actor.optimizer)
                    actor_loss.backward()
                    self._reduce_gradients(self.actor.parameters())
                    self.actor.optimizer.step()
                    actor_losses.append(actor_loss.detach())
                    actor_update_count += 1
                finally:
                    for parameter, state in zip(self.critic_ensemble.parameters(), critic_states):
                        parameter.requires_grad_(state)
                    for parameter, state in zip(self.critic_input_norms.parameters(), norm_states):
                        parameter.requires_grad_(state)

            if update_index % self.target_update_interval == 0:
                for critic, target in zip(self.critic_ensemble, self.critic_target_ensemble):
                    polyak_update(critic.parameters(), target.parameters(), self.tau)
        self._goal_sac_n_updates += int(gradient_steps)
        self._goal_sac_actor_updates += int(actor_update_count)
        self._n_updates += int(gradient_steps)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/goal_sac_actual_utd", float(gradient_steps))
        self.logger.record("train/goal_sac_actor_updates", float(actor_update_count))
        if q_means:
            self.logger.record("train/q_mean", float(th.stack(q_means).mean().cpu()))
            self.logger.record("train/q_std", float(th.stack(q_stds).mean().cpu()))
        if actor_losses:
            self.logger.record("train/actor_loss", float(th.stack(actor_losses).mean().detach().cpu()))
        self.logger.record("train/critic_loss", float(th.stack(critic_losses).mean().detach().cpu()))
        self.logger.record("train/ent_coef", float(th.stack(ent_coefs).mean().detach().cpu()))
        if ent_coef_losses:
            self.logger.record("train/ent_coef_loss", float(th.stack(ent_coef_losses).mean().detach().cpu()))
        stats = getattr(self.replay_buffer, "her_stats", {})
        if stats:
            samples = max(1, int(stats.get("samples", 0)))
            self.logger.record("replay/her_fraction", float(stats.get("her_samples", 0)) / samples)
            self.logger.record("replay/safe_candidates", float(stats.get("safe_candidates", 0)))
            self.logger.record("replay/rejected_candidates", float(stats.get("rejected_candidates", 0)))

    def _get_torch_save_params(self):
        names, state_dicts = super()._get_torch_save_params()
        return (
            list(dict.fromkeys(names + ["critic_ensemble", "critic_target_ensemble", "critic_input_norms"])),
            list(dict.fromkeys(state_dicts + ["critic.optimizer"])),
        )
