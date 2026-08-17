"""Stable-Baselines3 2.4 SAC with explicit cross-rank gradient averaging."""

from __future__ import annotations

import math
import time
from typing import Optional

import torch as th
from torch.nn import functional as F

from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update

from .context import DistributedContext
from .npu_performance import zero_optimizer_grad
from ..training_config import SAC_MAX_GRAD_NORM, SAC_MIN_ENT_COEF


class DistributedSAC(SAC):
    """One synchronized SAC learner over rank-local replay buffers.

    SB3 SAC owns separate actor, critic, and entropy optimizers. Wrapping only
    ``policy`` in DistributedDataParallel would miss some of those update paths,
    so this project-side subclass all-reduces each optimizer's gradients at the
    exact point between ``backward()`` and ``optimizer.step()``.
    """

    def __init__(self, *args, distributed_context: Optional[DistributedContext] = None, **kwargs):
        self.distributed_context = distributed_context
        self.max_grad_norm = float(kwargs.pop("max_grad_norm", SAC_MAX_GRAD_NORM))
        self.min_ent_coef = float(kwargs.pop("min_ent_coef", SAC_MIN_ENT_COEF))
        if self.min_ent_coef <= 0.0:
            raise ValueError("min_ent_coef must be positive.")
        super().__init__(*args, **kwargs)

    def set_distributed_context(self, context: DistributedContext) -> None:
        self.distributed_context = context

    def _excluded_save_params(self):
        # Keep SB3 .zip files portable: no process/rank context is serialized.
        return super()._excluded_save_params() + ["distributed_context"]

    def synchronize_parameters(self) -> None:
        context = self.distributed_context
        if context is None or not context.enabled:
            return
        context.broadcast_module(self.actor)
        context.broadcast_module(self.critic)
        context.broadcast_module(self.critic_target)
        context.broadcast_tensor(getattr(self, "log_ent_coef", None))
        context.broadcast_tensor(getattr(self, "ent_coef_tensor", None))
        context.barrier()

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        started = time.perf_counter()
        try:
            self._distributed_train(gradient_steps=gradient_steps, batch_size=batch_size)
        finally:
            if self.distributed_context is not None:
                self.distributed_context.record_update_time(time.perf_counter() - started)

    def _distributed_train(self, gradient_steps: int, batch_size: int = 64) -> None:
        context = self.distributed_context
        if context is None:
            raise RuntimeError("DistributedSAC requires a DistributedContext before training.")

        # ``gradient_steps`` is already the intended number of synchronized
        # optimizer updates for this rollout.  Every rank executes the same
        # update and gradient averaging turns the rank-local minibatches into
        # one global minibatch.  Multiplying this value by world size would do
        # world_size times more optimizer work and defeats data parallelism.

        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        actor_grad_norms, critic_grad_norms = [], []

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(
                    self.log_ent_coef * (log_prob + self.target_entropy).detach()
                ).mean()
                ent_coef_losses.append(ent_coef_loss.detach())
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(ent_coef.detach())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                zero_optimizer_grad(self.ent_coef_optimizer)
                ent_coef_loss.backward()
                context.average_tensor_gradient(self.log_ent_coef)
                self.ent_coef_optimizer.step()
                # Automatic entropy tuning may otherwise collapse exploration
                # long before a sparse navigation policy is established.
                with th.no_grad():
                    self.log_ent_coef.clamp_(min=math.log(self.min_ent_coef))

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(
                    replay_data.next_observations
                )
                next_q_values = th.cat(
                    self.critic_target(replay_data.next_observations, next_actions), dim=1
                )
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(
                F.mse_loss(current_q, target_q_values) for current_q in current_q_values
            )
            critic_losses.append(critic_loss.detach())

            zero_optimizer_grad(self.critic.optimizer)
            critic_loss.backward()
            context.average_gradients(self.critic.parameters())
            critic_grad_norms.append(
                th.nn.utils.clip_grad_norm_(
                    self.critic.parameters(),
                    max_norm=self.max_grad_norm,
                ).detach()
            )
            self.critic.optimizer.step()

            # The actor needs dQ/da, but never updates critic parameters in
            # this phase.  Temporarily freezing them preserves the exact actor
            # gradient while avoiding critic parameter-gradient computation
            # and storage that the next critic zero_grad would discard.
            critic_grad_states = [
                parameter.requires_grad for parameter in self.critic.parameters()
            ]
            for parameter in self.critic.parameters():
                parameter.requires_grad_(False)
            try:
                q_values_pi = th.cat(
                    self.critic(replay_data.observations, actions_pi), dim=1
                )
                min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
                actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
                actor_losses.append(actor_loss.detach())

                zero_optimizer_grad(self.actor.optimizer)
                actor_loss.backward()
                context.average_gradients(self.actor.parameters())
                actor_grad_norms.append(
                    th.nn.utils.clip_grad_norm_(
                        self.actor.parameters(),
                        max_norm=self.max_grad_norm,
                    ).detach()
                )
                self.actor.optimizer.step()
            finally:
                for parameter, requires_grad in zip(
                    self.critic.parameters(), critic_grad_states
                ):
                    parameter.requires_grad_(requires_grad)

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        # Keep metric reduction on the accelerator.  Calling .item() inside
        # every gradient step serializes the NPU pipeline with a device-to-host
        # synchronization; one transfer after the complete train block is
        # sufficient and yields the same logged means.
        zero = th.zeros((), dtype=actor_losses[0].dtype, device=actor_losses[0].device)
        local_metrics = th.stack(
            [
                th.stack(ent_coefs).mean(),
                th.stack(actor_losses).mean(),
                th.stack(critic_losses).mean(),
                th.stack(ent_coef_losses).mean() if ent_coef_losses else zero,
                th.stack(actor_grad_norms).mean(),
                th.stack(critic_grad_norms).mean(),
            ]
        )
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        metric_values = context.reduce_metrics_periodically(
            "sac_train",
            local_metrics,
            interval=64,
        )
        if metric_values is not None:
            (
                ent_coef_mean,
                actor_loss_mean,
                critic_loss_mean,
                ent_coef_loss_mean,
                actor_grad_norm_mean,
                critic_grad_norm_mean,
            ) = metric_values
            self.logger.record("train/ent_coef", ent_coef_mean)
            self.logger.record("train/actor_loss", actor_loss_mean)
            self.logger.record("train/critic_loss", critic_loss_mean)
            if len(ent_coef_losses) > 0:
                self.logger.record("train/ent_coef_loss", ent_coef_loss_mean)
            self.logger.record("train/actor_grad_norm", actor_grad_norm_mean)
            self.logger.record("train/critic_grad_norm", critic_grad_norm_mean)
