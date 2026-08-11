"""Stable-Baselines3 2.4 SAC with explicit cross-rank gradient averaging."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update

from .context import DistributedContext


class DistributedSAC(SAC):
    """One synchronized SAC learner over rank-local replay buffers.

    SB3 SAC owns separate actor, critic, and entropy optimizers. Wrapping only
    ``policy`` in DistributedDataParallel would miss some of those update paths,
    so this project-side subclass all-reduces each optimizer's gradients at the
    exact point between ``backward()`` and ``optimizer.step()``.
    """

    def __init__(self, *args, distributed_context: Optional[DistributedContext] = None, **kwargs):
        self.distributed_context = distributed_context
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
        context = self.distributed_context
        if context is None:
            raise RuntimeError("DistributedSAC requires a DistributedContext before training.")

        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

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
                ent_coef_losses.append(float(ent_coef_loss.item()))
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(float(ent_coef.item()))

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                context.average_tensor_gradient(self.log_ent_coef)
                self.ent_coef_optimizer.step()

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
            critic_losses.append(float(critic_loss.item()))

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            context.average_gradients(self.critic.parameters())
            self.critic.optimizer.step()

            q_values_pi = th.cat(
                self.critic(replay_data.observations, actions_pi), dim=1
            )
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(float(actor_loss.item()))

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            context.average_gradients(self.actor.parameters())
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        local_metrics = [
            float(np.mean(ent_coefs)),
            float(np.mean(actor_losses)),
            float(np.mean(critic_losses)),
            float(np.mean(ent_coef_losses)) if len(ent_coef_losses) > 0 else 0.0,
        ]
        ent_coef_mean, actor_loss_mean, critic_loss_mean, ent_coef_loss_mean = (
            context.average_metrics(local_metrics)
        )

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", ent_coef_mean)
        self.logger.record("train/actor_loss", actor_loss_mean)
        self.logger.record("train/critic_loss", critic_loss_mean)
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", ent_coef_loss_mean)
