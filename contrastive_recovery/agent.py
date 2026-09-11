"""DDP-compatible Contrastive Goal RL with a distinct Recovery-SAC policy."""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

from .networks import ContrastiveCritic, RecurrentActor, RecoveryCritic, RiskCritic


def _unwrap(module):
    return module.module if isinstance(module, DDP) else module


def _sample_actor(actor, obs, previous_actions, deterministic: bool = False):
    """Sample through ``forward`` so DDP installs its gradient reducer."""
    mean, log_std = actor(obs, previous_actions)
    normal = torch.distributions.Normal(mean, log_std.exp())
    raw = mean if deterministic else normal.rsample()
    action = torch.tanh(raw)
    log_prob = normal.log_prob(raw) - torch.log(1.0 - action.square() + 1e-6)
    return action, log_prob.sum(dim=-1, keepdim=True), torch.tanh(mean)


@dataclass
class AgentConfig:
    observation_dim: int
    action_dim: int
    num_envs: int
    device: str
    sequence_length: int = 32
    hidden_dim: int = 256
    embedding_dim: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    entropy_weight: float = 0.05
    risk_weight: float = 1.0
    recovery_gate_threshold: float = 0.40
    recovery_min_steps: int = 12
    distributed: bool = False


class ContrastiveRecoveryAgent:
    """Main direct policy + recovery direct policy, both recurrent and trainable.

    The gate is informed by a learned risk critic and measured SDF warning
    features.  It does not replace either RL action with a hand-designed
    steering command.
    """

    def __init__(self, config: AgentConfig):
        self.cfg = config
        self.device = torch.device(config.device)
        o, a, h = config.observation_dim, config.action_dim, config.hidden_dim
        sequence = config.sequence_length
        self.actor = RecurrentActor(o, a, h, sequence).to(self.device)
        self.contrastive = ContrastiveCritic(o, a, h, config.embedding_dim, sequence).to(self.device)
        self.risk = RiskCritic(o, a, h, sequence).to(self.device)
        self.recovery_actor = RecurrentActor(o, a, h, sequence).to(self.device)
        self.recovery_critic = RecoveryCritic(o, a, h, sequence).to(self.device)
        self.target_recovery_critic = copy.deepcopy(self.recovery_critic).to(self.device).eval()
        for parameter in self.target_recovery_critic.parameters():
            parameter.requires_grad_(False)

        if config.distributed:
            # Native DDP buckets overlap gradient AllReduce with backward.  Do
            # not call the legacy per-parameter DistributedContext reducer.
            kwargs = {"broadcast_buffers": False, "find_unused_parameters": False}
            if self.device.type in ("cuda", "npu"):
                kwargs["device_ids"] = [self.device.index]
            self.actor = DDP(self.actor, **kwargs)
            self.contrastive = DDP(self.contrastive, **kwargs)
            self.risk = DDP(self.risk, **kwargs)
            self.recovery_actor = DDP(self.recovery_actor, **kwargs)
            self.recovery_critic = DDP(self.recovery_critic, **kwargs)
            self.target_recovery_critic.load_state_dict(_unwrap(self.recovery_critic).state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.contrastive_opt = torch.optim.Adam(self.contrastive.parameters(), lr=config.learning_rate)
        self.risk_opt = torch.optim.Adam(self.risk.parameters(), lr=config.learning_rate)
        self.recovery_actor_opt = torch.optim.Adam(self.recovery_actor.parameters(), lr=config.learning_rate)
        self.recovery_critic_opt = torch.optim.Adam(self.recovery_critic.parameters(), lr=config.learning_rate)
        self.fused_adam_status = {}
        if self.device.type == "npu":
            # Kept lazy so the CPU-only structural smoke test is independent
            # of the full SOFA/SB3 installation.
            from mcr_sim.distributed.npu_performance import convert_to_npu_fused_adam
            for name in ("actor", "contrastive", "risk", "recovery_actor", "recovery_critic"):
                optimizer, status = convert_to_npu_fused_adam(getattr(self, f"{name}_opt"))
                setattr(self, f"{name}_opt", optimizer)
                self.fused_adam_status[name] = status
        self.update_count = 0
        self._obs_history = [deque(maxlen=config.sequence_length) for _ in range(config.num_envs)]
        self._action_history = [deque(maxlen=config.sequence_length) for _ in range(config.num_envs)]
        self._previous_action = np.zeros((config.num_envs, a), dtype=np.float32)
        self._recovery_remaining = np.zeros(config.num_envs, dtype=np.int32)

    def reset_envs(self, dones) -> None:
        for index, done in enumerate(np.asarray(dones, dtype=bool)):
            if done:
                self._obs_history[index].clear()
                self._action_history[index].clear()
                self._previous_action[index].fill(0.0)
                self._recovery_remaining[index] = 0

    def _online_sequences(self, observations: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_batch, action_batch = [], []
        for index, current in enumerate(np.asarray(observations, dtype=np.float32)):
            self._obs_history[index].append(current.copy())
            self._action_history[index].append(self._previous_action[index].copy())
            obs = list(self._obs_history[index])
            acts = list(self._action_history[index])
            while len(obs) < self.cfg.sequence_length:
                obs.insert(0, np.zeros(self.cfg.observation_dim, dtype=np.float32))
                acts.insert(0, np.zeros(self.cfg.action_dim, dtype=np.float32))
            obs_batch.append(np.asarray(obs, dtype=np.float32))
            action_batch.append(np.asarray(acts, dtype=np.float32))
        return (torch.as_tensor(np.asarray(obs_batch), device=self.device), torch.as_tensor(np.asarray(action_batch), device=self.device))

    @torch.no_grad()
    def act(self, observations: np.ndarray, sdf_warning: Optional[np.ndarray] = None, deterministic: bool = False):
        obs, previous = self._online_sequences(observations)
        task_actions, _, task_means = _sample_actor(self.actor, obs, previous, deterministic=deterministic)
        risk_logits = self.risk(obs, previous, task_actions)
        predicted_risk = torch.sigmoid(risk_logits[:, -1, 0]).detach().cpu().numpy()
        warnings = np.zeros(self.cfg.num_envs, dtype=np.float32) if sdf_warning is None else np.asarray(sdf_warning, dtype=np.float32)
        activate = (predicted_risk >= self.cfg.recovery_gate_threshold) | (warnings >= self.cfg.recovery_gate_threshold)
        self._recovery_remaining = np.maximum(self._recovery_remaining - 1, 0)
        self._recovery_remaining[activate] = int(self.cfg.recovery_min_steps)
        recover = self._recovery_remaining > 0
        recovery_actions, _, recovery_means = _sample_actor(self.recovery_actor, obs, previous, deterministic=deterministic)
        # Actors return an action for every element of the recurrent context;
        # the environment receives only the action for the newest observation.
        task_current = (task_means if deterministic else task_actions)[:, -1, :]
        recovery_current = (recovery_means if deterministic else recovery_actions)[:, -1, :]
        chosen = torch.where(
            torch.as_tensor(recover, device=self.device).view(-1, 1),
            recovery_current,
            task_current,
        )
        actions = chosen.detach().cpu().numpy().astype(np.float32)
        self._previous_action[:] = actions
        return actions, predicted_risk.astype(np.float32), recover.astype(np.float32)

    def _tensor_batch(self, batch):
        return {key: torch.as_tensor(getattr(batch, key), device=self.device, dtype=torch.float32) for key in batch.__dataclass_fields__}

    def update(self, batch) -> Dict[str, float]:
        data = self._tensor_batch(batch)
        obs, actions, next_obs = data["obs"], data["actions"], data["next_obs"]
        previous = torch.cat((torch.zeros_like(actions[:, :1]), actions[:, :-1]), dim=1)
        goals = data["future_goals"].unsqueeze(-1)
        last_action, last_goal = actions[:, -1:], goals[:, -1:]

        # Contrastive future-goal classification. Positives are feasible future
        # positions from the same unbroken episode; all other batch futures are
        # negatives. This is the actual learning signal for the main actor.
        query, goal_embed, temperature = self.contrastive(obs, previous, last_action, last_goal, True)
        query, goal_embed = query.squeeze(1), goal_embed.squeeze(1)
        logits = query @ goal_embed.t() / temperature
        labels = torch.arange(logits.shape[0], device=self.device)
        contrastive_loss = F.cross_entropy(logits, labels)
        self.contrastive_opt.zero_grad(set_to_none=True)
        contrastive_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.contrastive.parameters(), 10.0)
        self.contrastive_opt.step()

        risk_logits = self.risk(obs, previous, last_action)
        risk_loss = F.binary_cross_entropy_with_logits(risk_logits, data["risk_targets"][:, -1:].unsqueeze(-1))
        self.risk_opt.zero_grad(set_to_none=True)
        risk_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.risk.parameters(), 10.0)
        self.risk_opt.step()

        # Direct policy optimisation: seek future-goal reachability while the
        # risk model is frozen for this actor update.
        for parameter in list(self.risk.parameters()) + list(self.contrastive.parameters()):
            parameter.requires_grad_(False)
        proposed, log_prob, _ = _sample_actor(self.actor, obs, previous)
        reachability = self.contrastive(obs, previous, proposed, last_goal).mean()
        risk_cost = torch.sigmoid(self.risk(obs, previous, proposed)).mean()
        actor_loss = -reachability + self.cfg.risk_weight * risk_cost + self.cfg.entropy_weight * log_prob.mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_opt.step()
        for parameter in list(self.risk.parameters()) + list(self.contrastive.parameters()):
            parameter.requires_grad_(True)

        # Recovery SAC has its own critic and reward: leaving danger safely is
        # rewarded; an out-of-vessel event is strongly negative. It never
        # overwrites the main actor's objective.
        with torch.no_grad():
            next_previous = actions
            next_action, next_logp, _ = _sample_actor(self.recovery_actor, next_obs, next_previous)
            target_q1, target_q2 = self.target_recovery_critic(next_obs, next_previous, next_action)
            target = data["recovery_rewards"][:, -1:].unsqueeze(-1) + self.cfg.gamma * (1.0 - data["dones"][:, -1:].unsqueeze(-1)) * (torch.minimum(target_q1, target_q2) - self.cfg.entropy_weight * next_logp)
        q1, q2 = self.recovery_critic(obs, previous, last_action)
        recovery_critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.recovery_critic_opt.zero_grad(set_to_none=True)
        recovery_critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.recovery_critic.parameters(), 10.0)
        self.recovery_critic_opt.step()
        for parameter in self.recovery_critic.parameters():
            parameter.requires_grad_(False)
        rec_action, rec_logp, _ = _sample_actor(self.recovery_actor, obs, previous)
        rec_q1, rec_q2 = self.recovery_critic(obs, previous, rec_action)
        recovery_actor_loss = (self.cfg.entropy_weight * rec_logp - torch.minimum(rec_q1, rec_q2)).mean()
        self.recovery_actor_opt.zero_grad(set_to_none=True)
        recovery_actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.recovery_actor.parameters(), 10.0)
        self.recovery_actor_opt.step()
        for parameter in self.recovery_critic.parameters():
            parameter.requires_grad_(True)
        with torch.no_grad():
            for target_p, p in zip(self.target_recovery_critic.parameters(), _unwrap(self.recovery_critic).parameters()):
                target_p.mul_(1.0 - self.cfg.tau).add_(p, alpha=self.cfg.tau)
        self.update_count += 1
        return {"contrastive_loss": float(contrastive_loss.detach().cpu()), "risk_loss": float(risk_loss.detach().cpu()), "actor_loss": float(actor_loss.detach().cpu()), "risk_prediction": float(torch.sigmoid(risk_logits).mean().detach().cpu()), "recovery_critic_loss": float(recovery_critic_loss.detach().cpu()), "recovery_actor_loss": float(recovery_actor_loss.detach().cpu()), "reachability": float(reachability.detach().cpu())}

    def state_dict(self):
        return {"config": self.cfg.__dict__, "fused_adam": self.fused_adam_status, "actor": _unwrap(self.actor).state_dict(), "contrastive": _unwrap(self.contrastive).state_dict(), "risk": _unwrap(self.risk).state_dict(), "recovery_actor": _unwrap(self.recovery_actor).state_dict(), "recovery_critic": _unwrap(self.recovery_critic).state_dict(), "target_recovery_critic": self.target_recovery_critic.state_dict(), "optimizers": {"actor": self.actor_opt.state_dict(), "contrastive": self.contrastive_opt.state_dict(), "risk": self.risk_opt.state_dict(), "recovery_actor": self.recovery_actor_opt.state_dict(), "recovery_critic": self.recovery_critic_opt.state_dict()}, "updates": self.update_count}
