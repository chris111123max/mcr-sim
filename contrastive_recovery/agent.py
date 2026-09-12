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


def _zero_grad(optimizer) -> None:
    """Use the zeroing API supported by the deployed NpuFusedAdam."""
    optimizer_type = type(optimizer)
    fused = (
        optimizer_type.__name__.startswith("NpuFused")
        or optimizer_type.__module__.startswith("torch_npu.optim")
    )
    if fused:
        optimizer.zero_grad()
    else:
        optimizer.zero_grad(set_to_none=True)


def _choose_recovery(eligible, task_risk, recovery_risk, margin):
    """Never force a learned recovery action with higher predicted risk."""
    return np.asarray(eligible, dtype=bool) & (np.asarray(recovery_risk) + margin < np.asarray(task_risk))


def _bounded_goal_margin(future_score: torch.Tensor, current_score: torch.Tensor) -> torch.Tensor:
    """A common logit offset cannot inflate this bounded auxiliary signal."""
    return torch.sigmoid((future_score - current_score) / 10.0)


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
    task_entropy_weight: float = 0.002
    risk_weight: float = 0.10
    risk_warmup_updates: int = 2_000
    risk_ramp_updates: int = 4_000
    contrastive_actor_weight: float = 0.05
    recovery_gate_threshold: float = 0.65
    recovery_min_steps: int = 12
    recovery_gate_warmup_updates: int = 2_000
    recovery_risk_margin: float = 0.05
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
        self.task_critic = RecoveryCritic(o, a, h, sequence).to(self.device)
        self.target_task_critic = copy.deepcopy(self.task_critic).to(self.device).eval()
        for parameter in self.target_task_critic.parameters():
            parameter.requires_grad_(False)
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
            self.task_critic = DDP(self.task_critic, **kwargs)
            self.target_recovery_critic.load_state_dict(_unwrap(self.recovery_critic).state_dict())
            self.target_task_critic.load_state_dict(_unwrap(self.task_critic).state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.contrastive_opt = torch.optim.Adam(self.contrastive.parameters(), lr=config.learning_rate)
        self.risk_opt = torch.optim.Adam(self.risk.parameters(), lr=config.learning_rate)
        self.recovery_actor_opt = torch.optim.Adam(self.recovery_actor.parameters(), lr=config.learning_rate)
        self.recovery_critic_opt = torch.optim.Adam(self.recovery_critic.parameters(), lr=config.learning_rate)
        self.task_critic_opt = torch.optim.Adam(self.task_critic.parameters(), lr=config.learning_rate)
        self.fused_adam_status = {}
        if self.device.type == "npu":
            # Kept lazy so the CPU-only structural smoke test is independent
            # of the full SOFA/SB3 installation.
            from mcr_sim.distributed.npu_performance import convert_to_npu_fused_adam
            for name in ("actor", "contrastive", "risk", "recovery_actor", "recovery_critic", "task_critic"):
                optimizer, status = convert_to_npu_fused_adam(getattr(self, f"{name}_opt"))
                setattr(self, f"{name}_opt", optimizer)
                self.fused_adam_status[name] = status
        self.update_count = 0
        self._obs_history = [deque(maxlen=config.sequence_length) for _ in range(config.num_envs)]
        self._action_history = [deque(maxlen=config.sequence_length) for _ in range(config.num_envs)]
        self._previous_action = np.zeros((config.num_envs, a), dtype=np.float32)
        self._recovery_remaining = np.zeros(config.num_envs, dtype=np.int32)
        self.last_recovery_candidate_risk = np.zeros(config.num_envs, dtype=np.float32)
        self.last_recovery_rejected = np.zeros(config.num_envs, dtype=bool)

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
        warnings = np.zeros(self.cfg.num_envs, dtype=np.float32) if sdf_warning is None else np.asarray(sdf_warning, dtype=np.float32)
        task_current = (task_means if deterministic else task_actions)[:, -1, :]
        task_risk = torch.sigmoid(self.risk(obs, previous, task_current.unsqueeze(1))[:, -1, 0]).detach().cpu().numpy()
        # A randomly initialised BCE critic outputs ~0.5. Let the main policy
        # collect/learn safety transitions first; otherwise recovery would
        # incorrectly seize every environment at startup.  After warm-up,
        # either a calibrated learned-risk prediction or measured SDF warning
        # can activate the independently trained recovery policy.
        learned_gate = task_risk >= self.cfg.recovery_gate_threshold if self.update_count >= self.cfg.recovery_gate_warmup_updates else np.zeros_like(task_risk, dtype=bool)
        hard_warning_gate = warnings >= self.cfg.recovery_gate_threshold if self.update_count >= self.cfg.recovery_gate_warmup_updates else np.zeros_like(task_risk, dtype=bool)
        activate = learned_gate | hard_warning_gate
        self._recovery_remaining = np.maximum(self._recovery_remaining - 1, 0)
        self._recovery_remaining[activate] = int(self.cfg.recovery_min_steps)
        recovery_actions, _, recovery_means = _sample_actor(self.recovery_actor, obs, previous, deterministic=deterministic)
        # Actors return an action for every element of the recurrent context;
        # the environment receives only the action for the newest observation.
        recovery_current = (recovery_means if deterministic else recovery_actions)[:, -1, :]
        # The old gate forced the recovery actor to control for at least 12
        # steps, even when its proposed action was predicted to be riskier.
        # Keep the warning/hysteresis, but compare both RL candidates on the
        # same observation before each handoff.
        recovery_risk = torch.sigmoid(self.risk(obs, previous, recovery_current.unsqueeze(1))[:, -1, 0]).detach().cpu().numpy()
        eligible = self._recovery_remaining > 0
        recover = _choose_recovery(eligible, task_risk, recovery_risk, self.cfg.recovery_risk_margin)
        self.last_recovery_candidate_risk[:] = recovery_risk
        self.last_recovery_rejected[:] = eligible & ~recover
        predicted_risk = task_risk
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
        # With a scalar progress goal, two trajectories can legitimately have
        # the same achieved goal. Treating one as the other's negative gives
        # contradictory supervision and can collapse the goal embedding.
        same_goal = (last_goal[:, 0, 0][:, None] - last_goal[:, 0, 0][None, :]).abs() < 0.01
        false_negative_mask = same_goal & ~torch.eye(logits.shape[0], dtype=torch.bool, device=self.device)
        logits = logits.masked_fill(false_negative_mask, -1e4)
        contrastive_loss = F.cross_entropy(logits, labels)
        _zero_grad(self.contrastive_opt)
        contrastive_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.contrastive.parameters(), 10.0)
        self.contrastive_opt.step()

        risk_logits = self.risk(obs, previous, last_action)
        risk_loss = F.binary_cross_entropy_with_logits(risk_logits, data["risk_targets"][:, -1:].unsqueeze(-1))
        _zero_grad(self.risk_opt)
        risk_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.risk.parameters(), 10.0)
        self.risk_opt.step()

        # Bellman task critic uses actual progress on the physical route.  It
        # supplies a grounded action gradient even before any trajectory has
        # reached the real endpoint (the contrastive goal=1 is then unseen).
        with torch.no_grad():
            next_action_task, next_logp_task, _ = _sample_actor(self.actor, next_obs, actions)
            tq1, tq2 = self.target_task_critic(next_obs, actions, next_action_task)
            task_target = data["task_rewards"][:, -1:].unsqueeze(-1) + self.cfg.gamma * (1.0 - data["dones"][:, -1:].unsqueeze(-1)) * (torch.minimum(tq1, tq2) - self.cfg.task_entropy_weight * next_logp_task)
        task_q1, task_q2 = self.task_critic(obs, previous, last_action)
        task_critic_loss = F.mse_loss(task_q1, task_target) + F.mse_loss(task_q2, task_target)
        _zero_grad(self.task_critic_opt)
        task_critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.task_critic.parameters(), 10.0)
        self.task_critic_opt.step()

        # Direct policy optimisation: seek future-goal reachability while the
        # risk model is frozen for this actor update.
        frozen = list(self.risk.parameters()) + list(self.contrastive.parameters()) + list(self.task_critic.parameters())
        for parameter in frozen:
            parameter.requires_grad_(False)
        proposed, log_prob, _ = _sample_actor(self.actor, obs, previous)
        # Train the actor toward *observed, forward* future goals.  Requiring a
        # 0.90-progress example made this term identically zero in the pilot.
        # Do not score an unsupported goal=1 or reward a backward/unchanged
        # future goal as if it were forward navigation.
        forward_goal = (last_goal - obs[:, -1:, -2:-1]).detach() >= 0.005
        # A raw contrastive logit climbed above 22 in the pilot and dominated
        # the task Q (~0.15).  Compare the reachable future against *current*
        # progress and squash the margin to [0, 1]; absolute logit scale can
        # no longer be maximized without bound by the actor.
        candidate_goals = torch.cat((last_goal, obs[:, -1:, -2:-1].detach()), dim=1)
        contrastive_scores = self.contrastive(obs, previous, proposed, candidate_goals)
        contrastive_margin = contrastive_scores[:, :1] - contrastive_scores[:, 1:2]
        reachability_scores = _bounded_goal_margin(contrastive_scores[:, :1], contrastive_scores[:, 1:2])
        forward_count = forward_goal.float().sum().clamp_min(1.0)
        reachability = (reachability_scores * forward_goal.float()).sum() / forward_count
        goal_support = float(forward_goal.float().mean().detach().cpu().item())
        effective_contrastive_weight = self.cfg.contrastive_actor_weight * min(1.0, goal_support / 0.25)
        proposed_q1, proposed_q2 = self.task_critic(obs, previous, proposed)
        task_value = torch.minimum(proposed_q1, proposed_q2).mean()
        risk_cost = torch.sigmoid(self.risk(obs, previous, proposed)).mean()
        risk_fraction = max(0.0, min(1.0, (self.update_count - self.cfg.risk_warmup_updates) / float(max(1, self.cfg.risk_ramp_updates))))
        effective_risk_weight = self.cfg.risk_weight * risk_fraction
        actor_loss = -task_value - effective_contrastive_weight * reachability + effective_risk_weight * risk_cost + self.cfg.task_entropy_weight * log_prob.mean()
        _zero_grad(self.actor_opt)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_opt.step()
        for parameter in frozen:
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
        _zero_grad(self.recovery_critic_opt)
        recovery_critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.recovery_critic.parameters(), 10.0)
        self.recovery_critic_opt.step()
        for parameter in self.recovery_critic.parameters():
            parameter.requires_grad_(False)
        rec_action, rec_logp, _ = _sample_actor(self.recovery_actor, obs, previous)
        rec_q1, rec_q2 = self.recovery_critic(obs, previous, rec_action)
        recovery_actor_loss = (self.cfg.entropy_weight * rec_logp - torch.minimum(rec_q1, rec_q2)).mean()
        _zero_grad(self.recovery_actor_opt)
        recovery_actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.recovery_actor.parameters(), 10.0)
        self.recovery_actor_opt.step()
        for parameter in self.recovery_critic.parameters():
            parameter.requires_grad_(True)
        with torch.no_grad():
            for target_p, p in zip(self.target_task_critic.parameters(), _unwrap(self.task_critic).parameters()):
                target_p.mul_(1.0 - self.cfg.tau).add_(p, alpha=self.cfg.tau)
            for target_p, p in zip(self.target_recovery_critic.parameters(), _unwrap(self.recovery_critic).parameters()):
                target_p.mul_(1.0 - self.cfg.tau).add_(p, alpha=self.cfg.tau)
        self.update_count += 1
        tensor_metrics = {
            "contrastive_loss": contrastive_loss,
            "contrastive_false_negative_rate": false_negative_mask.float().mean(),
            "risk_loss": risk_loss,
            "actor_loss": actor_loss,
            "task_critic_loss": task_critic_loss,
            "task_value": task_value,
            "risk_prediction": torch.sigmoid(risk_logits).mean(),
            "risk_target_rate": data["risk_targets"].mean(),
            "recovery_critic_loss": recovery_critic_loss,
            "recovery_actor_loss": recovery_actor_loss,
            "reachability_supported_goal": reachability,
            "contrastive_margin": contrastive_margin.mean(),
            "contrastive_future_goal_mean": last_goal.mean(),
        }
        # One NPU-to-CPU transfer instead of a separate device synchronization
        # for each diagnostic metric on every learner update.
        metric_names = list(tensor_metrics)
        metric_values = torch.stack([tensor_metrics[name].detach().reshape(()) for name in metric_names]).cpu().tolist()
        metrics = dict(zip(metric_names, metric_values))
        metrics.update({
            "risk_weight_effective": float(effective_risk_weight),
            "contrastive_weight_effective": float(effective_contrastive_weight),
            "goal_support_rate": float(goal_support),
        })
        return metrics

    def state_dict(self):
        return {
            "config": self.cfg.__dict__,
            "fused_adam": self.fused_adam_status,
            "actor": _unwrap(self.actor).state_dict(),
            "contrastive": _unwrap(self.contrastive).state_dict(),
            "risk": _unwrap(self.risk).state_dict(),
            "task_critic": _unwrap(self.task_critic).state_dict(),
            "target_task_critic": self.target_task_critic.state_dict(),
            "recovery_actor": _unwrap(self.recovery_actor).state_dict(),
            "recovery_critic": _unwrap(self.recovery_critic).state_dict(),
            "target_recovery_critic": self.target_recovery_critic.state_dict(),
            "optimizers": {
                name: getattr(self, "{}_opt".format(name)).state_dict()
                for name in ("actor", "contrastive", "risk", "task_critic", "recovery_actor", "recovery_critic")
            },
            "updates": self.update_count,
        }
