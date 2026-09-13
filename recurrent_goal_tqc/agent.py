"""Manual-GRU, truncated-quantile actor critic with an auxiliary risk critic."""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP


def _unwrap(module):
    return module.module if isinstance(module, DDP) else module


def _zero_grad(optimizer):
    kind = type(optimizer)
    if kind.__name__.startswith("NpuFused") or kind.__module__.startswith("torch_npu.optim"):
        optimizer.zero_grad()
    else:
        optimizer.zero_grad(set_to_none=True)


class ManualGRU(nn.Module):
    """Actual recurrent state updates using basic ops supported by torch-npu 2.2.

    nn.GRU dispatched to an unsupported DynamicGRUV2 kernel on the user's 910B3.
    The explicit cell equations retain recurrent memory without that operator.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input = nn.Linear(input_dim, 3 * hidden_dim)
        self.hidden = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, obs, previous_actions):
        x = torch.cat((obs, previous_actions), dim=-1)
        h = x.new_zeros((x.shape[0], self.hidden_dim))
        for t in range(x.shape[1]):
            xr, xz, xn = self.input(x[:, t]).chunk(3, dim=-1)
            hr, hz, hn = self.hidden(h).chunk(3, dim=-1)
            reset = torch.sigmoid(xr + hr)
            update = torch.sigmoid(xz + hz)
            candidate = torch.tanh(xn + reset * hn)
            h = update * h + (1.0 - update) * candidate
        return h


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.encoder = ManualGRU(obs_dim + action_dim, hidden_dim)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                                  nn.Linear(hidden_dim, 2 * action_dim))

    def forward(self, obs, previous_actions):
        mean, log_std = self.head(self.encoder(obs, previous_actions)).chunk(2, dim=-1)
        return mean, log_std.clamp(-5.0, 1.0)


class QuantileEnsemble(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, n_critics, n_quantiles):
        super().__init__()
        self.encoders = nn.ModuleList(
            ManualGRU(obs_dim + action_dim, hidden_dim) for _ in range(n_critics))
        self.heads = nn.ModuleList(
            nn.Sequential(nn.Linear(hidden_dim + action_dim, hidden_dim), nn.SiLU(),
                          nn.Linear(hidden_dim, n_quantiles)) for _ in range(n_critics))

    def forward(self, obs, previous_actions, action):
        return torch.stack([
            head(torch.cat((encoder(obs, previous_actions), action), dim=-1))
            for encoder, head in zip(self.encoders, self.heads)
        ], dim=1)


class RiskCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.encoder = ManualGRU(obs_dim + action_dim, hidden_dim)
        self.head = nn.Sequential(nn.Linear(hidden_dim + action_dim, hidden_dim),
                                  nn.SiLU(), nn.Linear(hidden_dim, 1))

    def forward(self, obs, previous_actions, action):
        return self.head(torch.cat((self.encoder(obs, previous_actions), action), dim=-1))


def sample_actor(actor, obs, previous_actions, deterministic=False):
    mean, log_std = actor(obs, previous_actions)
    normal = torch.distributions.Normal(mean, log_std.exp())
    raw = mean if deterministic else normal.rsample()
    action = raw.tanh()
    logp = (normal.log_prob(raw) - torch.log(1.0 - action.square() + 1e-6)).sum(-1, keepdim=True)
    return action, logp


def quantile_huber(predicted, target):
    """Pairwise quantile regression over [batch, critic, quantile] atoms."""
    n_quantiles = predicted.shape[-1]
    residual = target[:, None, None, :] - predicted[:, :, :, None]
    tau = ((torch.arange(n_quantiles, device=predicted.device, dtype=predicted.dtype)
            + 0.5) / n_quantiles).view(1, 1, n_quantiles, 1)
    huber = F.smooth_l1_loss(residual, torch.zeros_like(residual), reduction="none")
    return ((tau - (residual.detach() < 0).to(predicted.dtype)).abs() * huber).mean()


@dataclass
class Config:
    observation_dim: int
    action_dim: int
    num_envs: int
    device: str
    sequence_length: int = 16
    hidden_dim: int = 128
    n_critics: int = 3
    n_quantiles: int = 16
    drop_top_quantiles_per_critic: int = 2
    learning_rate: float = 3e-4
    gamma: float = 0.995
    tau: float = 0.005
    entropy_coef: float = 0.05
    risk_weight: float = 0.05
    risk_warmup_updates: int = 2000
    distributed: bool = False


class RecurrentGoalTQC:
    def __init__(self, config: Config):
        self.cfg = config
        self.device = torch.device(config.device)
        o, a, h = config.observation_dim, config.action_dim, config.hidden_dim
        self.actor = Actor(o, a, h).to(self.device)
        self.critics = QuantileEnsemble(o, a, h, config.n_critics, config.n_quantiles).to(self.device)
        self.risk = RiskCritic(o, a, h).to(self.device)
        self.target_critics = copy.deepcopy(self.critics).to(self.device).eval()
        for p in self.target_critics.parameters():
            p.requires_grad_(False)
        if config.distributed:
            kwargs = {"broadcast_buffers": False, "find_unused_parameters": False}
            if self.device.type in ("npu", "cuda"):
                kwargs["device_ids"] = [self.device.index]
            self.actor = DDP(self.actor, **kwargs)
            self.critics = DDP(self.critics, **kwargs)
            self.risk = DDP(self.risk, **kwargs)
            self.target_critics.load_state_dict(_unwrap(self.critics).state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critics_opt = torch.optim.Adam(self.critics.parameters(), lr=config.learning_rate)
        self.risk_opt = torch.optim.Adam(self.risk.parameters(), lr=config.learning_rate)
        self.fused_adam = {}
        if self.device.type == "npu":
            from mcr_sim.distributed.npu_performance import convert_to_npu_fused_adam
            for name in ("actor", "critics", "risk"):
                opt, status = convert_to_npu_fused_adam(getattr(self, name + "_opt"))
                setattr(self, name + "_opt", opt)
                self.fused_adam[name] = status
        self.hist_obs = [deque(maxlen=config.sequence_length) for _ in range(config.num_envs)]
        self.hist_actions = [deque(maxlen=config.sequence_length) for _ in range(config.num_envs)]
        self.previous_action = np.zeros((config.num_envs, a), dtype=np.float32)
        self.update_count = 0

    def reset_envs(self, dones):
        for i, done in enumerate(dones):
            if done:
                self.hist_obs[i].clear()
                self.hist_actions[i].clear()
                self.previous_action[i].fill(0.0)

    @torch.no_grad()
    def act(self, observations, deterministic=False):
        obs_batch, prev_batch = [], []
        for i, current in enumerate(np.asarray(observations, dtype=np.float32)):
            self.hist_obs[i].append(current.copy())
            self.hist_actions[i].append(self.previous_action[i].copy())
            pad = self.cfg.sequence_length - len(self.hist_obs[i])
            obs_batch.append(np.asarray(
                [np.zeros_like(current) for _ in range(pad)] + list(self.hist_obs[i]), dtype=np.float32))
            prev_batch.append(np.asarray(
                [np.zeros_like(self.previous_action[i]) for _ in range(pad)]
                + list(self.hist_actions[i]), dtype=np.float32))
        obs = torch.as_tensor(np.asarray(obs_batch), device=self.device)
        previous = torch.as_tensor(np.asarray(prev_batch), device=self.device)
        action, _ = sample_actor(self.actor, obs, previous, deterministic)
        risk = torch.sigmoid(self.risk(obs, previous, action))[:, 0]
        actions = action.cpu().numpy().astype(np.float32)
        self.previous_action[:] = actions
        return actions, risk.cpu().numpy().astype(np.float32)

    def update(self, batch):
        data = {key: torch.as_tensor(getattr(batch, key), device=self.device,
                                     dtype=torch.float32) for key in batch.__dataclass_fields__}
        obs, next_obs = data["obs"], data["next_obs"]
        prev, actions = data["previous_actions"], data["actions"]
        last_action = actions[:, -1]
        with torch.no_grad():
            next_action, next_logp = sample_actor(self.actor, next_obs, actions)
            atoms = self.target_critics(next_obs, actions, next_action)
            kept = self.cfg.n_critics * (self.cfg.n_quantiles - self.cfg.drop_top_quantiles_per_critic)
            target_atoms = atoms.reshape(atoms.shape[0], -1).sort(dim=1).values[:, :kept]
            target = data["rewards"] + self.cfg.gamma * (1.0 - data["dones"]) * (
                target_atoms - self.cfg.entropy_coef * next_logp)
        predicted = self.critics(obs, prev, last_action)
        critic_loss = quantile_huber(predicted, target)
        _zero_grad(self.critics_opt)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critics.parameters(), 10.0)
        self.critics_opt.step()

        risk_logits = self.risk(obs, prev, last_action)
        risk_loss = F.binary_cross_entropy_with_logits(risk_logits, data["risk_targets"])
        _zero_grad(self.risk_opt)
        risk_loss.backward()
        nn.utils.clip_grad_norm_(self.risk.parameters(), 10.0)
        self.risk_opt.step()

        for p in self.critics.parameters():
            p.requires_grad_(False)
        for p in self.risk.parameters():
            p.requires_grad_(False)
        proposed, logp = sample_actor(self.actor, obs, prev)
        q = self.critics(obs, prev, proposed).mean(dim=-1).min(dim=1).values.mean()
        risk_cost = torch.sigmoid(self.risk(obs, prev, proposed)).mean()
        risk_weight = self.cfg.risk_weight if self.update_count >= self.cfg.risk_warmup_updates else 0.0
        actor_loss = self.cfg.entropy_coef * logp.mean() - q + risk_weight * risk_cost
        _zero_grad(self.actor_opt)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_opt.step()
        for p in self.critics.parameters():
            p.requires_grad_(True)
        for p in self.risk.parameters():
            p.requires_grad_(True)
        with torch.no_grad():
            for target_p, p in zip(self.target_critics.parameters(), _unwrap(self.critics).parameters()):
                target_p.mul_(1.0 - self.cfg.tau).add_(p, alpha=self.cfg.tau)
        self.update_count += 1
        names = ("critic_loss", "risk_loss", "actor_loss", "q_mean", "risk_prediction",
                 "risk_target", "her_fraction", "target_atom_mean")
        values = (critic_loss, risk_loss, actor_loss, q,
                  torch.sigmoid(risk_logits).mean(), data["risk_targets"].mean(),
                  data["her_mask"].mean(), target_atoms.mean())
        merged = torch.stack([value.detach().reshape(()) for value in values]).cpu().tolist()
        return dict(zip(names, merged))

    def state_dict(self):
        return {"config": self.cfg.__dict__, "updates": self.update_count,
                "actor": _unwrap(self.actor).state_dict(),
                "critics": _unwrap(self.critics).state_dict(),
                "target_critics": self.target_critics.state_dict(),
                "risk": _unwrap(self.risk).state_dict(),
                "optimizers": {name: getattr(self, name + "_opt").state_dict()
                               for name in ("actor", "critics", "risk")}}

    def load_state_dict(self, state, *, load_optimizers=False):
        _unwrap(self.actor).load_state_dict(state["actor"])
        _unwrap(self.critics).load_state_dict(state["critics"])
        self.target_critics.load_state_dict(state["target_critics"])
        _unwrap(self.risk).load_state_dict(state["risk"])
        if load_optimizers:
            for name in ("actor", "critics", "risk"):
                getattr(self, name + "_opt").load_state_dict(state["optimizers"][name])
        self.update_count = int(state["updates"])
