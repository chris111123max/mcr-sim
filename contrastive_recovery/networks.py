"""Small recurrent networks; deliberately fixed-shape for Ascend eager execution."""

from __future__ import annotations

import math
import torch
from torch import nn
from torch.distributions import Normal


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, out_dim))


class RecurrentTrunk(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.input = nn.Linear(observation_dim + action_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.hidden_dim = int(hidden_dim)

    def forward(self, obs: torch.Tensor, previous_actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat((obs, previous_actions), dim=-1)
        x = torch.nn.functional.silu(self.input(x))
        return self.gru(x)[0]


class RecurrentActor(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.trunk = RecurrentTrunk(observation_dim, action_dim, hidden_dim)
        self.head = _mlp(hidden_dim, hidden_dim, action_dim * 2)
        self.action_dim = int(action_dim)

    def distribution(self, obs: torch.Tensor, previous_actions: torch.Tensor):
        values = self.head(self.trunk(obs, previous_actions))
        mean, log_std = values.chunk(2, dim=-1)
        return mean, torch.clamp(log_std, -5.0, 1.5)

    def forward(self, obs: torch.Tensor, previous_actions: torch.Tensor):
        """DDP-visible distribution path; sampling is performed by the agent."""
        return self.distribution(obs, previous_actions)

    def sample(self, obs: torch.Tensor, previous_actions: torch.Tensor, deterministic: bool = False):
        mean, log_std = self.distribution(obs, previous_actions)
        normal = Normal(mean, log_std.exp())
        raw = mean if deterministic else normal.rsample()
        action = torch.tanh(raw)
        log_prob = normal.log_prob(raw) - torch.log(1.0 - action.square() + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True), torch.tanh(mean)


class ContrastiveCritic(nn.Module):
    """C(s, a, g) score trained by in-batch future-goal classification."""

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int, embedding_dim: int):
        super().__init__()
        self.trunk = RecurrentTrunk(observation_dim, action_dim, hidden_dim)
        self.query = _mlp(hidden_dim + action_dim, hidden_dim, embedding_dim)
        self.goal = _mlp(1, hidden_dim, embedding_dim)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(0.2)))

    def query_embedding(self, obs, previous_actions, action):
        h = self.trunk(obs, previous_actions)
        return torch.nn.functional.normalize(self.query(torch.cat((h, action), dim=-1)), dim=-1)

    def goal_embedding(self, goal):
        return torch.nn.functional.normalize(self.goal(goal), dim=-1)

    def score(self, obs, previous_actions, action, goal):
        q = self.query_embedding(obs, previous_actions, action)
        g = self.goal_embedding(goal)
        temperature = self.log_temperature.exp().clamp(0.03, 1.0)
        return (q * g).sum(dim=-1, keepdim=True) / temperature

    def forward(self, obs, previous_actions, action, goal, return_embeddings: bool = False):
        """DDP-visible contrastive path.

        Calling public helpers directly would bypass DDP's reducer, so all
        trainable paths deliberately enter through ``forward``.
        """
        q = self.query_embedding(obs, previous_actions, action)
        g = self.goal_embedding(goal)
        temperature = self.log_temperature.exp().clamp(0.03, 1.0)
        if return_embeddings:
            return q, g, temperature
        return (q * g).sum(dim=-1, keepdim=True) / temperature


class RiskCritic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.trunk = RecurrentTrunk(observation_dim, action_dim, hidden_dim)
        self.head = _mlp(hidden_dim + action_dim, hidden_dim, 1)

    def forward(self, obs, previous_actions, action):
        h = self.trunk(obs, previous_actions)
        return self.head(torch.cat((h, action), dim=-1))


class RecoveryCritic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.trunk = RecurrentTrunk(observation_dim, action_dim, hidden_dim)
        self.q1 = _mlp(hidden_dim + action_dim, hidden_dim, 1)
        self.q2 = _mlp(hidden_dim + action_dim, hidden_dim, 1)

    def forward(self, obs, previous_actions, action):
        h = self.trunk(obs, previous_actions)
        x = torch.cat((h, action), dim=-1)
        return self.q1(x), self.q2(x)
