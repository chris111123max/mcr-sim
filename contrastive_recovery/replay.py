"""Episode-preserving sequence replay for the contrastive/recovery experiment."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class SequenceBatch:
    obs: np.ndarray
    actions: np.ndarray
    next_obs: np.ndarray
    dones: np.ndarray
    recovery_rewards: np.ndarray
    risk_targets: np.ndarray
    future_goals: np.ndarray


class EpisodeSequenceReplay:
    """CPU episode replay.

    Ring buffers are convenient for ordinary SAC but can silently splice two
    unrelated trajectories when an RNN sequence crosses their write boundary.
    This store evicts complete old episodes only, which is more important than
    a few percent of memory efficiency for contrastive future sampling.
    """

    def __init__(self, capacity: int, num_envs: int, observation_dim: int, action_dim: int):
        self.capacity = int(capacity)
        self.num_envs = int(num_envs)
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self._pending: List[Dict[str, list]] = [self._empty_episode() for _ in range(num_envs)]
        self._episodes = deque()
        self._size = 0

    @staticmethod
    def _empty_episode() -> Dict[str, list]:
        return {key: [] for key in ("obs", "actions", "next_obs", "dones", "recovery_rewards", "risks", "goals")}

    @property
    def size(self) -> int:
        return int(self._size)

    @property
    def episode_count(self) -> int:
        return len(self._episodes)

    def add_batch(self, obs, actions, next_obs, dones, recovery_rewards, risks, goals) -> None:
        for index in range(self.num_envs):
            episode = self._pending[index]
            episode["obs"].append(np.asarray(obs[index], dtype=np.float32).copy())
            episode["actions"].append(np.asarray(actions[index], dtype=np.float32).copy())
            episode["next_obs"].append(np.asarray(next_obs[index], dtype=np.float32).copy())
            episode["dones"].append(float(bool(dones[index])))
            episode["recovery_rewards"].append(float(recovery_rewards[index]))
            episode["risks"].append(float(risks[index]))
            episode["goals"].append(float(goals[index]))
            if bool(dones[index]):
                self._commit(index)

    def _commit(self, index: int) -> None:
        pending = self._pending[index]
        length = len(pending["obs"])
        if length:
            episode = {key: np.asarray(value, dtype=np.float32) for key, value in pending.items()}
            self._episodes.append(episode)
            self._size += length
            while self._episodes and self._size > self.capacity:
                removed = self._episodes.popleft()
                self._size -= int(len(removed["obs"]))
        self._pending[index] = self._empty_episode()

    def can_sample(self, batch_size: int, sequence_length: int) -> bool:
        eligible = sum(len(ep["obs"]) >= int(sequence_length) for ep in self._episodes)
        return self._size >= max(int(batch_size), int(sequence_length) * 4) and eligible > 0

    def sample(self, batch_size: int, sequence_length: int, future_horizon: int, risk_horizon: int) -> SequenceBatch:
        eligible = [ep for ep in self._episodes if len(ep["obs"]) >= sequence_length]
        if not eligible:
            raise RuntimeError("Sequence replay has no complete episode long enough to sample.")
        rng = np.random.default_rng()
        selected = [eligible[int(rng.integers(len(eligible)))] for _ in range(int(batch_size))]
        result = {key: [] for key in ("obs", "actions", "next_obs", "dones", "recovery_rewards", "risk_targets", "future_goals")}
        for ep in selected:
            length = len(ep["obs"])
            start = int(rng.integers(0, length - sequence_length + 1))
            stop = start + sequence_length
            for key in ("obs", "actions", "next_obs", "dones", "recovery_rewards"):
                result[key].append(ep[key][start:stop])
            risks = ep["risks"]
            risk_values = []
            goals = []
            for t in range(start, stop):
                risk_end = min(length, t + max(1, int(risk_horizon)))
                risk_values.append(float(np.max(risks[t:risk_end])))
                future_low = min(length - 1, t + 1)
                future_high = min(length - 1, t + max(1, int(future_horizon)))
                future_index = int(rng.integers(future_low, future_high + 1))
                goals.append(float(ep["goals"][future_index]))
            result["risk_targets"].append(risk_values)
            result["future_goals"].append(goals)
        return SequenceBatch(**{key: np.asarray(value, dtype=np.float32) for key, value in result.items()})
