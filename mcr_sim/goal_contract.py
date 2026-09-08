"""Pure NumPy contract shared by rollout, HER and regression tests.

Route geometry describes a fixed selected path, not a movable goal.  The
goal-dependent distance at index 9 is replaced whenever a goal is relabelled.
The last two features are achieved route fraction and desired route fraction.
"""
from dataclasses import dataclass
import numpy as np


def condition_observation(base, achieved, goal):
    obs = np.asarray(base, dtype=np.float32).copy()
    achieved = np.asarray(achieved, dtype=np.float32)
    goal = np.asarray(goal, dtype=np.float32)
    if obs.shape[-1] < 12:
        raise ValueError("Expected MCR state with route distance at index 9")
    obs[..., 9] = np.clip(goal - achieved, -1.0, 1.0)
    tail = np.stack(np.broadcast_arrays(achieved, goal), axis=-1)
    return np.concatenate([obs, tail], axis=-1).astype(np.float32)


def relabel_observation(obs, goal):
    obs = np.asarray(obs, dtype=np.float32)
    return condition_observation(obs[..., :-2], obs[..., -2], goal)


@dataclass(frozen=True)
class GoalReward:
    gamma: float = 0.999
    step_cost: float = 0.002
    potential_scale: float = 5.0
    success_bonus: float = 20.0
    failure_penalty: float = 20.0

    def __post_init__(self):
        if not np.all(np.isfinite([
            self.gamma, self.step_cost, self.potential_scale, self.success_bonus, self.failure_penalty
        ])):
            raise ValueError("Reward parameters must be finite")
        if not 0 < self.gamma < 1:
            raise ValueError("Goal reward requires 0 < gamma < 1")
        if min(self.step_cost, self.potential_scale, self.success_bonus, self.failure_penalty) < 0:
            raise ValueError("Reward magnitudes must be nonnegative")

    def potential(self, achieved, goal):
        # Non-negative progress makes a stationary step slightly negative for
        # gamma < 1. Negative remaining distance creates a positive living
        # reward at rest, which can dominate a small step cost.
        return self.potential_scale * np.minimum(
            np.maximum(np.asarray(achieved), 0.0), np.asarray(goal)
        )

    def __call__(self, previous, achieved, goal, success, terminal, safety=0.0):
        terminal = np.asarray(terminal, dtype=bool)
        success = np.asarray(success, dtype=bool)
        # Finite-horizon tasks end at success, failure or time limit.  No
        # bootstrap across any of those boundaries; absorbing potential is 0.
        before = self.potential(previous, goal)
        after = np.where(terminal, 0.0, self.potential(achieved, goal))
        shaping = self.gamma * after - before
        base = np.where(success, self.success_bonus,
                        np.where(terminal, -self.failure_penalty, -self.step_cost))
        return np.asarray(base + shaping + safety, dtype=np.float32)
