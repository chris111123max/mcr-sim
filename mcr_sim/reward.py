"""Minimal stEVE navigation reward used by the magnetic environment."""
from dataclasses import dataclass
from typing import Optional

from .config import RewardConfig


@dataclass(frozen=True)
class RewardBreakdown:
    target: float
    path_delta: float
    step: float

    @property
    def total(self) -> float:
        return float(self.target + self.path_delta + self.step)


class StEVEReward:
    """Reward with exactly target, path-length delta and time-step terms."""

    def __init__(self, config: RewardConfig = RewardConfig()):
        self.config = config
        self._previous_path_length_mm: Optional[float] = None

    def reset(self, path_length_mm: float) -> None:
        self._previous_path_length_mm = float(path_length_mm)

    def step(self, path_length_mm: float, target_reached: bool) -> RewardBreakdown:
        current = float(path_length_mm)
        if self._previous_path_length_mm is None:
            raise RuntimeError("Reward must be reset before step")
        result = RewardBreakdown(
            target=self.config.target if target_reached else 0.0,
            path_delta=(self._previous_path_length_mm - current)
            * self.config.path_delta_per_mm,
            step=self.config.step,
        )
        self._previous_path_length_mm = current
        return result
