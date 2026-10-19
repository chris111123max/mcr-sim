"""Episode success and truncation rules for the unified task."""
from dataclasses import dataclass
import numpy as np

from .config import TaskConfig


@dataclass(frozen=True)
class TaskOutcome:
    terminated: bool
    truncated: bool
    success: bool
    reason: str
    distance_to_target_m: float


class NavigationTask:
    def __init__(self, mode: str = "train", config: TaskConfig = TaskConfig()):
        if mode not in ("train", "eval"):
            raise ValueError("mode must be 'train' or 'eval'")
        self.mode = mode
        self.config = config
        self.steps = 0

    @property
    def max_steps(self) -> int:
        return (
            self.config.max_episode_steps_train
            if self.mode == "train"
            else self.config.max_episode_steps_eval
        )

    def reset(self) -> None:
        self.steps = 0

    def step(
        self,
        tip_m,
        target_m,
        *,
        vessel_end: bool = False,
        simulation_error: bool = False,
    ) -> TaskOutcome:
        self.steps += 1
        distance = float(
            np.linalg.norm(
                np.asarray(tip_m, dtype=float).reshape(3)
                - np.asarray(target_m, dtype=float).reshape(3)
            )
        )
        # The requested success condition is intentionally strict (<, not <=).
        success = distance < self.config.target_threshold_m
        if simulation_error:
            return TaskOutcome(False, True, False, "simulation_error", distance)
        if success:
            return TaskOutcome(True, False, True, "target_reached", distance)
        if vessel_end:
            return TaskOutcome(False, True, False, "vessel_end", distance)
        if self.steps >= self.max_steps:
            return TaskOutcome(False, True, False, "max_steps", distance)
        return TaskOutcome(False, False, False, "running", distance)
