"""Episode metrics aligned with the stEVE Bench evaluation vocabulary."""
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class EpisodeMetrics:
    success: bool
    final_path_length_mm: float
    initial_path_length_mm: float
    path_length_ratio: float
    steps: int
    trajectory_length_m: float
    average_absolute_axial_command_m_s: float
    average_tip_speed_m_s: Optional[float]
    termination_reason: str


class EpisodeMetricTracker:
    def __init__(self, rl_dt_s: float):
        self.rl_dt_s = float(rl_dt_s)
        self.reset(np.zeros(3), 0.0)

    def reset(self, tip_m, initial_path_length_mm: float) -> None:
        self.initial_path_length_mm = float(initial_path_length_mm)
        self.previous_tip_m = np.asarray(tip_m, dtype=float).reshape(3).copy()
        self.trajectory_length_m = 0.0
        self.absolute_axial_speeds = []
        self.actual_tip_speeds = []
        self.steps = 0

    def step(self, tip_m, executed_insertion_m: float) -> None:
        current = np.asarray(tip_m, dtype=float).reshape(3)
        displacement = float(np.linalg.norm(current - self.previous_tip_m))
        self.trajectory_length_m += displacement
        self.actual_tip_speeds.append(displacement / self.rl_dt_s)
        self.absolute_axial_speeds.append(
            abs(float(executed_insertion_m)) / self.rl_dt_s
        )
        self.previous_tip_m = current.copy()
        self.steps += 1

    def finish(
        self,
        *,
        success: bool,
        final_path_length_mm: float,
        reason: str,
    ) -> EpisodeMetrics:
        initial = self.initial_path_length_mm
        ratio = (
            float(final_path_length_mm) / initial
            if initial > 1e-12
            else (0.0 if final_path_length_mm <= 1e-12 else float("inf"))
        )
        axial = float(np.mean(self.absolute_axial_speeds)) if self.steps else 0.0
        tip = float(np.mean(self.actual_tip_speeds)) if self.steps else None
        return EpisodeMetrics(
            success=bool(success),
            final_path_length_mm=float(final_path_length_mm),
            initial_path_length_mm=initial,
            path_length_ratio=ratio,
            steps=self.steps,
            trajectory_length_m=self.trajectory_length_m,
            average_absolute_axial_command_m_s=axial,
            average_tip_speed_m_s=tip,
            termination_reason=str(reason),
        )
