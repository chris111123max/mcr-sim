"""Read-only telemetry aggregation for production Beam safety training.

This module never participates in observation, reward, action, termination,
constraint construction, or solver control. It only consumes beam_safety
snapshots from env info and writes logger/TensorBoard scalars.
"""
from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


_COUNTER_FIELDS = (
    "physics_substeps_seen",
    "safe_free_substeps",
    "active_substeps",
    "near_wall_substeps",
    "unilateral_rows_built",
    "q_free_penetration_substeps",
    "committed_audits",
    "committed_audit_skips",
)


class BeamSafetyMetricsCollector:
    """Aggregate Beam safety info snapshots without affecting training."""

    def __init__(self, episode_window: int = 50):
        self.episode_window = max(1, int(episode_window))
        self._previous: dict[int, dict[str, float]] = {}
        self._rollout = {name: 0.0 for name in _COUNTER_FIELDS}
        self._total = {name: 0.0 for name in _COUNTER_FIELDS}
        self._rollout_worst_free_m = float("inf")
        self._rollout_worst_committed_m = float("inf")
        self._total_worst_free_m = float("inf")
        self._total_worst_committed_m = float("inf")
        self._recent_episodes = deque(maxlen=self.episode_window)
        self._episode_count = 0
        self._snapshot_count = 0

    @staticmethod
    def _finite_float(value, default=np.nan) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return float(default)
        return value if np.isfinite(value) else float(default)

    @staticmethod
    def _counter_value(snapshot: dict[str, Any], name: str) -> float:
        value = BeamSafetyMetricsCollector._finite_float(
            snapshot.get(name, 0.0), 0.0
        )
        return max(0.0, value)

    @staticmethod
    def _delta(current: float, previous: float | None) -> float:
        if previous is None or current < previous:
            return current
        return current - previous

    def observe(self, infos, dones=None) -> None:
        """Consume one vector-env step of read-only info dictionaries."""

        if infos is None:
            return
        infos = list(infos)
        if dones is None:
            done_flags = [False] * len(infos)
        else:
            done_flags = [bool(v) for v in np.asarray(dones).reshape(-1)]
            if len(done_flags) < len(infos):
                done_flags.extend([False] * (len(infos) - len(done_flags)))

        for env_index, info in enumerate(infos):
            if not isinstance(info, dict):
                continue
            snapshot = info.get("beam_safety")
            if not isinstance(snapshot, dict):
                continue

            previous = self._previous.get(env_index)
            current_counters: dict[str, float] = {}
            for name in _COUNTER_FIELDS:
                current = self._counter_value(snapshot, name)
                prev_value = None if previous is None else previous.get(name)
                delta = self._delta(current, prev_value)
                self._rollout[name] += delta
                self._total[name] += delta
                current_counters[name] = current

            worst_free = self._finite_float(
                snapshot.get("worst_free_clearance_m", np.nan)
            )
            if np.isfinite(worst_free):
                self._rollout_worst_free_m = min(
                    self._rollout_worst_free_m, worst_free
                )
                self._total_worst_free_m = min(
                    self._total_worst_free_m, worst_free
                )

            worst_committed = self._finite_float(
                snapshot.get("worst_audited_committed_clearance_m", np.nan)
            )
            if np.isfinite(worst_committed):
                self._rollout_worst_committed_m = min(
                    self._rollout_worst_committed_m, worst_committed
                )
                self._total_worst_committed_m = min(
                    self._total_worst_committed_m, worst_committed
                )

            self._snapshot_count += 1
            done = (
                bool(done_flags[env_index])
                if env_index < len(done_flags)
                else False
            )
            if done:
                episode = {
                    name: current_counters[name] for name in _COUNTER_FIELDS
                }
                episode["worst_free_clearance_m"] = worst_free
                episode[
                    "worst_audited_committed_clearance_m"
                ] = worst_committed
                self._recent_episodes.append(episode)
                self._episode_count += 1
                self._previous.pop(env_index, None)
            else:
                self._previous[env_index] = current_counters

    @staticmethod
    def _ratio(numerator: float, denominator: float) -> float:
        return (
            float(numerator / denominator)
            if denominator > 0.0
            else 0.0
        )

    @staticmethod
    def _window_mean(window, key: str) -> float:
        values = []
        for row in window:
            value = BeamSafetyMetricsCollector._finite_float(
                row.get(key, np.nan)
            )
            if np.isfinite(value):
                values.append(value)
        return float(np.mean(values)) if values else float("nan")

    @staticmethod
    def _window_min(window, key: str) -> float:
        values = []
        for row in window:
            value = BeamSafetyMetricsCollector._finite_float(
                row.get(key, np.nan)
            )
            if np.isfinite(value):
                values.append(value)
        return float(np.min(values)) if values else float("nan")

    def log_rollout(self, logger) -> None:
        """Record rollout/cumulative metrics and reset rollout accumulators."""

        physics = self._rollout["physics_substeps_seen"]
        audits = self._rollout["committed_audits"]
        skips = self._rollout["committed_audit_skips"]

        for key in (
            "physics_substeps_seen",
            "safe_free_substeps",
            "active_substeps",
            "near_wall_substeps",
            "unilateral_rows_built",
            "q_free_penetration_substeps",
            "committed_audits",
            "committed_audit_skips",
        ):
            logger.record(
                f"beam_safety/{key}_rollout",
                self._rollout[key],
                exclude="stdout",
            )

        logger.record(
            "beam_safety/active_fraction_rollout",
            self._ratio(self._rollout["active_substeps"], physics),
            exclude="stdout",
        )
        logger.record(
            "beam_safety/near_wall_fraction_rollout",
            self._ratio(self._rollout["near_wall_substeps"], physics),
            exclude="stdout",
        )
        logger.record(
            "beam_safety/q_free_penetration_fraction_rollout",
            self._ratio(
                self._rollout["q_free_penetration_substeps"], physics
            ),
            exclude="stdout",
        )
        logger.record(
            "beam_safety/committed_verification_fraction_rollout",
            self._ratio(audits, audits + skips),
            exclude="stdout",
        )

        if np.isfinite(self._rollout_worst_free_m):
            logger.record(
                "beam_safety/worst_free_clearance_mm_rollout",
                self._rollout_worst_free_m * 1000.0,
                exclude="stdout",
            )
        if np.isfinite(self._rollout_worst_committed_m):
            logger.record(
                "beam_safety/worst_audited_committed_clearance_mm_rollout",
                self._rollout_worst_committed_m * 1000.0,
                exclude="stdout",
            )

        for key in (
            "physics_substeps_seen",
            "active_substeps",
            "near_wall_substeps",
            "unilateral_rows_built",
            "q_free_penetration_substeps",
            "committed_audits",
            "committed_audit_skips",
        ):
            logger.record(
                f"beam_safety_total/{key}",
                self._total[key],
                exclude="stdout",
            )

        logger.record(
            "beam_safety_total/completed_episodes_observed",
            float(self._episode_count),
            exclude="stdout",
        )
        logger.record(
            "beam_safety_total/info_snapshots_observed",
            float(self._snapshot_count),
            exclude="stdout",
        )

        if np.isfinite(self._total_worst_free_m):
            logger.record(
                "beam_safety_total/worst_free_clearance_mm",
                self._total_worst_free_m * 1000.0,
                exclude="stdout",
            )
        if np.isfinite(self._total_worst_committed_m):
            logger.record(
                "beam_safety_total/worst_audited_committed_clearance_mm",
                self._total_worst_committed_m * 1000.0,
                exclude="stdout",
            )

        recent = list(self._recent_episodes)
        if recent:
            prefix = f"beam_episode_recent_w{self.episode_window}"
            for key in (
                "active_substeps",
                "near_wall_substeps",
                "unilateral_rows_built",
                "committed_audits",
                "committed_audit_skips",
            ):
                logger.record(
                    f"{prefix}/{key}_mean",
                    self._window_mean(recent, key),
                    exclude="stdout",
                )
            worst_free = self._window_min(
                recent, "worst_free_clearance_m"
            )
            if np.isfinite(worst_free):
                logger.record(
                    f"{prefix}/worst_free_clearance_mm",
                    worst_free * 1000.0,
                    exclude="stdout",
                )
            worst_committed = self._window_min(
                recent, "worst_audited_committed_clearance_m"
            )
            if np.isfinite(worst_committed):
                logger.record(
                    f"{prefix}/worst_audited_committed_clearance_mm",
                    worst_committed * 1000.0,
                    exclude="stdout",
                )

        self._rollout = {name: 0.0 for name in _COUNTER_FIELDS}
        self._rollout_worst_free_m = float("inf")
        self._rollout_worst_committed_m = float("inf")

    def snapshot(self) -> dict[str, Any]:
        """Return a test/debug snapshot; never used by the environment."""

        return {
            "episode_count": int(self._episode_count),
            "snapshot_count": int(self._snapshot_count),
            "total": dict(self._total),
            "total_worst_free_clearance_m": (
                float(self._total_worst_free_m)
                if np.isfinite(self._total_worst_free_m)
                else None
            ),
            "total_worst_audited_committed_clearance_m": (
                float(self._total_worst_committed_m)
                if np.isfinite(self._total_worst_committed_m)
                else None
            ),
        }


__all__ = ["BeamSafetyMetricsCollector"]
