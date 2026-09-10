"""Shared episode/epoch bookkeeping, checkpointing, and validation runtime."""

from __future__ import annotations

import csv
import json
from collections import deque
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback

from .evaluation import (
    ValidationResult,
    merge_validation_json,
    validation_selection_key,
    validation_result_to_json,
)
from ..training_config import (
    TRAINING_CURRICULUM_MIN_EPISODES_PER_VESSEL,
    TRAINING_CURRICULUM_MODELS,
    TRAINING_CURRICULUM_PROMOTION_MODES,
    TRAINING_CURRICULUM_ROLLING_EPISODES_PER_VESSEL,
    TRAINING_CURRICULUM_STAGE_NAMES,
    TRAINING_CURRICULUM_TARGET_FRACTIONS,
    curriculum_exploration_profile,
    curriculum_sampling_weights,
    update_curriculum_progress,
    update_validation_unlocked,
)


def _append_csv(path: Path, fieldnames, row) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _append_csv_rows(path: Path, fieldnames, rows) -> None:
    """Append one synchronized batch without reopening the file per episode."""

    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


_TERMINAL_REASON_TO_CODE = {
    "target": 1,
    "timeout": 2,
    "out_of_vessel": 3,
    "non_finite": 4,
    "other": 5,
}
_TERMINAL_CODE_TO_REASON = {
    code: reason for reason, code in _TERMINAL_REASON_TO_CODE.items()
}


class EpochExperimentCallback(BaseCallback):
    """Count global episodes and run parallel validation at epoch boundaries."""

    def __init__(
        self,
        context,
        algorithm_name: str,
        variant: str,
        epochs: int,
        episodes_per_epoch: int,
        model_dir: Path,
        run_dir: Path,
        validation_interval: int = 2,
        validation_min_train_success_rate: float = 0.20,
        validation_fn: Optional[Callable[[object, int], ValidationResult]] = None,
        resume_progress: bool = True,
        training_curriculum_enabled: bool = True,
        episode_sync_interval_steps: int = 256,
        performance_log_interval_steps: int = 64,
        terminal_trace_sample_interval: int = 20,
    ):
        super().__init__(verbose=0)
        self.context = context
        self.algorithm_name = str(algorithm_name).lower()
        self.variant = str(variant)
        self.epochs = int(epochs)
        self.episodes_per_epoch = int(episodes_per_epoch)
        self.target_episodes = self.epochs * self.episodes_per_epoch
        self.model_dir = Path(model_dir)
        self.run_dir = Path(run_dir)
        self.validation_interval = int(validation_interval)
        self.validation_min_train_success_rate = float(
            validation_min_train_success_rate
        )
        if not (0.0 <= self.validation_min_train_success_rate <= 1.0):
            raise ValueError("validation_min_train_success_rate must be in [0, 1].")
        self.validation_unlocked = update_validation_unlocked(
            False,
            0.0,
            self.validation_min_train_success_rate,
            full_task_ready=not bool(training_curriculum_enabled),
        )
        self.validation_fn = validation_fn
        self.resume_progress = bool(resume_progress)
        self.training_curriculum_enabled = bool(training_curriculum_enabled)
        self.curriculum_stage = 0
        self.curriculum_success_streak = 0
        self.curriculum_sampling_weights = {}
        self.curriculum_dr_profile = {}
        self.curriculum_exploration_profile = {}
        self.curriculum_target_fraction = 1.0
        self.episode_sync_interval_steps = max(1, int(episode_sync_interval_steps))
        self.performance_log_interval_steps = max(
            1, int(performance_log_interval_steps)
        )
        self._performance_step_count = 0
        self.terminal_trace_sample_interval = max(
            1, int(terminal_trace_sample_interval)
        )
        self._terminal_trace_counts = {}
        self._terminal_trace_records_written = 0
        self._terminal_trace_records_per_file = 100
        self.global_completed_episodes = 0
        self.next_epoch = 1
        self._epoch_success_count = 0
        self._epoch_reward_sum = 0.0
        self._epoch_episode_steps_sum = 0.0
        self._epoch_out_of_vessel_count = 0
        self._epoch_wrong_branch_count = 0
        self._epoch_non_finite_count = 0
        self._epoch_timeout_count = 0
        self._epoch_no_progress_count = 0
        self._epoch_reward_progress_sum = 0.0
        self._epoch_reward_terminal_sum = 0.0
        self._epoch_reward_safety_sum = 0.0
        self._epoch_reward_step_sum = 0.0
        self._epoch_insert_action_mean_sum = 0.0
        self._epoch_insert_positive_fraction_sum = 0.0
        self._epoch_insert_negative_fraction_sum = 0.0
        self._epoch_inserted_length_final_sum = 0.0
        self._epoch_route_completion_sum = 0.0
        self._epoch_positive_failure_count = 0
        self._epoch_reward_component_total_sum = 0.0
        self._epoch_route_potential_sum = 0.0
        self._epoch_route_jump_rejections_sum = 0.0
        self._epoch_safety_ratio_max_sum = 0.0
        self._epoch_safety_margin_min_sum = 0.0
        self._epoch_tip_clearance_min_sum = 0.0
        self._epoch_body_clearance_min_sum = 0.0
        self._epoch_body_warning_steps_sum = 0.0
        self._epoch_body_contact_steps_sum = 0.0
        self._epoch_body_warning_mean_sum = 0.0
        self._epoch_body_warning_positive_insert_steps_sum = 0.0
        self._epoch_raw_insert_action_mean_sum = 0.0
        self._curriculum_model_ids = tuple(
            dict.fromkeys(
                model_id
                for stage_models in TRAINING_CURRICULUM_MODELS
                for model_id in stage_models
            )
        )
        self._curriculum_model_to_index = {
            model_id: index
            for index, model_id in enumerate(self._curriculum_model_ids)
        }
        self._epoch_model_episode_counts = np.zeros(
            len(self._curriculum_model_ids), dtype=np.int64
        )
        self._epoch_model_success_counts = np.zeros(
            len(self._curriculum_model_ids), dtype=np.int64
        )
        self._epoch_model_jump_rejection_counts = np.zeros(
            len(self._curriculum_model_ids), dtype=np.float64
        )
        self._epoch_model_out_of_vessel_counts = np.zeros(
            len(self._curriculum_model_ids), dtype=np.int64
        )
        self._epoch_model_route_completion_sums = np.zeros(
            len(self._curriculum_model_ids), dtype=np.float64
        )
        self._epoch_model_body_clearance_min_sums = np.zeros(
            len(self._curriculum_model_ids), dtype=np.float64
        )
        self._epoch_model_safety_ratio_max_sums = np.zeros(
            len(self._curriculum_model_ids), dtype=np.float64
        )
        self._epoch_model_body_warning_steps_sums = np.zeros(
            len(self._curriculum_model_ids), dtype=np.float64
        )
        self._epoch_model_body_contact_steps_sums = np.zeros(
            len(self._curriculum_model_ids), dtype=np.float64
        )
        self._epoch_model_effective_insert_sums = np.zeros(
            len(self._curriculum_model_ids), dtype=np.float64
        )
        route_shape = (len(self._curriculum_model_ids), 6)
        self._epoch_route_episode_counts = np.zeros(route_shape, dtype=np.int64)
        self._epoch_route_success_counts = np.zeros(route_shape, dtype=np.int64)
        self._epoch_route_out_of_vessel_counts = np.zeros(route_shape, dtype=np.int64)
        self._epoch_route_completion_sums = np.zeros(route_shape, dtype=np.float64)
        self._curriculum_outcome_windows = {
            model_id: deque(
                maxlen=TRAINING_CURRICULUM_ROLLING_EPISODES_PER_VESSEL
            )
            for model_id in self._curriculum_model_ids
        }
        self._curriculum_route_outcome_windows = {
            f"{model_id}/target_{route_index:02d}": deque(
                maxlen=TRAINING_CURRICULUM_ROLLING_EPISODES_PER_VESSEL
            )
            for model_id in self._curriculum_model_ids
            if str(model_id).startswith("B")
            for route_index in range(1, 7)
        }
        self._pending_episode_events = []
        self._train_episode_fieldnames = [
            "global_episode", "epoch", "vessel_id", "target_route_id",
            "terminal_reason",
            "success", "steps", "reward", "route_completion",
            "route_potential", "final_dist_to_goal_m", "min_dist_to_goal_m",
            "inserted_length_m", "max_sdf_penetration_m", "route_progress_m",
            "route_projection_jump_rejections", "curriculum_stage",
            "route_start_progress_m", "route_target_progress_m",
            "route_progress_from_completion_m", "reward_progress",
            "reward_wall", "reward_branch", "reward_stagnation", "reward_step",
            "reward_success", "reward_out_of_vessel", "reward_non_finite",
            "reward_timeout", "sdf_tip_clearance_min_m",
            "sdf_body_clearance_min_m", "inserted_length_max_m",
        ]
        self._failure_episode_fieldnames = [
            "global_episode", "epoch", "vessel_id", "target_route_id",
            "terminal_reason",
            "steps", "reward", "route_completion", "route_progress_m",
            "route_projection_segment", "route_projection_distance_m",
            "final_dist_to_goal_m", "min_dist_to_goal_m",
            "sdf_tip_clearance_min_m", "sdf_body_clearance_min_m",
            "max_sdf_penetration_m", "centerline_safety_ratio_terminal",
            "centerline_safety_ratio_max_episode",
            "centerline_safety_margin_terminal",
            "centerline_safety_margin_min_episode",
            "sdf_tip_near_wall_steps_episode",
            "sdf_body_warning_steps_episode",
            "sdf_body_contact_steps_episode",
            "sdf_body_warning_mean_episode",
            "sdf_body_warning_positive_insert_steps_episode",
            "raw_insert_action_mean", "effective_insert_action_mean",
            "insert_positive_fraction", "insert_negative_fraction",
            "curve_bend_5mm_max_episode", "curve_bend_10mm_max_episode",
            "curve_bend_20mm_max_episode",
            "curve_alignment_error_20mm_max_episode",
            "reward_progress", "reward_wall", "reward_branch",
            "reward_stagnation", "reward_step", "reward_out_of_vessel",
            "reward_timeout",
        ]
        self.best_valid_success_rate = -1.0
        self.best_valid_route_completion = -1.0
        self.best_valid_route_potential = -1.0
        self.best_valid_final_distance_mm = float("inf")
        self.best_epoch = 0

    def _on_training_start(self) -> None:
        if self.resume_progress:
            self.best_valid_success_rate = float(
                getattr(self.model, "best_valid_success_rate", -1.0)
            )
            self.best_epoch = int(getattr(self.model, "best_epoch", 0))
            self.best_valid_route_completion = float(
                getattr(
                    self.model,
                    "best_valid_route_completion",
                    -1.0,
                )
            )
            self.best_valid_route_potential = float(
                getattr(self.model, "best_valid_route_potential", -1.0)
            )
            self.best_valid_final_distance_mm = float(
                getattr(
                    self.model,
                    "best_valid_final_distance_mm",
                    float("inf"),
                )
            )
            self.validation_unlocked = bool(
                getattr(
                    self.model,
                    "validation_unlocked",
                    float(getattr(self.model, "train_success_rate", 0.0))
                    >= self.validation_min_train_success_rate,
                )
            )
            saved_epoch = int(getattr(self.model, "epoch", 0))
            saved_episodes = int(getattr(self.model, "global_completed_episodes", 0))
            if saved_epoch > 0 and saved_episodes == saved_epoch * self.episodes_per_epoch:
                self.next_epoch = saved_epoch + 1
                self.global_completed_episodes = saved_episodes
            self.curriculum_stage = int(getattr(self.model, "curriculum_stage", 0))
            if (
                self.training_curriculum_enabled
                and self.curriculum_stage < len(TRAINING_CURRICULUM_MODELS) - 1
            ):
                self.validation_unlocked = False
            self.curriculum_success_streak = int(
                getattr(self.model, "curriculum_success_streak", 0)
            )
            saved_windows = getattr(
                self.model,
                "curriculum_rolling_outcomes",
                {},
            )
            if isinstance(saved_windows, dict):
                for model_id, outcomes in saved_windows.items():
                    window = self._curriculum_outcome_windows.get(
                        str(model_id).upper()
                    )
                    if window is None:
                        continue
                    try:
                        window.extend(int(bool(value)) for value in outcomes)
                    except TypeError:
                        continue
            saved_route_windows = getattr(
                self.model,
                "curriculum_route_rolling_outcomes",
                {},
            )
            if isinstance(saved_route_windows, dict):
                for route_key, outcomes in saved_route_windows.items():
                    window = self._curriculum_route_outcome_windows.get(str(route_key))
                    if window is None:
                        continue
                    try:
                        window.extend(int(bool(value)) for value in outcomes)
                    except TypeError:
                        continue
        self._apply_curriculum_stage(self.curriculum_stage)
        active_models = (
            TRAINING_CURRICULUM_MODELS[self.curriculum_stage]
            if self.training_curriculum_enabled
            else TRAINING_CURRICULUM_MODELS[-1]
        )
        rolling_rates, _, _, _ = self._rolling_curriculum_statistics(active_models)
        self._apply_curriculum_sampling(active_models, rolling_rates)

    def _apply_curriculum_stage(self, stage: int) -> None:
        if not self.training_curriculum_enabled:
            return
        # This callback runs on every distributed rank.  Calling env_method on
        # each local VecEnv therefore updates all workers before the epoch
        # barrier releases any rank back into rollout collection.
        self.training_env.env_method("set_curriculum_stage", int(stage))
        profiles = self.training_env.env_method(
            "get_curriculum_domain_randomization_profile"
        )
        self.curriculum_dr_profile = dict(profiles[0]) if profiles else {}
        target_fractions = self.training_env.env_method(
            "get_curriculum_target_fraction"
        )
        self.curriculum_target_fraction = (
            float(target_fractions[0]) if target_fractions else 1.0
        )
        self._apply_curriculum_exploration(stage)

    def _apply_curriculum_sampling(self, active_models, success_rates) -> None:
        if not self.training_curriculum_enabled:
            return
        weights = curriculum_sampling_weights(active_models, success_rates)
        self.training_env.env_method(
            "set_training_model_sampling_weights",
            weights,
        )
        self.curriculum_sampling_weights = dict(weights)

    def _apply_curriculum_exploration(self, stage: int) -> None:
        """Apply the same stage schedule to all supported algorithms."""

        profile = curriculum_exploration_profile(stage)
        self.curriculum_exploration_profile = dict(profile)
        if self.algorithm_name in {"ppo", "lstm_ppo"}:
            floor = float(profile["ppo_min_action_std"])
            self.model.min_action_std = floor
            log_std = getattr(getattr(self.model, "policy", None), "log_std", None)
            if log_std is not None:
                with th.no_grad():
                    log_std.clamp_(min=float(np.log(floor)))
        elif self.algorithm_name in {"sac", "goal_sac"}:
            floor = float(profile["sac_min_ent_coef"])
            if self.algorithm_name == "goal_sac":
                floor = float(self.model.min_ent_coef)
                profile["sac_min_ent_coef"] = floor
            self.model.min_ent_coef = floor
            log_ent_coef = getattr(self.model, "log_ent_coef", None)
            if log_ent_coef is not None:
                with th.no_grad():
                    log_ent_coef.clamp_(min=float(np.log(floor)))

    def _local_episode_events(self):
        dones = np.asarray(self.locals.get("dones", []), dtype=np.bool_).reshape(-1)
        infos = list(self.locals.get("infos", []))
        # Compact V13 event schema. Extra numeric diagnostics stay inside the
        # existing synchronized block, so they add no distributed collective.
        events = np.zeros((len(dones), 61), dtype=np.float32)
        for index, done in enumerate(dones):
            if not done:
                continue
            info = infos[index]
            episode_info = info.get("episode", {})
            self._append_sampled_terminal_trace(info, episode_info)
            route_ratio = float(info.get("route_progress_ratio", np.nan))
            route_completion = (
                float(np.clip(route_ratio, 0.0, 1.0))
                if np.isfinite(route_ratio)
                else 0.0
            )
            if self.algorithm_name == "goal_sac":
                reward_progress = float(info.get("episode_goal_reward_progress", 0.0))
                reward_terminal = float(info.get("episode_goal_reward_terminal", 0.0))
                reward_safety = float(info.get("episode_goal_reward_safety", 0.0))
                reward_step = float(info.get("episode_goal_reward_step", 0.0))
                reward_component_total = float(
                    info.get("episode_goal_reward_total_components", episode_info.get("r", 0.0))
                )
            else:
                reward_progress = float(info.get("episode_reward_route_progress", 0.0))
                reward_terminal = sum(
                    float(info.get(key, 0.0))
                    for key in (
                        "episode_reward_successful_task",
                        "episode_reward_out_of_vessel_penalty",
                        "episode_reward_non_finite_penalty",
                        "episode_reward_timeout_penalty",
                    )
                )
                reward_safety = sum(
                    float(info.get(key, 0.0))
                    for key in (
                        "episode_reward_wall_proximity_penalty",
                        "episode_reward_off_target_branch_penalty",
                    )
                )
                reward_step = (
                    float(info.get("episode_reward_step_penalty", 0.0))
                    + float(info.get("episode_reward_stagnation_penalty", 0.0))
                )
                reward_component_total = float(
                    info.get("episode_reward_total_components", episode_info.get("r", 0.0))
                )
            sampling_model = str(
                info.get(
                    "sampling_model",
                    info.get("chosen_model", info.get("task_id", "")),
                )
            ).upper()
            model_index = self._curriculum_model_to_index.get(sampling_model, -1)
            events[index] = [
                1.0,
                float(bool(info.get("done_by_target", False))),
                float(episode_info.get("r", 0.0)),
                float(episode_info.get("l", 0.0)),
                float(bool(info.get("done_by_out_of_vessel", False))),
                float(bool(info.get("wrong_branch_this_episode", False))),
                float(bool(info.get("done_by_non_finite", False))),
                float(bool(info.get("TimeLimit.truncated", False) or info.get("terminal_reason") == "timeout")),
                float(bool(info.get("no_progress_this_episode", False))),
                reward_progress,
                reward_terminal,
                reward_safety,
                reward_step,
                float(info.get("insert_action_mean_episode", 0.0)),
                float(info.get("insert_positive_fraction_episode", 0.0)),
                float(info.get("insert_negative_fraction_episode", 0.0)),
                float(info.get("inserted_length_final", 0.0)),
                1.0 if bool(info.get("done_by_target", False)) else route_completion,
                float(bool(info.get("positive_failure_return", False))),
                reward_component_total,
                float(info.get("route_potential", 0.0)),
                float(model_index),
                float(info.get("route_projection_jump_rejections_episode", 0.0)),
                float(info.get("curriculum_stage", self.curriculum_stage)),
                float(info.get("final_dist_to_goal", np.nan)),
                float(info.get("min_dist_to_goal", np.nan)),
                float(info.get("sdf_penetration_depth_max_episode", 0.0)),
                float(info.get("route_progress", np.nan)),
                float(
                    _TERMINAL_REASON_TO_CODE.get(
                        str(info.get("terminal_reason", "other")), 5
                    )
                ),
                float(info.get("route_start_progress", np.nan)),
                float(info.get("route_target_progress", np.nan)),
                float(
                    info.get("route_start_progress", 0.0)
                    + route_completion
                    * (
                        info.get("route_target_progress", 0.0)
                        - info.get("route_start_progress", 0.0)
                    )
                ),
                reward_progress,
                float(info.get("episode_reward_wall_proximity_penalty", 0.0)),
                float(info.get("episode_reward_off_target_branch_penalty", 0.0)),
                float(info.get("episode_reward_stagnation_penalty", 0.0)),
                float(info.get("episode_reward_step_penalty", 0.0)),
                float(info.get("episode_reward_successful_task", 0.0)),
                float(info.get("episode_reward_out_of_vessel_penalty", 0.0)),
                float(info.get("episode_reward_non_finite_penalty", 0.0)),
                float(info.get("episode_reward_timeout_penalty", 0.0)),
                float(info.get("sdf_surface_clearance_min_episode", np.nan)),
                float(info.get("sdf_body_surface_clearance_min_episode", np.nan)),
                float(info.get("inserted_length_max_episode", np.nan)),
                float(info.get("raw_insert_action_mean_episode", 0.0)),
                float(info.get("route_projection_segment", -1.0)),
                float(info.get("route_projection_distance", np.nan)),
                float(info.get("centerline_safety_ratio", np.nan)),
                float(info.get("centerline_safety_ratio_max_episode", np.nan)),
                float(info.get("centerline_safety_margin", np.nan)),
                float(info.get("centerline_safety_margin_min_episode", np.nan)),
                float(info.get("sdf_tip_near_wall_steps_episode", 0.0)),
                float(info.get("sdf_body_warning_steps_episode", 0.0)),
                float(info.get("sdf_body_contact_steps_episode", 0.0)),
                float(info.get("sdf_body_warning_mean_episode", 0.0)),
                float(info.get("sdf_body_warning_positive_insert_steps_episode", 0.0)),
                float(info.get("curve_bend_5mm_max_episode", 0.0)),
                float(info.get("curve_bend_10mm_max_episode", 0.0)),
                float(info.get("curve_bend_20mm_max_episode", 0.0)),
                float(info.get("curve_alignment_error_20mm_max_episode", 0.0)),
                float(info.get("target_route_index", 0.0)),
            ]
        return events

    def _append_sampled_terminal_trace(self, info, episode_info) -> None:
        """Persist a bounded sample of terminal histories without affecting RL."""

        trace = info.get("terminal_diagnostic_trace")
        if not trace:
            return
        reason = str(info.get("terminal_reason", "other"))
        key = (
            str(info.get("sampling_model", info.get("chosen_model", "unknown"))),
            str(info.get("target_route_id", "default")),
            reason,
        )
        count = int(self._terminal_trace_counts.get(key, 0)) + 1
        self._terminal_trace_counts[key] = count
        if (count - 1) % self.terminal_trace_sample_interval:
            return
        record = {
            "rank": int(self.context.rank),
            "local_terminal_index": count,
            "model_num_timesteps": int(getattr(self.model, "num_timesteps", 0)),
            "vessel_id": key[0],
            "target_route_id": key[1],
            "terminal_reason": key[2],
            "success": bool(info.get("done_by_target", False)),
            "episode_steps": int(episode_info.get("l", 0)),
            "episode_reward": float(episode_info.get("r", 0.0)),
            "trace": trace,
        }
        trace_dir = self.run_dir / "diagnostics" / "terminal_traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        part = self._terminal_trace_records_written // self._terminal_trace_records_per_file
        path = trace_dir / f"rank_{self.context.rank}_part_{part:04d}.jsonl"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")
        self._terminal_trace_records_written += 1

    def _append_train_episode_rows(self, accepted, global_episode_start: int) -> None:
        """Write diagnostics only; this data never feeds observations or updates."""

        if not self.context.is_main:
            return
        rows = []
        failure_rows = []
        for offset, event in enumerate(accepted):
            global_episode = int(global_episode_start + offset + 1)
            model_index = int(round(float(event[21])))
            vessel_id = (
                self._curriculum_model_ids[model_index]
                if 0 <= model_index < len(self._curriculum_model_ids)
                else "unknown"
            )
            route_index = int(round(float(event[60])))
            target_route_id = (
                f"target_{route_index:02d}"
                if 1 <= route_index <= 6
                else "default"
            )
            row = {
                "global_episode": global_episode,
                "epoch": int((global_episode - 1) // self.episodes_per_epoch + 1),
                "vessel_id": vessel_id,
                "target_route_id": target_route_id,
                "terminal_reason": _TERMINAL_CODE_TO_REASON.get(
                    int(round(float(event[28]))), "other"
                ),
                "success": int(event[1] > 0.5),
                "steps": int(round(float(event[3]))),
                "reward": float(event[2]),
                "route_completion": float(event[17]),
                "route_potential": float(event[20]),
                "final_dist_to_goal_m": float(event[24]),
                "min_dist_to_goal_m": float(event[25]),
                "inserted_length_m": float(event[16]),
                "max_sdf_penetration_m": float(event[26]),
                "route_progress_m": float(event[27]),
                "route_projection_jump_rejections": int(round(float(event[22]))),
                "curriculum_stage": int(round(float(event[23]))),
                "route_start_progress_m": float(event[29]),
                "route_target_progress_m": float(event[30]),
                "route_progress_from_completion_m": float(event[31]),
                "reward_progress": float(event[32]),
                "reward_wall": float(event[33]),
                "reward_branch": float(event[34]),
                "reward_stagnation": float(event[35]),
                "reward_step": float(event[36]),
                "reward_success": float(event[37]),
                "reward_out_of_vessel": float(event[38]),
                "reward_non_finite": float(event[39]),
                "reward_timeout": float(event[40]),
                "sdf_tip_clearance_min_m": float(event[41]),
                "sdf_body_clearance_min_m": float(event[42]),
                "inserted_length_max_m": float(event[43]),
            }
            rows.append(row)
            if event[1] <= 0.5:
                failure_rows.append({
                    "global_episode": global_episode,
                    "epoch": row["epoch"],
                    "vessel_id": vessel_id,
                    "target_route_id": target_route_id,
                    "terminal_reason": row["terminal_reason"],
                    "steps": row["steps"],
                    "reward": row["reward"],
                    "route_completion": row["route_completion"],
                    "route_progress_m": row["route_progress_m"],
                    "route_projection_segment": int(round(float(event[45]))),
                    "route_projection_distance_m": float(event[46]),
                    "final_dist_to_goal_m": row["final_dist_to_goal_m"],
                    "min_dist_to_goal_m": row["min_dist_to_goal_m"],
                    "sdf_tip_clearance_min_m": row["sdf_tip_clearance_min_m"],
                    "sdf_body_clearance_min_m": row["sdf_body_clearance_min_m"],
                    "max_sdf_penetration_m": row["max_sdf_penetration_m"],
                    "centerline_safety_ratio_terminal": float(event[47]),
                    "centerline_safety_ratio_max_episode": float(event[48]),
                    "centerline_safety_margin_terminal": float(event[49]),
                    "centerline_safety_margin_min_episode": float(event[50]),
                    "sdf_tip_near_wall_steps_episode": int(round(float(event[51]))),
                    "sdf_body_warning_steps_episode": int(round(float(event[52]))),
                    "sdf_body_contact_steps_episode": int(round(float(event[53]))),
                    "sdf_body_warning_mean_episode": float(event[54]),
                    "sdf_body_warning_positive_insert_steps_episode": int(round(float(event[55]))),
                    "raw_insert_action_mean": float(event[44]),
                    "effective_insert_action_mean": float(event[13]),
                    "insert_positive_fraction": float(event[14]),
                    "insert_negative_fraction": float(event[15]),
                    "curve_bend_5mm_max_episode": float(event[56]),
                    "curve_bend_10mm_max_episode": float(event[57]),
                    "curve_bend_20mm_max_episode": float(event[58]),
                    "curve_alignment_error_20mm_max_episode": float(event[59]),
                    "reward_progress": float(event[32]),
                    "reward_wall": float(event[33]),
                    "reward_branch": float(event[34]),
                    "reward_stagnation": float(event[35]),
                    "reward_step": float(event[36]),
                    "reward_out_of_vessel": float(event[38]),
                    "reward_timeout": float(event[40]),
                })
        _append_csv_rows(
            self.run_dir / "train_episodes.csv",
            self._train_episode_fieldnames,
            rows,
        )
        _append_csv_rows(
            self.run_dir / "diagnostics" / "failure_episodes.csv",
            self._failure_episode_fieldnames,
            failure_rows,
        )

    def _global_pending_episode_events(self):
        """Synchronize one fixed-size block of episode statistics."""

        local_np = np.stack(self._pending_episode_events, axis=0)
        local = th.as_tensor(
            local_np,
            dtype=th.float32,
            device=self.context.device.resolved,
        )
        if not self.context.enabled:
            return local_np.reshape(-1, local_np.shape[-1])
        gathered = [th.zeros_like(local) for _ in range(self.context.world_size)]
        self.context.all_gather(gathered, local)
        # Preserve the old per-step rank/env ordering while synchronizing many
        # steps at once: [rank, step, env, field] -> [step, rank, env, field].
        stacked = th.stack(gathered, dim=0).permute(1, 0, 2, 3).contiguous()
        return stacked.view(-1, stacked.shape[-1]).detach().cpu().numpy()

    def _checkpoint_path(self, epoch: int) -> Path:
        return self.model_dir / (
            f"{self.algorithm_name}_{self.variant}_epoch_{epoch:03d}_"
            f"episodes_{self.global_completed_episodes:05d}"
        )

    def _rolling_curriculum_statistics(self, active_models, promotion_mode=None):
        counts = {
            model_id: len(self._curriculum_outcome_windows[model_id])
            for model_id in active_models
        }
        rates = {
            model_id: (
                float(sum(self._curriculum_outcome_windows[model_id]))
                / float(counts[model_id])
                if counts[model_id] > 0
                else None
            )
            for model_id in active_models
        }
        ready = bool(
            counts
            and all(
                count >= TRAINING_CURRICULUM_MIN_EPISODES_PER_VESSEL
                for count in counts.values()
            )
            and all(rate is not None for rate in rates.values())
        )
        if ready and promotion_mode == "aggregate":
            total_count = sum(counts.values())
            mastery_rate = (
                sum(float(rates[model_id]) * counts[model_id] for model_id in active_models)
                / float(total_count)
                if total_count > 0
                else 0.0
            )
        else:
            mastery_rate = (
                min(float(rate) for rate in rates.values()) if ready else 0.0
            )
        return rates, counts, ready, mastery_rate

    def _record_model_metadata(self, epoch: int, train_success_rate: float) -> None:
        self.model.epoch = int(epoch)
        self.model.global_completed_episodes = int(self.global_completed_episodes)
        self.model.global_env_steps = int(self.model.num_timesteps * self.context.world_size)
        self.model.train_success_rate = float(train_success_rate)
        self.model.best_metric = (
            "valid_success_rate,route_completion,"
            "route_potential,-final_distance_mm"
        )
        self.model.best_valid_success_rate = float(self.best_valid_success_rate)
        self.model.best_valid_route_completion = float(
            self.best_valid_route_completion
        )
        self.model.best_valid_route_potential = float(
            self.best_valid_route_potential
        )
        self.model.best_valid_final_distance_mm = float(
            self.best_valid_final_distance_mm
        )
        self.model.best_epoch = int(self.best_epoch)
        self.model.validation_unlocked = bool(self.validation_unlocked)
        self.model.validation_min_train_success_rate = float(
            self.validation_min_train_success_rate
        )
        self.model.curriculum_stage = int(self.curriculum_stage)
        self.model.curriculum_stage_name = (
            str(TRAINING_CURRICULUM_STAGE_NAMES[self.curriculum_stage])
            if self.training_curriculum_enabled
            else "disabled_all_vessels"
        )
        self.model.curriculum_success_streak = int(self.curriculum_success_streak)
        self.model.curriculum_rolling_outcomes = {
            model_id: list(window)
            for model_id, window in self._curriculum_outcome_windows.items()
        }
        self.model.curriculum_route_rolling_outcomes = {
            route_key: list(window)
            for route_key, window in self._curriculum_route_outcome_windows.items()
        }
        self.model.curriculum_rolling_window_size = int(
            TRAINING_CURRICULUM_ROLLING_EPISODES_PER_VESSEL
        )
        self.model.curriculum_sampling_weights = dict(
            self.curriculum_sampling_weights
        )
        self.model.curriculum_dr_profile = dict(self.curriculum_dr_profile)
        self.model.curriculum_target_fraction = float(
            self.curriculum_target_fraction
        )
        self.model.curriculum_exploration_profile = dict(
            self.curriculum_exploration_profile
        )
        self.model.distributed_world_size_at_save = int(self.context.world_size)

    def _run_validation(self, epoch: int, checkpoint_path: Path) -> None:
        if self.validation_fn is None or epoch % self.validation_interval != 0:
            return
        if not self.validation_unlocked:
            if self.context.is_main:
                print(
                    f"[VALID][Epoch {epoch:03d}] status=skipped_locked "
                    f"unlock_train_success_rate="
                    f"{self.validation_min_train_success_rate:.6f}",
                    flush=True,
                )
            return
        error_text = ""
        local_result = None
        local_payload = ""
        try:
            local_result = self.validation_fn(self.model, epoch)
            if local_result is None:
                raise RuntimeError("validation_fn returned no result")
            local_payload = validation_result_to_json(local_result)
            local_vessels = sorted(
                {item.vessel_id for item in local_result.episodes}
            )
            print(
                f"[VALID][Rank {self.context.rank}][Epoch {epoch:03d}] "
                f"local_episodes={local_result.valid_episodes} "
                f"vessels={','.join(local_vessels)}"
            )
        except BaseException as exc:
            error_text = f"rank={self.context.rank} {type(exc).__name__}: {exc}"
            print(
                f"[VALID][Rank {self.context.rank}][Epoch {epoch:03d}] "
                f"status=failed error={error_text}",
                flush=True,
            )

        error_texts = self.context.all_gather_text(error_text)
        failures = [text for text in error_texts if text]
        if failures:
            # Validation must never terminate a long-running distributed
            # training job. Per-episode failures are normally converted into
            # ValidationEpisodeResult records by evaluate_policy; this is the
            # final guard for unexpected rank-local infrastructure errors.
            print(
                f"[VALID][Rank {self.context.rank}][Epoch {epoch:03d}] "
                "status=continuing_after_errors errors=" + " | ".join(failures),
                flush=True,
            )

        payloads = self.context.all_gather_text(local_payload)
        if self.context.is_main:
            try:
                result = merge_validation_json(payloads)
                valid_success_rate = float(result.valid_success_rate)
                candidate_key = validation_selection_key(result)
                best_key = (
                    float(self.best_valid_success_rate),
                    float(self.best_valid_route_completion),
                    float(self.best_valid_route_potential),
                    -float(self.best_valid_final_distance_mm),
                )
                is_new_best = candidate_key > best_key
                if is_new_best:
                    self.best_valid_success_rate = valid_success_rate
                    self.best_valid_route_completion = float(
                        result.valid_route_completion_mean
                    )
                    self.best_valid_route_potential = float(
                        result.valid_route_potential_mean
                    )
                    self.best_valid_final_distance_mm = float(
                        result.valid_final_distance_mm_mean
                    )
                    self.best_epoch = int(epoch)
                    self.model.best_valid_success_rate = valid_success_rate
                    self.model.best_valid_route_completion = float(
                        self.best_valid_route_completion
                    )
                    self.model.best_valid_route_potential = float(
                        self.best_valid_route_potential
                    )
                    self.model.best_valid_final_distance_mm = float(
                        self.best_valid_final_distance_mm
                    )
                    self.model.best_epoch = int(epoch)
                    self.model.best_metric = (
                        "valid_success_rate,route_completion,"
                        "route_potential,-final_distance_mm"
                    )
                    self.model.last_valid_success_rate = valid_success_rate
                    self.model.last_valid_epoch = int(epoch)
                    self.model.save(str(self.model_dir / "best_valid"))
                else:
                    self.model.last_valid_success_rate = valid_success_rate
                    self.model.last_valid_epoch = int(epoch)
                self.model.last_valid_route_completion = float(
                    result.valid_route_completion_mean
                )
                self.model.last_valid_route_potential = float(
                    result.valid_route_potential_mean
                )
                self.model.last_valid_final_distance_mm = float(
                    result.valid_final_distance_mm_mean
                )
                self.model.last_valid_min_distance_mm = float(
                    result.valid_min_distance_mm_mean
                )
                # Refresh the just-written epoch checkpoint with validation
                # metadata so resuming from it preserves strict tie behavior.
                self.model.best_valid_success_rate = float(self.best_valid_success_rate)
                self.model.best_valid_route_completion = float(
                    self.best_valid_route_completion
                )
                self.model.best_valid_route_potential = float(
                    self.best_valid_route_potential
                )
                self.model.best_valid_final_distance_mm = float(
                    self.best_valid_final_distance_mm
                )
                self.model.best_epoch = int(self.best_epoch)
                self.model.save(str(checkpoint_path))
                self.logger.record("valid/success_rate", valid_success_rate)
                self.logger.record(
                    "valid/route_completion",
                    float(result.valid_route_completion_mean),
                )
                self.logger.record(
                    "valid/route_potential",
                    float(result.valid_route_potential_mean),
                )
                self.logger.record(
                    "valid/final_distance_mm",
                    float(result.valid_final_distance_mm_mean),
                )
                self.logger.record(
                    "valid/min_distance_mm",
                    float(result.valid_min_distance_mm_mean),
                )
                self.logger.record("valid/episodes", float(result.valid_episodes), exclude="stdout")
                _append_csv(
                    self.run_dir / "valid_summary.csv",
                    [
                        "epoch", "valid_vessels", "valid_episodes", "valid_success_count",
                        "valid_success_rate", "valid_route_completion_mean",
                        "valid_route_potential_mean", "valid_final_distance_mm_mean",
                        "valid_min_distance_mm_mean", "best_valid_success_rate",
                        "best_valid_route_completion", "best_valid_route_potential",
                        "best_valid_final_distance_mm", "is_new_best", "checkpoint",
                    ],
                    {
                        "epoch": epoch,
                        "valid_vessels": result.valid_vessels,
                        "valid_episodes": result.valid_episodes,
                        "valid_success_count": result.valid_success_count,
                        "valid_success_rate": valid_success_rate,
                        "valid_route_completion_mean": result.valid_route_completion_mean,
                        "valid_route_potential_mean": result.valid_route_potential_mean,
                        "valid_final_distance_mm_mean": result.valid_final_distance_mm_mean,
                        "valid_min_distance_mm_mean": result.valid_min_distance_mm_mean,
                        "best_valid_success_rate": self.best_valid_success_rate,
                        "best_valid_route_completion": self.best_valid_route_completion,
                        "best_valid_route_potential": self.best_valid_route_potential,
                        "best_valid_final_distance_mm": self.best_valid_final_distance_mm,
                        "is_new_best": is_new_best,
                        "checkpoint": str(checkpoint_path) + ".zip",
                    },
                )
                for episode_result in result.episodes:
                    _append_csv(
                        self.run_dir / "valid_episodes.csv",
                        [
                            "epoch", "vessel_id", "episode_index", "seed", "success",
                            "terminal_reason", "steps", "reward",
                            "route_completion", "route_potential",
                            "final_distance_mm", "min_distance_mm", "error",
                        ],
                        {
                            "epoch": epoch,
                            "vessel_id": episode_result.vessel_id,
                            "episode_index": episode_result.episode_index,
                            "seed": episode_result.seed,
                            "success": episode_result.success,
                            "terminal_reason": episode_result.terminal_reason,
                            "steps": episode_result.steps,
                            "reward": episode_result.reward,
                            "route_completion": episode_result.route_completion,
                            "route_potential": episode_result.route_potential,
                            "final_distance_mm": episode_result.final_distance_mm,
                            "min_distance_mm": episode_result.min_distance_mm,
                            "error": episode_result.error,
                        },
                    )
                print(
                    f"[VALID][Epoch {epoch:03d}] valid_vessels={result.valid_vessels} "
                    f"valid_episodes={result.valid_episodes} "
                    f"valid_success_count={result.valid_success_count} "
                    f"valid_success_rate={valid_success_rate:.6f} "
                    f"valid_route_completion={result.valid_route_completion_mean:.6f} "
                    f"valid_route_potential={result.valid_route_potential_mean:.6f} "
                    f"valid_final_distance_mm={result.valid_final_distance_mm_mean:.3f} "
                    f"valid_min_distance_mm={result.valid_min_distance_mm_mean:.3f} "
                    f"best_valid_success_rate={self.best_valid_success_rate:.6f} "
                    f"is_new_best={is_new_best}"
                )
            except BaseException as exc:
                error_text = f"{type(exc).__name__}: {exc}"
        error_text = self.context.broadcast_text(error_text)
        self.context.barrier()
        if error_text:
            print(
                f"[VALID][Rank {self.context.rank}][Epoch {epoch:03d}] "
                "status=result_handling_failed_continuing error=" + error_text,
                flush=True,
            )

    def _finish_epoch(self, epoch: int) -> None:
        train_success_rate = float(self._epoch_success_count / self.episodes_per_epoch)
        train_reward_mean = float(self._epoch_reward_sum / self.episodes_per_epoch)
        train_episode_steps_mean = float(
            self._epoch_episode_steps_sum / self.episodes_per_epoch
        )
        epoch_divisor = float(self.episodes_per_epoch)
        reward_progress_mean = self._epoch_reward_progress_sum / epoch_divisor
        reward_terminal_mean = self._epoch_reward_terminal_sum / epoch_divisor
        reward_safety_mean = self._epoch_reward_safety_sum / epoch_divisor
        reward_step_mean = self._epoch_reward_step_sum / epoch_divisor
        insert_action_mean = self._epoch_insert_action_mean_sum / epoch_divisor
        insert_positive_fraction = self._epoch_insert_positive_fraction_sum / epoch_divisor
        insert_negative_fraction = self._epoch_insert_negative_fraction_sum / epoch_divisor
        inserted_length_final_mean_mm = (
            self._epoch_inserted_length_final_sum / epoch_divisor * 1000.0
        )
        route_completion_mean = self._epoch_route_completion_sum / epoch_divisor
        positive_failure_rate = self._epoch_positive_failure_count / epoch_divisor
        reward_component_total_mean = self._epoch_reward_component_total_sum / epoch_divisor
        route_potential_final_mean = self._epoch_route_potential_sum / epoch_divisor
        route_jump_rejections_mean = (
            self._epoch_route_jump_rejections_sum / epoch_divisor
        )
        raw_insert_action_mean = self._epoch_raw_insert_action_mean_sum / epoch_divisor
        safety_ratio_max_mean = self._epoch_safety_ratio_max_sum / epoch_divisor
        safety_margin_min_mean = self._epoch_safety_margin_min_sum / epoch_divisor
        tip_clearance_min_mm_mean = (
            self._epoch_tip_clearance_min_sum / epoch_divisor * 1000.0
        )
        body_clearance_min_mm_mean = (
            self._epoch_body_clearance_min_sum / epoch_divisor * 1000.0
        )
        body_warning_steps_mean = self._epoch_body_warning_steps_sum / epoch_divisor
        body_contact_steps_mean = self._epoch_body_contact_steps_sum / epoch_divisor
        body_warning_mean = self._epoch_body_warning_mean_sum / epoch_divisor
        warning_positive_insert_steps_mean = (
            self._epoch_body_warning_positive_insert_steps_sum / epoch_divisor
        )
        warning_positive_insert_fraction = float(
            self._epoch_body_warning_positive_insert_steps_sum
            / max(self._epoch_body_warning_steps_sum, 1.0)
        )
        progress_to_safety_abs_ratio = float(
            abs(self._epoch_reward_progress_sum)
            / max(abs(self._epoch_reward_safety_sum), 1e-9)
        )
        out_of_vessel_rate = self._epoch_out_of_vessel_count / epoch_divisor
        timeout_rate = self._epoch_timeout_count / epoch_divisor
        if out_of_vessel_rate >= 0.50 and warning_positive_insert_fraction >= 0.50:
            diagnostic_assessment = "body_escape_and_forward_insertion_after_warning"
        elif out_of_vessel_rate >= 0.50:
            diagnostic_assessment = "body_escape_dominant"
        elif timeout_rate >= 0.50:
            diagnostic_assessment = "timeout_dominant"
        else:
            diagnostic_assessment = "mixed_or_improving"
        curriculum_stage_used = self.curriculum_stage
        curriculum_stage_name_used = (
            str(TRAINING_CURRICULUM_STAGE_NAMES[curriculum_stage_used])
            if self.training_curriculum_enabled
            else "disabled_all_vessels"
        )
        curriculum_target_fraction_used = float(
            TRAINING_CURRICULUM_TARGET_FRACTIONS[curriculum_stage_used]
            if self.training_curriculum_enabled
            else 1.0
        )
        active_models = (
            TRAINING_CURRICULUM_MODELS[curriculum_stage_used]
            if self.training_curriculum_enabled
            else TRAINING_CURRICULUM_MODELS[-1]
        )
        curriculum_vessel_episode_counts = {
            model_id: int(
                self._epoch_model_episode_counts[
                    self._curriculum_model_to_index[model_id]
                ]
            )
            for model_id in active_models
        }
        curriculum_vessel_success_rates = {}
        curriculum_vessel_route_jump_rejections_mean = {}
        for model_id in active_models:
            model_index = self._curriculum_model_to_index[model_id]
            episode_count = int(self._epoch_model_episode_counts[model_index])
            curriculum_vessel_success_rates[model_id] = (
                float(self._epoch_model_success_counts[model_index] / episode_count)
                if episode_count > 0
                else None
            )
            curriculum_vessel_route_jump_rejections_mean[model_id] = (
                float(
                    self._epoch_model_jump_rejection_counts[model_index]
                    / episode_count
                )
                if episode_count > 0
                else None
            )
        (
            curriculum_rolling_success_rates,
            curriculum_rolling_episode_counts,
            curriculum_mastery_ready,
            curriculum_mastery_success_rate,
        ) = self._rolling_curriculum_statistics(
            active_models,
            promotion_mode=(
                TRAINING_CURRICULUM_PROMOTION_MODES[curriculum_stage_used]
                if curriculum_stage_used < len(TRAINING_CURRICULUM_PROMOTION_MODES)
                else None
            ),
        )
        # A branching vessel is six distinct control tasks.  Its aggregate
        # success can hide route-mode collapse, so curriculum promotion uses
        # the weakest sufficiently sampled branch route while vessel sampling
        # remains balanced at the outer level.
        promotion_rates = dict(curriculum_rolling_success_rates)
        promotion_counts = dict(curriculum_rolling_episode_counts)
        for model_id in active_models:
            if not str(model_id).startswith("B"):
                continue
            route_windows = [
                self._curriculum_route_outcome_windows[
                    f"{model_id}/target_{route_index:02d}"
                ]
                for route_index in range(1, 7)
            ]
            route_counts = [len(window) for window in route_windows]
            if any(count == 0 for count in route_counts):
                promotion_rates[model_id] = None
                promotion_counts[model_id] = 0
            else:
                promotion_rates[model_id] = min(
                    float(sum(window)) / float(len(window))
                    for window in route_windows
                )
                promotion_counts[model_id] = min(route_counts)
        promotion_ready = bool(
            active_models
            and all(
                promotion_rates.get(model_id) is not None
                and promotion_counts.get(model_id, 0)
                >= TRAINING_CURRICULUM_MIN_EPISODES_PER_VESSEL
                for model_id in active_models
            )
        )
        if promotion_ready:
            curriculum_mastery_success_rate = min(
                float(promotion_rates[model_id]) for model_id in active_models
            )
        else:
            curriculum_mastery_success_rate = 0.0
        curriculum_mastery_ready = promotion_ready
        curriculum_mastery_ready = bool(curriculum_mastery_ready)
        if self.training_curriculum_enabled:
            # The curriculum helper historically aggregates vessel scores.
            # Give every active vessel the same global worst-route score so an
            # easy route can never hide an unlearned target branch.
            strict_promotion_rates = {
                model_id: (
                    float(curriculum_mastery_success_rate)
                    if promotion_ready
                    else None
                )
                for model_id in promotion_rates
            }
            next_curriculum_stage, self.curriculum_success_streak = (
                update_curriculum_progress(
                    self.curriculum_stage,
                    self.curriculum_success_streak,
                    train_success_rate,
                    strict_promotion_rates,
                    promotion_counts,
                )
            )
            if next_curriculum_stage != self.curriculum_stage:
                self.curriculum_stage = next_curriculum_stage
                for window in self._curriculum_outcome_windows.values():
                    window.clear()
                for window in self._curriculum_route_outcome_windows.values():
                    window.clear()
                self._apply_curriculum_stage(self.curriculum_stage)
            sampling_models = TRAINING_CURRICULUM_MODELS[self.curriculum_stage]
            sampling_rates, _, _, _ = self._rolling_curriculum_statistics(
                sampling_models
            )
            self._apply_curriculum_sampling(sampling_models, sampling_rates)
        self.model.curriculum_mastery_success_rate = float(
            curriculum_mastery_success_rate
        )
        self.model.curriculum_mastery_ready = bool(curriculum_mastery_ready)
        self.model.curriculum_rolling_success_rates = dict(
            curriculum_rolling_success_rates
        )
        self.model.curriculum_rolling_episode_counts = dict(
            curriculum_rolling_episode_counts
        )
        self.model.curriculum_vessel_success_rates = dict(
            curriculum_vessel_success_rates
        )
        self.validation_unlocked = update_validation_unlocked(
            self.validation_unlocked,
            train_success_rate,
            self.validation_min_train_success_rate,
            full_task_ready=(
                not self.training_curriculum_enabled
                or curriculum_stage_used >= len(TRAINING_CURRICULUM_MODELS) - 1
            ),
        )
        self._record_model_metadata(epoch, train_success_rate)
        checkpoint_path = self._checkpoint_path(epoch)
        if self.context.is_main:
            self.model.save(str(checkpoint_path))
            self.logger.record("train/success_rate", train_success_rate)
            self.logger.record("train/reward_mean", train_reward_mean)
            self.logger.record("train/episode_steps_mean", train_episode_steps_mean)
            self.logger.record("train/no_progress_rate", self._epoch_no_progress_count / epoch_divisor)
            self.logger.record("train/reward_progress_mean", reward_progress_mean, exclude="stdout")
            self.logger.record("train/reward_terminal_mean", reward_terminal_mean, exclude="stdout")
            self.logger.record("train/reward_safety_mean", reward_safety_mean, exclude="stdout")
            self.logger.record("train/reward_step_mean", reward_step_mean, exclude="stdout")
            self.logger.record("train/insert_action_mean", insert_action_mean, exclude="stdout")
            self.logger.record("train/insert_positive_fraction", insert_positive_fraction, exclude="stdout")
            self.logger.record("train/insert_negative_fraction", insert_negative_fraction, exclude="stdout")
            self.logger.record("train/inserted_length_final_mean_mm", inserted_length_final_mean_mm, exclude="stdout")
            self.logger.record("train/route_completion_mean", route_completion_mean, exclude="stdout")
            self.logger.record("train/positive_failure_rate", positive_failure_rate)
            self.logger.record("train/reward_component_total_mean", reward_component_total_mean, exclude="stdout")
            self.logger.record("train/route_potential_final_mean", route_potential_final_mean, exclude="stdout")
            self.logger.record(
                "train/route_projection_jump_rejections_mean",
                route_jump_rejections_mean,
                exclude="stdout",
            )
            self.logger.record("train/curriculum_stage", float(self.curriculum_stage))
            self.logger.record(
                "train/curriculum_dr_fraction",
                float(self.curriculum_dr_profile.get("fraction", 1.0)),
            )
            self.logger.record(
                "train/curriculum_target_fraction",
                float(self.curriculum_target_fraction),
            )
            for model_id, probability in self.curriculum_sampling_weights.items():
                self.logger.record(
                    f"train/curriculum_sampling_probability_{model_id}",
                    float(probability),
                    exclude="stdout",
                )
                jump_mean = curriculum_vessel_route_jump_rejections_mean.get(
                    model_id
                )
                if jump_mean is not None:
                    self.logger.record(
                        f"train/route_projection_jump_rejections_mean_{model_id}",
                        float(jump_mean),
                        exclude="stdout",
                    )
            exploration_floor = (
                self.curriculum_exploration_profile.get("sac_min_ent_coef")
                if self.algorithm_name == "sac"
                else self.curriculum_exploration_profile.get("ppo_min_action_std")
            )
            if exploration_floor is not None:
                self.logger.record(
                    "train/curriculum_exploration_floor",
                    float(exploration_floor),
                    exclude="stdout",
                )
            self.logger.record(
                "train/curriculum_mastery_success_rate",
                float(curriculum_mastery_success_rate),
            )
            self.logger.record(
                "train/curriculum_success_streak",
                float(self.curriculum_success_streak),
            )
            self.logger.record(
                "train/curriculum_mastery_ready",
                float(curriculum_mastery_ready),
            )
            self.logger.record("train/episodes", float(self.episodes_per_epoch), exclude="stdout")
            self.logger.record("train/global_completed_episodes", float(self.global_completed_episodes), exclude="stdout")
            self.logger.record("train/global_env_steps", float(self.model.global_env_steps), exclude="stdout")
            _append_csv(
                self.run_dir / "train_summary.csv",
                ["epoch", "train_episodes", "train_success_count", "train_success_rate", "train_reward_mean", "train_episode_steps_mean", "out_of_vessel_count", "wrong_branch_count", "non_finite_count", "timeout_count", "no_progress_count", "positive_failure_count", "positive_failure_rate", "reward_progress_mean", "reward_terminal_mean", "reward_safety_mean", "reward_step_mean", "reward_component_total_mean", "route_potential_final_mean", "route_projection_jump_rejections_mean", "curriculum_stage_used", "curriculum_stage_name_used", "curriculum_stage", "curriculum_stage_name", "curriculum_target_fraction_used", "curriculum_target_fraction", "curriculum_success_streak", "curriculum_mastery_ready", "curriculum_mastery_success_rate", "curriculum_vessel_success_rates", "curriculum_vessel_episode_counts", "curriculum_vessel_route_jump_rejections_mean", "curriculum_rolling_success_rates", "curriculum_rolling_episode_counts", "curriculum_sampling_weights", "curriculum_dr_profile", "curriculum_exploration_profile", "insert_action_mean", "insert_positive_fraction", "insert_negative_fraction", "inserted_length_final_mean_mm", "route_completion_mean", "global_completed_episodes", "global_env_steps", "checkpoint"],
                {
                    "epoch": epoch,
                    "train_episodes": self.episodes_per_epoch,
                    "train_success_count": self._epoch_success_count,
                    "train_success_rate": train_success_rate,
                    "train_reward_mean": train_reward_mean,
                    "train_episode_steps_mean": train_episode_steps_mean,
                    "out_of_vessel_count": self._epoch_out_of_vessel_count,
                    "wrong_branch_count": self._epoch_wrong_branch_count,
                    "non_finite_count": self._epoch_non_finite_count,
                    "timeout_count": self._epoch_timeout_count,
                    "no_progress_count": self._epoch_no_progress_count,
                    "positive_failure_count": self._epoch_positive_failure_count,
                    "positive_failure_rate": positive_failure_rate,
                    "reward_progress_mean": reward_progress_mean,
                    "reward_terminal_mean": reward_terminal_mean,
                    "reward_safety_mean": reward_safety_mean,
                    "reward_step_mean": reward_step_mean,
                    "reward_component_total_mean": reward_component_total_mean,
                    "route_potential_final_mean": route_potential_final_mean,
                    "route_projection_jump_rejections_mean": route_jump_rejections_mean,
                    "curriculum_stage_used": curriculum_stage_used,
                    "curriculum_stage_name_used": curriculum_stage_name_used,
                    "curriculum_stage": self.curriculum_stage,
                    "curriculum_stage_name": (
                        TRAINING_CURRICULUM_STAGE_NAMES[self.curriculum_stage]
                        if self.training_curriculum_enabled
                        else "disabled_all_vessels"
                    ),
                    "curriculum_target_fraction_used": curriculum_target_fraction_used,
                    "curriculum_target_fraction": self.curriculum_target_fraction,
                    "curriculum_success_streak": self.curriculum_success_streak,
                    "curriculum_mastery_ready": curriculum_mastery_ready,
                    "curriculum_mastery_success_rate": curriculum_mastery_success_rate,
                    "curriculum_vessel_success_rates": json.dumps(
                        curriculum_vessel_success_rates,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "curriculum_vessel_episode_counts": json.dumps(
                        curriculum_vessel_episode_counts,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "curriculum_vessel_route_jump_rejections_mean": json.dumps(
                        curriculum_vessel_route_jump_rejections_mean,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "curriculum_rolling_success_rates": json.dumps(
                        curriculum_rolling_success_rates,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "curriculum_rolling_episode_counts": json.dumps(
                        curriculum_rolling_episode_counts,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "curriculum_sampling_weights": json.dumps(
                        self.curriculum_sampling_weights,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "curriculum_dr_profile": json.dumps(
                        self.curriculum_dr_profile,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "curriculum_exploration_profile": json.dumps(
                        self.curriculum_exploration_profile,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "insert_action_mean": insert_action_mean,
                    "insert_positive_fraction": insert_positive_fraction,
                    "insert_negative_fraction": insert_negative_fraction,
                    "inserted_length_final_mean_mm": inserted_length_final_mean_mm,
                    "route_completion_mean": route_completion_mean,
                    "global_completed_episodes": self.global_completed_episodes,
                    "global_env_steps": self.model.global_env_steps,
                    "checkpoint": str(checkpoint_path) + ".zip",
                },
            )
            _append_csv(
                self.run_dir / "diagnostics" / "safety_summary.csv",
                [
                    "epoch", "episodes", "success_rate", "out_of_vessel_rate",
                    "timeout_rate", "diagnostic_assessment",
                    "route_completion_mean", "centerline_safety_ratio_max_mean",
                    "centerline_safety_margin_min_mean",
                    "sdf_tip_clearance_min_mm_mean",
                    "sdf_body_clearance_min_mm_mean",
                    "sdf_body_warning_steps_mean", "sdf_body_contact_steps_mean",
                    "sdf_body_warning_mean",
                    "sdf_body_warning_positive_insert_steps_mean",
                    "sdf_body_warning_positive_insert_fraction",
                    "raw_insert_action_mean", "effective_insert_action_mean",
                    "insert_positive_fraction", "insert_negative_fraction",
                    "reward_progress_mean", "reward_safety_mean",
                    "progress_to_safety_abs_ratio",
                ],
                {
                    "epoch": epoch,
                    "episodes": self.episodes_per_epoch,
                    "success_rate": train_success_rate,
                    "out_of_vessel_rate": out_of_vessel_rate,
                    "timeout_rate": timeout_rate,
                    "diagnostic_assessment": diagnostic_assessment,
                    "route_completion_mean": route_completion_mean,
                    "centerline_safety_ratio_max_mean": safety_ratio_max_mean,
                    "centerline_safety_margin_min_mean": safety_margin_min_mean,
                    "sdf_tip_clearance_min_mm_mean": tip_clearance_min_mm_mean,
                    "sdf_body_clearance_min_mm_mean": body_clearance_min_mm_mean,
                    "sdf_body_warning_steps_mean": body_warning_steps_mean,
                    "sdf_body_contact_steps_mean": body_contact_steps_mean,
                    "sdf_body_warning_mean": body_warning_mean,
                    "sdf_body_warning_positive_insert_steps_mean": warning_positive_insert_steps_mean,
                    "sdf_body_warning_positive_insert_fraction": warning_positive_insert_fraction,
                    "raw_insert_action_mean": raw_insert_action_mean,
                    "effective_insert_action_mean": insert_action_mean,
                    "insert_positive_fraction": insert_positive_fraction,
                    "insert_negative_fraction": insert_negative_fraction,
                    "reward_progress_mean": reward_progress_mean,
                    "reward_safety_mean": reward_safety_mean,
                    "progress_to_safety_abs_ratio": progress_to_safety_abs_ratio,
                },
            )
            vessel_rows = []
            for model_id in active_models:
                model_index = self._curriculum_model_to_index[model_id]
                episode_count = int(self._epoch_model_episode_counts[model_index])
                if episode_count <= 0:
                    continue
                divisor = float(episode_count)
                vessel_rows.append({
                    "epoch": epoch,
                    "vessel_id": model_id,
                    "episodes": episode_count,
                    "success_count": int(self._epoch_model_success_counts[model_index]),
                    "success_rate": float(self._epoch_model_success_counts[model_index] / divisor),
                    "out_of_vessel_count": int(self._epoch_model_out_of_vessel_counts[model_index]),
                    "out_of_vessel_rate": float(self._epoch_model_out_of_vessel_counts[model_index] / divisor),
                    "route_completion_mean": float(self._epoch_model_route_completion_sums[model_index] / divisor),
                    "sdf_body_clearance_min_mm_mean": float(self._epoch_model_body_clearance_min_sums[model_index] / divisor * 1000.0),
                    "centerline_safety_ratio_max_mean": float(self._epoch_model_safety_ratio_max_sums[model_index] / divisor),
                    "sdf_body_warning_steps_mean": float(self._epoch_model_body_warning_steps_sums[model_index] / divisor),
                    "sdf_body_contact_steps_mean": float(self._epoch_model_body_contact_steps_sums[model_index] / divisor),
                    "effective_insert_action_mean": float(self._epoch_model_effective_insert_sums[model_index] / divisor),
                })
            _append_csv_rows(
                self.run_dir / "diagnostics" / "vessel_summary.csv",
                [
                    "epoch", "vessel_id", "episodes", "success_count",
                    "success_rate", "out_of_vessel_count", "out_of_vessel_rate",
                    "route_completion_mean", "sdf_body_clearance_min_mm_mean",
                    "centerline_safety_ratio_max_mean",
                    "sdf_body_warning_steps_mean", "sdf_body_contact_steps_mean",
                    "effective_insert_action_mean",
                ],
                vessel_rows,
            )
            route_rows = []
            for model_id in active_models:
                if not str(model_id).startswith("B"):
                    continue
                model_index = self._curriculum_model_to_index[model_id]
                for route_offset in range(6):
                    episode_count = int(
                        self._epoch_route_episode_counts[model_index, route_offset]
                    )
                    if episode_count <= 0:
                        continue
                    divisor = float(episode_count)
                    route_rows.append({
                        "epoch": epoch,
                        "vessel_id": model_id,
                        "target_route_id": f"target_{route_offset + 1:02d}",
                        "episodes": episode_count,
                        "success_count": int(
                            self._epoch_route_success_counts[model_index, route_offset]
                        ),
                        "success_rate": float(
                            self._epoch_route_success_counts[model_index, route_offset]
                            / divisor
                        ),
                        "out_of_vessel_count": int(
                            self._epoch_route_out_of_vessel_counts[model_index, route_offset]
                        ),
                        "out_of_vessel_rate": float(
                            self._epoch_route_out_of_vessel_counts[model_index, route_offset]
                            / divisor
                        ),
                        "route_completion_mean": float(
                            self._epoch_route_completion_sums[model_index, route_offset]
                            / divisor
                        ),
                    })
            _append_csv_rows(
                self.run_dir / "diagnostics" / "route_summary.csv",
                [
                    "epoch", "vessel_id", "target_route_id", "episodes",
                    "success_count", "success_rate", "out_of_vessel_count",
                    "out_of_vessel_rate", "route_completion_mean",
                ],
                route_rows,
            )
            print(
                f"[TRAIN][Epoch {epoch:03d}] train_episodes={self.episodes_per_epoch} "
                f"train_success_count={self._epoch_success_count} "
                f"train_success_rate={train_success_rate:.6f} "
                f"train_reward_mean={train_reward_mean:.3f} "
                f"train_episode_steps_mean={train_episode_steps_mean:.1f} "
                f"out_of_vessel_rate={out_of_vessel_rate:.6f} "
                f"body_clearance_min_mm_mean={body_clearance_min_mm_mean:.3f} "
                f"body_warning_steps_mean={body_warning_steps_mean:.1f} "
                f"warning_forward_fraction={warning_positive_insert_fraction:.3f} "
                f"diagnostic={diagnostic_assessment} "
                f"positive_failure_rate={positive_failure_rate:.6f} "
                f"curriculum_stage_used={curriculum_stage_used} "
                f"curriculum_stage_name_used={curriculum_stage_name_used} "
                f"curriculum_stage={self.curriculum_stage} "
                f"curriculum_stage_name={TRAINING_CURRICULUM_STAGE_NAMES[self.curriculum_stage] if self.training_curriculum_enabled else 'disabled_all_vessels'} "
                f"curriculum_target_fraction_used={curriculum_target_fraction_used:.2f} "
                f"curriculum_target_fraction={self.curriculum_target_fraction:.2f} "
                f"curriculum_mastery_ready={curriculum_mastery_ready} "
                f"curriculum_mastery_success_rate={curriculum_mastery_success_rate:.6f} "
                f"curriculum_vessel_success_rates={json.dumps(curriculum_vessel_success_rates, sort_keys=True, separators=(',', ':'))} "
                f"curriculum_rolling_success_rates={json.dumps(curriculum_rolling_success_rates, sort_keys=True, separators=(',', ':'))} "
                f"curriculum_rolling_episode_counts={json.dumps(curriculum_rolling_episode_counts, sort_keys=True, separators=(',', ':'))} "
                f"curriculum_sampling_weights={json.dumps(self.curriculum_sampling_weights, sort_keys=True, separators=(',', ':'))} "
                f"curriculum_dr_profile={json.dumps(self.curriculum_dr_profile, sort_keys=True, separators=(',', ':'))} "
                f"curriculum_exploration_profile={json.dumps(self.curriculum_exploration_profile, sort_keys=True, separators=(',', ':'))} "
                f"curriculum_success_streak={self.curriculum_success_streak} "
                f"global_completed_episodes={self.global_completed_episodes} "
                f"global_env_steps={self.model.global_env_steps} "
                f"checkpoint={checkpoint_path}.zip"
            )
        # No rank may collect more rollout data until the epoch checkpoint is
        # complete.  Even epochs then enter the validation barriers below.
        self.context.barrier()
        self._run_validation(epoch, checkpoint_path)
        self._epoch_success_count = 0
        self._epoch_reward_sum = 0.0
        self._epoch_episode_steps_sum = 0.0
        self._epoch_out_of_vessel_count = 0
        self._epoch_wrong_branch_count = 0
        self._epoch_non_finite_count = 0
        self._epoch_timeout_count = 0
        self._epoch_no_progress_count = 0
        self._epoch_reward_progress_sum = 0.0
        self._epoch_reward_terminal_sum = 0.0
        self._epoch_reward_safety_sum = 0.0
        self._epoch_reward_step_sum = 0.0
        self._epoch_insert_action_mean_sum = 0.0
        self._epoch_insert_positive_fraction_sum = 0.0
        self._epoch_insert_negative_fraction_sum = 0.0
        self._epoch_inserted_length_final_sum = 0.0
        self._epoch_route_completion_sum = 0.0
        self._epoch_positive_failure_count = 0
        self._epoch_reward_component_total_sum = 0.0
        self._epoch_route_potential_sum = 0.0
        self._epoch_route_jump_rejections_sum = 0.0
        self._epoch_safety_ratio_max_sum = 0.0
        self._epoch_safety_margin_min_sum = 0.0
        self._epoch_tip_clearance_min_sum = 0.0
        self._epoch_body_clearance_min_sum = 0.0
        self._epoch_body_warning_steps_sum = 0.0
        self._epoch_body_contact_steps_sum = 0.0
        self._epoch_body_warning_mean_sum = 0.0
        self._epoch_body_warning_positive_insert_steps_sum = 0.0
        self._epoch_raw_insert_action_mean_sum = 0.0
        self._epoch_model_episode_counts.fill(0)
        self._epoch_model_success_counts.fill(0)
        self._epoch_model_jump_rejection_counts.fill(0.0)
        self._epoch_model_out_of_vessel_counts.fill(0)
        self._epoch_model_route_completion_sums.fill(0.0)
        self._epoch_model_body_clearance_min_sums.fill(0.0)
        self._epoch_model_safety_ratio_max_sums.fill(0.0)
        self._epoch_model_body_warning_steps_sums.fill(0.0)
        self._epoch_model_body_contact_steps_sums.fill(0.0)
        self._epoch_model_effective_insert_sums.fill(0.0)
        self._epoch_route_episode_counts.fill(0)
        self._epoch_route_success_counts.fill(0)
        self._epoch_route_out_of_vessel_counts.fill(0)
        self._epoch_route_completion_sums.fill(0.0)

    def _on_step(self) -> bool:
        self._pending_episode_events.append(self._local_episode_events())
        self._performance_step_count += 1
        if self._performance_step_count >= self.performance_log_interval_steps:
            performance = self.context.performance_snapshot(reset=True)
            print(
                f"[PERF][Rank {self.context.rank}] "
                f"vector_steps={self._performance_step_count} "
                f"elapsed_s={performance['elapsed_seconds']:.3f} "
                f"rollout_other_s={performance['rollout_other_seconds']:.3f} "
                f"update_s={performance['update_seconds']:.3f} "
                f"communication_s={performance['communication_seconds']:.3f} "
                f"communication_calls={performance['communication_calls']}"
            )
            self._performance_step_count = 0
        sync_interval = self.episode_sync_interval_steps if self.context.enabled else 1
        if len(self._pending_episode_events) < sync_interval:
            return True

        events = self._global_pending_episode_events()
        self._pending_episode_events.clear()
        completed_events = events[events[:, 0] != 0]
        cursor = 0
        while cursor < len(completed_events) and self.global_completed_episodes < self.target_episodes:
            remaining = self.episodes_per_epoch - (
                self.global_completed_episodes % self.episodes_per_epoch
            )
            take = min(remaining, len(completed_events) - cursor)
            accepted = completed_events[cursor : cursor + take]
            global_episode_start = self.global_completed_episodes
            self.global_completed_episodes += len(accepted)
            self._append_train_episode_rows(accepted, global_episode_start)
            self._epoch_success_count += int(accepted[:, 1].sum())
            self._epoch_reward_sum += float(accepted[:, 2].sum())
            self._epoch_episode_steps_sum += float(accepted[:, 3].sum())
            self._epoch_out_of_vessel_count += int(accepted[:, 4].sum())
            self._epoch_wrong_branch_count += int(accepted[:, 5].sum())
            self._epoch_non_finite_count += int(accepted[:, 6].sum())
            self._epoch_timeout_count += int(accepted[:, 7].sum())
            self._epoch_no_progress_count += int(accepted[:, 8].sum())
            self._epoch_reward_progress_sum += float(accepted[:, 9].sum())
            self._epoch_reward_terminal_sum += float(accepted[:, 10].sum())
            self._epoch_reward_safety_sum += float(accepted[:, 11].sum())
            self._epoch_reward_step_sum += float(accepted[:, 12].sum())
            self._epoch_insert_action_mean_sum += float(accepted[:, 13].sum())
            self._epoch_insert_positive_fraction_sum += float(accepted[:, 14].sum())
            self._epoch_insert_negative_fraction_sum += float(accepted[:, 15].sum())
            self._epoch_inserted_length_final_sum += float(accepted[:, 16].sum())
            self._epoch_route_completion_sum += float(accepted[:, 17].sum())
            self._epoch_positive_failure_count += int(accepted[:, 18].sum())
            self._epoch_reward_component_total_sum += float(accepted[:, 19].sum())
            self._epoch_route_potential_sum += float(accepted[:, 20].sum())
            self._epoch_route_jump_rejections_sum += float(accepted[:, 22].sum())
            finite_diagnostics = np.nan_to_num(
                accepted,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            self._epoch_raw_insert_action_mean_sum += float(finite_diagnostics[:, 44].sum())
            self._epoch_safety_ratio_max_sum += float(finite_diagnostics[:, 48].sum())
            self._epoch_safety_margin_min_sum += float(finite_diagnostics[:, 50].sum())
            self._epoch_tip_clearance_min_sum += float(finite_diagnostics[:, 41].sum())
            self._epoch_body_clearance_min_sum += float(finite_diagnostics[:, 42].sum())
            self._epoch_body_warning_steps_sum += float(finite_diagnostics[:, 52].sum())
            self._epoch_body_contact_steps_sum += float(finite_diagnostics[:, 53].sum())
            self._epoch_body_warning_mean_sum += float(finite_diagnostics[:, 54].sum())
            self._epoch_body_warning_positive_insert_steps_sum += float(
                finite_diagnostics[:, 55].sum()
            )
            for event in accepted:
                model_index = int(round(float(event[21])))
                episode_curriculum_stage = int(round(float(event[23])))
                if (
                    episode_curriculum_stage == self.curriculum_stage
                    and 0 <= model_index < len(self._curriculum_model_ids)
                ):
                    self._epoch_model_episode_counts[model_index] += 1
                    success = int(event[1] > 0.5)
                    self._epoch_model_success_counts[model_index] += success
                    self._epoch_model_jump_rejection_counts[model_index] += float(
                        event[22]
                    )
                    finite_event = np.nan_to_num(
                        event,
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )
                    self._epoch_model_out_of_vessel_counts[model_index] += int(
                        event[4] > 0.5
                    )
                    self._epoch_model_route_completion_sums[model_index] += float(
                        finite_event[17]
                    )
                    self._epoch_model_body_clearance_min_sums[model_index] += float(
                        finite_event[42]
                    )
                    self._epoch_model_safety_ratio_max_sums[model_index] += float(
                        finite_event[48]
                    )
                    self._epoch_model_body_warning_steps_sums[model_index] += float(
                        finite_event[52]
                    )
                    self._epoch_model_body_contact_steps_sums[model_index] += float(
                        finite_event[53]
                    )
                    self._epoch_model_effective_insert_sums[model_index] += float(
                        finite_event[13]
                    )
                    model_id = self._curriculum_model_ids[model_index]
                    self._curriculum_outcome_windows[model_id].append(success)
                    route_index = int(round(float(event[60])))
                    if 1 <= route_index <= 6 and str(model_id).startswith("B"):
                        route_offset = route_index - 1
                        self._epoch_route_episode_counts[model_index, route_offset] += 1
                        self._epoch_route_success_counts[model_index, route_offset] += success
                        self._epoch_route_out_of_vessel_counts[model_index, route_offset] += int(
                            event[4] > 0.5
                        )
                        self._epoch_route_completion_sums[model_index, route_offset] += float(
                            finite_event[17]
                        )
                        route_key = f"{model_id}/target_{route_index:02d}"
                        self._curriculum_route_outcome_windows[route_key].append(success)
                    if success:
                        self.curriculum_success_streak += 1
                    else:
                        self.curriculum_success_streak = 0
            cursor += take
            if (
                self.next_epoch <= self.epochs
                and self.global_completed_episodes
                == self.next_epoch * self.episodes_per_epoch
            ):
                self._finish_epoch(self.next_epoch)
                self.next_epoch += 1
        return self.global_completed_episodes < self.target_episodes
