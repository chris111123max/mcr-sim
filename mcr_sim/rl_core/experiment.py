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
    TRAINING_CURRICULUM_ROLLING_EPISODES_PER_VESSEL,
    TRAINING_CURRICULUM_TARGET_FRACTIONS,
    curriculum_exploration_profile,
    curriculum_sampling_weights,
    update_curriculum_progress,
    update_validation_unlocked,
)


def _append_csv(path: Path, fieldnames, row) -> None:
    path = Path(path)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


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
        self._epoch_reward_waypoints_sum = 0.0
        self._epoch_reward_terminal_sum = 0.0
        self._epoch_reward_safety_sum = 0.0
        self._epoch_reward_behavior_sum = 0.0
        self._epoch_reward_retraction_sum = 0.0
        self._epoch_reward_no_progress_dense_sum = 0.0
        self._epoch_reward_step_sum = 0.0
        self._epoch_insert_action_mean_sum = 0.0
        self._epoch_insert_positive_fraction_sum = 0.0
        self._epoch_insert_negative_fraction_sum = 0.0
        self._epoch_inserted_length_final_sum = 0.0
        self._epoch_waypoint_ratio_sum = 0.0
        self._epoch_positive_failure_count = 0
        self._epoch_reward_component_total_sum = 0.0
        self._epoch_route_potential_sum = 0.0
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
        self._curriculum_outcome_windows = {
            model_id: deque(
                maxlen=TRAINING_CURRICULUM_ROLLING_EPISODES_PER_VESSEL
            )
            for model_id in self._curriculum_model_ids
        }
        self._pending_episode_events = []
        self.best_valid_success_rate = -1.0
        self.best_valid_waypoint_reached_ratio = -1.0
        self.best_valid_route_potential = -1.0
        self.best_valid_final_distance_mm = float("inf")
        self.best_epoch = 0

    def _on_training_start(self) -> None:
        if self.resume_progress:
            self.best_valid_success_rate = float(
                getattr(self.model, "best_valid_success_rate", -1.0)
            )
            self.best_epoch = int(getattr(self.model, "best_epoch", 0))
            self.best_valid_waypoint_reached_ratio = float(
                getattr(
                    self.model,
                    "best_valid_waypoint_reached_ratio",
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
                and TRAINING_CURRICULUM_TARGET_FRACTIONS[
                    min(
                        max(self.curriculum_stage, 0),
                        len(TRAINING_CURRICULUM_TARGET_FRACTIONS) - 1,
                    )
                ]
                < 1.0 - 1e-9
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
        self._apply_curriculum_stage(self.curriculum_stage)
        active_models = TRAINING_CURRICULUM_MODELS[self.curriculum_stage]
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
        elif self.algorithm_name == "sac":
            floor = float(profile["sac_min_ent_coef"])
            self.model.min_ent_coef = floor
            log_ent_coef = getattr(self.model, "log_ent_coef", None)
            if log_ent_coef is not None:
                with th.no_grad():
                    log_ent_coef.clamp_(min=float(np.log(floor)))

    def _local_episode_events(self):
        dones = np.asarray(self.locals.get("dones", []), dtype=np.bool_).reshape(-1)
        infos = list(self.locals.get("infos", []))
        events = np.zeros((len(dones), 26), dtype=np.float32)
        for index, done in enumerate(dones):
            if not done:
                continue
            info = infos[index]
            episode_info = info.get("episode", {})
            waypoint_count = float(info.get("waypoint_reached_count_episode", 0.0))
            waypoint_num = float(info.get("waypoint_num", 0.0))
            waypoint_ratio = (
                float(np.clip(waypoint_count / max(1.0, waypoint_num - 1.0), 0.0, 1.0))
                if waypoint_num > 1.0
                else 0.0
            )
            reward_progress = float(info.get("episode_reward_waypoint_approach", 0.0)) + float(
                info.get("episode_reward_target_approach", 0.0)
            )
            reward_terminal = sum(
                float(info.get(key, 0.0))
                for key in (
                    "episode_reward_successful_task",
                    "episode_reward_out_of_vessel_penalty",
                    "episode_reward_wrong_branch_penalty",
                    "episode_reward_non_finite_penalty",
                    "episode_reward_timeout_penalty",
                    "episode_reward_no_progress_terminal_penalty",
                )
            )
            reward_safety = sum(
                float(info.get(key, 0.0))
                for key in (
                    "episode_reward_wall_proximity_penalty",
                    "episode_reward_wall_penetration_penalty",
                    "episode_reward_off_target_branch_penalty",
                )
            )
            reward_retraction = float(
                info.get("episode_reward_retraction_penalty", 0.0)
            )
            reward_no_progress_dense = float(
                info.get("episode_reward_no_progress_penalty", 0.0)
            )
            reward_behavior = reward_retraction + reward_no_progress_dense
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
                float(bool(info.get("done_by_wrong_branch", False))),
                float(bool(info.get("done_by_non_finite", False))),
                float(bool(info.get("TimeLimit.truncated", False) or info.get("terminal_reason") == "timeout")),
                float(bool(info.get("done_by_no_progress", False))),
                reward_progress,
                float(info.get("episode_reward_waypoint_reached", 0.0)),
                reward_terminal,
                reward_safety,
                reward_behavior,
                float(info.get("episode_reward_step_penalty", 0.0)),
                float(info.get("insert_action_mean_episode", 0.0)),
                float(info.get("insert_positive_fraction_episode", 0.0)),
                float(info.get("insert_negative_fraction_episode", 0.0)),
                float(info.get("inserted_length_final", 0.0)),
                1.0 if bool(info.get("done_by_target", False)) else waypoint_ratio,
                float(bool(info.get("positive_failure_return", False))),
                float(info.get("episode_reward_total_components", episode_info.get("r", 0.0))),
                float(info.get("route_potential", 0.0)),
                float(model_index),
                reward_retraction,
                reward_no_progress_dense,
            ]
        return events

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

    def _rolling_curriculum_statistics(self, active_models):
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
            "valid_success_rate,waypoint_reached_ratio,"
            "route_potential,-final_distance_mm"
        )
        self.model.best_valid_success_rate = float(self.best_valid_success_rate)
        self.model.best_valid_waypoint_reached_ratio = float(
            self.best_valid_waypoint_reached_ratio
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
        self.model.curriculum_success_streak = int(self.curriculum_success_streak)
        self.model.curriculum_rolling_outcomes = {
            model_id: list(window)
            for model_id, window in self._curriculum_outcome_windows.items()
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
                    float(self.best_valid_waypoint_reached_ratio),
                    float(self.best_valid_route_potential),
                    -float(self.best_valid_final_distance_mm),
                )
                is_new_best = candidate_key > best_key
                if is_new_best:
                    self.best_valid_success_rate = valid_success_rate
                    self.best_valid_waypoint_reached_ratio = float(
                        result.valid_waypoint_reached_ratio_mean
                    )
                    self.best_valid_route_potential = float(
                        result.valid_route_potential_mean
                    )
                    self.best_valid_final_distance_mm = float(
                        result.valid_final_distance_mm_mean
                    )
                    self.best_epoch = int(epoch)
                    self.model.best_valid_success_rate = valid_success_rate
                    self.model.best_valid_waypoint_reached_ratio = float(
                        self.best_valid_waypoint_reached_ratio
                    )
                    self.model.best_valid_route_potential = float(
                        self.best_valid_route_potential
                    )
                    self.model.best_valid_final_distance_mm = float(
                        self.best_valid_final_distance_mm
                    )
                    self.model.best_epoch = int(epoch)
                    self.model.best_metric = (
                        "valid_success_rate,waypoint_reached_ratio,"
                        "route_potential,-final_distance_mm"
                    )
                    self.model.last_valid_success_rate = valid_success_rate
                    self.model.last_valid_epoch = int(epoch)
                    self.model.save(str(self.model_dir / "best_valid"))
                else:
                    self.model.last_valid_success_rate = valid_success_rate
                    self.model.last_valid_epoch = int(epoch)
                self.model.last_valid_waypoint_reached_ratio = float(
                    result.valid_waypoint_reached_ratio_mean
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
                self.model.best_valid_waypoint_reached_ratio = float(
                    self.best_valid_waypoint_reached_ratio
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
                    "valid/waypoint_reached_ratio",
                    float(result.valid_waypoint_reached_ratio_mean),
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
                        "valid_success_rate", "valid_waypoint_reached_ratio_mean",
                        "valid_route_potential_mean", "valid_final_distance_mm_mean",
                        "valid_min_distance_mm_mean", "best_valid_success_rate",
                        "best_valid_waypoint_reached_ratio", "best_valid_route_potential",
                        "best_valid_final_distance_mm", "is_new_best", "checkpoint",
                    ],
                    {
                        "epoch": epoch,
                        "valid_vessels": result.valid_vessels,
                        "valid_episodes": result.valid_episodes,
                        "valid_success_count": result.valid_success_count,
                        "valid_success_rate": valid_success_rate,
                        "valid_waypoint_reached_ratio_mean": result.valid_waypoint_reached_ratio_mean,
                        "valid_route_potential_mean": result.valid_route_potential_mean,
                        "valid_final_distance_mm_mean": result.valid_final_distance_mm_mean,
                        "valid_min_distance_mm_mean": result.valid_min_distance_mm_mean,
                        "best_valid_success_rate": self.best_valid_success_rate,
                        "best_valid_waypoint_reached_ratio": self.best_valid_waypoint_reached_ratio,
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
                            "waypoint_reached_ratio", "route_potential",
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
                            "waypoint_reached_ratio": episode_result.waypoint_reached_ratio,
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
                    f"valid_waypoint_reached_ratio={result.valid_waypoint_reached_ratio_mean:.6f} "
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
        reward_waypoints_mean = self._epoch_reward_waypoints_sum / epoch_divisor
        reward_terminal_mean = self._epoch_reward_terminal_sum / epoch_divisor
        reward_safety_mean = self._epoch_reward_safety_sum / epoch_divisor
        reward_behavior_mean = self._epoch_reward_behavior_sum / epoch_divisor
        reward_retraction_mean = self._epoch_reward_retraction_sum / epoch_divisor
        reward_no_progress_dense_mean = (
            self._epoch_reward_no_progress_dense_sum / epoch_divisor
        )
        reward_step_mean = self._epoch_reward_step_sum / epoch_divisor
        insert_action_mean = self._epoch_insert_action_mean_sum / epoch_divisor
        insert_positive_fraction = self._epoch_insert_positive_fraction_sum / epoch_divisor
        insert_negative_fraction = self._epoch_insert_negative_fraction_sum / epoch_divisor
        inserted_length_final_mean_mm = (
            self._epoch_inserted_length_final_sum / epoch_divisor * 1000.0
        )
        waypoint_reached_ratio_mean = self._epoch_waypoint_ratio_sum / epoch_divisor
        positive_failure_rate = self._epoch_positive_failure_count / epoch_divisor
        reward_component_total_mean = self._epoch_reward_component_total_sum / epoch_divisor
        route_potential_final_mean = self._epoch_route_potential_sum / epoch_divisor
        curriculum_stage_used = self.curriculum_stage
        curriculum_target_fraction_used = float(
            TRAINING_CURRICULUM_TARGET_FRACTIONS[curriculum_stage_used]
            if self.training_curriculum_enabled
            else 1.0
        )
        active_models = TRAINING_CURRICULUM_MODELS[curriculum_stage_used]
        curriculum_vessel_episode_counts = {
            model_id: int(
                self._epoch_model_episode_counts[
                    self._curriculum_model_to_index[model_id]
                ]
            )
            for model_id in active_models
        }
        curriculum_vessel_success_rates = {}
        for model_id in active_models:
            model_index = self._curriculum_model_to_index[model_id]
            episode_count = int(self._epoch_model_episode_counts[model_index])
            curriculum_vessel_success_rates[model_id] = (
                float(self._epoch_model_success_counts[model_index] / episode_count)
                if episode_count > 0
                else None
            )
        (
            curriculum_rolling_success_rates,
            curriculum_rolling_episode_counts,
            curriculum_mastery_ready,
            curriculum_mastery_success_rate,
        ) = self._rolling_curriculum_statistics(active_models)
        curriculum_mastery_ready = bool(curriculum_mastery_ready)
        if not curriculum_mastery_ready:
            self.curriculum_success_streak = 0
        if self.training_curriculum_enabled:
            next_curriculum_stage, self.curriculum_success_streak = (
                update_curriculum_progress(
                    self.curriculum_stage,
                    self.curriculum_success_streak,
                    train_success_rate,
                    curriculum_rolling_success_rates,
                    curriculum_rolling_episode_counts,
                )
            )
            if next_curriculum_stage != self.curriculum_stage:
                self.curriculum_stage = next_curriculum_stage
                for window in self._curriculum_outcome_windows.values():
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
            full_task_ready=curriculum_target_fraction_used >= 1.0 - 1e-9,
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
            self.logger.record("train/reward_waypoints_mean", reward_waypoints_mean, exclude="stdout")
            self.logger.record("train/reward_terminal_mean", reward_terminal_mean, exclude="stdout")
            self.logger.record("train/reward_safety_mean", reward_safety_mean, exclude="stdout")
            self.logger.record("train/reward_behavior_mean", reward_behavior_mean, exclude="stdout")
            self.logger.record("train/reward_retraction_mean", reward_retraction_mean, exclude="stdout")
            self.logger.record("train/reward_no_progress_dense_mean", reward_no_progress_dense_mean, exclude="stdout")
            self.logger.record("train/reward_step_mean", reward_step_mean, exclude="stdout")
            self.logger.record("train/insert_action_mean", insert_action_mean, exclude="stdout")
            self.logger.record("train/insert_positive_fraction", insert_positive_fraction, exclude="stdout")
            self.logger.record("train/insert_negative_fraction", insert_negative_fraction, exclude="stdout")
            self.logger.record("train/inserted_length_final_mean_mm", inserted_length_final_mean_mm, exclude="stdout")
            self.logger.record("train/waypoint_reached_ratio_mean", waypoint_reached_ratio_mean, exclude="stdout")
            self.logger.record("train/positive_failure_rate", positive_failure_rate)
            self.logger.record("train/reward_component_total_mean", reward_component_total_mean, exclude="stdout")
            self.logger.record("train/route_potential_final_mean", route_potential_final_mean, exclude="stdout")
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
                ["epoch", "train_episodes", "train_success_count", "train_success_rate", "train_reward_mean", "train_episode_steps_mean", "out_of_vessel_count", "wrong_branch_count", "non_finite_count", "timeout_count", "no_progress_count", "positive_failure_count", "positive_failure_rate", "reward_progress_mean", "reward_waypoints_mean", "reward_terminal_mean", "reward_safety_mean", "reward_behavior_mean", "reward_retraction_mean", "reward_no_progress_dense_mean", "reward_step_mean", "reward_component_total_mean", "route_potential_final_mean", "curriculum_stage_used", "curriculum_stage", "curriculum_target_fraction_used", "curriculum_target_fraction", "curriculum_success_streak", "curriculum_mastery_ready", "curriculum_mastery_success_rate", "curriculum_vessel_success_rates", "curriculum_vessel_episode_counts", "curriculum_rolling_success_rates", "curriculum_rolling_episode_counts", "curriculum_sampling_weights", "curriculum_dr_profile", "curriculum_exploration_profile", "insert_action_mean", "insert_positive_fraction", "insert_negative_fraction", "inserted_length_final_mean_mm", "waypoint_reached_ratio_mean", "global_completed_episodes", "global_env_steps", "checkpoint"],
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
                    "reward_waypoints_mean": reward_waypoints_mean,
                    "reward_terminal_mean": reward_terminal_mean,
                    "reward_safety_mean": reward_safety_mean,
                    "reward_behavior_mean": reward_behavior_mean,
                    "reward_retraction_mean": reward_retraction_mean,
                    "reward_no_progress_dense_mean": reward_no_progress_dense_mean,
                    "reward_step_mean": reward_step_mean,
                    "reward_component_total_mean": reward_component_total_mean,
                    "route_potential_final_mean": route_potential_final_mean,
                    "curriculum_stage_used": curriculum_stage_used,
                    "curriculum_stage": self.curriculum_stage,
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
                    "waypoint_reached_ratio_mean": waypoint_reached_ratio_mean,
                    "global_completed_episodes": self.global_completed_episodes,
                    "global_env_steps": self.model.global_env_steps,
                    "checkpoint": str(checkpoint_path) + ".zip",
                },
            )
            print(
                f"[TRAIN][Epoch {epoch:03d}] train_episodes={self.episodes_per_epoch} "
                f"train_success_count={self._epoch_success_count} "
                f"train_success_rate={train_success_rate:.6f} "
                f"train_reward_mean={train_reward_mean:.3f} "
                f"train_episode_steps_mean={train_episode_steps_mean:.1f} "
                f"positive_failure_rate={positive_failure_rate:.6f} "
                f"curriculum_stage_used={curriculum_stage_used} "
                f"curriculum_stage={self.curriculum_stage} "
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
        self._epoch_reward_waypoints_sum = 0.0
        self._epoch_reward_terminal_sum = 0.0
        self._epoch_reward_safety_sum = 0.0
        self._epoch_reward_behavior_sum = 0.0
        self._epoch_reward_retraction_sum = 0.0
        self._epoch_reward_no_progress_dense_sum = 0.0
        self._epoch_reward_step_sum = 0.0
        self._epoch_insert_action_mean_sum = 0.0
        self._epoch_insert_positive_fraction_sum = 0.0
        self._epoch_insert_negative_fraction_sum = 0.0
        self._epoch_inserted_length_final_sum = 0.0
        self._epoch_waypoint_ratio_sum = 0.0
        self._epoch_positive_failure_count = 0
        self._epoch_reward_component_total_sum = 0.0
        self._epoch_route_potential_sum = 0.0
        self._epoch_model_episode_counts.fill(0)
        self._epoch_model_success_counts.fill(0)

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
            self.global_completed_episodes += len(accepted)
            self._epoch_success_count += int(accepted[:, 1].sum())
            self._epoch_reward_sum += float(accepted[:, 2].sum())
            self._epoch_episode_steps_sum += float(accepted[:, 3].sum())
            self._epoch_out_of_vessel_count += int(accepted[:, 4].sum())
            self._epoch_wrong_branch_count += int(accepted[:, 5].sum())
            self._epoch_non_finite_count += int(accepted[:, 6].sum())
            self._epoch_timeout_count += int(accepted[:, 7].sum())
            self._epoch_no_progress_count += int(accepted[:, 8].sum())
            self._epoch_reward_progress_sum += float(accepted[:, 9].sum())
            self._epoch_reward_waypoints_sum += float(accepted[:, 10].sum())
            self._epoch_reward_terminal_sum += float(accepted[:, 11].sum())
            self._epoch_reward_safety_sum += float(accepted[:, 12].sum())
            self._epoch_reward_behavior_sum += float(accepted[:, 13].sum())
            self._epoch_reward_step_sum += float(accepted[:, 14].sum())
            self._epoch_insert_action_mean_sum += float(accepted[:, 15].sum())
            self._epoch_insert_positive_fraction_sum += float(accepted[:, 16].sum())
            self._epoch_insert_negative_fraction_sum += float(accepted[:, 17].sum())
            self._epoch_inserted_length_final_sum += float(accepted[:, 18].sum())
            self._epoch_waypoint_ratio_sum += float(accepted[:, 19].sum())
            self._epoch_positive_failure_count += int(accepted[:, 20].sum())
            self._epoch_reward_component_total_sum += float(accepted[:, 21].sum())
            self._epoch_route_potential_sum += float(accepted[:, 22].sum())
            self._epoch_reward_retraction_sum += float(accepted[:, 24].sum())
            self._epoch_reward_no_progress_dense_sum += float(accepted[:, 25].sum())
            for event in accepted:
                model_index = int(round(float(event[23])))
                if 0 <= model_index < len(self._curriculum_model_ids):
                    self._epoch_model_episode_counts[model_index] += 1
                    success = int(event[1] > 0.5)
                    self._epoch_model_success_counts[model_index] += success
                    model_id = self._curriculum_model_ids[model_index]
                    self._curriculum_outcome_windows[model_id].append(success)
            cursor += take
            if (
                self.next_epoch <= self.epochs
                and self.global_completed_episodes
                == self.next_epoch * self.episodes_per_epoch
            ):
                self._finish_epoch(self.next_epoch)
                self.next_epoch += 1
        return self.global_completed_episodes < self.target_episodes
