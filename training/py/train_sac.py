import argparse
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Union

# This executable lives in python/training/py/.  Resolve imports and the
# complete project root from the file itself, independent of caller cwd.
TRAINING_PY_DIR = Path(__file__).resolve().parent
TRAINING_DIR = TRAINING_PY_DIR.parent
PYTHON_ROOT = TRAINING_DIR.parent
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from collections import deque, defaultdict

import numpy as np
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import get_schedule_fn

from mcr_sim.mcr_rl_env import MCREnv, ObservationType, ActionType, EnvType
from mcr_sim.distributed import DistributedSAC, initialize_distributed
from mcr_sim.paths import PROJECT_ROOT, TRAINING_RUNS_DIR
from mcr_sim.rl_core.base import RenderMode, RenderFramework
from mcr_sim.training_config import (
    ACTOR_HISTORY_STEPS,
    ENTRY_TANGENT_POINTS,
    FRAME_SKIP,
    INITIAL_ORIENTATION_MAX_ANGLE_DEG,
    MAX_EPISODE_STEPS,
    RADIUS_OBSERVATION_SCALE_M,
    REWARD_OUT_OF_VESSEL,
    REWARD_OFF_TARGET_BRANCH,
    REWARD_STEP,
    REWARD_SUCCESS,
    REWARD_TARGET_APPROACH,
    REWARD_TIMEOUT,
    REWARD_WALL_PENETRATION,
    REWARD_WALL_PROXIMITY,
    REWARD_WAYPOINT_APPROACH,
    REWARD_WAYPOINT_REACHED,
    REWARD_WRONG_BRANCH,
    SAC_BATCH_SIZE,
    SAC_BUFFER_SIZE,
    SAC_EPOCHS,
    SAC_GAMMA,
    SAC_GRADIENT_STEPS,
    SAC_LEARNING_RATE,
    SAC_LEARNING_STARTS,
    SAC_N_ENVS,
    SAC_STEPS_PER_EPOCH,
    SAC_TAU,
    SAC_TRAIN_FREQ,
    SETTLE_STEPS,
    SDF_OUTSIDE_CENTER_TOLERANCE_M,
    SDF_OUTSIDE_CONFIRM_STEPS,
    SOFA_TIME_STEP_S,
    START_WINDOW_DISTANCE_M,
    TARGET_THRESHOLD_M,
    TARGET_WINDOW_DISTANCE_M,
    VESSEL_SCALE_MAX,
    VESSEL_SCALE_MIN,
    WRONG_BRANCH_CONFIRM_STEPS,
    WRONG_BRANCH_DISTANCE_MARGIN_M,
)

DEFAULT_LOG_ROOT = TRAINING_RUNS_DIR

ARTIFICIAL_MODEL_IDS = [f"C{i:02d}" for i in range(1, 6)] + [
    f"B{i:02d}" for i in range(1, 6)
]

ALL_MODEL_CHOICES = [
    "",
    "0207",
    "0207_left",
    "0207_right",
    "0210",
    "V1",
    "0021",
    "0028",
    "0038",
    "0230",
    "0231",
    "0237",
    "aorta6",
    "S1",
] + ARTIFICIAL_MODEL_IDS

TASK_IDS_FOR_LOGGING = [
    "0207_left",
    "0207_right",
    "0210",
    "V1",
    "0021",
    "0028",
    "0038",
    "0230",
    "0231",
    "0237",
    "aorta6",
] + ARTIFICIAL_MODEL_IDS

MODEL_IDS_FOR_LOGGING = [
    "0207",
    "0207_left",
    "0207_right",
    "0210",
    "V1",
    "0021",
    "0028",
    "0038",
    "0230",
    "0231",
    "0237",
    "aorta6",
    "S1",
] + ARTIFICIAL_MODEL_IDS


class ExtraRolloutMetricsCallback(BaseCallback):
    """Compact TensorBoard metrics for waypoint SAC training."""

    def __init__(self, window_size: int = 50, success_label: str = "target", verbose: int = 0):
        super().__init__(verbose)
        self.window_size = int(window_size)
        self.success_label = str(success_label)
        self.recent_episodes = deque(maxlen=self.window_size)
        self.task_windows = defaultdict(lambda: deque(maxlen=self.window_size))
        self.model_windows = defaultdict(lambda: deque(maxlen=self.window_size))
        self.total_episodes = 0
        self.total_success_target = 0

    @staticmethod
    def _safe_float(value, default=np.nan) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _mean(values, default=np.nan) -> float:
        cleaned = []
        for value in values:
            try:
                value = float(value)
            except Exception:
                continue
            if not np.isnan(value):
                cleaned.append(value)
        return float(np.mean(cleaned)) if cleaned else float(default)

    @staticmethod
    def _max(values, default=np.nan) -> float:
        cleaned = []
        for value in values:
            try:
                value = float(value)
            except Exception:
                continue
            if not np.isnan(value):
                cleaned.append(value)
        return float(np.max(cleaned)) if cleaned else float(default)

    @staticmethod
    def _rate(values) -> float:
        values = list(values)
        return float(np.mean(values)) if values else float("nan")

    def _episode_from_info(self, info: dict) -> dict:
        episode_info = info.get("episode", {}) if isinstance(info.get("episode", {}), dict) else {}

        waypoint_reached_count = self._safe_float(info.get("waypoint_reached_count_episode", np.nan))
        waypoint_num = self._safe_float(info.get("waypoint_num", np.nan))

        # Single clean waypoint-progress metric:
        #   0.0 = no waypoint reached in this episode
        #   1.0 = all waypoints/final target reached
        # This is based only on actually reached waypoints, not on the active waypoint index
        # initialized by centerline projection.
        if np.isfinite(waypoint_num) and waypoint_num > 1:
            waypoint_reached_ratio = float(np.clip(waypoint_reached_count / max(1.0, waypoint_num - 1.0), 0.0, 1.0))
        else:
            waypoint_reached_ratio = float("nan")
        if bool(info.get("done_by_target", False)):
            waypoint_reached_ratio = 1.0

        return {
            "task_id": str(info.get("task_id", "unknown")),
            "chosen_model": str(info.get("chosen_model", info.get("task_id", "unknown"))),
            "sampling_model": str(info.get("sampling_model", info.get("chosen_model", "unknown"))),
            "success_target": bool(info.get("done_by_target", False)),
            "success_2mm": bool(info.get("success_2mm", False)),
            "min_dist_m": self._safe_float(info.get("min_dist_to_goal", np.nan)),
            "final_dist_m": self._safe_float(info.get("final_dist_to_goal", np.nan)),
            "terminal_reason": str(info.get("terminal_reason", "unknown")),
            "done_by_target": bool(info.get("done_by_target", False)),
            "done_by_timeout": bool(info.get("done_by_timeout", False)),
            "done_by_out_of_vessel": bool(info.get("done_by_out_of_vessel", False)),
            "done_by_wrong_branch": bool(info.get("done_by_wrong_branch", False)),
            "done_by_non_finite": bool(info.get("done_by_non_finite", False)),
            "ep_len": self._safe_float(episode_info.get("l", np.nan)),
            "ep_reward": self._safe_float(episode_info.get("r", np.nan)),
            "waypoint_reached_ratio": waypoint_reached_ratio,
            "waypoint_reached_count": waypoint_reached_count,
            "waypoint_num": waypoint_num,
            "waypoint_distance_m": self._safe_float(info.get("waypoint_distance", np.nan)),
            "centerline_safety_ratio": self._safe_float(info.get("centerline_safety_ratio", np.nan)),
            "centerline_safety_ratio_max_episode": self._safe_float(info.get("centerline_safety_ratio_max_episode", np.nan)),
            "centerline_safety_margin": self._safe_float(info.get("centerline_safety_margin", np.nan)),
            "centerline_safety_margin_min_episode": self._safe_float(info.get("centerline_safety_margin_min_episode", np.nan)),
            "out_of_vessel": bool(info.get("out_of_vessel_this_episode", False)),
            "wrong_branch": bool(info.get("wrong_branch_this_episode", False)),
            "sdf_surface_clearance_m": self._safe_float(
                info.get("sdf_surface_clearance", np.nan)
            ),
            "sdf_surface_clearance_min_m": self._safe_float(
                info.get("sdf_surface_clearance_min_episode", np.nan)
            ),
            "sdf_wall_contact_steps": self._safe_float(
                info.get("sdf_wall_contact_steps_episode", 0.0)
            ),
            "route_graph_distance_gap_m": self._safe_float(
                info.get("route_graph_distance_gap", np.nan)
            ),
            "vessel_scale_factor": self._safe_float(info.get("vessel_scale_factor", np.nan)),
            "reward_progress": self._safe_float(
                info.get("episode_reward_waypoint_approach", 0.0)
            )
            + self._safe_float(info.get("episode_reward_target_approach", 0.0)),
            "reward_waypoints": self._safe_float(
                info.get("episode_reward_waypoint_reached", 0.0)
            ),
            "reward_terminal": self._safe_float(
                info.get("episode_reward_successful_task", 0.0)
            )
            + self._safe_float(info.get("episode_reward_out_of_vessel_penalty", 0.0))
            + self._safe_float(info.get("episode_reward_wrong_branch_penalty", 0.0))
            + self._safe_float(info.get("episode_reward_timeout_penalty", 0.0)),
            "reward_safety": self._safe_float(
                info.get("episode_reward_wall_proximity_penalty", 0.0)
            )
            + self._safe_float(
                info.get("episode_reward_wall_penetration_penalty", 0.0)
            )
            + self._safe_float(
                info.get("episode_reward_off_target_branch_penalty", 0.0)
            ),
            "reward_step": self._safe_float(info.get("episode_reward_step_penalty", 0.0)),
        }

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        dones = self.locals.get("dones")
        if infos is not None and dones is not None:
            for done, info in zip(dones, infos):
                if not done:
                    continue
                ep = self._episode_from_info(info)
                self.recent_episodes.append(ep)
                self.task_windows[ep["task_id"]].append(ep)

                # Avoid double-counting fixed-vessel runs where chosen_model == sampling_model.
                for model_key in {ep["chosen_model"], ep["sampling_model"]}:
                    self.model_windows[model_key].append(ep)

                self.total_episodes += 1
                if ep["success_target"]:
                    self.total_success_target += 1
        return True

    def _log_window(self, prefix: str, window) -> None:
        window = list(window)
        if len(window) == 0:
            return
        self.logger.record(f"{prefix}/success_{self.success_label}_w{self.window_size}", self._rate(ep["success_target"] for ep in window))
        self.logger.record(f"{prefix}/min_dist_mm_w{self.window_size}", self._mean(ep["min_dist_m"] * 1000.0 for ep in window))

        # The only waypoint progress metric kept in TensorBoard.
        # It is the recent-episode average of:
        #   reached_waypoints / (waypoint_num - 1)
        self.logger.record(
            f"{prefix}/waypoint_reached_ratio_w{self.window_size}",
            self._mean(ep["waypoint_reached_ratio"] for ep in window),
        )

    def _on_rollout_end(self) -> None:
        self._log_window("rollout_recent", self.recent_episodes)

        recent = list(self.recent_episodes)
        if len(recent) > 0:
            self.logger.record(f"rollout_recent/final_dist_mm_w{self.window_size}", self._mean(ep["final_dist_m"] * 1000.0 for ep in recent))
            self.logger.record(f"terminal/target_rate_w{self.window_size}", self._rate(ep["done_by_target"] for ep in recent))
            self.logger.record(f"terminal/timeout_rate_w{self.window_size}", self._rate(ep["done_by_timeout"] for ep in recent))
            self.logger.record(f"terminal/out_of_vessel_rate_w{self.window_size}", self._rate(ep["done_by_out_of_vessel"] for ep in recent))
            self.logger.record(f"terminal/wrong_branch_rate_w{self.window_size}", self._rate(ep["done_by_wrong_branch"] for ep in recent))
            self.logger.record(f"terminal/non_finite_rate_w{self.window_size}", self._rate(ep["done_by_non_finite"] for ep in recent))
            self.logger.record(f"rollout_recent/centerline_safety_ratio_mean_w{self.window_size}", self._mean(ep["centerline_safety_ratio"] for ep in recent), exclude="stdout")
            self.logger.record(f"rollout_recent/centerline_safety_ratio_max_w{self.window_size}", self._max(ep["centerline_safety_ratio_max_episode"] for ep in recent), exclude="stdout")
            self.logger.record(f"rollout_recent/centerline_safety_margin_mean_w{self.window_size}", self._mean(ep["centerline_safety_margin"] for ep in recent), exclude="stdout")
            self.logger.record(f"rollout_recent/centerline_safety_margin_min_w{self.window_size}", self._mean(ep["centerline_safety_margin_min_episode"] for ep in recent), exclude="stdout")
            self.logger.record(f"rollout_recent/vessel_scale_mean_w{self.window_size}", self._mean(ep["vessel_scale_factor"] for ep in recent), exclude="stdout")
            self.logger.record(f"rollout_recent/sdf_clearance_min_mm_w{self.window_size}", self._mean(ep["sdf_surface_clearance_min_m"] * 1000.0 for ep in recent), exclude="stdout")
            self.logger.record(f"rollout_recent/sdf_wall_contact_steps_w{self.window_size}", self._mean(ep["sdf_wall_contact_steps"] for ep in recent), exclude="stdout")
            self.logger.record(f"rollout_recent/route_graph_gap_mm_w{self.window_size}", self._mean(ep["route_graph_distance_gap_m"] * 1000.0 for ep in recent), exclude="stdout")
            self.logger.record(f"reward_components/progress_w{self.window_size}", self._mean(ep["reward_progress"] for ep in recent), exclude="stdout")
            self.logger.record(f"reward_components/waypoints_w{self.window_size}", self._mean(ep["reward_waypoints"] for ep in recent), exclude="stdout")
            self.logger.record(f"reward_components/terminal_w{self.window_size}", self._mean(ep["reward_terminal"] for ep in recent), exclude="stdout")
            self.logger.record(f"reward_components/safety_w{self.window_size}", self._mean(ep["reward_safety"] for ep in recent), exclude="stdout")
            self.logger.record(f"reward_components/step_w{self.window_size}", self._mean(ep["reward_step"] for ep in recent), exclude="stdout")

        if self.total_episodes > 0:
            self.logger.record(f"rollout_cumulative/success_{self.success_label}", float(self.total_success_target / self.total_episodes))
            self.logger.record("rollout_cumulative/episodes", float(self.total_episodes))

        # Per-vessel diagnostics for mixed-vessel training. These metrics do not
        # change the training algorithm. They are TensorBoard-only to keep
        # console output compact and avoid long-key truncation conflicts.
        active_task_stats = []
        for task_id in TASK_IDS_FOR_LOGGING:
            task_window = list(self.task_windows.get(task_id, []))
            if len(task_window) == 0:
                continue

            prefix = f"task/{task_id}"
            success_rate = self._rate(ep["success_target"] for ep in task_window)
            timeout_rate = self._rate(ep["done_by_timeout"] for ep in task_window)
            out_rate = self._rate(ep["done_by_out_of_vessel"] for ep in task_window)
            wrong_branch_rate = self._rate(ep["done_by_wrong_branch"] for ep in task_window)
            non_finite_rate = self._rate(ep["done_by_non_finite"] for ep in task_window)
            final_dist_mm = self._mean(ep["final_dist_m"] * 1000.0 for ep in task_window)
            min_dist_mm = self._mean(ep["min_dist_m"] * 1000.0 for ep in task_window)
            waypoint_ratio = self._mean(ep["waypoint_reached_ratio"] for ep in task_window)
            safety_ratio_max = self._max(ep["centerline_safety_ratio_max_episode"] for ep in task_window)
            safety_margin_min = self._mean(ep["centerline_safety_margin_min_episode"] for ep in task_window)
            sdf_clearance_min_mm = self._mean(
                ep["sdf_surface_clearance_min_m"] * 1000.0
                for ep in task_window
            )

            self.logger.record(f"{prefix}/success_{self.success_label}_w{self.window_size}", success_rate, exclude="stdout")
            self.logger.record(f"{prefix}/target_rate_w{self.window_size}", success_rate, exclude="stdout")
            self.logger.record(f"{prefix}/timeout_rate_w{self.window_size}", timeout_rate, exclude="stdout")
            self.logger.record(f"{prefix}/out_of_vessel_rate_w{self.window_size}", out_rate, exclude="stdout")
            self.logger.record(f"{prefix}/wrong_branch_rate_w{self.window_size}", wrong_branch_rate, exclude="stdout")
            self.logger.record(f"{prefix}/non_finite_rate_w{self.window_size}", non_finite_rate, exclude="stdout")
            self.logger.record(f"{prefix}/final_dist_mm_w{self.window_size}", final_dist_mm, exclude="stdout")
            self.logger.record(f"{prefix}/min_dist_mm_w{self.window_size}", min_dist_mm, exclude="stdout")
            self.logger.record(f"{prefix}/waypoint_reached_ratio_w{self.window_size}", waypoint_ratio, exclude="stdout")
            self.logger.record(f"{prefix}/centerline_safety_ratio_max_w{self.window_size}", safety_ratio_max, exclude="stdout")
            self.logger.record(f"{prefix}/centerline_safety_margin_min_w{self.window_size}", safety_margin_min, exclude="stdout")
            self.logger.record(f"{prefix}/sdf_clearance_min_mm_w{self.window_size}", sdf_clearance_min_mm, exclude="stdout")
            self.logger.record(f"{prefix}/episodes_w{self.window_size}", float(len(task_window)), exclude="stdout")

            if np.isfinite(success_rate):
                active_task_stats.append({
                    "task_id": task_id,
                    "success_rate": float(success_rate),
                    "timeout_rate": float(timeout_rate),
                    "out_of_vessel_rate": float(out_rate),
                    "wrong_branch_rate": float(wrong_branch_rate),
                    "final_dist_mm": float(final_dist_mm),
                    "waypoint_ratio": float(waypoint_ratio),
                })

        # Bottleneck summary for quickly detecting whether mixed training is being
        # held back by one or two hard vessels. TensorBoard scalar names cannot
        # carry strings, so each task gets an is_worst flag and deficit value.
        if len(active_task_stats) > 0:
            success_values = [x["success_rate"] for x in active_task_stats]
            mean_success = float(np.mean(success_values))
            best_success = float(np.max(success_values))
            worst_success = float(np.min(success_values))
            success_gap = float(best_success - worst_success)
            worst_idx = int(np.argmin(success_values))
            worst_task_id = active_task_stats[worst_idx]["task_id"]

            self.logger.record(f"mix_bottleneck/mean_success_{self.success_label}_w{self.window_size}", mean_success, exclude="stdout")
            self.logger.record(f"mix_bottleneck/best_success_{self.success_label}_w{self.window_size}", best_success, exclude="stdout")
            self.logger.record(f"mix_bottleneck/worst_success_{self.success_label}_w{self.window_size}", worst_success, exclude="stdout")
            self.logger.record(f"mix_bottleneck/success_gap_w{self.window_size}", success_gap, exclude="stdout")
            self.logger.record(f"mix_bottleneck/worst_timeout_rate_w{self.window_size}", active_task_stats[worst_idx]["timeout_rate"], exclude="stdout")
            self.logger.record(f"mix_bottleneck/worst_out_of_vessel_rate_w{self.window_size}", active_task_stats[worst_idx]["out_of_vessel_rate"], exclude="stdout")
            self.logger.record(f"mix_bottleneck/worst_final_dist_mm_w{self.window_size}", active_task_stats[worst_idx]["final_dist_mm"], exclude="stdout")
            self.logger.record(f"mix_bottleneck/worst_waypoint_ratio_w{self.window_size}", active_task_stats[worst_idx]["waypoint_ratio"], exclude="stdout")

            for item in active_task_stats:
                prefix = f"bottleneck/{item['task_id']}"
                self.logger.record(f"{prefix}/is_worst_w{self.window_size}", 1.0 if item["task_id"] == worst_task_id else 0.0, exclude="stdout")
                self.logger.record(f"{prefix}/success_deficit_from_mean_w{self.window_size}", float(mean_success - item["success_rate"]), exclude="stdout")
                self.logger.record(f"{prefix}/success_deficit_from_best_w{self.window_size}", float(best_success - item["success_rate"]), exclude="stdout")


class DistributedRuntimeCallback(BaseCallback):
    """Rank-0 TensorBoard scalars describing global distributed progress."""

    def __init__(
        self,
        world_size: int,
        total_n_envs: int,
        global_batch_size: int,
        local_batch_size: int,
        steps_per_epoch: int,
    ):
        super().__init__(verbose=0)
        self.world_size = int(world_size)
        self.total_n_envs = int(total_n_envs)
        self.global_batch_size = int(global_batch_size)
        self.local_batch_size = int(local_batch_size)
        self.steps_per_epoch = int(steps_per_epoch)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        global_timesteps = float(self.model.num_timesteps * self.world_size)
        self.logger.record("distributed/global_timesteps", global_timesteps, exclude="stdout")
        self.logger.record(
            "distributed/global_epoch",
            global_timesteps / float(self.steps_per_epoch),
            exclude="stdout",
        )
        self.logger.record("distributed/world_size", float(self.world_size), exclude="stdout")
        self.logger.record("distributed/total_n_envs", float(self.total_n_envs), exclude="stdout")
        self.logger.record("distributed/global_batch_size", float(self.global_batch_size), exclude="stdout")
        self.logger.record("distributed/local_batch_size", float(self.local_batch_size), exclude="stdout")


def _parse_ent_coef(value: str) -> Union[str, float]:
    """Parse SAC ent_coef argument.

    Supported values:
    - "auto": automatic entropy tuning, initialized from checkpoint value when resuming.
    - "auto_0.001": automatic entropy tuning initialized at 0.001.
    - numeric string, e.g. "0.001": fixed entropy coefficient.
    """
    value = str(value).strip().lower()
    if value == "auto" or value.startswith("auto_"):
        return value
    return float(value)


def _current_ent_coef_value(model: SAC, fallback: float = 1.0) -> float:
    """Best-effort read of the currently stored entropy coefficient."""
    try:
        log_ent_coef = getattr(model, "log_ent_coef", None)
        if log_ent_coef is not None:
            return float(th.exp(log_ent_coef.detach()).cpu().item())
    except Exception:
        pass

    try:
        ent_coef_tensor = getattr(model, "ent_coef_tensor", None)
        if ent_coef_tensor is not None:
            return float(ent_coef_tensor.detach().cpu().item())
    except Exception:
        pass

    try:
        ent_coef = getattr(model, "ent_coef", None)
        if isinstance(ent_coef, (float, int)):
            return float(ent_coef)
    except Exception:
        pass

    return float(fallback)


def _override_learning_rate(model: SAC, learning_rate: float, announce: bool = True) -> None:
    """Override learning rate of a loaded SAC model when resuming.

    SB3 checkpoints keep the old lr_schedule and optimizer learning rates.
    This function makes --learning-rate effective for resume training.
    """
    learning_rate = float(learning_rate)
    model.learning_rate = learning_rate
    model.lr_schedule = get_schedule_fn(learning_rate)

    optimizers = [
        getattr(getattr(model, "actor", None), "optimizer", None),
        getattr(getattr(model, "critic", None), "optimizer", None),
        getattr(model, "ent_coef_optimizer", None),
    ]

    for optimizer in optimizers:
        if optimizer is None:
            continue
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate

    if announce:
        print(f"Resume learning_rate overridden to: {learning_rate:.6g}")


def _force_auto_ent_coef(model: SAC, ent_coef_arg: Union[str, float], announce: bool = True) -> None:
    """Force a loaded SAC model to use automatic entropy tuning.

    This is useful when:
    - the checkpoint was originally trained with fixed ent_coef;
    - or we want resume training to start auto entropy from a specific value, e.g. auto_0.001.

    If ent_coef_arg == "auto", initialize from checkpoint current value.
    If ent_coef_arg == "auto_xxx", initialize from xxx.
    """
    if not (isinstance(ent_coef_arg, str) and ent_coef_arg.startswith("auto")):
        return

    if ent_coef_arg.startswith("auto_"):
        init_value = float(ent_coef_arg.split("_", 1)[1])
    else:
        init_value = _current_ent_coef_value(model, fallback=1.0)

    init_value = max(float(init_value), 1e-12)

    model.ent_coef = ent_coef_arg
    model.log_ent_coef = th.log(th.ones(1, device=model.device) * init_value).requires_grad_(True)
    model.ent_coef_optimizer = th.optim.Adam([model.log_ent_coef], lr=model.lr_schedule(1))
    model.ent_coef_tensor = None

    if announce:
        print(f"Automatic ent_coef tuning enabled. init_ent_coef={init_value:.6g}")


def _force_fixed_ent_coef(model: SAC, ent_coef_value: float, announce: bool = True) -> None:
    """Force a loaded SAC model to use a fixed entropy coefficient."""
    ent_coef_value = float(ent_coef_value)
    model.ent_coef = ent_coef_value
    model.ent_coef_tensor = th.tensor(ent_coef_value, device=model.device)
    model.log_ent_coef = None
    model.ent_coef_optimizer = None
    if announce:
        print(f"Fixed ent_coef enabled. ent_coef={ent_coef_value:.6g}")


def _adapt_resume_timestep_counter(
    model: SAC,
    current_world_size: int,
    reset_num_timesteps: bool,
    announce: bool = True,
) -> None:
    """Preserve global timestep meaning across single/distributed checkpoints."""
    current_world_size = max(1, int(current_world_size))
    saved_world_size = max(1, int(getattr(model, "distributed_world_size_at_save", 1)))
    if not reset_num_timesteps and saved_world_size != current_world_size:
        old_local_steps = int(model.num_timesteps)
        completed_global_steps = old_local_steps * saved_world_size
        model.num_timesteps = int(math.ceil(completed_global_steps / current_world_size))
        if announce:
            print(
                "Resume timestep counter adapted: "
                f"saved_world_size={saved_world_size}, current_world_size={current_world_size}, "
                f"saved_local_steps={old_local_steps}, completed_global_steps={completed_global_steps}, "
                f"current_local_steps={model.num_timesteps}"
            )
    model.distributed_world_size_at_save = current_world_size


def parse_args():
    parser = argparse.ArgumentParser(description="Train MCR agent with SAC")
    parser.add_argument("--env-type", choices=["aortic", "flat"], default="aortic")
    parser.add_argument(
        "--force-model",
        type=str,
        default="",
        choices=ALL_MODEL_CHOICES,
        help=(
            "Force one vessel model/path for single-vessel fine-tuning. "
            "Empty string means uniform multi-vessel training."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=SAC_EPOCHS,
        help=(
            "Number of training epochs. One epoch is --steps-per-epoch global "
            "environment transitions across all ranks (default: 50)."
        ),
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=SAC_STEPS_PER_EPOCH,
        help="Global environment transitions per epoch (default: 100000).",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help=(
            "Compatibility override for total global transitions. When omitted, "
            "total transitions = epochs * steps-per-epoch."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable one synchronized multi-process SAC learner under torchrun.",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=4,
        help="Number of learner ranks/devices. The launcher defaults to four for Ascend 910B3.",
    )
    parser.add_argument(
        "--local-rank",
        "--local_rank",
        dest="local_rank",
        type=int,
        default=0,
        help="Local rank fallback; torchrun LOCAL_RANK takes precedence.",
    )
    parser.add_argument(
        "--dist-backend",
        type=str,
        default="",
        choices=["", "hccl", "nccl", "gloo"],
        help="Distributed backend. Empty selects hccl/nccl/gloo from the resolved device.",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=SAC_N_ENVS,
        help=(
            "Global number of SOFA environments. Distributed mode divides it "
            "equally across ranks; single-process mode uses all locally."
        ),
    )

    # SAC-specific arguments
    parser.add_argument("--learning-rate", type=float, default=SAC_LEARNING_RATE)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=SAC_BATCH_SIZE,
        help=(
            "Global SAC minibatch size. Distributed mode requires divisibility "
            "by world size and uses batch_size/world_size per rank."
        ),
    )
    parser.add_argument("--buffer-size", type=int, default=SAC_BUFFER_SIZE)
    parser.add_argument("--learning-starts", type=int, default=SAC_LEARNING_STARTS)
    parser.add_argument("--train-freq", type=int, default=SAC_TRAIN_FREQ)
    parser.add_argument(
        "--gradient-steps",
        type=int,
        default=SAC_GRADIENT_STEPS,
        help="SAC gradient updates per rollout; -1 matches updates to collected transitions.",
    )
    parser.add_argument("--tau", type=float, default=SAC_TAU)
    parser.add_argument(
        "--ent-coef",
        type=str,
        default="auto",
        help="Entropy coefficient: 'auto', 'auto_0.001', or fixed float such as 0.001",
    )
    parser.add_argument("--gamma", type=float, default=SAC_GAMMA)

    parser.add_argument("--frame-skip", type=int, default=FRAME_SKIP)
    parser.add_argument("--time-step", type=float, default=SOFA_TIME_STEP_S)
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=SETTLE_STEPS,
        help="SOFA settle steps after reset. Increase to 10-20 if initialization becomes unstable.",
    )
    parser.add_argument("--target-threshold", type=float, default=TARGET_THRESHOLD_M)
    parser.add_argument("--max-episode-steps", type=int, default=MAX_EPISODE_STEPS)

    # Environment safety option kept because it is part of the actor state.
    parser.add_argument(
        "--radius-observation-scale",
        type=float,
        default=RADIUS_OBSERVATION_SCALE_M,
        help="Local vessel radius observation scale in meters. Default 5 mm.",
    )

    # Domain randomization for sim-to-sim generalization.
    dr_group = parser.add_mutually_exclusive_group()
    dr_group.add_argument(
        "--randomize-start-target",
        dest="randomize_start_target",
        action="store_true",
        help="Randomize start/target by choosing nearby centerline endpoint samples.",
    )
    dr_group.add_argument(
        "--no-randomize-start-target",
        dest="randomize_start_target",
        action="store_false",
        help="Disable start/target endpoint randomization.",
    )
    parser.set_defaults(randomize_start_target=True)
    parser.add_argument(
        "--start-window-mm",
        type=float,
        default=START_WINDOW_DISTANCE_M * 1000.0,
        help="Physical centerline randomization window from the nominal start, in mm.",
    )
    parser.add_argument(
        "--target-window-mm",
        type=float,
        default=TARGET_WINDOW_DISTANCE_M * 1000.0,
        help="Physical centerline randomization window from the nominal target, in mm.",
    )
    parser.add_argument("--start-window-points", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--target-window-points", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--start-target-random-radius",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )

    init_group = parser.add_mutually_exclusive_group()
    init_group.add_argument(
        "--randomize-initial-orientation",
        dest="randomize_initial_orientation",
        action="store_true",
        help="Randomize initial mcR orientation within a cone around the entry tangent.",
    )
    init_group.add_argument(
        "--no-randomize-initial-orientation",
        dest="randomize_initial_orientation",
        action="store_false",
        help="Disable initial orientation randomization.",
    )
    parser.set_defaults(randomize_initial_orientation=True)
    parser.add_argument(
        "--initial-orientation-max-angle-deg",
        type=float,
        default=INITIAL_ORIENTATION_MAX_ANGLE_DEG,
        help="Max initial orientation deviation from entry tangent in degrees.",
    )
    parser.add_argument(
        "--entry-tangent-points",
        type=int,
        default=ENTRY_TANGENT_POINTS,
        help="Number of initial centerline points used to estimate entry tangent. Default 5.",
    )
    soft_group = parser.add_mutually_exclusive_group()
    soft_group.add_argument(
        "--soft-randomize-single-vessel",
        dest="soft_randomize_single_vessel",
        action="store_true",
        help=(
            "In forced single-vessel mode, randomize start/target/orientation inside "
            "the existing SOFA scene instead of rebuilding the STL/collision scene each episode."
        ),
    )
    soft_group.add_argument(
        "--no-soft-randomize-single-vessel",
        dest="soft_randomize_single_vessel",
        action="store_false",
        help="Use old behavior: rebuild the SOFA scene when start/target/orientation randomization is enabled.",
    )
    parser.set_defaults(soft_randomize_single_vessel=True)

    parser.add_argument(
        "--vessel-scale-min",
        type=float,
        default=VESSEL_SCALE_MIN,
        help="Minimum isotropic vessel scale sampled per episode (default: 0.90).",
    )
    parser.add_argument(
        "--vessel-scale-max",
        type=float,
        default=VESSEL_SCALE_MAX,
        help="Maximum isotropic vessel scale sampled per episode (default: 1.00).",
    )

    # All-vessel local-observation curriculum runs.
    parser.add_argument(
        "--log-root",
        type=str,
        default=str(DEFAULT_LOG_ROOT),
        help="Training output root. Defaults to the project-level training_runs directory.",
    )
    parser.add_argument("--exp-name", type=str, default="sac_waypoint_uniform")
    parser.add_argument(
        "--save-freq",
        type=int,
        default=0,
        help="Global transitions between checkpoints; 0 saves once per epoch.",
    )
    parser.add_argument("--render", choices=["headless", "human"], default="headless")
    parser.add_argument(
        "--resume-from",
        type=str,
        default="",
        help="Path to existing .zip SAC model/checkpoint to continue training. Must match current observation shape.",
    )
    parser.add_argument("--reset-num-timesteps", action="store_true", help="Reset timestep counter when resuming")

    # Console/progress controls. Keeping progress bars off by default reduces
    # terminal I/O during long headless training, especially when stdout is also
    # receiving SOFA messages from C++ plugins.
    parser.add_argument(
        "--progress-bar",
        action="store_true",
        help="Show SB3 progress bar. Default is off to reduce terminal overhead.",
    )
    parser.add_argument(
        "--sb3-verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Stable-Baselines3 verbosity. Use 0 for quieter long training.",
    )
    parser.add_argument(
        "--scene-verbose",
        action="store_true",
        help=(
            "Print detailed SOFA scene, pose, centerline, and collision "
            "diagnostics. Default is off for clean training logs."
        ),
    )

    # Scene/env sampling is uniform over the active training vessels. No priority sampling.

    args = parser.parse_args()
    if args.epochs <= 0 or args.steps_per_epoch <= 0:
        parser.error("--epochs and --steps-per-epoch must be positive integers")
    if args.timesteps is None:
        args.timesteps = int(args.epochs) * int(args.steps_per_epoch)
    elif args.timesteps <= 0:
        parser.error("--timesteps must be positive when provided")
    if args.save_freq <= 0:
        args.save_freq = int(args.steps_per_epoch)
    if args.max_episode_steps <= 0:
        parser.error("--max-episode-steps must be a positive integer")
    if not (0.5 <= args.vessel_scale_min <= args.vessel_scale_max <= 1.0):
        parser.error("vessel scale bounds must satisfy 0.5 <= min <= max <= 1.0")
    if args.start_window_mm < 0.0 or args.target_window_mm < 0.0:
        parser.error("start/target window distances must be non-negative")
    if args.start_window_points is not None or args.target_window_points is not None:
        print(
            "[WARNING] --start-window-points/--target-window-points are deprecated "
            "and ignored; use --start-window-mm/--target-window-mm."
        )
    if args.start_target_random_radius is not None:
        print(
            "[WARNING] --start-target-random-radius is deprecated and ignored; "
            "use --start-window-mm/--target-window-mm."
        )
    args.ent_coef = _parse_ent_coef(args.ent_coef)
    return args


def build_env(args):
    env_type = EnvType.AORTIC if args.env_type == "aortic" else EnvType.FLAT
    render_mode = RenderMode.HUMAN if args.render == "human" else RenderMode.NONE
    n_envs = max(1, int(getattr(args, "local_n_envs", args.n_envs)))
    global_env_offset = int(getattr(args, "distributed_rank", 0)) * n_envs

    if n_envs > 1 and args.render == "human":
        raise ValueError("Parallel SOFA environments require --render headless. Do not use GUI/human render with --n-envs > 1.")

    def _make(rank: int = 0):
        def _init():
            # Give each subprocess a different NumPy seed. SB3 will also manage
            # environment seeding, but this avoids identical default RNG streams
            # during SOFA scene construction/randomization.
            try:
                env_seed = int(args.seed) + global_env_offset + int(rank)
                np.random.seed(env_seed)
                random.seed(env_seed)
            except Exception:
                pass

            create_scene_kwargs = {
                "radius_observation_scale": float(args.radius_observation_scale),
                "actor_history_steps": ACTOR_HISTORY_STEPS,
                "randomize_start_target": bool(args.randomize_start_target),
                "start_window_distance_m": float(args.start_window_mm) / 1000.0,
                "target_window_distance_m": float(args.target_window_mm) / 1000.0,
                "randomize_initial_orientation": bool(args.randomize_initial_orientation),
                "initial_orientation_max_angle_deg": float(args.initial_orientation_max_angle_deg),
                "entry_tangent_points": int(args.entry_tangent_points),
                "soft_randomize_single_vessel": bool(args.soft_randomize_single_vessel),
                "vessel_scale_min": float(args.vessel_scale_min),
                "vessel_scale_max": float(args.vessel_scale_max),
                "verbose_scene": bool(args.scene_verbose),
            }
            # If running with GUI (human), enable debug_rendering so the scene
            # creates the visual OglModel and ensure vessels are sufficiently
            # opaque by default so they are visible at startup.
            if args.render == "human":
                create_scene_kwargs["debug_rendering"] = True
                create_scene_kwargs["positioning_camera"] = True
                create_scene_kwargs["vessel_alpha"] = 0.8
            else:
                create_scene_kwargs["debug_rendering"] = False
                create_scene_kwargs["positioning_camera"] = False
            if args.force_model:
                create_scene_kwargs["force_model"] = args.force_model

            env = MCREnv(
                env_type=env_type,
                observation_type=ObservationType.STATE,
                action_type=ActionType.CONTINUOUS,
                time_step=args.time_step,
                frame_skip=args.frame_skip,
                settle_steps=args.settle_steps,
                render_mode=render_mode,
                render_framework=RenderFramework.PYGLET,
                target_distance_threshold=args.target_threshold,
                max_episode_steps=args.max_episode_steps,
                create_scene_kwargs=create_scene_kwargs,
            )
            return Monitor(env)

        return _init

    if n_envs == 1:
        return DummyVecEnv([_make(0)])

    # Use spawn instead of fork for SOFA/C++ plugin safety.
    return SubprocVecEnv([_make(i) for i in range(n_envs)], start_method="spawn")

def main():
    args = parse_args()
    # The Python environment and the SOFA scene must advance with the same dt.
    # Set this before spawning worker processes so every scene inherits it.
    os.environ["MCR_SOFA_DT"] = str(float(args.time_step))
    context = initialize_distributed(
        enabled=bool(args.distributed),
        requested_device=args.device,
        requested_world_size=int(args.world_size),
        cli_local_rank=int(args.local_rank),
        requested_backend=args.dist_backend,
    )

    args.distributed_rank = int(context.rank)
    args.resolved_device = context.device.resolved
    total_n_envs = max(1, int(args.n_envs))
    global_batch_size = int(args.batch_size)

    if context.enabled:
        if total_n_envs % context.world_size != 0:
            raise ValueError(
                f"--n-envs={total_n_envs} must be divisible by world_size={context.world_size}."
            )
        if global_batch_size % context.world_size != 0:
            raise ValueError(
                f"--batch-size={global_batch_size} must be divisible by world_size={context.world_size}."
            )
        args.local_n_envs = total_n_envs // context.world_size
        local_batch_size = global_batch_size // context.world_size
        local_total_timesteps = int(math.ceil(float(args.timesteps) / context.world_size))
    else:
        args.local_n_envs = total_n_envs
        local_batch_size = global_batch_size
        local_total_timesteps = int(args.timesteps)
    args.rank_seed = int(args.seed) + int(context.rank) * int(args.local_n_envs)

    if args.local_n_envs > 1 and args.render == "human":
        raise ValueError("Parallel SOFA environments require --render headless.")
    if local_batch_size <= 0:
        raise ValueError("Local batch size must be positive.")

    threshold_mm = int(round(float(args.target_threshold) * 1000.0))
    threshold_tag = f"{threshold_mm}mm"
    forced_tag = f"_{args.force_model}_only" if args.force_model else ""

    run_timestamp = str(os.environ.get("MCR_RUN_TIMESTAMP", "")).strip()
    if not run_timestamp and context.is_main:
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_timestamp = context.broadcast_text(run_timestamp)

    log_root = Path(args.log_root).expanduser()
    if not log_root.is_absolute():
        log_root = PROJECT_ROOT / log_root
    run_dir = log_root.resolve() / (
        f"{args.exp_name}{forced_tag}_{args.env_type}_{threshold_tag}_{run_timestamp}"
    )
    model_dir = run_dir / "models"
    tb_dir = run_dir / "tb"
    if context.is_main:
        model_dir.mkdir(parents=True, exist_ok=True)
        tb_dir.mkdir(parents=True, exist_ok=True)
    context.barrier()

    print(
        f"[RANK {context.rank}] local_rank={context.local_rank} "
        f"device={context.device.resolved} local_n_envs={args.local_n_envs} "
        f"seed={args.rank_seed}"
    )

    env = None
    try:
        env = build_env(args)

        callback_list = None
        if context.is_main:
            checkpoint_prefix = f"sac_mcr_{threshold_tag}_all_vessels{forced_tag}_ckpt"
            # One callback call represents total_n_envs transitions across all
            # synchronized ranks, so save_freq retains global-timestep semantics.
            effective_save_freq = max(
                1, int(math.ceil(float(args.save_freq) / float(total_n_envs)))
            )
            checkpoint_callback = CheckpointCallback(
                save_freq=effective_save_freq,
                save_path=str(model_dir),
                name_prefix=checkpoint_prefix,
                save_replay_buffer=False,
                save_vecnormalize=False,
            )
            extra_metrics_callback = ExtraRolloutMetricsCallback(
                window_size=50, success_label=threshold_tag
            )
            distributed_callback = DistributedRuntimeCallback(
                world_size=context.world_size,
                total_n_envs=total_n_envs,
                global_batch_size=global_batch_size,
                local_batch_size=local_batch_size,
                steps_per_epoch=args.steps_per_epoch,
            )
            callback_list = CallbackList(
                [checkpoint_callback, extra_metrics_callback, distributed_callback]
            )

        algorithm_class = DistributedSAC if context.enabled else SAC
        tensorboard_log = str(tb_dir) if context.is_main else None
        sb3_verbose = int(args.sb3_verbose) if context.is_main else 0

        if args.resume_from:
            resume_path = Path(args.resume_from).expanduser()
            if not resume_path.is_absolute():
                resume_path = PROJECT_ROOT / resume_path
            resume_path = resume_path.resolve()
            if not resume_path.is_file():
                raise FileNotFoundError(f"Resume model not found: {resume_path}")

            custom_objects = {
                "learning_rate": args.learning_rate,
                "buffer_size": args.buffer_size,
                "batch_size": local_batch_size,
                "learning_starts": args.learning_starts,
                "train_freq": args.train_freq,
                "gradient_steps": args.gradient_steps,
                "tau": args.tau,
                "gamma": args.gamma,
            }
            model = algorithm_class.load(
                str(resume_path),
                env=env,
                device=args.resolved_device,
                custom_objects=custom_objects,
            )
            model.tensorboard_log = tensorboard_log
            model.verbose = sb3_verbose
            model.seed = int(args.rank_seed)
            model.set_random_seed(int(args.rank_seed))
            reset_num_timesteps = args.reset_num_timesteps
            _adapt_resume_timestep_counter(
                model,
                current_world_size=context.world_size,
                reset_num_timesteps=reset_num_timesteps,
                announce=context.is_main,
            )

            if context.is_main:
                print(f"Resuming SAC training from: {resume_path}")
                print("IMPORTANT: resume model must have the same observation shape as current env.")
                print(f"Original ent_coef from checkpoint: {_current_ent_coef_value(model):.6g}")

            _override_learning_rate(model, args.learning_rate, announce=context.is_main)
            if isinstance(args.ent_coef, str) and args.ent_coef.startswith("auto"):
                _force_auto_ent_coef(model, args.ent_coef, announce=context.is_main)
            else:
                _force_fixed_ent_coef(model, float(args.ent_coef), announce=context.is_main)

            if context.is_main:
                print(
                    f"[MCR TRAIN] resume local_batch={model.batch_size} "
                    f"buffer_per_rank={model.buffer_size}"
                )
        else:
            model_kwargs = dict(
                policy="MlpPolicy",
                env=env,
                learning_rate=args.learning_rate,
                buffer_size=args.buffer_size,
                learning_starts=args.learning_starts,
                batch_size=local_batch_size,
                tau=args.tau,
                gamma=args.gamma,
                train_freq=args.train_freq,
                gradient_steps=args.gradient_steps,
                ent_coef=args.ent_coef,
                tensorboard_log=tensorboard_log,
                seed=args.rank_seed,
                device=args.resolved_device,
                verbose=sb3_verbose,
            )
            if context.enabled:
                model_kwargs["distributed_context"] = context
            model = algorithm_class(**model_kwargs)
            model.distributed_world_size_at_save = int(context.world_size)
            reset_num_timesteps = True
            if context.is_main:
                print(
                    f"[MCR TRAIN] new SAC ent_coef={args.ent_coef} "
                    f"lr={args.learning_rate:g} "
                    f"global_batch={global_batch_size}, local_batch={local_batch_size}, "
                    f"buffer_per_rank={args.buffer_size}"
                )

        if context.enabled:
            model.set_distributed_context(context)
            model.synchronize_parameters()

        if context.is_main:
            print(
                f"[MCR TRAIN] device={context.device.resolved} "
                f"distributed={context.enabled} backend={context.backend} "
                f"world={context.world_size} envs={total_n_envs} "
                f"envs_per_rank={args.local_n_envs}"
            )
            print(
                f"[MCR TRAIN] task={args.env_type} "
                f"model={args.force_model or 'uniform(B01-B05,C01-C05)'} "
                f"obs={env.observation_space.shape} action={env.action_space.shape}"
            )
            print(
                f"[MCR TRAIN] epochs={args.epochs} steps_per_epoch={args.steps_per_epoch} "
                f"global_steps={args.timesteps} local_steps={local_total_timesteps} "
                f"max_episode_steps={args.max_episode_steps} checkpoint={args.save_freq}"
            )
            print(
                f"[MCR TRAIN] SAC lr={args.learning_rate:g} "
                f"batch={global_batch_size}/{local_batch_size} "
                f"buffer_per_rank={args.buffer_size} ent_coef={args.ent_coef}"
            )
            print(
                f"[MCR TRAIN] reward progress={REWARD_WAYPOINT_APPROACH:g}/"
                f"{REWARD_TARGET_APPROACH:g} waypoint={REWARD_WAYPOINT_REACHED:g} "
                f"wall={REWARD_WALL_PROXIMITY:g}/{REWARD_WALL_PENETRATION:g} "
                f"branch={REWARD_OFF_TARGET_BRANCH:g}/{REWARD_WRONG_BRANCH:g} "
                f"success={REWARD_SUCCESS:g} out={REWARD_OUT_OF_VESSEL:g} "
                f"timeout={REWARD_TIMEOUT:g} step={REWARD_STEP:g}"
            )
            print(
                f"[MCR TRAIN] terminal SDF>{SDF_OUTSIDE_CENTER_TOLERANCE_M*1000.0:.1f}mm "
                f"x{SDF_OUTSIDE_CONFIRM_STEPS}; wrong_branch>"
                f"{WRONG_BRANCH_DISTANCE_MARGIN_M*1000.0:.1f}mm "
                f"x{WRONG_BRANCH_CONFIRM_STEPS}; target={args.target_threshold*1000.0:.1f}mm"
            )
            print(
                f"[MCR TRAIN] randomization start/target="
                f"{float(args.start_window_mm):.1f}/{float(args.target_window_mm):.1f}mm "
                f"orientation={float(args.initial_orientation_max_angle_deg):.1f}deg "
                f"vessel_scale={float(args.vessel_scale_min):.2f}-"
                f"{float(args.vessel_scale_max):.2f} scene_verbose={args.scene_verbose}"
            )
            print(f"[MCR TRAIN] output={run_dir}")

        model.learn(
            total_timesteps=local_total_timesteps,
            callback=callback_list,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=bool(args.progress_bar and context.is_main),
        )

        context.barrier()
        final_model_path = model_dir / f"sac_mcr_{threshold_tag}_all_vessels{forced_tag}_final"
        if context.is_main:
            model.save(str(final_model_path))
            print(f"[DONE] model={final_model_path}.zip")
            print(f"[DONE] tensorboard={tb_dir}")
        context.barrier()
    finally:
        if env is not None:
            env.close()
        context.close()


if __name__ == "__main__":
    main()
