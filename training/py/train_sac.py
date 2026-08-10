import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Union

# This executable lives in python/training/py/.  Resolve imports and the
# complete project root from the file itself, independent of caller cwd.
TRAINING_PY_DIR = Path(__file__).resolve().parent
TRAINING_DIR = TRAINING_PY_DIR.parent
PYTHON_ROOT = TRAINING_DIR.parent
PROJECT_ROOT = PYTHON_ROOT.parent
DEFAULT_LOG_ROOT = PROJECT_ROOT / "training_runs"
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
from mcr_sim.rl_core.base import RenderMode, RenderFramework


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
]

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
]

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
]


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
            self.logger.record(f"terminal/non_finite_rate_w{self.window_size}", self._rate(ep["done_by_non_finite"] for ep in recent))
            self.logger.record(f"rollout_recent/centerline_safety_ratio_mean_w{self.window_size}", self._mean(ep["centerline_safety_ratio"] for ep in recent), exclude="stdout")
            self.logger.record(f"rollout_recent/centerline_safety_ratio_max_w{self.window_size}", self._max(ep["centerline_safety_ratio_max_episode"] for ep in recent), exclude="stdout")
            self.logger.record(f"rollout_recent/centerline_safety_margin_mean_w{self.window_size}", self._mean(ep["centerline_safety_margin"] for ep in recent), exclude="stdout")
            self.logger.record(f"rollout_recent/centerline_safety_margin_min_w{self.window_size}", self._mean(ep["centerline_safety_margin_min_episode"] for ep in recent), exclude="stdout")

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
            non_finite_rate = self._rate(ep["done_by_non_finite"] for ep in task_window)
            final_dist_mm = self._mean(ep["final_dist_m"] * 1000.0 for ep in task_window)
            min_dist_mm = self._mean(ep["min_dist_m"] * 1000.0 for ep in task_window)
            waypoint_ratio = self._mean(ep["waypoint_reached_ratio"] for ep in task_window)
            safety_ratio_max = self._max(ep["centerline_safety_ratio_max_episode"] for ep in task_window)
            safety_margin_min = self._mean(ep["centerline_safety_margin_min_episode"] for ep in task_window)

            self.logger.record(f"{prefix}/success_{self.success_label}_w{self.window_size}", success_rate, exclude="stdout")
            self.logger.record(f"{prefix}/target_rate_w{self.window_size}", success_rate, exclude="stdout")
            self.logger.record(f"{prefix}/timeout_rate_w{self.window_size}", timeout_rate, exclude="stdout")
            self.logger.record(f"{prefix}/out_of_vessel_rate_w{self.window_size}", out_rate, exclude="stdout")
            self.logger.record(f"{prefix}/non_finite_rate_w{self.window_size}", non_finite_rate, exclude="stdout")
            self.logger.record(f"{prefix}/final_dist_mm_w{self.window_size}", final_dist_mm, exclude="stdout")
            self.logger.record(f"{prefix}/min_dist_mm_w{self.window_size}", min_dist_mm, exclude="stdout")
            self.logger.record(f"{prefix}/waypoint_reached_ratio_w{self.window_size}", waypoint_ratio, exclude="stdout")
            self.logger.record(f"{prefix}/centerline_safety_ratio_max_w{self.window_size}", safety_ratio_max, exclude="stdout")
            self.logger.record(f"{prefix}/centerline_safety_margin_min_w{self.window_size}", safety_margin_min, exclude="stdout")
            self.logger.record(f"{prefix}/episodes_w{self.window_size}", float(len(task_window)), exclude="stdout")

            if np.isfinite(success_rate):
                active_task_stats.append({
                    "task_id": task_id,
                    "success_rate": float(success_rate),
                    "timeout_rate": float(timeout_rate),
                    "out_of_vessel_rate": float(out_rate),
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


def _override_learning_rate(model: SAC, learning_rate: float) -> None:
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

    print(f"Resume learning_rate overridden to: {learning_rate:.6g}")


def _force_auto_ent_coef(model: SAC, ent_coef_arg: Union[str, float]) -> None:
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

    print(f"Automatic ent_coef tuning enabled. init_ent_coef={init_value:.6g}")


def _force_fixed_ent_coef(model: SAC, ent_coef_value: float) -> None:
    """Force a loaded SAC model to use a fixed entropy coefficient."""
    ent_coef_value = float(ent_coef_value)
    model.ent_coef = ent_coef_value
    model.ent_coef_tensor = th.tensor(ent_coef_value, device=model.device)
    model.log_ent_coef = None
    model.ent_coef_optimizer = None
    print(f"Fixed ent_coef enabled. ent_coef={ent_coef_value:.6g}")


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
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Number of parallel SOFA environments. Use 1 for DummyVecEnv, >1 for SubprocVecEnv headless training.",
    )

    # SAC-specific arguments
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buffer-size", type=int, default=300_000)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument(
        "--ent-coef",
        type=str,
        default="auto",
        help="Entropy coefficient: 'auto', 'auto_0.001', or fixed float such as 0.001",
    )
    parser.add_argument("--gamma", type=float, default=0.99)

    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--time-step", type=float, default=0.01)
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=8,
        help="SOFA settle steps after reset. Increase to 10-20 if initialization becomes unstable.",
    )
    parser.add_argument("--target-threshold", type=float, default=0.010)
    parser.add_argument("--max-episode-steps", type=int, default=2048)

    # Environment safety option kept because it is part of the actor state.
    parser.add_argument(
        "--radius-observation-scale",
        type=float,
        default=0.005,
        help="Local vessel radius observation scale in meters. Default 5 mm.",
    )

    # Domain randomization for sim-to-sim generalization.
    dr_group = parser.add_mutually_exclusive_group()
    dr_group.add_argument(
        "--randomize-start-target",
        dest="randomize_start_target",
        action="store_true",
        help="Randomize start and target inside a small ball around nominal endpoints.",
    )
    dr_group.add_argument(
        "--no-randomize-start-target",
        dest="randomize_start_target",
        action="store_false",
        help="Disable start/target endpoint randomization.",
    )
    parser.set_defaults(randomize_start_target=False)
    parser.add_argument(
        "--start-target-random-radius",
        type=float,
        default=0.002,
        help="Start/target randomization radius in meters. Default 0.002 = 2 mm.",
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
    parser.set_defaults(randomize_initial_orientation=False)
    parser.add_argument(
        "--initial-orientation-max-angle-deg",
        type=float,
        default=20.0,
        help="Max initial orientation deviation from entry tangent in degrees. Default 20.",
    )
    parser.add_argument(
        "--entry-tangent-points",
        type=int,
        default=5,
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

    # All-vessel local-observation curriculum runs.
    parser.add_argument(
        "--log-root",
        type=str,
        default=str(DEFAULT_LOG_ROOT),
        help="Training output root. Defaults to the project-level training_runs directory.",
    )
    parser.add_argument("--exp-name", type=str, default="sac_waypoint_uniform")
    parser.add_argument("--save-freq", type=int, default=50_000)
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

    # Scene/env sampling is uniform over the active training vessels. No priority sampling.

    args = parser.parse_args()
    args.ent_coef = _parse_ent_coef(args.ent_coef)
    return args


def build_env(args):
    env_type = EnvType.AORTIC if args.env_type == "aortic" else EnvType.FLAT
    render_mode = RenderMode.HUMAN if args.render == "human" else RenderMode.NONE
    n_envs = max(1, int(getattr(args, "n_envs", 1)))

    if n_envs > 1 and args.render == "human":
        raise ValueError("Parallel SOFA environments require --render headless. Do not use GUI/human render with --n-envs > 1.")

    def _make(rank: int = 0):
        def _init():
            # Give each subprocess a different NumPy seed. SB3 will also manage
            # environment seeding, but this avoids identical default RNG streams
            # during SOFA scene construction/randomization.
            try:
                np.random.seed(int(args.seed) + int(rank))
            except Exception:
                pass

            create_scene_kwargs = {
                "radius_observation_scale": float(args.radius_observation_scale),
                "actor_history_steps": 4,
                "randomize_start_target": bool(args.randomize_start_target),
                "start_target_random_radius": float(args.start_target_random_radius),
                "randomize_initial_orientation": bool(args.randomize_initial_orientation),
                "initial_orientation_max_angle_deg": float(args.initial_orientation_max_angle_deg),
                "entry_tangent_points": int(args.entry_tangent_points),
                "soft_randomize_single_vessel": bool(args.soft_randomize_single_vessel),
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

    threshold_mm = int(round(float(args.target_threshold) * 1000.0))
    threshold_tag = f"{threshold_mm}mm"
    forced_tag = f"_{args.force_model}_only" if args.force_model else ""

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_root = Path(args.log_root).expanduser()
    if not log_root.is_absolute():
        log_root = PROJECT_ROOT / log_root
    run_dir = log_root.resolve() / f"{args.exp_name}{forced_tag}_{args.env_type}_{threshold_tag}_{now}"
    model_dir = run_dir / "models"
    tb_dir = run_dir / "tb"
    model_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    env = build_env(args)

    checkpoint_prefix = f"sac_mcr_{threshold_tag}_all_vessels{forced_tag}_ckpt"
    # CheckpointCallback is called once per vectorized env step. With n_envs > 1,
    # each callback step contains n_envs transitions, so divide save_freq to keep
    # the checkpoint interval measured in total environment timesteps.
    effective_save_freq = max(1, int(args.save_freq) // max(1, int(args.n_envs)))
    checkpoint_callback = CheckpointCallback(
        save_freq=effective_save_freq,
        save_path=str(model_dir),
        name_prefix=checkpoint_prefix,
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    extra_metrics_callback = ExtraRolloutMetricsCallback(window_size=50, success_label=threshold_tag)
    callback_list = CallbackList([checkpoint_callback, extra_metrics_callback])

    if args.resume_from:
        resume_path = Path(args.resume_from).expanduser()
        if not resume_path.is_absolute():
            resume_path = PROJECT_ROOT / resume_path
        resume_path = resume_path.resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume model not found: {resume_path}")

        # custom_objects makes resume-time hyperparameters such as buffer_size/batch_size visible
        # to the loaded model instead of silently keeping checkpoint defaults.
        custom_objects = {
            "learning_rate": args.learning_rate,
            "buffer_size": args.buffer_size,
            "batch_size": args.batch_size,
            "learning_starts": args.learning_starts,
            "train_freq": args.train_freq,
            "gradient_steps": args.gradient_steps,
            "tau": args.tau,
            "gamma": args.gamma,
        }

        model = SAC.load(
            str(resume_path),
            env=env,
            device=args.device,
            custom_objects=custom_objects,
        )
        model.tensorboard_log = str(tb_dir)
        model.verbose = int(args.sb3_verbose)
        reset_num_timesteps = args.reset_num_timesteps

        print(f"Resuming SAC training from: {resume_path}")
        print("IMPORTANT: resume model must have the same observation shape as current env.")
        print(f"Original ent_coef from checkpoint: {_current_ent_coef_value(model):.6g}")

        # Important: override lr before rebuilding automatic ent_coef optimizer,
        # because _force_auto_ent_coef uses model.lr_schedule(1).
        _override_learning_rate(model, args.learning_rate)

        if isinstance(args.ent_coef, str) and args.ent_coef.startswith("auto"):
            _force_auto_ent_coef(model, args.ent_coef)
        else:
            _force_fixed_ent_coef(model, float(args.ent_coef))

        print(f"Resume hyperparams: batch_size={model.batch_size}, buffer_size={model.buffer_size}")

    else:
        model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            tau=args.tau,
            gamma=args.gamma,
            train_freq=args.train_freq,
            gradient_steps=args.gradient_steps,
            ent_coef=args.ent_coef,
            tensorboard_log=str(tb_dir),
            seed=args.seed,
            device=args.device,
            verbose=int(args.sb3_verbose),
        )
        reset_num_timesteps = True
        print(f"New SAC model: ent_coef={args.ent_coef}, lr={args.learning_rate:g}, batch={args.batch_size}, buffer={args.buffer_size}")

    print("[RUN]")
    print(f"  stage={threshold_tag}  env={args.env_type}  force_model={args.force_model or 'uniform/all'}")
    print(f"  obs={env.observation_space}  action={env.action_space}")
    print("  algorithm=standard SAC: actor and critic use the same 52-D waypoint observation")
    print(f"  timesteps={args.timesteps}  n_envs={int(args.n_envs)}  lr={args.learning_rate:g}  batch={args.batch_size}  buffer={args.buffer_size}  ent_coef={args.ent_coef}")
    print(f"  max_steps={args.max_episode_steps}")
    print(f"  radius_obs_scale={float(args.radius_observation_scale)*1000.0:.2f}mm")
    print(
        f"  active_train_models=0207_left,0207_right,0210,V1  randomize_start_target={bool(args.randomize_start_target)} "
        f"radius={float(args.start_target_random_radius)*1000.0:.2f}mm  "
        f"randomize_initial_orientation={bool(args.randomize_initial_orientation)} "
        f"max_angle={float(args.initial_orientation_max_angle_deg):.1f}deg  "
        f"soft_randomize_single_vessel={bool(args.soft_randomize_single_vessel)}"
    )
    print(f"  run_dir={run_dir}")

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback_list,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=bool(args.progress_bar),
        )

        final_model_path = model_dir / f"sac_mcr_{threshold_tag}_all_vessels{forced_tag}_final"
        model.save(str(final_model_path))

        print(f"[DONE] model={final_model_path}.zip")
        print(f"[DONE] tensorboard={tb_dir}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
