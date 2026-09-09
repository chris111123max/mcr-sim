"""Post-hoc deterministic/stochastic evaluation on fixed training vessels.

This diagnostic never updates a model.  It is intentionally separate from the
validation protocol so B01/B02 behavior can be measured even while validation
remains locked by the curriculum.
"""

from __future__ import annotations

import copy
import csv
import math
from pathlib import Path
import sys

import numpy as np

TRAINING_PY_DIR = Path(__file__).resolve().parent
PYTHON_ROOT = TRAINING_PY_DIR.parents[1]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from mcr_sim.distributed import DistributedPPO, configure_npu_execution, initialize_distributed
from mcr_sim.rl_core.evaluation import evaluate_vector_policy

from train_ppo import build_env, parse_args


def _configure_parser(parser) -> None:
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--eval-vessels", nargs="+", default=["B01", "B02"])
    parser.add_argument("--eval-episodes-per-vessel", type=int, default=32)
    parser.add_argument("--eval-n-envs", type=int, default=32)
    parser.add_argument("--eval-output-dir", default="")


def _append_rows(path: Path, fieldnames, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


DIAGNOSTIC_FIELDS = [
    "route_progress_m", "route_projection_segment",
    "route_projection_distance_mm", "route_projection_jump_rejections",
    "centerline_local_radius_mm", "centerline_safety_ratio",
    "centerline_safety_ratio_max_episode", "centerline_safety_margin",
    "centerline_safety_margin_min_episode", "curve_bend_5mm",
    "curve_bend_10mm", "curve_bend_20mm", "curve_alignment_error_20mm",
    "sdf_tip_clearance_min_episode_mm", "sdf_body_clearance_min_episode_mm",
    "sdf_penetration_depth_max_episode_mm", "sdf_near_wall_steps_episode",
    "sdf_wall_contact_steps_episode", "insert_action_mean_episode",
    "insert_positive_fraction_episode", "insert_negative_fraction_episode",
    "insert_near_zero_fraction_episode", "inserted_length_final_mm",
    "inserted_length_max_episode_mm", "model_action_rot_n_mean",
    "model_action_rot_n_abs_mean", "model_action_rot_n_abs_max",
    "model_action_rot_b_mean", "model_action_rot_b_abs_mean",
    "model_action_rot_b_abs_max", "model_action_insert_mean",
    "model_action_insert_abs_mean", "model_action_insert_abs_max",
]

TRACE_FIELDS = [
    "step", "route_completion", "route_progress_m", "route_progress_delta_m",
    "route_projection_segment", "route_projection_distance_m",
    "centerline_local_radius_m", "centerline_safety_ratio",
    "centerline_safety_margin", "curve_bend_5mm", "curve_bend_10mm",
    "curve_bend_20mm", "curve_alignment_error_20mm", "rot_n", "rot_b",
    "raw_insert", "effective_insert", "inserted_length_m", "tip_clearance_m",
    "body_clearance_m", "body_warning", "off_target_branch", "no_progress",
    "reward_progress", "reward_wall", "reward_stagnation", "reward_step",
    "reward_total",
]


def _finite_mean(values, default=math.nan):
    values = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(values)) if values else float(default)


def _policy_std(model):
    log_std = getattr(getattr(model, "policy", None), "log_std", None)
    if log_std is None:
        values = np.asarray([], dtype=np.float64)
    else:
        values = np.exp(
            log_std.detach().float().cpu().numpy().reshape(-1).astype(np.float64)
        )
    names = ("rot_n", "rot_b", "insert")
    result = {
        f"policy_std_{name}": (
            float(values[index]) if index < values.size else math.nan
        )
        for index, name in enumerate(names)
    }
    result["policy_std_mean"] = _finite_mean(values)
    return result


def _aggregate(episodes):
    episodes = tuple(episodes)
    count = len(episodes)
    successes = sum(int(item.success) for item in episodes)
    out_of_vessel = sum(item.terminal_reason == "out_of_vessel" for item in episodes)
    timeouts = sum(item.terminal_reason in {"timeout", "evaluation_limit"} for item in episodes)
    near_target_failures = sum(
        (not item.success) and math.isfinite(item.min_distance_mm)
        and item.min_distance_mm <= 10.0
        for item in episodes
    )
    return {
        "episodes": count,
        "success_count": successes,
        "success_rate": float(successes / count) if count else 0.0,
        "out_of_vessel_count": out_of_vessel,
        "out_of_vessel_rate": float(out_of_vessel / count) if count else 0.0,
        "timeout_count": timeouts,
        "near_target_failure_count": near_target_failures,
        "reward_mean": _finite_mean(item.reward for item in episodes),
        "steps_mean": _finite_mean(item.steps for item in episodes),
        "route_completion_mean": _finite_mean(
            item.route_completion for item in episodes
        ),
        "route_potential_mean": _finite_mean(item.route_potential for item in episodes),
        "final_distance_mm_mean": _finite_mean(
            item.final_distance_mm for item in episodes
        ),
        "min_distance_mm_mean": _finite_mean(item.min_distance_mm for item in episodes),
        "centerline_safety_ratio_max_episode_mean": _finite_mean(
            item.diagnostics.get("centerline_safety_ratio_max_episode", math.nan)
            for item in episodes
        ),
        "centerline_safety_margin_min_episode_mean": _finite_mean(
            item.diagnostics.get(
                "centerline_safety_margin_min_episode", math.nan
            )
            for item in episodes
        ),
        "sdf_body_clearance_min_episode_mm_mean": _finite_mean(
            item.diagnostics.get("sdf_body_clearance_min_episode_mm", math.nan)
            for item in episodes
        ),
        "sdf_near_wall_steps_episode_mean": _finite_mean(
            item.diagnostics.get("sdf_near_wall_steps_episode", math.nan)
            for item in episodes
        ),
    }


def main() -> None:
    args = parse_args(_configure_parser)
    if args.eval_episodes_per_vessel <= 0:
        raise ValueError("--eval-episodes-per-vessel must be positive")
    if args.eval_n_envs <= 0:
        raise ValueError("--eval-n-envs must be positive")

    checkpoints = [Path(value).expanduser().resolve() for value in args.checkpoints]
    missing = [str(path) for path in checkpoints if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing checkpoints: " + ", ".join(missing))

    context = initialize_distributed(
        enabled=False,
        requested_device=args.device,
        requested_world_size=1,
        cli_local_rank=0,
        requested_backend="",
    )
    configure_npu_execution(
        context.device.accelerator,
        enabled=bool(args.npu_fast_execution),
    )
    args.resolved_device = context.device.resolved
    args.distributed_rank = 0
    args.rank_seed = int(args.seed)
    args.n_envs = 1
    args.local_n_envs = 1
    args.training_curriculum = False
    args.randomize_start_target = False
    args.randomize_initial_orientation = False
    args.soft_randomize_single_vessel = False
    args.start_window_mm = 0.0
    args.target_window_mm = 0.0
    args.initial_orientation_max_angle_deg = 0.0
    args.vessel_scale_min = 1.0
    args.vessel_scale_max = 1.0
    args.render = "headless"

    output_dir = (
        Path(args.eval_output_dir).expanduser().resolve()
        if args.eval_output_dir
        else checkpoints[0].parent.parent / "diagnostics"
    )
    summary_path = output_dir / "checkpoint_safety_summary.csv"
    vessel_path = output_dir / "checkpoint_safety_vessels.csv"
    episode_path = output_dir / "checkpoint_safety_episodes.csv"
    trace_path = output_dir / "checkpoint_safety_terminal_trace.csv"
    summary_fields = [
        "checkpoint", "mode", "vessels", "episodes", "success_count",
        "success_rate", "out_of_vessel_count", "out_of_vessel_rate",
        "timeout_count", "near_target_failure_count", "reward_mean",
        "steps_mean", "route_completion_mean", "route_potential_mean",
        "final_distance_mm_mean", "min_distance_mm_mean",
        "centerline_safety_ratio_max_episode_mean",
        "centerline_safety_margin_min_episode_mean",
        "sdf_body_clearance_min_episode_mm_mean",
        "sdf_near_wall_steps_episode_mean", "policy_std_rot_n",
        "policy_std_rot_b", "policy_std_insert", "policy_std_mean",
    ]
    vessel_fields = ["checkpoint", "mode", "vessel_id"] + summary_fields[3:]
    episode_fields = [
        "checkpoint", "mode", "vessel_id", "episode_index", "seed",
        "success", "terminal_reason", "steps", "reward",
        "route_completion", "route_potential", "final_distance_mm",
        "min_distance_mm", "error",
    ] + DIAGNOSTIC_FIELDS
    trace_fields = [
        "checkpoint", "mode", "vessel_id", "episode_index", "seed",
        "success", "terminal_reason", "trace_offset_from_end",
    ] + TRACE_FIELDS

    for checkpoint in checkpoints:
        model = DistributedPPO.load(
            str(checkpoint),
            device=args.resolved_device,
        )
        model.set_distributed_context(context)
        model.policy.set_training_mode(False)
        policy_std = _policy_std(model)
        for mode_index, deterministic in enumerate((True, False)):
            mode = "deterministic" if deterministic else "stochastic"
            model.set_random_seed(int(args.seed) + mode_index * 1_000_000)

            def env_factory(vessel_id, n_envs):
                eval_args = copy.copy(args)
                eval_args.force_model = str(vessel_id).upper()
                eval_args.n_envs = int(n_envs)
                eval_args.local_n_envs = int(n_envs)
                return build_env(eval_args)

            result = evaluate_vector_policy(
                tuple(str(value).upper() for value in args.eval_vessels),
                env_factory,
                lambda observation, flag=deterministic: model.predict(
                    observation, deterministic=flag
                )[0],
                episodes_per_vessel=int(args.eval_episodes_per_vessel),
                max_parallel_envs=int(args.eval_n_envs),
                max_episode_steps=int(args.max_episode_steps),
                base_seed=int(args.seed) + 200_000,
            )
            _append_rows(
                summary_path,
                summary_fields,
                [{
                    "checkpoint": str(checkpoint),
                    "mode": mode,
                    "vessels": result.valid_vessels,
                    **_aggregate(result.episodes),
                    **policy_std,
                }],
            )
            vessel_rows = []
            for vessel_id in sorted({item.vessel_id for item in result.episodes}):
                vessel_episodes = tuple(
                    item for item in result.episodes if item.vessel_id == vessel_id
                )
                vessel_rows.append({
                    "checkpoint": str(checkpoint),
                    "mode": mode,
                    "vessel_id": vessel_id,
                    **_aggregate(vessel_episodes),
                    **policy_std,
                })
            _append_rows(vessel_path, vessel_fields, vessel_rows)
            _append_rows(
                episode_path,
                episode_fields,
                [{
                    "checkpoint": str(checkpoint),
                    "mode": mode,
                    "vessel_id": item.vessel_id,
                    "episode_index": item.episode_index,
                    "seed": item.seed,
                    "success": item.success,
                    "terminal_reason": item.terminal_reason,
                    "steps": item.steps,
                    "reward": item.reward,
                    "route_completion": item.route_completion,
                    "route_potential": item.route_potential,
                    "final_distance_mm": item.final_distance_mm,
                    "min_distance_mm": item.min_distance_mm,
                    "error": item.error,
                    **{
                        key: item.diagnostics.get(key, math.nan)
                        for key in DIAGNOSTIC_FIELDS
                    },
                } for item in result.episodes],
            )
            trace_rows = []
            for item in result.episodes:
                trace = tuple(item.diagnostics.get("terminal_trace", ()))
                for trace_index, trace_item in enumerate(trace):
                    trace_rows.append({
                        "checkpoint": str(checkpoint),
                        "mode": mode,
                        "vessel_id": item.vessel_id,
                        "episode_index": item.episode_index,
                        "seed": item.seed,
                        "success": item.success,
                        "terminal_reason": item.terminal_reason,
                        "trace_offset_from_end": trace_index - len(trace),
                        **{
                            key: trace_item.get(key, math.nan)
                            for key in TRACE_FIELDS
                        },
                    })
            _append_rows(trace_path, trace_fields, trace_rows)
            print(
                f"[TRAIN EVAL] checkpoint={checkpoint.name} mode={mode} "
                f"episodes={result.valid_episodes} "
                f"success_rate={result.valid_success_rate:.6f} "
                f"route_completion={result.valid_route_completion_mean:.6f}",
                flush=True,
            )

    print(f"[TRAIN EVAL] summary={summary_path}", flush=True)
    print(f"[TRAIN EVAL] vessels={vessel_path}", flush=True)
    print(f"[TRAIN EVAL] episodes={episode_path}", flush=True)
    print(f"[TRAIN EVAL] terminal_trace={trace_path}", flush=True)


if __name__ == "__main__":
    main()
