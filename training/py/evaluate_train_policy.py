"""Post-hoc deterministic/stochastic evaluation on fixed training vessels.

This diagnostic never updates a model.  It is intentionally separate from the
validation protocol so B01/B02 behavior can be measured even while validation
remains locked by the curriculum.
"""

from __future__ import annotations

import copy
import csv
from pathlib import Path
import sys

TRAINING_PY_DIR = Path(__file__).resolve().parent
PYTHON_ROOT = TRAINING_PY_DIR.parents[1]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from mcr_sim.distributed import DistributedPPO, configure_npu_execution, initialize_distributed
from mcr_sim.rl_core.evaluation import evaluate_policy

from train_ppo import build_env, parse_args


def _configure_parser(parser) -> None:
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--eval-vessels", nargs="+", default=["B01", "B02"])
    parser.add_argument("--eval-episodes-per-vessel", type=int, default=10)
    parser.add_argument("--eval-output-dir", default="")


def _append_rows(path: Path, fieldnames, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args(_configure_parser)
    if args.eval_episodes_per_vessel <= 0:
        raise ValueError("--eval-episodes-per-vessel must be positive")

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
    summary_path = output_dir / "train_policy_eval_summary.csv"
    episode_path = output_dir / "train_policy_eval_episodes.csv"
    summary_fields = [
        "checkpoint", "mode", "vessels", "episodes", "success_count",
        "success_rate", "route_completion_mean", "route_potential_mean",
        "final_distance_mm_mean", "min_distance_mm_mean",
    ]
    episode_fields = [
        "checkpoint", "mode", "vessel_id", "episode_index", "seed",
        "success", "terminal_reason", "steps", "reward",
        "route_completion", "route_potential", "final_distance_mm",
        "min_distance_mm", "error",
    ]

    for checkpoint in checkpoints:
        model = DistributedPPO.load(
            str(checkpoint),
            device=args.resolved_device,
        )
        model.set_distributed_context(context)
        model.policy.set_training_mode(False)
        for mode_index, deterministic in enumerate((True, False)):
            mode = "deterministic" if deterministic else "stochastic"
            model.set_random_seed(int(args.seed) + mode_index * 1_000_000)

            def env_factory(vessel_id):
                eval_args = copy.copy(args)
                eval_args.force_model = str(vessel_id).upper()
                return build_env(eval_args)

            result = evaluate_policy(
                tuple(str(value).upper() for value in args.eval_vessels),
                env_factory,
                lambda observation, flag=deterministic: model.predict(
                    observation, deterministic=flag
                )[0],
                episodes_per_vessel=int(args.eval_episodes_per_vessel),
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
                    "episodes": result.valid_episodes,
                    "success_count": result.valid_success_count,
                    "success_rate": result.valid_success_rate,
                    "route_completion_mean": result.valid_route_completion_mean,
                    "route_potential_mean": result.valid_route_potential_mean,
                    "final_distance_mm_mean": result.valid_final_distance_mm_mean,
                    "min_distance_mm_mean": result.valid_min_distance_mm_mean,
                }],
            )
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
                } for item in result.episodes],
            )
            print(
                f"[TRAIN EVAL] checkpoint={checkpoint.name} mode={mode} "
                f"episodes={result.valid_episodes} "
                f"success_rate={result.valid_success_rate:.6f} "
                f"route_completion={result.valid_route_completion_mean:.6f}",
                flush=True,
            )

    print(f"[TRAIN EVAL] summary={summary_path}", flush=True)
    print(f"[TRAIN EVAL] episodes={episode_path}", flush=True)


if __name__ == "__main__":
    main()
