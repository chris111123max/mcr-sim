"""Train the isolated Goal-conditioned SAC + Safe HER experiment.

The baseline ``train_sac.py`` and MCREnv are imported only as reusable
infrastructure.  This entry point changes neither their defaults nor Reward
V11; it wraps the vector environment and uses a separate replay/algorithm.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

TRAINING_PY_DIR = Path(__file__).resolve().parent
PYTHON_ROOT = TRAINING_PY_DIR.parent.parent
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from mcr_sim.goal_conditioned_env import GoalConditionedVecEnv, SafeHerReplayBuffer
from mcr_sim.goal_sac_config import *  # noqa: F401,F403 - explicit run metadata
from mcr_sim.distributed import (
    GoalConditionedSAC,
    configure_npu_execution,
    convert_to_npu_fused_adam,
    initialize_distributed,
)
from mcr_sim.paths import PROJECT_ROOT, TRAINING_RUNS_DIR
from mcr_sim.rl_core.run_logging import start_run_log_capture, write_run_config
from mcr_sim.rl_core.experiment import EpochExperimentCallback
from mcr_sim.rl_core.evaluation import (
    discover_validation_vessels,
    evaluate_policy,
)
from mcr_sim.training_config import (
    ENTRY_TANGENT_POINTS,
    FRAME_SKIP,
    INITIAL_ORIENTATION_MAX_ANGLE_DEG,
    RADIUS_OBSERVATION_SCALE_M,
    SETTLE_STEPS,
    SOFA_TIME_STEP_S,
    START_WINDOW_DISTANCE_M,
    TARGET_THRESHOLD_M,
    TARGET_WINDOW_DISTANCE_M,
    VESSEL_SCALE_MAX,
    VESSEL_SCALE_MIN,
    TRAINING_CURRICULUM_ENABLED,
    curriculum_protocol_profile,
)
from training.py.train_sac import build_env


def parse_args():
    parser = argparse.ArgumentParser(description="Goal-conditioned SAC + Safe HER")
    parser.add_argument("--env-type", choices=["aortic", "flat"], default="aortic")
    parser.add_argument("--force-model", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--local-rank", "--local_rank", dest="local_rank", type=int, default=0)
    parser.add_argument("--dist-backend", default="", choices=["", "hccl", "nccl", "gloo"])
    parser.add_argument("--n-envs", type=int, default=GOAL_SAC_N_ENVS)
    parser.add_argument("--epochs", type=int, default=GOAL_SAC_EPOCHS)
    parser.add_argument("--episodes-per-epoch", type=int, default=GOAL_SAC_EPISODES_PER_EPOCH)
    parser.add_argument("--timesteps", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=GOAL_SAC_LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=GOAL_SAC_BATCH_SIZE)
    parser.add_argument("--buffer-size", type=int, default=GOAL_SAC_BUFFER_SIZE)
    parser.add_argument("--learning-starts", type=int, default=GOAL_SAC_LEARNING_STARTS)
    parser.add_argument("--train-freq", type=int, default=GOAL_SAC_TRAIN_FREQ)
    parser.add_argument("--tau", type=float, default=GOAL_SAC_TAU)
    parser.add_argument("--gamma", type=float, default=GOAL_SAC_GAMMA)
    parser.add_argument("--ent-coef", default=GOAL_SAC_ENT_COEF)
    parser.add_argument("--her-ratio", type=float, default=GOAL_SAC_HER_RATIO)
    parser.add_argument("--her-safe-margin-mm", type=float, default=GOAL_SAC_HER_SAFE_MARGIN_M * 1000.0)
    parser.add_argument("--goal-tolerance", type=float, default=GOAL_SAC_GOAL_TOLERANCE)
    parser.add_argument(
        "--her-min-goal-advance",
        type=float,
        default=GOAL_SAC_HER_MIN_GOAL_ADVANCE,
    )
    parser.add_argument("--goal-step-cost", type=float, default=GOAL_SAC_STEP_COST)
    parser.add_argument("--goal-potential-scale", type=float, default=GOAL_SAC_POTENTIAL_SCALE)
    parser.add_argument("--goal-success-bonus", type=float, default=GOAL_SAC_SUCCESS_BONUS)
    parser.add_argument("--goal-safety-weight", type=float, default=GOAL_SAC_SAFETY_WEIGHT)
    parser.add_argument(
        "--goal-action-smoothness-weight",
        type=float,
        default=GOAL_SAC_ACTION_SMOOTHNESS_WEIGHT,
    )
    parser.add_argument(
        "--goal-failure-terminal-penalty",
        type=float,
        default=GOAL_SAC_FAILURE_TERMINAL_PENALTY,
    )
    parser.add_argument("--goal-timeout-penalty", type=float, default=GOAL_SAC_TIMEOUT_PENALTY)
    parser.add_argument(
        "--goal-out-of-vessel-penalty",
        type=float,
        default=GOAL_SAC_OUT_OF_VESSEL_PENALTY,
    )
    parser.add_argument(
        "--goal-non-finite-penalty",
        type=float,
        default=GOAL_SAC_NON_FINITE_PENALTY,
    )
    parser.add_argument("--critic-ensemble-size", type=int, default=GOAL_SAC_CRITIC_ENSEMBLE_SIZE)
    parser.add_argument("--target-critic-subset-size", type=int, default=GOAL_SAC_TARGET_CRITIC_SUBSET_SIZE)
    parser.add_argument("--utd-ratio", type=int, default=GOAL_SAC_UTD_RATIO)
    parser.add_argument("--utd-warmup-steps", type=int, default=GOAL_SAC_UTD_WARMUP_STEPS)
    parser.add_argument("--actor-update-interval", type=int, default=GOAL_SAC_ACTOR_UPDATE_INTERVAL)
    parser.add_argument("--min-ent-coef", type=float, default=GOAL_SAC_MIN_ENT_COEF)
    parser.add_argument("--metric-log-interval", type=int, default=GOAL_SAC_METRIC_LOG_INTERVAL)
    parser.add_argument(
        "--performance-log-interval-steps",
        type=int,
        default=GOAL_SAC_PERFORMANCE_LOG_INTERVAL_STEPS,
    )
    parser.add_argument("--no-critic-layer-norm", action="store_true", default=True)
    curriculum = parser.add_mutually_exclusive_group()
    curriculum.add_argument(
        "--training-curriculum", dest="training_curriculum", action="store_true"
    )
    curriculum.add_argument(
        "--no-training-curriculum", dest="training_curriculum", action="store_false"
    )
    parser.set_defaults(training_curriculum=TRAINING_CURRICULUM_ENABLED)
    parser.add_argument("--randomize-start-target", action="store_true", default=True)
    parser.add_argument("--no-randomize-start-target", dest="randomize_start_target", action="store_false")
    parser.add_argument("--randomize-initial-orientation", action="store_true", default=True)
    parser.add_argument("--no-randomize-initial-orientation", dest="randomize_initial_orientation", action="store_false")
    parser.add_argument("--start-window-mm", type=float, default=START_WINDOW_DISTANCE_M * 1000.0)
    parser.add_argument("--target-window-mm", type=float, default=TARGET_WINDOW_DISTANCE_M * 1000.0)
    parser.add_argument("--initial-orientation-max-angle-deg", type=float, default=INITIAL_ORIENTATION_MAX_ANGLE_DEG)
    parser.add_argument("--entry-tangent-points", type=int, default=ENTRY_TANGENT_POINTS)
    parser.add_argument("--vessel-scale-min", type=float, default=VESSEL_SCALE_MIN)
    parser.add_argument("--vessel-scale-max", type=float, default=VESSEL_SCALE_MAX)
    parser.add_argument("--radius-observation-scale", type=float, default=RADIUS_OBSERVATION_SCALE_M)
    parser.add_argument("--frame-skip", type=int, default=FRAME_SKIP)
    parser.add_argument("--time-step", type=float, default=SOFA_TIME_STEP_S)
    parser.add_argument("--settle-steps", type=int, default=SETTLE_STEPS)
    parser.add_argument("--target-threshold", type=float, default=TARGET_THRESHOLD_M)
    parser.add_argument("--max-episode-steps", type=int, default=GOAL_SAC_MAX_EPISODE_STEPS)
    parser.add_argument("--render", choices=["headless", "human"], default="headless")
    parser.add_argument("--log-root", default=str(TRAINING_RUNS_DIR))
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--save-freq", type=int, default=0)
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--valid-dir", default=str(PROJECT_ROOT / "mesh" / "valid"))
    parser.add_argument("--valid-episodes-per-vessel", type=int, default=2)
    parser.add_argument("--validation-interval", type=int, default=2)
    parser.add_argument("--valid-min-train-success-rate", type=float, default=0.20)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--no-npu-fused-adam", action="store_true")
    parser.add_argument("--no-npu-fast-execution", action="store_true")
    parser.add_argument("--no-npu-replay-buffer", action="store_true")
    return parser.parse_args()


def _baseline_args(args, context):
    """Build the small Namespace expected by the proven baseline env factory."""
    return SimpleNamespace(
        env_type=args.env_type, render=args.render, n_envs=args.n_envs,
        local_n_envs=args.local_n_envs, distributed_rank=context.rank,
        seed=args.seed, radius_observation_scale=args.radius_observation_scale,
        gamma=args.gamma, randomize_start_target=args.randomize_start_target,
        start_window_mm=args.start_window_mm, target_window_mm=args.target_window_mm,
        randomize_initial_orientation=args.randomize_initial_orientation,
        initial_orientation_max_angle_deg=args.initial_orientation_max_angle_deg,
        entry_tangent_points=args.entry_tangent_points,
        soft_randomize_single_vessel=True, vessel_scale_min=args.vessel_scale_min,
        vessel_scale_max=args.vessel_scale_max,
        training_curriculum=args.training_curriculum,
        scene_verbose=False, force_model=args.force_model, asset_root="",
        time_step=args.time_step, frame_skip=args.frame_skip,
        settle_steps=args.settle_steps, target_threshold=args.target_threshold,
        max_episode_steps=args.max_episode_steps,
    )


def main():
    args = parse_args()
    if args.resume_from:
        raise NotImplementedError(
            "Goal-SAC resume is intentionally disabled in the first isolated implementation; "
            "start a new run so HER/ensemble replay state cannot be mixed with baseline checkpoints."
        )
    context = initialize_distributed(
        enabled=bool(args.distributed), requested_device=args.device,
        requested_world_size=args.world_size, cli_local_rank=args.local_rank,
        requested_backend=args.dist_backend,
    )
    args.npu_execution = configure_npu_execution(
        context.device.accelerator, enabled=not args.no_npu_fast_execution
    )
    if args.n_envs % context.world_size != 0:
        raise ValueError("--n-envs must be divisible by --world-size in distributed mode.")
    if args.batch_size % context.world_size != 0:
        raise ValueError("--batch-size must be divisible by --world-size in distributed mode.")
    args.local_n_envs = args.n_envs // context.world_size if context.enabled else args.n_envs
    local_batch = args.batch_size // context.world_size if context.enabled else args.batch_size
    total_timesteps = int(args.timesteps or args.epochs * args.episodes_per_epoch * args.max_episode_steps)
    local_timesteps = int(math.ceil(total_timesteps / context.world_size)) if context.enabled else total_timesteps
    args.resolved_device = context.device.resolved
    args.algorithm = "goal_sac"
    args.goal_sac = True
    args.goal_conditioning = "route_context_achieved_desired_v3_obs47"
    args.curriculum_protocol = curriculum_protocol_profile()
    args.reward_profile = {
        "version": "goal_potential_antistall_v4",
        "goal_reward": "base + gamma*Phi(next)-Phi(previous); terminal Phi=0",
        "potential": "scale*min(max(achieved,0),goal)",
        "dense_route_progress": True,
        "potential_scale": args.goal_potential_scale,
        "success_bonus": args.goal_success_bonus,
        "gamma": args.gamma,
        "safety_weight": args.goal_safety_weight,
        "action_smoothness_weight": args.goal_action_smoothness_weight,
        "failure_terminal_penalty": args.goal_failure_terminal_penalty,
        "timeout_penalty": args.goal_timeout_penalty,
        "out_of_vessel_penalty": args.goal_out_of_vessel_penalty,
        "non_finite_penalty": args.goal_non_finite_penalty,
        "failure_horizon_compensation": False,
        "timeout_bootstrap": False,
        "her": "disabled" if args.her_ratio == 0.0 else "safe_future",
        "her_min_goal_advance": args.her_min_goal_advance,
        "original_her_ratio": [1.0 - args.her_ratio, args.her_ratio],
    }
    args.goal_sac_config = {
        "critic_ensemble_size": args.critic_ensemble_size,
        "target_critic_subset_size": args.target_critic_subset_size,
        "utd_ratio": args.utd_ratio,
        "utd_warmup_steps": args.utd_warmup_steps,
        "actor_update_interval": args.actor_update_interval,
        "min_ent_coef": args.min_ent_coef,
        "metric_log_interval": args.metric_log_interval,
        "performance_log_interval_steps": args.performance_log_interval_steps,
        "critic_layer_norm": not args.no_critic_layer_norm,
    }
    run_root = Path(args.log_root).expanduser()
    if not run_root.is_absolute():
        run_root = PROJECT_ROOT / run_root
    timestamp = os.environ.get("MCR_RUN_TIMESTAMP", "") if context.is_main else ""
    if context.is_main and not timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp = context.broadcast_text(timestamp)
    run_dir = run_root.resolve() / f"{args.exp_name}_{timestamp}"
    log_dir, model_dir, tb_dir = run_dir / "logs", run_dir / "models", run_dir / "tb"
    if context.is_main:
        log_dir.mkdir(parents=True, exist_ok=True); model_dir.mkdir(parents=True, exist_ok=True); tb_dir.mkdir(parents=True, exist_ok=True)
        write_run_config(log_dir / "run_config.json", args, algorithm="goal_sac", run_dir=run_dir, model_dir=model_dir, tensorboard_dir=tb_dir)
    context.barrier()
    start_run_log_capture(log_dir, context.rank)
    env = None
    reward_kwargs = dict(
        step_cost=args.goal_step_cost, gamma=args.gamma,
        potential_scale=args.goal_potential_scale, success_bonus=args.goal_success_bonus,
        failure_terminal_penalty=args.goal_failure_terminal_penalty)
    try:
        env = GoalConditionedVecEnv(
            build_env(_baseline_args(args, context)),
            **reward_kwargs,
            safety_weight=args.goal_safety_weight,
            action_smoothness_weight=args.goal_action_smoothness_weight,
            timeout_penalty=args.goal_timeout_penalty,
            out_of_vessel_penalty=args.goal_out_of_vessel_penalty,
            non_finite_penalty=args.goal_non_finite_penalty,
            max_episode_steps=args.max_episode_steps,
        )
        model = GoalConditionedSAC(
            policy="MlpPolicy", env=env, learning_rate=args.learning_rate,
            buffer_size=args.buffer_size, learning_starts=args.learning_starts,
            batch_size=local_batch, tau=args.tau, gamma=args.gamma,
            train_freq=args.train_freq, gradient_steps=1, ent_coef=args.ent_coef,
            replay_buffer_class=SafeHerReplayBuffer,
            replay_buffer_kwargs=dict(
                her_ratio=args.her_ratio,
                her_safe_margin_m=args.her_safe_margin_mm / 1000.0,
                goal_tolerance=args.goal_tolerance,
                min_goal_advance=args.her_min_goal_advance,
                **reward_kwargs,
            ), tensorboard_log=str(tb_dir) if context.is_main else None,
            seed=args.seed + context.rank, device=args.resolved_device,
            verbose=1 if context.is_main else 0, distributed_context=context,
            critic_ensemble_size=args.critic_ensemble_size,
            target_critic_subset_size=args.target_critic_subset_size,
            utd_ratio=args.utd_ratio, utd_warmup_steps=args.utd_warmup_steps,
            actor_update_interval=args.actor_update_interval,
            critic_layer_norm=not args.no_critic_layer_norm,
            metric_log_interval=args.metric_log_interval,
            min_ent_coef=args.min_ent_coef,
        )
        if context.enabled:
            model.synchronize_parameters()
        if context.device.accelerator == "npu" and not args.no_npu_fused_adam:
            actor_opt, actor_status = convert_to_npu_fused_adam(model.actor.optimizer)
            critic_opt, critic_status = convert_to_npu_fused_adam(model.critic.optimizer)
            if actor_status in {"enabled", "already_enabled"}:
                model.actor.optimizer = actor_opt
            if critic_status in {"enabled", "already_enabled"}:
                model.critic.optimizer = critic_opt
            print(f"[GOAL SAC] fused_adam actor={actor_status} critic={critic_status}")
        valid_dir = Path(args.valid_dir).expanduser()
        if not valid_dir.is_absolute():
            valid_dir = PROJECT_ROOT / valid_dir
        valid_vessels = [] if args.skip_validation else discover_validation_vessels(
            valid_dir, expected_vessels=5
        )

        def run_validation(current_model, epoch):
            def valid_factory(vessel_id):
                valid_args = _baseline_args(args, context)
                valid_args.force_model = str(vessel_id)
                valid_args.asset_root = str(valid_dir.resolve())
                valid_args.local_n_envs = 1
                valid_args.n_envs = 1
                valid_args.render = "headless"
                valid_args.training_curriculum = False
                return GoalConditionedVecEnv(
                    build_env(valid_args),
                    **reward_kwargs,
                    safety_weight=args.goal_safety_weight,
                    action_smoothness_weight=args.goal_action_smoothness_weight,
                    timeout_penalty=args.goal_timeout_penalty,
                    out_of_vessel_penalty=args.goal_out_of_vessel_penalty,
                    non_finite_penalty=args.goal_non_finite_penalty,
                    max_episode_steps=args.max_episode_steps,
                )

            return evaluate_policy(
                vessel_ids=valid_vessels,
                env_factory=valid_factory,
                deterministic_action=lambda observation: current_model.predict(
                    observation, deterministic=True
                )[0],
                episodes_per_vessel=args.valid_episodes_per_vessel,
                max_episode_steps=args.max_episode_steps,
                task_rank=context.rank,
                task_world_size=context.world_size,
            )

        callback = EpochExperimentCallback(
            context=context,
            algorithm_name="goal_sac",
            variant="noher" if args.her_ratio == 0.0 else "safeher",
            epochs=args.epochs,
            episodes_per_epoch=args.episodes_per_epoch,
            model_dir=model_dir,
            run_dir=log_dir,
            validation_interval=args.validation_interval,
            validation_min_train_success_rate=args.valid_min_train_success_rate,
            validation_fn=None if args.skip_validation else run_validation,
            resume_progress=False,
            training_curriculum_enabled=args.training_curriculum,
            performance_log_interval_steps=args.performance_log_interval_steps,
        )
        print(
            f"[GOAL SAC] device={context.device.resolved} world={context.world_size} "
            f"envs={args.n_envs}/{args.local_n_envs} obs={env.observation_space.shape} "
            f"batch={args.batch_size}/{local_batch} q_ensemble={args.critic_ensemble_size} "
            f"critic_modules={model._critic_module_count} "
            f"target_subset={args.target_critic_subset_size} utd={args.utd_ratio} "
            f"her={args.her_ratio:.2f} her_min_advance={args.her_min_goal_advance:.3f} "
            f"min_ent_coef={args.min_ent_coef:.3f} "
            f"curriculum={'on' if args.training_curriculum else 'off'} output={run_dir}"
        )
        print(
            "[GOAL SAC REWARD] version=goal_potential_antistall_v4 "
            f"step=-{args.goal_step_cost:g} success=+{args.goal_success_bonus:g} "
            f"timeout=-{args.goal_timeout_penalty:g} "
            f"out=-{args.goal_out_of_vessel_penalty:g} "
            f"non_finite=-{args.goal_non_finite_penalty:g} "
            f"potential={args.goal_potential_scale:g} "
            f"safety={args.goal_safety_weight:g} "
            f"smooth={args.goal_action_smoothness_weight:g} ent_coef={args.ent_coef}",
            flush=True,
        )
        model.learn(total_timesteps=local_timesteps, callback=callback, reset_num_timesteps=True)
        context.barrier()
        if context.is_main:
            final_path = model_dir / f"goal_sac_final_{args.epochs:03d}ep"
            model.save(str(final_path))
            print(f"[DONE] model={final_path}.zip")
    finally:
        if env is not None:
            env.close()
        context.close()


if __name__ == "__main__":
    main()
