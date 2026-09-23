"""Four-device Ascend PPO baseline using the shared MCR experiment protocol."""

from __future__ import annotations

import argparse
import copy
import math
import os
import sys
from datetime import datetime
from pathlib import Path

TRAINING_PY_DIR = Path(__file__).resolve().parent
PYTHON_ROOT = TRAINING_PY_DIR.parents[1]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

import numpy as np
import torch as th
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.utils import get_schedule_fn

from mcr_sim.distributed import (
    DistributedPPO,
    configure_npu_execution,
    convert_to_npu_fused_adam,
    initialize_distributed,
)
from mcr_sim.paths import PROJECT_ROOT, TRAINING_RUNS_DIR, VALID_MESH_DIR
from mcr_sim.rl_core.evaluation import discover_validation_vessels, evaluate_policy
from mcr_sim.rl_core.experiment import EpochExperimentCallback
from mcr_sim.rl_core.run_logging import start_run_log_capture, write_run_config
from mcr_sim.training_config import (
    FRAME_SKIP,
    PHYSICS_SUBSTEPS,
    INITIAL_ORIENTATION_MAX_ANGLE_DEG,
    MAX_EPISODE_STEPS,
    PPO_BATCH_SIZE,
    PPO_CLIP_RANGE,
    PPO_ENT_COEF,
    PPO_EPOCHS,
    PPO_EPISODES_PER_EPOCH,
    PPO_GAE_LAMBDA,
    PPO_GAMMA,
    PPO_LEARNING_RATE,
    PPO_MAX_GRAD_NORM,
    PPO_MAX_ACTION_STD,
    PPO_INITIAL_ACTION_STD,
    PPO_MIN_ACTION_STD,
    PPO_N_ENVS,
    PPO_N_EPOCHS,
    PPO_N_STEPS,
    PPO_VF_COEF,
    RADIUS_OBSERVATION_SCALE_M,
    SETTLE_STEPS,
    SOFA_TIME_STEP_S,
    START_WINDOW_DISTANCE_M,
    TARGET_THRESHOLD_M,
    TARGET_WINDOW_DISTANCE_M,
    TRAINING_CURRICULUM_ENABLED,
    VALID_EPISODES_PER_VESSEL,
    VALID_INTERVAL,
    VALID_MIN_TRAIN_SUCCESS_RATE,
    VALID_VESSELS,
    VESSEL_SCALE_MAX,
    VESSEL_SCALE_MIN,
    curriculum_protocol_profile,
    reward_profile,
)

# Reuse the established SOFA environment construction and detailed rollout
# logger instead of maintaining a second algorithm-specific environment path.
from train_sac import (  # noqa: E402
    ALL_MODEL_CHOICES,
    DistributedRuntimeCallback,
    ExtraRolloutMetricsCallback,
    build_env,
)


def parse_args(configure_parser=None):
    parser = argparse.ArgumentParser(description="Train MCR agent with PPO")
    parser.add_argument("--env-type", choices=["aortic", "flat"], default="aortic")
    parser.add_argument("--force-model", choices=ALL_MODEL_CHOICES, default="")
    parser.add_argument("--epochs", type=int, default=PPO_EPOCHS)
    parser.add_argument("--episodes-per-epoch", type=int, default=PPO_EPISODES_PER_EPOCH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    fused_adam = parser.add_mutually_exclusive_group()
    fused_adam.add_argument(
        "--npu-fused-adam", dest="npu_fused_adam", action="store_true"
    )
    fused_adam.add_argument(
        "--no-npu-fused-adam", dest="npu_fused_adam", action="store_false"
    )
    npu_execution = parser.add_mutually_exclusive_group()
    npu_execution.add_argument(
        "--npu-fast-execution", dest="npu_fast_execution", action="store_true"
    )
    npu_execution.add_argument(
        "--no-npu-fast-execution", dest="npu_fast_execution", action="store_false"
    )
    parser.set_defaults(npu_fused_adam=True, npu_fast_execution=True)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--local-rank", "--local_rank", dest="local_rank", type=int, default=0)
    parser.add_argument("--dist-backend", choices=["", "hccl", "nccl", "gloo"], default="")
    parser.add_argument("--n-envs", type=int, default=PPO_N_ENVS)

    parser.add_argument("--learning-rate", type=float, default=PPO_LEARNING_RATE)
    parser.add_argument("--n-steps", type=int, default=PPO_N_STEPS)
    parser.add_argument("--batch-size", type=int, default=PPO_BATCH_SIZE)
    parser.add_argument("--n-epochs", type=int, default=PPO_N_EPOCHS)
    parser.add_argument("--gamma", type=float, default=PPO_GAMMA)
    parser.add_argument("--gae-lambda", type=float, default=PPO_GAE_LAMBDA)
    parser.add_argument("--clip-range", type=float, default=PPO_CLIP_RANGE)
    parser.add_argument("--ent-coef", type=float, default=PPO_ENT_COEF)
    parser.add_argument("--vf-coef", type=float, default=PPO_VF_COEF)
    parser.add_argument("--max-grad-norm", type=float, default=PPO_MAX_GRAD_NORM)
    parser.add_argument("--min-action-std", type=float, default=PPO_MIN_ACTION_STD)
    parser.add_argument("--max-action-std", type=float, default=PPO_MAX_ACTION_STD)
    parser.add_argument("--initial-action-std", type=float, default=PPO_INITIAL_ACTION_STD)

    parser.add_argument("--frame-skip", type=int, default=FRAME_SKIP)
    parser.add_argument("--physics-substeps", type=int, default=PHYSICS_SUBSTEPS)
    parser.add_argument("--time-step", type=float, default=SOFA_TIME_STEP_S)
    parser.add_argument("--settle-steps", type=int, default=SETTLE_STEPS)
    parser.add_argument("--target-threshold", type=float, default=TARGET_THRESHOLD_M)
    parser.add_argument("--max-episode-steps", type=int, default=MAX_EPISODE_STEPS)
    parser.add_argument("--radius-observation-scale", type=float, default=RADIUS_OBSERVATION_SCALE_M)

    endpoints = parser.add_mutually_exclusive_group()
    endpoints.add_argument("--randomize-start-target", dest="randomize_start_target", action="store_true")
    endpoints.add_argument("--no-randomize-start-target", dest="randomize_start_target", action="store_false")
    parser.set_defaults(randomize_start_target=True)
    parser.add_argument("--start-window-mm", type=float, default=START_WINDOW_DISTANCE_M * 1000.0)
    parser.add_argument("--target-window-mm", type=float, default=TARGET_WINDOW_DISTANCE_M * 1000.0)
    parser.add_argument("--start-window-points", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--target-window-points", type=int, default=None, help=argparse.SUPPRESS)

    orientation = parser.add_mutually_exclusive_group()
    orientation.add_argument("--randomize-initial-orientation", dest="randomize_initial_orientation", action="store_true")
    orientation.add_argument("--no-randomize-initial-orientation", dest="randomize_initial_orientation", action="store_false")
    parser.set_defaults(randomize_initial_orientation=True)
    parser.add_argument("--initial-orientation-max-angle-deg", type=float, default=INITIAL_ORIENTATION_MAX_ANGLE_DEG)
    parser.add_argument("--entry-tangent-points", type=int, default=5)
    parser.add_argument("--soft-randomize-single-vessel", dest="soft_randomize_single_vessel", action="store_true")
    parser.add_argument("--no-soft-randomize-single-vessel", dest="soft_randomize_single_vessel", action="store_false")
    parser.set_defaults(soft_randomize_single_vessel=True)
    parser.add_argument("--vessel-scale-min", type=float, default=VESSEL_SCALE_MIN)
    parser.add_argument("--vessel-scale-max", type=float, default=VESSEL_SCALE_MAX)
    curriculum = parser.add_mutually_exclusive_group()
    curriculum.add_argument(
        "--training-curriculum",
        dest="training_curriculum",
        action="store_true",
        help=(
            "Advance the five-stage B01/B02 target-route curriculum from "
            "shallow targets 01/04 to all twelve routes. Promotion requires "
            "every active route to reach its rolling success threshold."
        ),
    )
    curriculum.add_argument(
        "--no-training-curriculum",
        dest="training_curriculum",
        action="store_false",
    )
    parser.set_defaults(training_curriculum=TRAINING_CURRICULUM_ENABLED)

    parser.add_argument("--log-root", default=str(TRAINING_RUNS_DIR))
    parser.add_argument("--variant", default="base")
    parser.add_argument("--exp-name", default="")
    parser.add_argument("--render", choices=["headless", "human"], default="headless")
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--reset-num-timesteps", action="store_true")
    parser.add_argument("--valid-dir", default=str(VALID_MESH_DIR))
    parser.add_argument(
        "--valid-min-train-success-rate",
        type=float,
        default=VALID_MIN_TRAIN_SUCCESS_RATE,
    )
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--sb3-verbose", type=int, choices=[0, 1, 2], default=1)
    parser.add_argument("--scene-verbose", action="store_true")
    if configure_parser is not None:
        configure_parser(parser)
    args = parser.parse_args()

    if args.epochs <= 0 or args.episodes_per_epoch <= 0:
        parser.error("--epochs and --episodes-per-epoch must be positive")
    if args.n_steps <= 0 or args.batch_size <= 0 or args.n_epochs <= 0:
        parser.error("PPO rollout/update sizes must be positive")
    if args.max_episode_steps <= 0:
        parser.error("--max-episode-steps must be positive")
    if not (0.0 <= args.valid_min_train_success_rate <= 1.0):
        parser.error("--valid-min-train-success-rate must be in [0, 1]")
    if not (0.5 <= args.vessel_scale_min <= args.vessel_scale_max <= 1.0):
        parser.error("vessel scale bounds must satisfy 0.5 <= min <= max <= 1.0")
    if not (0.0 < args.min_action_std <= args.max_action_std):
        parser.error("PPO action std bounds must satisfy 0 < min <= max")
    if not (args.min_action_std <= args.initial_action_std <= args.max_action_std):
        parser.error("--initial-action-std must lie within the action std bounds")
    args.steps_per_epoch = args.episodes_per_epoch * args.max_episode_steps
    args.episode_mode = True
    return args


def _override_learning_rate(model: PPO, learning_rate: float) -> None:
    model.learning_rate = float(learning_rate)
    model.lr_schedule = get_schedule_fn(float(learning_rate))
    for group in model.policy.optimizer.param_groups:
        group["lr"] = float(learning_rate)


def main():
    args = parse_args()
    args.reward_profile = reward_profile(args.gamma)
    args.curriculum_protocol = curriculum_protocol_profile()
    args.policy_net_arch = {"pi": [512, 512, 256], "vf": [512, 512, 256]}
    args.policy_activation = "SiLU"
    os.environ["MCR_SOFA_DT"] = str(float(args.time_step))
    context = initialize_distributed(
        enabled=args.distributed,
        requested_device=args.device,
        requested_world_size=args.world_size,
        cli_local_rank=args.local_rank,
        requested_backend=args.dist_backend,
    )
    args.npu_execution = configure_npu_execution(
        context.device.accelerator,
        enabled=bool(args.npu_fast_execution),
    )
    args.distributed_rank = context.rank
    args.resolved_device = context.device.resolved
    total_n_envs = int(args.n_envs)
    global_batch_size = int(args.batch_size)
    if context.enabled:
        if total_n_envs % context.world_size:
            raise ValueError("--n-envs must be divisible by world size")
        if global_batch_size % context.world_size:
            raise ValueError("--batch-size must be divisible by world size")
        args.local_n_envs = total_n_envs // context.world_size
        local_batch_size = global_batch_size // context.world_size
    else:
        args.local_n_envs = total_n_envs
        local_batch_size = global_batch_size
    local_rollout_size = int(args.n_steps) * int(args.local_n_envs)
    if local_rollout_size % local_batch_size:
        raise ValueError(
            f"local rollout size {local_rollout_size} must be divisible by local batch {local_batch_size}"
        )
    args.rank_seed = int(args.seed) + context.rank * args.local_n_envs
    local_total_timesteps = int(
        math.ceil(args.epochs * args.episodes_per_epoch * args.max_episode_steps / context.world_size)
    )

    valid_dir = Path(args.valid_dir).expanduser()
    if not valid_dir.is_absolute():
        valid_dir = PROJECT_ROOT / valid_dir
    valid_dir = valid_dir.resolve()
    valid_vessels = [] if args.skip_validation else discover_validation_vessels(
        valid_dir, expected_vessels=VALID_VESSELS
    )

    timestamp = os.environ.get("MCR_RUN_TIMESTAMP", "").strip()
    if not timestamp and context.is_main:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp = context.broadcast_text(timestamp)
    if not args.exp_name:
        device_tag = f"{context.world_size}npu" if context.enabled else "1device"
        args.exp_name = f"ppo_{args.variant}_{total_n_envs}env_{device_tag}_ep{args.episodes_per_epoch}"
    log_root = Path(args.log_root).expanduser()
    if not log_root.is_absolute():
        log_root = PROJECT_ROOT / log_root
    run_dir = log_root.resolve() / f"{args.exp_name}_{timestamp}"
    model_dir, tb_dir, log_dir = run_dir / "models", run_dir / "tb", run_dir / "logs"
    if context.is_main:
        model_dir.mkdir(parents=True, exist_ok=True)
        tb_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
    context.barrier()

    run_log = start_run_log_capture(log_dir, context.rank)
    if context.is_main:
        write_run_config(
            log_dir / "run_config.json",
            args,
            algorithm="ppo",
            run_dir=run_dir,
            model_dir=model_dir,
            tensorboard_dir=tb_dir,
        )

    env = None
    try:
        env = build_env(args)
        observation_shape = getattr(env.observation_space, "shape", None)
        args.observation_space_shape = (
            list(observation_shape) if observation_shape is not None else None
        )
        args.observation_space_dtype = str(
            getattr(env.observation_space, "dtype", "unknown")
        )

        def run_validation(current_model, epoch):
            def env_factory(vessel_id):
                valid_args = copy.copy(args)
                valid_args.force_model = str(vessel_id)
                valid_args.asset_root = str(valid_dir)
                valid_args.local_n_envs = 1
                valid_args.distributed_rank = 0
                valid_args.render = "headless"
                valid_args.seed = int(args.seed) + 100_000
                valid_args.training_curriculum = False
                return build_env(valid_args)

            was_training = bool(current_model.policy.training)
            try:
                return evaluate_policy(
                    valid_vessels,
                    env_factory,
                    lambda observation: current_model.predict(observation, deterministic=True)[0],
                    episodes_per_vessel=VALID_EPISODES_PER_VESSEL,
                    max_episode_steps=args.max_episode_steps,
                    base_seed=int(args.seed) + 100_000,
                    task_rank=context.rank,
                    task_world_size=context.world_size,
                )
            finally:
                current_model.policy.set_training_mode(was_training)

        epoch_callback = EpochExperimentCallback(
            context=context,
            algorithm_name="ppo",
            variant=args.variant,
            epochs=args.epochs,
            episodes_per_epoch=args.episodes_per_epoch,
            model_dir=model_dir,
            run_dir=log_dir,
            validation_interval=VALID_INTERVAL,
            validation_min_train_success_rate=args.valid_min_train_success_rate,
            validation_fn=None if args.skip_validation else run_validation,
            resume_progress=not args.reset_num_timesteps,
            training_curriculum_enabled=args.training_curriculum,
        )
        callbacks = [epoch_callback]
        if context.is_main:
            callbacks.extend(
                [
                    ExtraRolloutMetricsCallback(window_size=50, success_label="target"),
                    DistributedRuntimeCallback(
                        context.world_size,
                        total_n_envs,
                        global_batch_size,
                        local_batch_size,
                        args.steps_per_epoch,
                        args.episodes_per_epoch,
                        epoch_callback,
                    ),
                ]
            )
        callback = CallbackList(callbacks) if len(callbacks) > 1 else callbacks[0]
        # Use the shared implementation on one or many devices so exploration
        # bounds and optimizer semantics remain identical.
        algorithm_class = DistributedPPO
        tensorboard_log = str(tb_dir) if context.is_main else None
        verbose = args.sb3_verbose if context.is_main else 0

        if args.resume_from:
            resume_path = Path(args.resume_from).expanduser()
            if not resume_path.is_absolute():
                resume_path = PROJECT_ROOT / resume_path
            resume_path = resume_path.resolve()
            if not resume_path.is_file():
                raise FileNotFoundError(f"Resume model not found: {resume_path}")
            model = algorithm_class.load(
                str(resume_path),
                env=env,
                device=args.resolved_device,
                custom_objects={
                    "n_steps": args.n_steps,
                    "batch_size": local_batch_size,
                    "n_epochs": args.n_epochs,
                    "gamma": args.gamma,
                    "gae_lambda": args.gae_lambda,
                    "clip_range": get_schedule_fn(args.clip_range),
                    "ent_coef": args.ent_coef,
                    "vf_coef": args.vf_coef,
                    "max_grad_norm": args.max_grad_norm,
                },
            )
            model.tensorboard_log = tensorboard_log
            model.verbose = verbose
            model.seed = int(args.rank_seed)
            model.set_random_seed(int(args.rank_seed))
            model.min_action_std = float(args.min_action_std)
            model.max_action_std = float(args.max_action_std)
            model.clamp_action_std()
            _override_learning_rate(model, args.learning_rate)
            reset_num_timesteps = args.reset_num_timesteps
            saved_world_size = max(
                1, int(getattr(model, "distributed_world_size_at_save", 1))
            )
            if not reset_num_timesteps and saved_world_size != context.world_size:
                completed_global_steps = int(model.num_timesteps) * saved_world_size
                model.num_timesteps = int(
                    math.ceil(completed_global_steps / context.world_size)
                )
        else:
            policy_kwargs = {
                "log_std_init": math.log(float(args.initial_action_std)),
                # The old SB3 64x64 default was undersized for the long-history
                # multi-branch controller.  Separate high-capacity actor/value
                # trunks avoid forcing incompatible route modes through a tiny
                # shared bottleneck.
                "net_arch": dict(pi=[512, 512, 256], vf=[512, 512, 256]),
                "activation_fn": th.nn.SiLU,
            }
            kwargs = dict(
                policy="MlpPolicy",
                env=env,
                learning_rate=args.learning_rate,
                n_steps=args.n_steps,
                batch_size=local_batch_size,
                n_epochs=args.n_epochs,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                clip_range=args.clip_range,
                ent_coef=args.ent_coef,
                vf_coef=args.vf_coef,
                max_grad_norm=args.max_grad_norm,
                policy_kwargs=policy_kwargs,
                tensorboard_log=tensorboard_log,
                seed=args.rank_seed,
                device=args.resolved_device,
                verbose=verbose,
                distributed_context=context,
                min_action_std=args.min_action_std,
                max_action_std=args.max_action_std,
            )
            model = algorithm_class(**kwargs)
            reset_num_timesteps = True

        fused_status = "not_requested"
        if args.npu_fused_adam and context.device.accelerator == "npu":
            original_optimizer = model.policy.optimizer
            fused_optimizer, fused_status = convert_to_npu_fused_adam(
                original_optimizer
            )
            local_fused = fused_status in ("enabled", "already_enabled")
            globally_fused = context.average_metrics([float(local_fused)])[0] == 1.0
            if globally_fused:
                model.policy.optimizer = fused_optimizer
            else:
                model.policy.optimizer = original_optimizer
                fused_status = f"{fused_status};global_fallback"
        args.npu_fused_adam_status = fused_status
        args.npu_fused_adam_enabled = fused_status in ("enabled", "already_enabled")
        if context.is_main:
            write_run_config(
                log_dir / "run_config.json",
                args,
                algorithm="ppo",
                run_dir=run_dir,
                model_dir=model_dir,
                tensorboard_dir=tb_dir,
            )
            print(
                f"[MCR NPU][PPO] execution={args.npu_execution} "
                f"fused_adam={args.npu_fused_adam_status}"
            )
        model.distributed_world_size_at_save = context.world_size
        # ``DistributedPPO`` is also used for the single-device path so that
        # optimizer/observation semantics stay identical.  The context is a
        # no-op when distributed execution is disabled, but it must still be
        # attached after loading a checkpoint: ``distributed_context`` is an
        # intentionally excluded save parameter and otherwise resumed
        # single-device runs fail on the first optimizer update.
        model.set_distributed_context(context)
        if context.enabled:
            model.synchronize_parameters()

        if context.is_main:
            print(
                f"[MCR TRAIN][PPO] device={context.device.resolved} distributed={context.enabled} "
                f"world={context.world_size} envs={total_n_envs} epochs={args.epochs} "
                f"episodes_per_epoch={args.episodes_per_epoch} max_episode_steps={args.max_episode_steps}"
            )
            print(
                f"[MCR TRAIN][PPO] n_steps={args.n_steps} n_epochs={args.n_epochs} "
                f"batch={global_batch_size}/{local_batch_size} "
                f"ent_coef={args.ent_coef:g} "
                f"action_std={args.min_action_std:g}-{args.max_action_std:g} "
                f"output={run_dir}"
            )
        model.learn(
            total_timesteps=local_total_timesteps,
            callback=callback,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=bool(args.progress_bar and context.is_main),
        )
        context.barrier()
        final_path = model_dir / f"ppo_{args.variant}_final_epoch_{args.epochs:03d}"
        if context.is_main:
            model.save(str(final_path))
            print(f"[DONE] model={final_path}.zip")
        context.barrier()
    finally:
        if env is not None:
            env.close()
        context.close()


if __name__ == "__main__":
    main()
