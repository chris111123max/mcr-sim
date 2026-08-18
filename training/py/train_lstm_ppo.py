"""Four-device Ascend LSTM-PPO using the shared MCR experiment protocol."""

from __future__ import annotations

import copy
import math
import os
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path

TRAINING_PY_DIR = Path(__file__).resolve().parent
PYTHON_ROOT = TRAINING_PY_DIR.parents[1]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

import numpy as np
import stable_baselines3
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.utils import get_schedule_fn

from mcr_sim.distributed import (
    configure_npu_execution,
    convert_to_npu_fused_adam,
    initialize_distributed,
)
try:
    from mcr_sim.distributed.recurrent_ppo import DistributedRecurrentPPO
except ModuleNotFoundError as exc:
    if exc.name and exc.name.startswith("sb3_contrib"):
        raise RuntimeError(
            "LSTM-PPO requires the server-compatible sb3-contrib 2.4.x package."
        ) from exc
    raise
from mcr_sim.paths import PROJECT_ROOT
from mcr_sim.rl_core.evaluation import discover_validation_vessels, evaluate_policy
from mcr_sim.rl_core.experiment import EpochExperimentCallback
from mcr_sim.rl_core.run_logging import start_run_log_capture, write_run_config
from mcr_sim.training_config import (
    VALID_EPISODES_PER_VESSEL,
    VALID_INTERVAL,
    VALID_VESSELS,
    reward_profile,
)

# Reuse the exact MLP-PPO common argument parser, environment construction,
# callbacks, and learning-rate resume semantics.  Structural LSTM arguments are
# parsed separately below and are the only formal-experiment additions.
from train_ppo import (  # noqa: E402
    DistributedRuntimeCallback,
    ExtraRolloutMetricsCallback,
    _override_learning_rate,
    build_env,
    parse_args as parse_mlp_ppo_args,
)


LSTM_HIDDEN_SIZE_DEFAULT = 128
LSTM_NUM_LAYERS_DEFAULT = 1
EXPECTED_SB3_MINOR = (2, 4)
EXPECTED_SB3_CONTRIB_MINOR = (2, 4)


def _major_minor(version_text: str):
    parts = str(version_text).split("+", 1)[0].split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError) as exc:
        raise RuntimeError(f"Cannot parse package version {version_text!r}.") from exc


def _dependency_versions():
    try:
        contrib_version = metadata.version("sb3-contrib")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "sb3-contrib is not installed. This implementation targets "
            "sb3-contrib==2.4.0 paired with stable-baselines3==2.4.0."
        ) from exc
    sb3_version = str(stable_baselines3.__version__)
    if _major_minor(sb3_version) != EXPECTED_SB3_MINOR:
        raise RuntimeError(
            f"Incompatible stable-baselines3={sb3_version}; expected 2.4.x."
        )
    if _major_minor(contrib_version) != EXPECTED_SB3_CONTRIB_MINOR:
        raise RuntimeError(
            f"Incompatible sb3-contrib={contrib_version}; expected 2.4.x."
        )
    return sb3_version, contrib_version


def _configure_recurrent_parser(parser) -> None:
    parser.add_argument(
        "--lstm-hidden-size",
        type=int,
        default=LSTM_HIDDEN_SIZE_DEFAULT,
    )
    parser.add_argument(
        "--lstm-num-layers",
        type=int,
        default=LSTM_NUM_LAYERS_DEFAULT,
    )


def parse_args():
    """Reuse every MLP-PPO common option and add only LSTM structure."""

    args = parse_mlp_ppo_args(configure_parser=_configure_recurrent_parser)
    if args.lstm_hidden_size <= 0:
        raise ValueError("--lstm-hidden-size must be positive")
    if args.lstm_num_layers <= 0:
        raise ValueError("--lstm-num-layers must be positive")
    args.lstm_hidden_size = int(args.lstm_hidden_size)
    args.lstm_num_layers = int(args.lstm_num_layers)
    args.lstm_bidirectional = False
    args.policy_type = "MlpLstmPolicy"
    return args


class RecurrentValidationPredictor:
    """Maintain one actor LSTM state for a single validation environment."""

    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self) -> None:
        self.lstm_state = None
        self.episode_start = np.ones((1,), dtype=bool)

    def __call__(self, observation):
        action, self.lstm_state = self.model.predict(
            observation,
            state=self.lstm_state,
            episode_start=self.episode_start,
            deterministic=True,
        )
        self.episode_start.fill(False)
        return action


def _assert_loaded_lstm_architecture(model, hidden_size: int, num_layers: int):
    actor_lstm = model.policy.lstm_actor
    critic_lstm = model.policy.lstm_critic
    if actor_lstm is None or critic_lstm is None:
        raise RuntimeError("Formal LSTM-PPO requires separate actor and critic LSTMs.")
    actual = (
        int(actor_lstm.hidden_size),
        int(actor_lstm.num_layers),
        bool(actor_lstm.bidirectional),
        int(critic_lstm.hidden_size),
        int(critic_lstm.num_layers),
        bool(critic_lstm.bidirectional),
    )
    expected = (
        int(hidden_size),
        int(num_layers),
        False,
        int(hidden_size),
        int(num_layers),
        False,
    )
    if actual != expected:
        raise ValueError(
            "Loaded recurrent checkpoint architecture does not match CLI: "
            f"actual={actual} expected={expected}."
        )


def main():
    args = parse_args()
    sb3_version, sb3_contrib_version = _dependency_versions()
    args.stable_baselines3_version = sb3_version
    args.sb3_contrib_version = sb3_contrib_version
    args.reward_profile = reward_profile()
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
            f"local rollout size {local_rollout_size} must be divisible by "
            f"local batch {local_batch_size}"
        )
    args.global_batch_size = global_batch_size
    args.local_batch_size = local_batch_size
    args.rank_seed = int(args.seed) + context.rank * args.local_n_envs
    local_total_timesteps = int(
        math.ceil(
            args.epochs
            * args.episodes_per_epoch
            * args.max_episode_steps
            / context.world_size
        )
    )

    valid_dir = Path(args.valid_dir).expanduser()
    if not valid_dir.is_absolute():
        valid_dir = PROJECT_ROOT / valid_dir
    valid_dir = valid_dir.resolve()
    valid_vessels = (
        []
        if args.skip_validation
        else discover_validation_vessels(
            valid_dir,
            expected_vessels=VALID_VESSELS,
        )
    )

    timestamp = os.environ.get("MCR_RUN_TIMESTAMP", "").strip()
    if not timestamp and context.is_main:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp = context.broadcast_text(timestamp)
    if not args.exp_name:
        device_tag = f"{context.world_size}npu" if context.enabled else "1device"
        args.exp_name = (
            f"lstm_ppo_{args.variant}_{total_n_envs}env_{device_tag}_"
            f"ep{args.episodes_per_epoch}"
        )
    log_root = Path(args.log_root).expanduser()
    if not log_root.is_absolute():
        log_root = PROJECT_ROOT / log_root
    run_dir = log_root.resolve() / f"{args.exp_name}_{timestamp}"
    model_dir, tb_dir, log_dir = (
        run_dir / "models",
        run_dir / "tb",
        run_dir / "logs",
    )
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
            algorithm="lstm_ppo",
            run_dir=run_dir,
            model_dir=model_dir,
            tensorboard_dir=tb_dir,
        )

    env = None
    try:
        env = build_env(args)

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
            predictor = RecurrentValidationPredictor(current_model)
            try:
                return evaluate_policy(
                    valid_vessels,
                    env_factory,
                    predictor,
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
            algorithm_name="lstm_ppo",
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
                    ExtraRolloutMetricsCallback(
                        window_size=50,
                        success_label="target",
                    ),
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
        tensorboard_log = str(tb_dir) if context.is_main else None
        verbose = args.sb3_verbose if context.is_main else 0

        if args.resume_from:
            resume_path = Path(args.resume_from).expanduser()
            if not resume_path.is_absolute():
                resume_path = PROJECT_ROOT / resume_path
            resume_path = resume_path.resolve()
            if not resume_path.is_file():
                raise FileNotFoundError(f"Resume model not found: {resume_path}")
            model = DistributedRecurrentPPO.load(
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
            _assert_loaded_lstm_architecture(
                model,
                args.lstm_hidden_size,
                args.lstm_num_layers,
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
                1,
                int(getattr(model, "distributed_world_size_at_save", 1)),
            )
            if not reset_num_timesteps and saved_world_size != context.world_size:
                completed_global_steps = int(model.num_timesteps) * saved_world_size
                model.num_timesteps = int(
                    math.ceil(completed_global_steps / context.world_size)
                )
        else:
            policy_kwargs = {
                "lstm_hidden_size": args.lstm_hidden_size,
                "n_lstm_layers": args.lstm_num_layers,
                "shared_lstm": False,
                "enable_critic_lstm": True,
                "lstm_kwargs": {"bidirectional": False},
            }
            model = DistributedRecurrentPPO(
                policy="MlpLstmPolicy",
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
            reset_num_timesteps = True

        recurrent_parameter_names = model.assert_recurrent_parameters_registered()
        fused_status = "not_requested"
        if args.npu_fused_adam and context.device.accelerator == "npu":
            original_optimizer = model.policy.optimizer
            fused_optimizer, fused_status = convert_to_npu_fused_adam(
                original_optimizer
            )
            local_fused = fused_status in ("enabled", "already_enabled")
            globally_fused = (
                context.average_metrics([float(local_fused)])[0] == 1.0
            )
            if globally_fused:
                model.policy.optimizer = fused_optimizer
            else:
                model.policy.optimizer = original_optimizer
                fused_status = f"{fused_status};global_fallback"
        args.npu_fused_adam_status = fused_status
        args.npu_fused_adam_enabled = fused_status in (
            "enabled",
            "already_enabled",
        )
        args.recurrent_parameter_names = recurrent_parameter_names
        if context.is_main:
            write_run_config(
                log_dir / "run_config.json",
                args,
                algorithm="lstm_ppo",
                run_dir=run_dir,
                model_dir=model_dir,
                tensorboard_dir=tb_dir,
            )
            print(
                f"[MCR NPU][LSTM-PPO] execution={args.npu_execution} "
                f"fused_adam={args.npu_fused_adam_status}"
            )
        model.distributed_world_size_at_save = context.world_size
        model.set_distributed_context(context)
        if context.enabled:
            model.synchronize_parameters()

        if context.is_main:
            state_shape = (
                args.lstm_num_layers,
                args.local_n_envs,
                args.lstm_hidden_size,
            )
            print(
                f"[MCR TRAIN][LSTM-PPO] device={context.device.resolved} "
                f"distributed={context.enabled} world={context.world_size} "
                f"envs={total_n_envs} epochs={args.epochs} "
                f"episodes_per_epoch={args.episodes_per_epoch} "
                f"max_episode_steps={args.max_episode_steps}"
            )
            print(
                f"[MCR TRAIN][LSTM-PPO] n_steps={args.n_steps} "
                f"n_epochs={args.n_epochs} "
                f"batch={global_batch_size}/{local_batch_size} "
                f"ent_coef={args.ent_coef:g} "
                f"action_std={args.min_action_std:g}-{args.max_action_std:g} "
                f"lstm_state_shape={state_shape} actor_and_critic=True "
                f"recurrent_parameters={len(recurrent_parameter_names)} "
                f"output={run_dir}"
            )
        model.learn(
            total_timesteps=local_total_timesteps,
            callback=callback,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=bool(args.progress_bar and context.is_main),
        )
        context.barrier()
        final_path = (
            model_dir
            / f"lstm_ppo_{args.variant}_final_epoch_{args.epochs:03d}"
        )
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
