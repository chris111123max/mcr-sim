"""PPO entry for production Fast-FD Beam/SDF unilateral safety.

Legacy training/py/train_ppo.py is intentionally unchanged. This entry uses
identical PPO/training semantics but swaps MCREnv for BeamSafetyMCREnv and
forces GenericConstraintSolver before SOFA workers are spawned.
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

TRAINING_PY_DIR = Path(__file__).resolve().parent
PYTHON_ROOT = TRAINING_PY_DIR.parents[1]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import train_ppo as baseline
import train_sac as shared
from mcr_sim.beam_safety_env import BeamSafetyMCREnv
from mcr_sim.beam_sdf_unilateral_fast import (
    ACTIVE_BAND_M,
    MARGIN_M,
    resolve_plugin_path,
)
from mcr_sim.rl_core.beam_safety_metrics import BeamSafetyMetricsCollector


_BASE_PARSE_ARGS = baseline.parse_args
_BASE_ROLLOUT_METRICS_CALLBACK = baseline.ExtraRolloutMetricsCallback


class BeamSafetyRolloutMetricsCallback(_BASE_ROLLOUT_METRICS_CALLBACK):
    """Legacy PPO rollout metrics plus read-only Beam safety telemetry."""

    def __init__(
        self,
        window_size: int = 50,
        success_label: str = "target",
        verbose: int = 0,
    ):
        super().__init__(
            window_size=window_size,
            success_label=success_label,
            verbose=verbose,
        )
        self._beam_metrics = BeamSafetyMetricsCollector(
            episode_window=window_size
        )

    def _on_step(self) -> bool:
        keep_training = super()._on_step()
        self._beam_metrics.observe(
            self.locals.get("infos"),
            self.locals.get("dones"),
        )
        return bool(keep_training)

    def _on_rollout_end(self) -> None:
        super()._on_rollout_end()
        self._beam_metrics.log_rollout(self.logger)


def _configure_beam_parser(parser) -> None:
    parser.add_argument(
        "--beam-safety-validation",
        choices=["audit", "debug", "off"],
        default="audit",
        help=(
            "Independent committed 10um verifier mode. audit keeps physical "
            "Beam constraints on every substep but validates committed states "
            "on active/near-wall/periodic/reset/terminal events; debug validates "
            "every substep; off disables only the independent verifier."
        ),
    )
    parser.add_argument(
        "--beam-safety-audit-interval-substeps",
        type=int,
        default=16,
        help="Periodic committed audit interval for far-safe substeps.",
    )
    parser.add_argument(
        "--beam-safety-audit-near-clearance-mm",
        type=float,
        default=(MARGIN_M + ACTIVE_BAND_M) * 1000.0,
        help=(
            "In audit mode, committed validation is forced whenever q_free "
            "clearance is at or below this threshold. Default 0.300 mm."
        ),
    )
    parser.add_argument(
        "--beam-safety-plugin",
        default="",
        help=(
            "Optional explicit libMCRBeamLinearizedUnilateral.so path. Empty "
            "uses mcr_sim/native/beam_unilateral/_build."
        ),
    )


def parse_args():
    args = _BASE_PARSE_ARGS(_configure_beam_parser)
    if int(args.physics_substeps) != 2:
        raise ValueError(
            "Beam-unilateral training is validated only for exactly 2 physics substeps."
        )
    if int(args.frame_skip) != 1:
        raise ValueError(
            "Beam-unilateral training requires frame_skip=1 with 2 physics substeps."
        )
    if int(args.beam_safety_audit_interval_substeps) < 1:
        raise ValueError("--beam-safety-audit-interval-substeps must be >= 1")
    near_m = float(args.beam_safety_audit_near_clearance_mm) / 1000.0
    if not np.isfinite(near_m) or near_m < MARGIN_M:
        raise ValueError(
            "--beam-safety-audit-near-clearance-mm must be >= 0.100 mm"
        )
    args.beam_safety_audit_near_clearance_m = near_m
    args.beam_safety_plugin = str(
        resolve_plugin_path(args.beam_safety_plugin or None)
    )
    if str(args.variant) == "base":
        args.variant = f"beam_unilateral_{args.beam_safety_validation}"
    return args


def build_env(args):
    """Match the established train_sac/train_ppo VecEnv construction."""
    env_type = (
        shared.EnvType.AORTIC
        if args.env_type == "aortic"
        else shared.EnvType.FLAT
    )
    render_mode = (
        shared.RenderMode.HUMAN
        if args.render == "human"
        else shared.RenderMode.NONE
    )
    n_envs = max(1, int(getattr(args, "local_n_envs", args.n_envs)))
    global_env_offset = int(getattr(args, "distributed_rank", 0)) * n_envs

    if n_envs > 1 and args.render == "human":
        raise ValueError(
            "Parallel SOFA environments require --render headless."
        )

    def _make(rank: int = 0):
        def _init():
            try:
                env_seed = int(args.seed) + global_env_offset + int(rank)
                np.random.seed(env_seed)
                random.seed(env_seed)
            except Exception:
                pass

            create_scene_kwargs = {
                "radius_observation_scale": float(args.radius_observation_scale),
                "actor_history_steps": shared.ACTOR_HISTORY_STEPS,
                "reward_discount_gamma": float(args.gamma),
                "randomize_start_target": bool(args.randomize_start_target),
                "start_window_distance_m": float(args.start_window_mm) / 1000.0,
                "target_window_distance_m": float(args.target_window_mm) / 1000.0,
                "randomize_initial_orientation": bool(
                    args.randomize_initial_orientation
                ),
                "initial_orientation_max_angle_deg": float(
                    args.initial_orientation_max_angle_deg
                ),
                "entry_tangent_points": int(args.entry_tangent_points),
                "soft_randomize_single_vessel": bool(
                    args.soft_randomize_single_vessel
                ),
                "vessel_scale_min": float(args.vessel_scale_min),
                "vessel_scale_max": float(args.vessel_scale_max),
                "training_curriculum_enabled": bool(
                    getattr(
                        args,
                        "training_curriculum",
                        shared.TRAINING_CURRICULUM_ENABLED,
                    )
                ),
                "verbose_scene": bool(args.scene_verbose),
                "sampling_slot": global_env_offset + int(rank),
            }
            if args.render == "human":
                create_scene_kwargs["debug_rendering"] = True
                create_scene_kwargs["positioning_camera"] = True
                create_scene_kwargs["vessel_alpha"] = 0.8
            else:
                create_scene_kwargs["debug_rendering"] = False
                create_scene_kwargs["positioning_camera"] = False
            if args.force_model:
                create_scene_kwargs["force_model"] = args.force_model
            if getattr(args, "asset_root", ""):
                create_scene_kwargs["asset_root"] = str(args.asset_root)

            env = BeamSafetyMCREnv(
                env_type=env_type,
                observation_type=shared.ObservationType.STATE,
                action_type=shared.ActionType.CONTINUOUS,
                time_step=args.time_step,
                frame_skip=args.frame_skip,
                physics_substeps=args.physics_substeps,
                settle_steps=args.settle_steps,
                render_mode=render_mode,
                render_framework=shared.RenderFramework.PYGLET,
                target_distance_threshold=args.target_threshold,
                max_episode_steps=args.max_episode_steps,
                create_scene_kwargs=create_scene_kwargs,
                beam_safety_plugin=args.beam_safety_plugin,
                beam_safety_validation_mode=args.beam_safety_validation,
                beam_safety_audit_interval_substeps=(
                    args.beam_safety_audit_interval_substeps
                ),
                beam_safety_audit_near_clearance_m=(
                    args.beam_safety_audit_near_clearance_m
                ),
            )
            return Monitor(env)

        return _init

    if n_envs == 1:
        return DummyVecEnv([_make(0)])
    return SubprocVecEnv(
        [_make(i) for i in range(n_envs)],
        start_method="spawn",
    )


def main() -> None:
    # This must be set in the parent before any spawn worker creates SOFA.
    os.environ["MCR_CONSTRAINT_SOLVER"] = "generic"

    # Reuse the established PPO implementation without editing the legacy
    # baseline. Its validation closure also resolves this patched build_env.
    # Only the Beam-specific entry point swaps in a telemetry-extended rollout
    # callback; observation/reward/action/done and PPO optimization remain the
    # legacy baseline implementation.
    baseline.parse_args = parse_args
    baseline.build_env = build_env
    baseline.ExtraRolloutMetricsCallback = BeamSafetyRolloutMetricsCallback
    baseline.main()


if __name__ == "__main__":
    main()
