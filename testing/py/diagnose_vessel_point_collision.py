"""Short V15.2-C vessel PointCollisionModel diagnostic.

This script is intentionally diagnostic-only:
- production scene/training code is not modified;
- the process monkey-patches the vessel Environment class so the vessel uses
  TriangleCollisionModel + PointCollisionModel only (no vessel LineCollisionModel);
- the policy/physics configuration stays at V15.2-C (1 RL action = 2 x 5 ms);
- only the target action is sampled in detail after replaying the deterministic
  prefix needed to reach that state.

The known V15.2-C step-1186 baseline is kept in the output for direct comparison:
first substep contact=0, constraint rows=0, contact-free body penetration=0.3551 mm.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

PYTHON_ROOT = Path(__file__).resolve().parents[2]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from mcr_sim.distributed import DistributedPPO
from mcr_sim.mcr_rl_env import EnvType, MCREnv
from mcr_sim.rl_core.base import RenderMode
from mcr_sim import mcr_environment as mcr_environment_module


BASELINE_REFERENCE = {
    "configuration": "V15.2-C LocalMinDistance, vessel Triangle only",
    "target_step": 1186,
    "substep_contacts": [0, 1],
    "substep_constraint_rows": [0, 6],
    "max_contact_free_body_penetration_mm": 0.3551,
    "source_note": (
        "Previously reproduced deterministic V15.2-C step-1186 diagnostic. "
        "This script does not rerun baseline unless the user does so separately."
    ),
}


def _as_array(data):
    try:
        return np.asarray(data.array(), dtype=np.float64).copy()
    except Exception:
        return np.asarray(data.value, dtype=np.float64).copy()


def _count_constraint_rows(dofs) -> int:
    try:
        value = dofs.constraint.value
    except Exception:
        return -1
    return len([line for line in str(value).splitlines() if line.strip()])


def _default_output(checkpoint: Path, seed: int, target_step: int) -> Path:
    for parent in checkpoint.parents:
        if parent.parent.name == "training_runs" or parent.name.startswith("ppo_"):
            diagnostics = parent / "diagnostics"
            return diagnostics / (
                f"v15_2c_vessel_point_step{target_step}_seed{seed}.json"
            )
    return (
        Path.cwd()
        / "diagnostics"
        / f"v15_2c_vessel_point_step{target_step}_seed{seed}.json"
    )


def _install_point_only_vessel_environment() -> type:
    """Monkey-patch scene construction for this process only.

    The production Environment class remains untouched on disk.  The wrapper
    forces the legacy combined vessel Line/Point switch off, lets the normal
    TriangleCollisionModel be created, then adds only PointCollisionModel.
    """

    original = mcr_environment_module.Environment

    class PointOnlyVesselEnvironment(original):
        def __init__(self, *args, **kwargs):
            kwargs = dict(kwargs)
            kwargs["use_line_point_collision"] = False
            super().__init__(*args, **kwargs)
            self.CollisionModel.addObject(
                "PointCollisionModel",
                name="DiagnosticVesselPointCollision",
                moving=False,
                simulated=False,
                proximity=float(self.line_point_collision_proximity),
            )

    mcr_environment_module.Environment = PointOnlyVesselEnvironment
    return original


def _collision_model_summary(env: MCREnv) -> dict:
    vessel = env.scene_creation_result["mcr_environment"].CollisionModel
    catheter = env.mcr_controller_sofa.instrument.InstrumentCombined.getChild(
        "mcr_collis"
    )

    def classes(node):
        return [obj.getClassName() for obj in node.objects]

    vessel_classes = classes(vessel)
    catheter_classes = classes(catheter)
    return {
        "vessel_classes": vessel_classes,
        "catheter_classes": catheter_classes,
        "vessel_triangle_count": vessel_classes.count("TriangleCollisionModel"),
        "vessel_point_count": vessel_classes.count("PointCollisionModel"),
        "vessel_line_count": vessel_classes.count("LineCollisionModel"),
        "catheter_point_count": catheter_classes.count("PointCollisionModel"),
        "catheter_line_count": catheter_classes.count("LineCollisionModel"),
    }


def _capture(env: MCREnv, label: str) -> dict:
    controller = env.mcr_controller_sofa
    instrument = controller.instrument.InstrumentCombined
    collision_node = instrument.getChild("mcr_collis")
    collision_dofs = collision_node.getObject("CollisionDOFs")
    beam_dofs = instrument.getObject("DOFs")

    tip = np.asarray(
        controller.get_pos_quat_catheter_tip()[:3], dtype=np.float64
    ).copy()
    env._sdf_geometry_cache_step = -1
    env._update_sdf_safety_state(tip, advance_failure_counters=False)

    beam_pos = _as_array(beam_dofs.position)
    beam_free = _as_array(beam_dofs.free_position)
    collision_pos = _as_array(collision_dofs.position)
    collision_free = _as_array(collision_dofs.free_position)
    contacts = [
        obj
        for obj in collision_node.objects
        if obj.getClassName() == "FrictionContact"
    ]

    tip_clearance_mm = float(env.current_sdf_surface_clearance * 1000.0)
    body_clearance_mm = float(env.current_sdf_body_min_surface_clearance * 1000.0)

    return {
        "label": label,
        "root_time_s": float(env._sofa_root_node.getTime()),
        "root_dt_s": float(env._sofa_root_node.getDt()),
        "tip_mm": (tip * 1000.0).tolist(),
        "tip_surface_clearance_mm": tip_clearance_mm,
        "body_min_surface_clearance_mm": body_clearance_mm,
        "body_penetration_mm": max(0.0, -body_clearance_mm),
        "friction_contact_count": len(contacts),
        "collision_constraint_rows": _count_constraint_rows(collision_dofs),
        "beam_constraint_rows": _count_constraint_rows(beam_dofs),
        "beam_tip_correction_mm": float(
            np.linalg.norm((beam_pos[-1, :3] - beam_free[-1, :3]) * 1000.0)
        ),
        "beam_max_correction_mm": float(
            np.max(
                np.linalg.norm(
                    (beam_pos[:, :3] - beam_free[:, :3]) * 1000.0,
                    axis=1,
                )
            )
        ),
        "collision_tip_correction_mm": float(
            np.linalg.norm(
                (collision_pos[-1, :3] - collision_free[-1, :3]) * 1000.0
            )
        ),
        "xtip_m": float(controller._getXTipValue()),
        "pending_insert_m": float(controller.pending_insert_delta),
    }


def _prepare_smoothed_action(env: MCREnv, raw_action) -> np.ndarray:
    action = np.asarray(raw_action, dtype=np.float32).reshape(3)
    action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
    action = np.clip(action, -1.0, 1.0).astype(np.float32)

    previous = env._last_smoothed_action.copy()
    delta = np.clip(
        action - previous,
        -float(env.max_action_delta),
        float(env.max_action_delta),
    )
    smoothed = np.clip(previous + delta, -1.0, 1.0).astype(np.float32)
    env._prev_smoothed_action = previous
    env._last_smoothed_action = smoothed.copy()
    return smoothed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=15204)
    parser.add_argument("--target-step", type=int, default=1186)
    parser.add_argument("--wall-stiffness", type=float, default=10.0)
    parser.add_argument(
        "--baseline-json",
        default=None,
        help="Optional existing baseline diagnostic JSON to preserve alongside this result.",
    )
    args = parser.parse_args()

    if args.target_step < 1:
        raise ValueError("--target-step must be >= 1")

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else _default_output(checkpoint, args.seed, args.target_step).resolve()
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    # V15.2-C: policy/controller period remains 10 ms, physics is 2 x 5 ms.
    os.environ["MCR_SOFA_DT"] = "0.005"

    original_environment = _install_point_only_vessel_environment()
    env = None
    started = time.perf_counter()

    try:
        model = DistributedPPO.load(str(checkpoint), device="cpu")
        model.policy.set_training_mode(False)

        env = MCREnv(
            create_scene_kwargs={
                "force_model": "B02",
                "centerline_file": "target_04_centerline.vtk",
                "verbose_scene": False,
                "training_curriculum_enabled": False,
                "vessel_scale_min": 1.0,
                "vessel_scale_max": 1.0,
                "start_window_distance_m": 0.0,
                "target_window_distance_m": 0.0,
                "initial_orientation_max_angle_deg": 0.0,
                "sdf_physics_wall_enabled": True,
                "sdf_wall_stiffness_n_per_m": float(args.wall_stiffness),
                "diagnostic_intersection_method": "local_min_distance",
                # Keep the production combined vessel Line/Point switch disabled.
                "use_vessel_line_point_collision": False,
            },
            env_type=EnvType.AORTIC,
            render_mode=RenderMode.NONE,
            max_episode_steps=2048,
            time_step=0.005,
            frame_skip=1,
            physics_substeps=2,
        )

        observation, info = env.reset(seed=int(args.seed))
        models = _collision_model_summary(env)
        if models["vessel_triangle_count"] != 1:
            raise RuntimeError(
                f"Expected one vessel TriangleCollisionModel, got {models}"
            )
        if models["vessel_point_count"] != 1:
            raise RuntimeError(
                f"Expected one vessel PointCollisionModel, got {models}"
            )
        if models["vessel_line_count"] != 0:
            raise RuntimeError(
                f"Point-only test accidentally enabled vessel LineCollisionModel: {models}"
            )

        # Replay only the deterministic prefix required to reconstruct the known
        # state.  We stop immediately after the single target action.
        prefix_steps = int(args.target_step) - 1
        for step in range(1, prefix_steps + 1):
            raw_action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = env.step(raw_action)
            if step % 250 == 0:
                print(f"[PREFIX] step={step}", flush=True)
            if terminated or truncated:
                raise RuntimeError(
                    f"Point-only branch stopped before target step at {step}: "
                    f"{info.get('terminal_reason')}"
                )

        controller = env.mcr_controller_sofa
        raw_action, _ = model.predict(observation, deterministic=True)
        raw_action = np.asarray(raw_action, dtype=np.float32).reshape(3)
        smoothed_action = _prepare_smoothed_action(env, raw_action)

        before = _capture(env, "before_target_action")
        env._do_action(smoothed_action)
        requested_insert_m = float(controller.pending_insert_delta)
        original_chunk = float(controller.insert_substep_max)
        if abs(requested_insert_m) > 1e-12:
            controller.insert_substep_max = abs(requested_insert_m) / 2.0

        rows = []
        previous_tip = np.asarray(
            controller.get_pos_quat_catheter_tip()[:3], dtype=np.float64
        ).copy()
        try:
            for substep in (1, 2):
                env.sofa_simulation.animate(
                    env._sofa_root_node, env._sofa_root_node.getDt()
                )
                row = _capture(env, f"physics_substep_{substep}")
                tip = np.asarray(row["tip_mm"], dtype=np.float64) / 1000.0
                row["tip_move_mm"] = float(
                    np.linalg.norm(tip - previous_tip) * 1000.0
                )
                row["contact_free_body_penetration_mm"] = (
                    row["body_penetration_mm"]
                    if row["friction_contact_count"] == 0
                    else 0.0
                )
                previous_tip = tip
                rows.append(row)
        finally:
            controller.insert_substep_max = original_chunk

        max_contact_free_penetration = max(
            row["contact_free_body_penetration_mm"] for row in rows
        )
        first_substep_contact = rows[0]["friction_contact_count"] > 0
        baseline_penetration = float(
            BASELINE_REFERENCE["max_contact_free_body_penetration_mm"]
        )
        improvement_mm = baseline_penetration - max_contact_free_penetration

        baseline_json_data = None
        if args.baseline_json:
            baseline_path = Path(args.baseline_json).expanduser().resolve()
            if baseline_path.is_file():
                baseline_json_data = json.loads(baseline_path.read_text())

        result = {
            "test": "V15.2-C vessel PointCollisionModel only",
            "diagnostic_only": True,
            "production_code_modified": False,
            "checkpoint": str(checkpoint),
            "seed": int(args.seed),
            "target_step": int(args.target_step),
            "physics_dt_s": 0.005,
            "physics_substeps": 2,
            "rl_control_period_s": 0.01,
            "wall_stiffness_n_per_m": float(args.wall_stiffness),
            "collision_models": models,
            "baseline_reference": BASELINE_REFERENCE,
            "baseline_json": baseline_json_data,
            "raw_action": raw_action.tolist(),
            "smoothed_action": smoothed_action.tolist(),
            "requested_insert_m": requested_insert_m,
            "before": before,
            "substeps": rows,
            "point_test_summary": {
                "first_substep_contact_established": bool(first_substep_contact),
                "substep_contacts": [
                    int(row["friction_contact_count"]) for row in rows
                ],
                "substep_constraint_rows": [
                    int(row["collision_constraint_rows"]) for row in rows
                ],
                "max_contact_free_body_penetration_mm": float(
                    max_contact_free_penetration
                ),
                "baseline_contact_free_body_penetration_mm": baseline_penetration,
                "penetration_improvement_mm": float(improvement_mm),
                "penetration_improvement_fraction": float(
                    improvement_mm / baseline_penetration
                    if baseline_penetration > 0.0
                    else math.nan
                ),
                "finite": bool(
                    all(
                        np.isfinite(
                            [
                                row["tip_surface_clearance_mm"],
                                row["body_min_surface_clearance_mm"],
                                row["beam_max_correction_mm"],
                            ]
                        ).all()
                        for row in rows
                    )
                ),
            },
            "runtime_s": float(time.perf_counter() - started),
            "result_file": str(output),
        }

        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, sort_keys=True), flush=True)

    finally:
        mcr_environment_module.Environment = original_environment
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
