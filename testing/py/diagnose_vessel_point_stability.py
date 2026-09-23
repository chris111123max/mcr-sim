"""Bounded V15.2-C stability test with vessel PointCollisionModel enabled at scene creation.

Diagnostic only:
- no training;
- V15.2-C stays at 1 RL action = 2 x 5 ms physics substeps;
- vessel collision is Triangle + Point, never vessel Line;
- Point exists from scene construction (no runtime add/init);
- every physics substep is traced for contact/constraint/correction/clearance;
- output is a JSON artifact suitable for later Codex analysis.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
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


def _as_array(data):
    try:
        return np.asarray(data.array(), dtype=np.float64).copy()
    except Exception:
        return np.asarray(data.value, dtype=np.float64).copy()


def _constraint_rows(dofs) -> int:
    try:
        value = dofs.constraint.value
    except Exception:
        return -1
    return len([line for line in str(value).splitlines() if line.strip()])


def _finite_float(value):
    try:
        value = float(value)
    except Exception:
        return None
    return value if math.isfinite(value) else None


def _collision_summary(env: MCREnv) -> dict:
    vessel_node = env.scene_creation_result["mcr_environment"].CollisionModel
    catheter_node = env.mcr_controller_sofa.instrument.InstrumentCombined.getChild(
        "mcr_collis"
    )
    vessel = [obj.getClassName() for obj in vessel_node.objects]
    catheter = [obj.getClassName() for obj in catheter_node.objects]
    return {
        "vessel_classes": vessel,
        "catheter_classes": catheter,
        "vessel_triangle_count": vessel.count("TriangleCollisionModel"),
        "vessel_point_count": vessel.count("PointCollisionModel"),
        "vessel_line_count": vessel.count("LineCollisionModel"),
        "catheter_point_count": catheter.count("PointCollisionModel"),
        "catheter_line_count": catheter.count("LineCollisionModel"),
    }


def _solver_object(env: MCREnv):
    for obj in env._sofa_root_node.objects:
        if obj.getClassName() in ("LCPConstraintSolver", "GenericConstraintSolver"):
            return obj
    return None


def _solver_snapshot(solver) -> dict:
    if solver is None:
        return {}
    result = {"class": solver.getClassName()}
    # SOFA versions expose different runtime statistics. Record only fields
    # actually available in this build; missing fields are harmless.
    for name in (
        "currentIterations",
        "currentError",
        "iterations",
        "numIterations",
        "error",
        "residual",
    ):
        try:
            data = getattr(solver, name)
            value = data.value if hasattr(data, "value") else data
            arr = np.asarray(value)
            if arr.size == 1:
                scalar = arr.reshape(-1)[0]
                if np.issubdtype(arr.dtype, np.number):
                    scalar = float(scalar)
                    result[name] = scalar if math.isfinite(scalar) else None
                else:
                    result[name] = str(scalar)
        except Exception:
            pass
    return result


def _default_output(checkpoint: Path, seed: int, max_steps: int) -> Path:
    run_dir = checkpoint.parent.parent
    return (
        run_dir
        / "diagnostics"
        / f"v15_2c_vessel_point_stability_seed{seed}_steps{max_steps}.json"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=15204)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1285,
        help="Bounded diagnostic horizon; this is evaluation only, not training.",
    )
    parser.add_argument("--wall-stiffness", type=float, default=10.0)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print one compact progress line every N RL steps.",
    )
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if args.max_steps < 1:
        raise ValueError("--max-steps must be >= 1")

    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else _default_output(checkpoint, args.seed, args.max_steps).resolve()
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    # V15.2-C production control period: 10 ms = 2 x 5 ms.
    os.environ["MCR_SOFA_DT"] = "0.005"

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
            # New explicit point-only vessel mode. Legacy combined switch stays off.
            "use_vessel_line_point_collision": False,
            "use_vessel_point_collision": True,
            "use_vessel_line_collision": False,
        },
        env_type=EnvType.AORTIC,
        render_mode=RenderMode.NONE,
        max_episode_steps=max(int(args.max_steps), 2048),
        time_step=0.005,
        frame_skip=1,
        physics_substeps=2,
    )

    observation, info = env.reset(seed=int(args.seed))

    models = _collision_summary(env)
    if (
        models["vessel_triangle_count"] != 1
        or models["vessel_point_count"] != 1
        or models["vessel_line_count"] != 0
    ):
        raise RuntimeError(
            "Expected vessel Triangle + Point only, without vessel Line: "
            f"{models}"
        )

    controller = env.mcr_controller_sofa
    instrument = controller.instrument.InstrumentCombined
    collision_node = instrument.getChild("mcr_collis")
    collision_dofs = collision_node.getObject("CollisionDOFs")
    beam_dofs = instrument.getObject("DOFs")
    solver = _solver_object(env)

    original_animate = env.sofa_simulation.animate
    substep_counter = defaultdict(int)
    substep_trace = []
    previous_tip = np.asarray(
        controller.get_pos_quat_catheter_tip()[:3], dtype=np.float64
    ).copy()
    capture_non_finite = False

    aggregates = {
        "substeps": 0,
        "contact_active_substeps": 0,
        "contact_free_penetration_substeps": 0,
        "max_friction_contact_count": 0,
        "max_collision_constraint_rows": 0,
        "max_beam_constraint_rows": 0,
        "max_beam_correction_mm": 0.0,
        "max_beam_tip_correction_mm": 0.0,
        "max_tip_move_mm": 0.0,
        "min_body_clearance_mm": math.inf,
        "min_tip_clearance_mm": math.inf,
        "max_contact_free_body_penetration_mm": 0.0,
    }

    def traced_animate(root, dt):
        nonlocal previous_tip, capture_non_finite

        result = original_animate(root, dt)

        rl_step = int(env._elapsed_steps) + 1
        substep_counter[rl_step] += 1
        substep = int(substep_counter[rl_step])

        tip = np.asarray(
            controller.get_pos_quat_catheter_tip()[:3], dtype=np.float64
        ).copy()

        # Refresh SDF geometry only; do not advance failure counters.
        env._sdf_geometry_cache_step = -1
        env._update_sdf_safety_state(tip, advance_failure_counters=False)

        beam_pos = _as_array(beam_dofs.position)
        beam_free = _as_array(beam_dofs.free_position)
        coll_pos = _as_array(collision_dofs.position)
        contacts = [
            obj
            for obj in collision_node.objects
            if obj.getClassName() == "FrictionContact"
        ]

        body_clearance_mm = float(
            env.current_sdf_body_min_surface_clearance * 1000.0
        )
        tip_clearance_mm = float(env.current_sdf_surface_clearance * 1000.0)
        body_penetration_mm = max(0.0, -body_clearance_mm)
        contact_free_penetration_mm = (
            body_penetration_mm if len(contacts) == 0 else 0.0
        )

        beam_correction = np.linalg.norm(
            (beam_pos[:, :3] - beam_free[:, :3]) * 1000.0, axis=1
        )
        max_beam_correction_mm = float(np.max(beam_correction))
        beam_tip_correction_mm = float(beam_correction[-1])
        tip_move_mm = float(np.linalg.norm(tip - previous_tip) * 1000.0)
        previous_tip = tip

        collision_rows = int(_constraint_rows(collision_dofs))
        beam_rows = int(_constraint_rows(beam_dofs))

        finite = bool(
            np.all(np.isfinite(beam_pos))
            and np.all(np.isfinite(beam_free))
            and np.all(np.isfinite(coll_pos))
            and np.all(np.isfinite(tip))
            and math.isfinite(body_clearance_mm)
            and math.isfinite(tip_clearance_mm)
            and math.isfinite(max_beam_correction_mm)
        )
        capture_non_finite = capture_non_finite or (not finite)

        row = {
            "rl_step": rl_step,
            "substep": substep,
            "root_time_s": float(root.getTime()),
            "friction_contact_count": len(contacts),
            "collision_constraint_rows": collision_rows,
            "beam_constraint_rows": beam_rows,
            "body_clearance_mm": body_clearance_mm,
            "tip_clearance_mm": tip_clearance_mm,
            "body_penetration_mm": body_penetration_mm,
            "contact_free_body_penetration_mm": contact_free_penetration_mm,
            "beam_max_correction_mm": max_beam_correction_mm,
            "beam_tip_correction_mm": beam_tip_correction_mm,
            "tip_move_mm": tip_move_mm,
            "solver": _solver_snapshot(solver),
            "finite": finite,
        }
        substep_trace.append(row)

        aggregates["substeps"] += 1
        aggregates["contact_active_substeps"] += int(len(contacts) > 0)
        aggregates["contact_free_penetration_substeps"] += int(
            contact_free_penetration_mm > 0.0
        )
        aggregates["max_friction_contact_count"] = max(
            aggregates["max_friction_contact_count"], len(contacts)
        )
        aggregates["max_collision_constraint_rows"] = max(
            aggregates["max_collision_constraint_rows"], collision_rows
        )
        aggregates["max_beam_constraint_rows"] = max(
            aggregates["max_beam_constraint_rows"], beam_rows
        )
        aggregates["max_beam_correction_mm"] = max(
            aggregates["max_beam_correction_mm"], max_beam_correction_mm
        )
        aggregates["max_beam_tip_correction_mm"] = max(
            aggregates["max_beam_tip_correction_mm"], beam_tip_correction_mm
        )
        aggregates["max_tip_move_mm"] = max(
            aggregates["max_tip_move_mm"], tip_move_mm
        )
        aggregates["min_body_clearance_mm"] = min(
            aggregates["min_body_clearance_mm"], body_clearance_mm
        )
        aggregates["min_tip_clearance_mm"] = min(
            aggregates["min_tip_clearance_mm"], tip_clearance_mm
        )
        aggregates["max_contact_free_body_penetration_mm"] = max(
            aggregates["max_contact_free_body_penetration_mm"],
            contact_free_penetration_mm,
        )

        return result

    env.sofa_simulation.animate = traced_animate

    started = time.perf_counter()
    step_runtime_s = []
    terminal_reason = "diagnostic_limit"
    completed_steps = 0
    non_finite_observation = False

    try:
        for step in range(1, int(args.max_steps) + 1):
            raw_action, _ = model.predict(observation, deterministic=True)

            step_started = time.perf_counter()
            observation, reward, terminated, truncated, info = env.step(raw_action)
            step_runtime_s.append(time.perf_counter() - step_started)
            completed_steps = step

            if not np.all(np.isfinite(np.asarray(observation))):
                non_finite_observation = True

            if args.progress_every > 0 and step % int(args.progress_every) == 0:
                print(
                    "[VESSEL_POINT_STABILITY]",
                    "step=", step,
                    "contacts=", aggregates["max_friction_contact_count"],
                    "rows_max=", aggregates["max_collision_constraint_rows"],
                    "body_min_mm=", aggregates["min_body_clearance_mm"],
                    "cf_pen_max_mm=", aggregates[
                        "max_contact_free_body_penetration_mm"
                    ],
                    "beam_corr_max_mm=", aggregates["max_beam_correction_mm"],
                    "step_runtime_max_s=", max(step_runtime_s),
                    flush=True,
                )

            if capture_non_finite or non_finite_observation:
                terminal_reason = "non_finite_diagnostic"
                break

            if terminated or truncated:
                terminal_reason = str(info.get("terminal_reason", "environment_stop"))
                break
    finally:
        env.sofa_simulation.animate = original_animate

    runtime_total_s = time.perf_counter() - started
    step_runtime_arr = np.asarray(step_runtime_s, dtype=np.float64)

    # Make infinities JSON-safe if the run stopped before a finite SDF sample.
    for key in ("min_body_clearance_mm", "min_tip_clearance_mm"):
        if not math.isfinite(float(aggregates[key])):
            aggregates[key] = None

    result = {
        "test": "V15.2-C vessel Point scene-start stability",
        "diagnostic_only": True,
        "training_started": False,
        "checkpoint": str(checkpoint),
        "seed": int(args.seed),
        "requested_max_steps": int(args.max_steps),
        "completed_steps": int(completed_steps),
        "physics_dt_s": 0.005,
        "physics_substeps": 2,
        "rl_control_period_s": 0.01,
        "wall_stiffness_n_per_m": float(args.wall_stiffness),
        "collision_models": models,
        "solver_initial": _solver_snapshot(solver),
        "terminal_reason": terminal_reason,
        "capture_non_finite": bool(capture_non_finite),
        "non_finite_observation": bool(non_finite_observation),
        "aggregates": aggregates,
        "step_runtime": {
            "mean_s": _finite_float(np.mean(step_runtime_arr))
            if step_runtime_arr.size
            else None,
            "p95_s": _finite_float(np.percentile(step_runtime_arr, 95))
            if step_runtime_arr.size
            else None,
            "max_s": _finite_float(np.max(step_runtime_arr))
            if step_runtime_arr.size
            else None,
        },
        "episode_info": {
            "episode_min_tip_clearance_m": _finite_float(
                info.get("episode_min_tip_clearance")
            ),
            "episode_min_body_clearance_m": _finite_float(
                info.get("episode_min_body_clearance")
            ),
            "episode_max_contact_free_penetration_m": _finite_float(
                info.get("episode_max_contact_free_penetration")
            ),
            "contact_active_substeps": int(info.get("contact_active_substeps", 0)),
            "successful_task": bool(info.get("successful_task", False)),
        },
        "runtime_total_s": float(runtime_total_s),
        "substep_trace": substep_trace,
        "result_file": str(output),
    }

    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "substep_trace"}, sort_keys=True), flush=True)

    try:
        env.close()
    except Exception:
        pass

    os._exit(0)


if __name__ == "__main__":
    main()
