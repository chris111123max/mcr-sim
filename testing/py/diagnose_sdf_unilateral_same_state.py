"""Same-state causal A/B for the true SDF unilateral constraint.

Shared prefix:
  V15.2-C, B02/target04/seed15204
  catheter Line+Point collision
  vessel Triangle-only collision
  vessel Point/Line disabled
  soft SDF wall preserved
  tangent-triangle SDF hard wall disabled
  C++ SDF unilateral component constructed but disabled

Fork after step 1185:
  A: LCPConstraintSolver + unilateral OFF (historical baseline)
  B: GenericConstraintSolver + unilateral OFF (solver-only control)
  C: GenericConstraintSolver + unilateral ON  (unilateral causal branch)

All three inherit the exact same forked catheter state and execute the same
step-1186 policy action with 2 x 5 ms physics.  Generic branches replace the
solver only after fork, then re-init FreeMotionAnimationLoop and verify the
catheter state fingerprint is unchanged before applying the action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

PYTHON_ROOT = Path(__file__).resolve().parents[2]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from mcr_sim.distributed import DistributedPPO
from mcr_sim.mcr_rl_env import EnvType, MCREnv
from mcr_sim.rl_core.base import RenderMode
from mcr_sim.training_config import (
    CONSTRAINT_MAX_ITERATIONS,
    CONSTRAINT_TOLERANCE,
)


KNOWN_BASELINE = {
    "step": 1186,
    "substep_contacts": [0, 1],
    "substep_constraint_rows": [0, 6],
    "first_substep_body_clearance_mm": -0.3551,
}


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


def _state_fingerprint(env: MCREnv) -> str:
    controller = env.mcr_controller_sofa
    instrument = controller.instrument.InstrumentCombined
    collision_dofs = instrument.getChild("mcr_collis").getObject("CollisionDOFs")
    beam_dofs = instrument.getObject("DOFs")
    payload = b"".join(
        [
            _as_array(collision_dofs.position).tobytes(),
            _as_array(collision_dofs.free_position).tobytes(),
            _as_array(beam_dofs.position).tobytes(),
            _as_array(beam_dofs.free_position).tobytes(),
            _as_array(beam_dofs.velocity).tobytes(),
            np.asarray(
                [
                    float(controller._getXTipValue()),
                    float(controller.pending_insert_delta),
                    float(env._sofa_root_node.getTime()),
                ],
                dtype=np.float64,
            ).tobytes(),
        ]
    )
    return hashlib.sha256(payload).hexdigest()


def _data_scalar(obj, name, default):
    try:
        data = getattr(obj, name)
        value = data.value
        if isinstance(value, (list, tuple, np.ndarray)):
            value = np.asarray(value).reshape(-1)[0]
        return value
    except Exception:
        return default


def _root_object_by_class(env: MCREnv, class_names):
    wanted = set(class_names)
    for obj in env._sofa_root_node.objects:
        if obj.getClassName() in wanted:
            return obj
    return None


def _solver_object(env: MCREnv):
    return _root_object_by_class(
        env, ("LCPConstraintSolver", "GenericConstraintSolver")
    )


def _solver_summary(env: MCREnv) -> dict:
    solver = _solver_object(env)
    if solver is None:
        return {"class": None}
    class_name = solver.getClassName()
    summary = {"class": class_name}
    if class_name == "LCPConstraintSolver":
        summary.update(
            {
                "tolerance": float(
                    _data_scalar(
                        solver, "tolerance", CONSTRAINT_TOLERANCE
                    )
                ),
                "max_iterations": int(
                    _data_scalar(
                        solver, "maxIt", CONSTRAINT_MAX_ITERATIONS
                    )
                ),
                "build_lcp": bool(
                    _data_scalar(solver, "build_lcp", False)
                ),
            }
        )
    elif class_name == "GenericConstraintSolver":
        summary.update(
            {
                "tolerance": float(
                    _data_scalar(
                        solver, "tolerance", CONSTRAINT_TOLERANCE
                    )
                ),
                "max_iterations": int(
                    _data_scalar(
                        solver, "maxIterations", CONSTRAINT_MAX_ITERATIONS
                    )
                ),
            }
        )
    return summary


def _switch_to_generic_solver(env: MCREnv, expected_fingerprint: str) -> dict:
    before_fingerprint = _state_fingerprint(env)
    if before_fingerprint != expected_fingerprint:
        raise RuntimeError(
            "Solver switch requested from a state different from the fork"
        )

    root = env._sofa_root_node
    old_solver = _solver_object(env)
    if old_solver is None:
        raise RuntimeError("No constraint solver found before Generic switch")

    old_summary = _solver_summary(env)
    tolerance = float(
        old_summary.get("tolerance", CONSTRAINT_TOLERANCE)
    )
    max_iterations = int(
        old_summary.get("max_iterations", CONSTRAINT_MAX_ITERATIONS)
    )

    animation_loop = _root_object_by_class(
        env, ("FreeMotionAnimationLoop",)
    )
    if animation_loop is None:
        raise RuntimeError("FreeMotionAnimationLoop not found")

    root.removeObject(old_solver)
    generic = root.addObject(
        "GenericConstraintSolver",
        tolerance=str(tolerance),
        maxIterations=str(max_iterations),
        printLog="false",
    )

    # FreeMotionAnimationLoop caches a ConstraintSolver* during init().  Refresh
    # that pointer after replacing the solver; otherwise it may still reference
    # the removed LCPConstraintSolver.
    animation_loop.init()

    new_summary = _solver_summary(env)
    if new_summary.get("class") != "GenericConstraintSolver":
        raise RuntimeError(
            f"Generic solver replacement failed: {new_summary}"
        )

    after_fingerprint = _state_fingerprint(env)
    if after_fingerprint != expected_fingerprint:
        raise RuntimeError(
            "Replacing/reinitializing the constraint solver changed the "
            "catheter state before the target action"
        )

    return {
        "old": old_summary,
        "new": new_summary,
        "state_unchanged": True,
        "generic_object_class": generic.getClassName(),
    }


def _collision_summary(env: MCREnv) -> dict:
    vessel = env.scene_creation_result["mcr_environment"].CollisionModel
    catheter = env.mcr_controller_sofa.instrument.InstrumentCombined.getChild(
        "mcr_collis"
    )
    hard = env.scene_creation_result.get("sdf_hard_constraint_controller")
    unilateral = env.scene_creation_result.get(
        "sdf_unilateral_constraint_controller"
    )

    vessel_classes = [obj.getClassName() for obj in vessel.objects]
    catheter_classes = [obj.getClassName() for obj in catheter.objects]
    return {
        "vessel_classes": vessel_classes,
        "catheter_classes": catheter_classes,
        "vessel_triangle_count": vessel_classes.count("TriangleCollisionModel"),
        "vessel_point_count": vessel_classes.count("PointCollisionModel"),
        "vessel_line_count": vessel_classes.count("LineCollisionModel"),
        "catheter_point_count": catheter_classes.count("PointCollisionModel"),
        "catheter_line_count": catheter_classes.count("LineCollisionModel"),
        "sdf_hard_triangle_controller_present": hard is not None,
        "sdf_unilateral_controller_present": unilateral is not None,
        "sdf_unilateral_component_class": (
            unilateral.constraint.getClassName() if unilateral is not None else None
        ),
    }


def _refresh_sdf(env):
    tip = np.asarray(
        env.mcr_controller_sofa.get_pos_quat_catheter_tip()[:3],
        dtype=np.float64,
    ).copy()
    env._sdf_geometry_cache_step = -1
    env._update_sdf_safety_state(tip, advance_failure_counters=False)


def _capture(env: MCREnv, label: str) -> dict:
    controller = env.mcr_controller_sofa
    instrument = controller.instrument.InstrumentCombined
    collision_node = instrument.getChild("mcr_collis")
    collision_dofs = collision_node.getObject("CollisionDOFs")
    beam_dofs = instrument.getObject("DOFs")
    unilateral = env.scene_creation_result.get(
        "sdf_unilateral_constraint_controller"
    )

    _refresh_sdf(env)

    contacts = [
        obj
        for obj in collision_node.objects
        if obj.getClassName() == "FrictionContact"
    ]
    beam_pos = _as_array(beam_dofs.position)
    beam_free = _as_array(beam_dofs.free_position)
    coll_pos = _as_array(collision_dofs.position)
    coll_free = _as_array(collision_dofs.free_position)
    tip = np.asarray(
        controller.get_pos_quat_catheter_tip()[:3], dtype=np.float64
    ).copy()

    body_clearance_mm = float(
        env.current_sdf_body_min_surface_clearance * 1000.0
    )
    tip_clearance_mm = float(env.current_sdf_surface_clearance * 1000.0)
    udiag = unilateral.get_diagnostics() if unilateral is not None else {}

    return {
        "label": label,
        "root_time_s": float(env._sofa_root_node.getTime()),
        "state_fingerprint_sha256": _state_fingerprint(env),
        "solver": _solver_summary(env),
        "tip_mm": (tip * 1000.0).tolist(),
        "body_clearance_mm": body_clearance_mm,
        "tip_clearance_mm": tip_clearance_mm,
        "body_penetration_mm": max(0.0, -body_clearance_mm),
        "friction_contact_count": len(contacts),
        "collision_constraint_rows": _constraint_rows(collision_dofs),
        "beam_constraint_rows": _constraint_rows(beam_dofs),
        "beam_max_correction_mm": float(
            np.max(
                np.linalg.norm(
                    (beam_pos[:, :3] - beam_free[:, :3]) * 1000.0,
                    axis=1,
                )
            )
        ),
        "collision_max_correction_mm": float(
            np.max(
                np.linalg.norm(
                    (coll_pos[:, :3] - coll_free[:, :3]) * 1000.0,
                    axis=1,
                )
            )
        ),
        "xtip_m": float(controller._getXTipValue()),
        "pending_insert_m": float(controller.pending_insert_delta),
        "unilateral": udiag,
    }


def _prepare_action(env, raw_action):
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


def _run_target(
    env,
    raw_action,
    branch,
    enabled,
    fingerprint,
    solver_mode="lcp",
):
    unilateral = env.scene_creation_result["sdf_unilateral_constraint_controller"]
    controller = env.mcr_controller_sofa

    if _state_fingerprint(env) != fingerprint:
        raise RuntimeError(f"{branch}: fork state changed before branch")

    solver_switch = None
    if solver_mode == "generic":
        solver_switch = _switch_to_generic_solver(env, fingerprint)
    elif solver_mode != "lcp":
        raise ValueError(f"Unsupported solver_mode={solver_mode}")

    unilateral.set_enabled(enabled)

    if _state_fingerprint(env) != fingerprint:
        raise RuntimeError(
            f"{branch}: solver/unilateral toggle changed catheter state"
        )

    before = _capture(env, "before_step_1186")
    action = _prepare_action(env, raw_action)
    env._do_action(action)

    requested_insert = float(controller.pending_insert_delta)
    original_chunk = float(controller.insert_substep_max)
    controller.insert_substep_max = (
        abs(requested_insert) / 2.0
        if abs(requested_insert) > 1e-12
        else original_chunk
    )

    rows = []
    try:
        for substep in (1, 2):
            env.sofa_simulation.animate(
                env._sofa_root_node, env._sofa_root_node.getDt()
            )
            rows.append(_capture(env, f"physics_substep_{substep}"))
    finally:
        controller.insert_substep_max = original_chunk

    return {
        "branch": branch,
        "solver_mode": solver_mode,
        "solver_switch": solver_switch,
        "solver_before_action": _solver_summary(env),
        "unilateral_enabled": bool(enabled),
        "fork_fingerprint_sha256": fingerprint,
        "raw_action": np.asarray(raw_action, dtype=np.float32).tolist(),
        "smoothed_action": action.tolist(),
        "requested_insert_m": requested_insert,
        "before": before,
        "substeps": rows,
        "summary": {
            "solver_class": _solver_summary(env).get("class"),
            "substep_friction_contacts": [
                int(r["friction_contact_count"]) for r in rows
            ],
            "substep_constraint_rows": [
                int(r["collision_constraint_rows"]) for r in rows
            ],
            "substep_body_clearance_mm": [
                float(r["body_clearance_mm"]) for r in rows
            ],
            "unilateral_candidates": [
                int(r["unilateral"].get("candidate_constraints", 0))
                for r in rows
            ],
            "unilateral_python_active": [
                int(r["unilateral"].get("active_constraints", 0))
                for r in rows
            ],
            "unilateral_cpp_active": [
                int(r["unilateral"].get("cpp_active_count", -1))
                for r in rows
            ],
            "unilateral_sample_kinds": [
                list(r["unilateral"].get("selected_sample_kinds", []))
                for r in rows
            ],
            "max_body_penetration_mm": float(
                max(r["body_penetration_mm"] for r in rows)
            ),
            "max_beam_correction_mm": float(
                max(r["beam_max_correction_mm"] for r in rows)
            ),
            "finite": bool(
                all(
                    np.isfinite(
                        [
                            r["body_clearance_mm"],
                            r["tip_clearance_mm"],
                            r["beam_max_correction_mm"],
                            r["collision_max_correction_mm"],
                        ]
                    ).all()
                    for r in rows
                )
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=15204)
    parser.add_argument("--target-step", type=int, default=1186)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else (
            checkpoint.parent.parent
            / "diagnostics"
            / f"v15_2c_sdf_unilateral_same_state_step{args.target_step}_seed{args.seed}.json"
        ).resolve()
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output_a = output.with_name(output.stem + "_A_lcp_baseline.json")
    output_b = output.with_name(output.stem + "_B_generic_baseline.json")
    output_c = output.with_name(output.stem + "_C_generic_unilateral.json")

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
            "sdf_wall_stiffness_n_per_m": 10.0,
            "diagnostic_intersection_method": "local_min_distance",
            "use_vessel_line_point_collision": False,
            "use_vessel_point_collision": False,
            "use_vessel_line_collision": False,
            "sdf_hard_constraint_construct": False,
            "sdf_hard_constraint_enabled": False,
            "sdf_unilateral_constraint_construct": True,
            "sdf_unilateral_constraint_enabled": False,
        },
        env_type=EnvType.AORTIC,
        render_mode=RenderMode.NONE,
        max_episode_steps=2048,
        time_step=0.005,
        frame_skip=1,
        physics_substeps=2,
    )

    started = time.perf_counter()
    observation, info = env.reset(seed=int(args.seed))

    models = _collision_summary(env)
    if (
        models["vessel_triangle_count"] != 1
        or models["vessel_point_count"] != 0
        or models["vessel_line_count"] != 0
        or models["catheter_point_count"] < 1
        or models["catheter_line_count"] < 1
        or models["sdf_hard_triangle_controller_present"]
        or not models["sdf_unilateral_controller_present"]
    ):
        raise RuntimeError(f"Unexpected collision/constraint configuration: {models}")

    unilateral = env.scene_creation_result["sdf_unilateral_constraint_controller"]
    if unilateral.enabled:
        raise RuntimeError("Unilateral constraint must be disabled through prefix")

    prefix_steps = int(args.target_step) - 1
    for step in range(1, prefix_steps + 1):
        raw_action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = env.step(raw_action)
        if step % 250 == 0:
            print(f"[PREFIX] step={step}", flush=True)
        if terminated or truncated:
            raise RuntimeError(
                f"Baseline stopped before fork at step {step}: "
                f"{info.get('terminal_reason')}"
            )

    fingerprint = _state_fingerprint(env)
    fork_capture = _capture(env, f"fork_after_step_{prefix_steps}")
    raw_action, _ = model.predict(observation, deterministic=True)
    raw_action = np.asarray(raw_action, dtype=np.float32).reshape(3)

    child_specs = [
        (
            "B_generic_baseline",
            False,
            "generic",
            output_b,
        ),
        (
            "C_generic_unilateral",
            True,
            "generic",
            output_c,
        ),
    ]
    child_pids = {}
    for branch, enabled, solver_mode, child_output in child_specs:
        child_pid = os.fork()
        if child_pid == 0:
            try:
                child_result = _run_target(
                    env,
                    raw_action,
                    branch,
                    enabled,
                    fingerprint,
                    solver_mode=solver_mode,
                )
                child_output.write_text(
                    json.dumps(child_result, indent=2, sort_keys=True) + "\n"
                )
                os._exit(0)
            except BaseException:
                child_output.write_text(
                    json.dumps(
                        {"error": traceback.format_exc()},
                        indent=2,
                    )
                    + "\n"
                )
                os._exit(1)
        child_pids[branch] = child_pid

    result_a = _run_target(
        env,
        raw_action,
        "A_lcp_baseline",
        False,
        fingerprint,
        solver_mode="lcp",
    )
    output_a.write_text(json.dumps(result_a, indent=2, sort_keys=True) + "\n")

    child_statuses = {}
    for branch, child_pid in child_pids.items():
        _, status = os.waitpid(child_pid, 0)
        child_statuses[branch] = int(status)

    result_b = json.loads(output_b.read_text())
    result_c = json.loads(output_c.read_text())

    comparison = None
    children_ok = (
        all(status == 0 for status in child_statuses.values())
        and "error" not in result_b
        and "error" not in result_c
    )
    if children_ok:
        a = result_a["summary"]
        b = result_b["summary"]
        cc = result_c["summary"]
        comparison = {
            "same_fork_fingerprint_all": (
                result_a["fork_fingerprint_sha256"]
                == result_b["fork_fingerprint_sha256"]
                == result_c["fork_fingerprint_sha256"]
                == fingerprint
            ),
            "same_raw_action_all": (
                np.array_equal(
                    np.asarray(result_a["raw_action"], dtype=np.float32),
                    np.asarray(result_b["raw_action"], dtype=np.float32),
                )
                and np.array_equal(
                    np.asarray(result_a["raw_action"], dtype=np.float32),
                    np.asarray(result_c["raw_action"], dtype=np.float32),
                )
            ),
            "same_smoothed_action_all": (
                np.array_equal(
                    np.asarray(result_a["smoothed_action"], dtype=np.float32),
                    np.asarray(result_b["smoothed_action"], dtype=np.float32),
                )
                and np.array_equal(
                    np.asarray(result_a["smoothed_action"], dtype=np.float32),
                    np.asarray(result_c["smoothed_action"], dtype=np.float32),
                )
            ),
            "same_requested_insert_all": (
                math.isclose(
                    float(result_a["requested_insert_m"]),
                    float(result_b["requested_insert_m"]),
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
                and math.isclose(
                    float(result_a["requested_insert_m"]),
                    float(result_c["requested_insert_m"]),
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
            ),
            "A_solver": a["solver_class"],
            "B_solver": b["solver_class"],
            "C_solver": cc["solver_class"],
            "A_friction_contacts": a["substep_friction_contacts"],
            "B_friction_contacts": b["substep_friction_contacts"],
            "C_friction_contacts": cc["substep_friction_contacts"],
            "A_constraint_rows": a["substep_constraint_rows"],
            "B_constraint_rows": b["substep_constraint_rows"],
            "C_constraint_rows": cc["substep_constraint_rows"],
            "A_body_clearance_mm": a["substep_body_clearance_mm"],
            "B_body_clearance_mm": b["substep_body_clearance_mm"],
            "C_body_clearance_mm": cc["substep_body_clearance_mm"],
            "C_unilateral_python_active": cc["unilateral_python_active"],
            "C_unilateral_cpp_active": cc["unilateral_cpp_active"],
            "C_unilateral_sample_kinds": cc["unilateral_sample_kinds"],
            "A_max_penetration_mm": a["max_body_penetration_mm"],
            "B_max_penetration_mm": b["max_body_penetration_mm"],
            "C_max_penetration_mm": cc["max_body_penetration_mm"],
            "generic_unilateral_penetration_reduction_mm": (
                b["max_body_penetration_mm"]
                - cc["max_body_penetration_mm"]
            ),
            "A_max_beam_correction_mm": a["max_beam_correction_mm"],
            "B_max_beam_correction_mm": b["max_beam_correction_mm"],
            "C_max_beam_correction_mm": cc["max_beam_correction_mm"],
            "all_finite": bool(
                a["finite"] and b["finite"] and cc["finite"]
            ),
            "A_first_clearance_error_vs_known_mm": (
                a["substep_body_clearance_mm"][0]
                - KNOWN_BASELINE["first_substep_body_clearance_mm"]
            ),
            "B_solver_switch_state_unchanged": bool(
                result_b.get("solver_switch", {}).get(
                    "state_unchanged", False
                )
            ),
            "C_solver_switch_state_unchanged": bool(
                result_c.get("solver_switch", {}).get(
                    "state_unchanged", False
                )
            ),
        }

    combined = {
        "test": (
            "V15.2-C same-state solver compatibility + "
            "true SDF unilateral causal fork"
        ),
        "diagnostic_only": True,
        "training_started": False,
        "checkpoint": str(checkpoint),
        "seed": int(args.seed),
        "target_step": int(args.target_step),
        "collision_models": models,
        "fork_capture": fork_capture,
        "A_lcp_baseline": result_a,
        "B_generic_baseline": result_b,
        "C_generic_unilateral": result_c,
        "comparison": comparison,
        "child_statuses": child_statuses,
        "runtime_s": float(time.perf_counter() - started),
        "files": {
            "combined": str(output),
            "A_lcp_baseline": str(output_a),
            "B_generic_baseline": str(output_b),
            "C_generic_unilateral": str(output_c),
        },
    }
    output.write_text(json.dumps(combined, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"comparison": comparison, "files": combined["files"]}, sort_keys=True), flush=True)

    try:
        env.close()
    except Exception:
        pass

    if not children_ok:
        raise RuntimeError(
            "Generic diagnostic branch failed; inspect B/C JSON"
        )

    os._exit(0)


if __name__ == "__main__":
    main()
