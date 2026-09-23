"""Same-state causal test for the SDF-derived native hard-contact wall.

Baseline collision configuration is preserved:
  catheter: LineCollisionModel + PointCollisionModel
  vessel:   TriangleCollisionModel only
  response: FrictionContactConstraint
  solver:   LCPConstraintSolver
  physics:  V15.2-C, 2 x 5 ms per RL action

The SDF hard-wall collision model is constructed from scene start but disabled
and parked far away. After deterministic replay through step 1185, the process
forks:
  A: hard wall stays disabled
  B: hard wall is enabled without scene re-init

Both branches execute the exact same step-1186 action.
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


KNOWN_BASELINE = {
    "step": 1186,
    "substep_contacts": [0, 1],
    "substep_constraint_rows": [0, 6],
    "first_substep_body_clearance_mm": -0.3551,
    "max_contact_free_body_penetration_mm": 0.3551,
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


def _data_scalar(obj, name, default=None):
    try:
        value = getattr(obj, name)
        if hasattr(value, "value"):
            value = value.value
        arr = np.asarray(value).reshape(-1)
        if not arr.size:
            return default
        scalar = arr[0]
        if isinstance(scalar, (np.integer, int)):
            return int(scalar)
        if isinstance(scalar, (np.floating, float)):
            value = float(scalar)
            return value if math.isfinite(value) else default
        return str(scalar)
    except Exception:
        return default


def _catheter_state_fingerprint(env: MCREnv) -> str:
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


def _collision_summary(env: MCREnv) -> dict:
    vessel_node = env.scene_creation_result["mcr_environment"].CollisionModel
    catheter_node = env.mcr_controller_sofa.instrument.InstrumentCombined.getChild(
        "mcr_collis"
    )
    hard = env.scene_creation_result.get("sdf_hard_constraint_controller")

    vessel = [obj.getClassName() for obj in vessel_node.objects]
    catheter = [obj.getClassName() for obj in catheter_node.objects]
    hard_classes = (
        [obj.getClassName() for obj in hard.collision_node.objects]
        if hard is not None
        else []
    )
    vessel_env = env.scene_creation_result["mcr_environment"]
    return {
        "vessel_classes": vessel,
        "catheter_classes": catheter,
        "hard_wall_classes": hard_classes,
        "vessel_triangle_count": vessel.count("TriangleCollisionModel"),
        "vessel_point_count": vessel.count("PointCollisionModel"),
        "vessel_line_count": vessel.count("LineCollisionModel"),
        "catheter_point_count": catheter.count("PointCollisionModel"),
        "catheter_line_count": catheter.count("LineCollisionModel"),
        "hard_triangle_count": hard_classes.count("TriangleCollisionModel"),
        "vessel_triangle_group": _data_scalar(
            vessel_env.TriangleCollisionModel, "group"
        ),
        "hard_triangle_group": (
            _data_scalar(hard.collision_model, "group")
            if hard is not None
            else None
        ),
        "hard_triangle_both_side": (
            _data_scalar(hard.collision_model, "bothSide")
            if hard is not None
            else None
        ),
    }


def _refresh_sdf(env: MCREnv) -> None:
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
    hard = env.scene_creation_result.get("sdf_hard_constraint_controller")

    _refresh_sdf(env)

    tip = np.asarray(
        controller.get_pos_quat_catheter_tip()[:3], dtype=np.float64
    ).copy()
    beam_pos = _as_array(beam_dofs.position)
    beam_free = _as_array(beam_dofs.free_position)
    collision_pos = _as_array(collision_dofs.position)
    collision_free = _as_array(collision_dofs.free_position)
    contacts = [
        obj
        for obj in collision_node.objects
        if obj.getClassName() == "FrictionContact"
    ]

    body_clearance_mm = float(
        env.current_sdf_body_min_surface_clearance * 1000.0
    )
    tip_clearance_mm = float(env.current_sdf_surface_clearance * 1000.0)

    hard_diag = hard.get_diagnostics() if hard is not None else {}

    return {
        "label": label,
        "root_time_s": float(env._sofa_root_node.getTime()),
        "root_dt_s": float(env._sofa_root_node.getDt()),
        "catheter_state_fingerprint_sha256": _catheter_state_fingerprint(env),
        "tip_mm": (tip * 1000.0).tolist(),
        "tip_surface_clearance_mm": tip_clearance_mm,
        "body_min_surface_clearance_mm": body_clearance_mm,
        "body_penetration_mm": max(0.0, -body_clearance_mm),
        "friction_contact_count": len(contacts),
        "collision_constraint_rows": _constraint_rows(collision_dofs),
        "beam_constraint_rows": _constraint_rows(beam_dofs),
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
        "hard_wall": hard_diag,
    }


def _prepare_action(env: MCREnv, raw_action) -> np.ndarray:
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


def _run_target_action(
    env: MCREnv,
    raw_action: np.ndarray,
    branch_name: str,
    enable_hard_wall: bool,
    fork_fingerprint: str,
) -> dict:
    hard = env.scene_creation_result["sdf_hard_constraint_controller"]
    controller = env.mcr_controller_sofa

    pre_enable_fingerprint = _catheter_state_fingerprint(env)
    if pre_enable_fingerprint != fork_fingerprint:
        raise RuntimeError(f"{branch_name}: catheter state changed before branch")

    hard.set_enabled(enable_hard_wall)

    post_enable_fingerprint = _catheter_state_fingerprint(env)
    if post_enable_fingerprint != fork_fingerprint:
        raise RuntimeError(
            f"{branch_name}: enabling hard wall directly changed catheter state"
        )

    before = _capture(env, "before_step_1186")
    smoothed_action = _prepare_action(env, raw_action)
    env._do_action(smoothed_action)

    requested_insert_m = float(controller.pending_insert_delta)
    original_chunk = float(controller.insert_substep_max)
    controller.insert_substep_max = (
        abs(requested_insert_m) / 2.0
        if abs(requested_insert_m) > 1e-12
        else original_chunk
    )

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

    numeric = []
    for row in rows:
        numeric.extend(
            [
                row["tip_surface_clearance_mm"],
                row["body_min_surface_clearance_mm"],
                row["beam_tip_correction_mm"],
                row["beam_max_correction_mm"],
                row["tip_move_mm"],
            ]
        )

    return {
        "branch": branch_name,
        "hard_wall_enabled": bool(enable_hard_wall),
        "fork_fingerprint_sha256": fork_fingerprint,
        "pre_enable_fingerprint_sha256": pre_enable_fingerprint,
        "post_enable_fingerprint_sha256": post_enable_fingerprint,
        "raw_action": np.asarray(raw_action, dtype=np.float32).tolist(),
        "smoothed_action": smoothed_action.tolist(),
        "requested_insert_m": requested_insert_m,
        "before": before,
        "substeps": rows,
        "summary": {
            "substep_contacts": [
                int(row["friction_contact_count"]) for row in rows
            ],
            "substep_constraint_rows": [
                int(row["collision_constraint_rows"]) for row in rows
            ],
            "substep_body_clearance_mm": [
                float(row["body_min_surface_clearance_mm"]) for row in rows
            ],
            "substep_tip_clearance_mm": [
                float(row["tip_surface_clearance_mm"]) for row in rows
            ],
            "substep_hard_candidate_patches": [
                int(row["hard_wall"].get("candidate_patches", 0)) for row in rows
            ],
            "substep_hard_active_patches": [
                int(row["hard_wall"].get("active_patches", 0)) for row in rows
            ],
            "substep_hard_dropped_patches": [
                int(row["hard_wall"].get("dropped_patches", 0)) for row in rows
            ],
            "substep_hard_selected_indices": [
                list(row["hard_wall"].get("selected_sample_indices", []))
                for row in rows
            ],
            "substep_hard_selected_clearances_mm": [
                [
                    float(value) * 1000.0
                    for value in row["hard_wall"].get(
                        "selected_clearances_m", []
                    )
                ]
                for row in rows
            ],
            "substep_hard_contact_count": [
                row["hard_wall"].get("hard_wall_contact_count") for row in rows
            ],
            "max_contact_free_body_penetration_mm": float(
                max(row["contact_free_body_penetration_mm"] for row in rows)
            ),
            "max_body_penetration_mm": float(
                max(row["body_penetration_mm"] for row in rows)
            ),
            "max_beam_correction_mm": float(
                max(row["beam_max_correction_mm"] for row in rows)
            ),
            "finite": bool(np.all(np.isfinite(numeric))),
        },
    }


def _default_output(checkpoint: Path, seed: int, target_step: int) -> Path:
    return (
        checkpoint.parent.parent
        / "diagnostics"
        / f"v15_2c_sdf_hard_same_state_step{target_step}_seed{seed}.json"
    )


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
        else _default_output(checkpoint, args.seed, args.target_step).resolve()
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output_a = output.with_name(output.stem + "_A_baseline.json")
    output_b = output.with_name(output.stem + "_B_sdf_hard.json")

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
            # Preserve the existing V15.2-C soft SDF wall in both branches.
            "sdf_physics_wall_enabled": True,
            "sdf_wall_stiffness_n_per_m": 10.0,
            "diagnostic_intersection_method": "local_min_distance",
            # Explicitly exclude the previous vessel Point/Line experiment.
            "use_vessel_line_point_collision": False,
            "use_vessel_point_collision": False,
            "use_vessel_line_collision": False,
            # Construct hard-wall collision geometry but keep it parked/disabled
            # throughout the deterministic prefix.
            "sdf_hard_constraint_construct": True,
            "sdf_hard_constraint_enabled": False,
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
        or models["catheter_point_count"] != 1
        or models["catheter_line_count"] != 1
        or models["hard_triangle_count"] != 1
    ):
        raise RuntimeError(
            "Unexpected collision configuration; this test requires catheter "
            "Line+Point, vessel Triangle-only, plus one disabled SDF hard-wall "
            f"Triangle model. Got: {models}"
        )

    hard = env.scene_creation_result.get("sdf_hard_constraint_controller")
    if hard is None or hard.enabled:
        raise RuntimeError("SDF hard wall must exist but be disabled before replay")

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

    fork_fingerprint = _catheter_state_fingerprint(env)
    fork_capture = _capture(env, f"fork_after_step_{prefix_steps}")
    raw_action, _ = model.predict(observation, deterministic=True)
    raw_action = np.asarray(raw_action, dtype=np.float32).reshape(3)

    child_pid = os.fork()
    if child_pid == 0:
        try:
            result_b = _run_target_action(
                env,
                raw_action,
                "B_sdf_hard_enabled",
                True,
                fork_fingerprint,
            )
            output_b.write_text(
                json.dumps(result_b, indent=2, sort_keys=True) + "\n"
            )
            os._exit(0)
        except BaseException:
            output_b.write_text(
                json.dumps(
                    {"error": traceback.format_exc()},
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            os._exit(1)

    result_a = _run_target_action(
        env,
        raw_action,
        "A_baseline_hard_disabled",
        False,
        fork_fingerprint,
    )
    output_a.write_text(
        json.dumps(result_a, indent=2, sort_keys=True) + "\n"
    )

    _, child_status = os.waitpid(child_pid, 0)
    result_b = json.loads(output_b.read_text())

    comparison = None
    if child_status == 0 and "error" not in result_b:
        a = result_a["summary"]
        b = result_b["summary"]
        a_first_clearance = float(a["substep_body_clearance_mm"][0])
        baseline_reproduced = bool(
            a["substep_contacts"] == KNOWN_BASELINE["substep_contacts"]
            and a["substep_constraint_rows"]
            == KNOWN_BASELINE["substep_constraint_rows"]
            and abs(
                a_first_clearance
                - KNOWN_BASELINE["first_substep_body_clearance_mm"]
            )
            <= 0.05
        )
        comparison = {
            "same_fork_fingerprint": bool(
                result_a["fork_fingerprint_sha256"]
                == result_b["fork_fingerprint_sha256"]
                == fork_fingerprint
            ),
            "same_raw_action": bool(
                np.array_equal(
                    np.asarray(result_a["raw_action"], dtype=np.float32),
                    np.asarray(result_b["raw_action"], dtype=np.float32),
                )
            ),
            "same_smoothed_action": bool(
                np.array_equal(
                    np.asarray(result_a["smoothed_action"], dtype=np.float32),
                    np.asarray(result_b["smoothed_action"], dtype=np.float32),
                )
            ),
            "same_requested_insert": bool(
                math.isclose(
                    float(result_a["requested_insert_m"]),
                    float(result_b["requested_insert_m"]),
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
            ),
            "A_reproduced_known_step1186_baseline": baseline_reproduced,
            "A_substep_contacts": a["substep_contacts"],
            "B_substep_contacts": b["substep_contacts"],
            "A_substep_constraint_rows": a["substep_constraint_rows"],
            "B_substep_constraint_rows": b["substep_constraint_rows"],
            "A_body_clearance_mm": a["substep_body_clearance_mm"],
            "B_body_clearance_mm": b["substep_body_clearance_mm"],
            "B_hard_candidate_patches": b[
                "substep_hard_candidate_patches"
            ],
            "B_hard_active_patches": b["substep_hard_active_patches"],
            "B_hard_dropped_patches": b["substep_hard_dropped_patches"],
            "B_hard_selected_indices": b["substep_hard_selected_indices"],
            "B_hard_selected_clearances_mm": b[
                "substep_hard_selected_clearances_mm"
            ],
            "B_hard_contact_count": b["substep_hard_contact_count"],
            "A_max_body_penetration_mm": a["max_body_penetration_mm"],
            "B_max_body_penetration_mm": b["max_body_penetration_mm"],
            "body_penetration_reduction_mm": float(
                a["max_body_penetration_mm"]
                - b["max_body_penetration_mm"]
            ),
            "A_max_beam_correction_mm": a["max_beam_correction_mm"],
            "B_max_beam_correction_mm": b["max_beam_correction_mm"],
            "B_removed_first_substep_penetration": bool(
                a["substep_body_clearance_mm"][0] < 0.0
                and b["substep_body_clearance_mm"][0] >= 0.0
            ),
            "both_finite": bool(a["finite"] and b["finite"]),
        }

    combined = {
        "test": "V15.2-C same-state SDF native hard-contact wall",
        "diagnostic_only": True,
        "training_started": False,
        "direct_position_projection": False,
        "vessel_point_collision_enabled": False,
        "vessel_line_collision_enabled": False,
        "checkpoint": str(checkpoint),
        "seed": int(args.seed),
        "target_step": int(args.target_step),
        "fork_after_step": prefix_steps,
        "physics_dt_s": 0.005,
        "physics_substeps": 2,
        "rl_control_period_s": 0.01,
        "known_baseline": KNOWN_BASELINE,
        "collision_models": models,
        "fork_fingerprint_sha256": fork_fingerprint,
        "fork_capture": fork_capture,
        "A": result_a,
        "B": result_b,
        "child_status": int(child_status),
        "comparison": comparison,
        "runtime_s": float(time.perf_counter() - started),
        "files": {
            "combined": str(output),
            "A": str(output_a),
            "B": str(output_b),
        },
    }
    output.write_text(json.dumps(combined, indent=2, sort_keys=True) + "\n")
    print(json.dumps(combined, sort_keys=True), flush=True)

    try:
        env.close()
    except Exception:
        pass

    if child_status != 0 or "error" in result_b:
        raise RuntimeError(
            "SDF hard-wall branch failed; inspect the B JSON for details"
        )

    os._exit(0)


if __name__ == "__main__":
    main()
