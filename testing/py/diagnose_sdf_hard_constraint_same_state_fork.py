"""Same-state causal A/B for SDF-derived hard contact at V15.2-C step 1186.

A and B share the exact catheter state through step 1185.

A:
    original vessel Triangle collision only
    + original catheter Line/Point collision
    + existing soft SDF physics wall
    + SDF hard-contact geometry constructed but disabled/parked

B:
    identical state/action/insertion
    + enable SDF-derived local tangent Triangle patches immediately before
      step 1186

Important:
- vessel PointCollisionModel is forbidden in both branches;
- vessel LineCollisionModel is forbidden in both branches;
- no position projection, rollback, teleport or RL safety logic;
- the hard wall is enforced only through the existing
  FrictionContactConstraint + LCPConstraintSolver pipeline;
- only one RL action is evaluated after the fork (2 x 5 ms).
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


def _count_constraint_rows(dofs) -> int:
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
        if arr.size:
            scalar = arr[0]
            if isinstance(scalar, (np.integer, int)):
                return int(scalar)
            if isinstance(scalar, (np.floating, float)):
                value = float(scalar)
                return value if math.isfinite(value) else default
            return str(scalar)
    except Exception:
        pass
    return default


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


def _collision_summary(env: MCREnv) -> dict:
    vessel_env = env.scene_creation_result["mcr_environment"]
    vessel_node = vessel_env.CollisionModel
    catheter_node = env.mcr_controller_sofa.instrument.InstrumentCombined.getChild(
        "mcr_collis"
    )
    hard = env.scene_creation_result.get("sdf_hard_constraint_controller")

    vessel_classes = [obj.getClassName() for obj in vessel_node.objects]
    catheter_classes = [obj.getClassName() for obj in catheter_node.objects]

    result = {
        "vessel_classes": vessel_classes,
        "catheter_classes": catheter_classes,
        "vessel_triangle_count": vessel_classes.count("TriangleCollisionModel"),
        "vessel_point_count": vessel_classes.count("PointCollisionModel"),
        "vessel_line_count": vessel_classes.count("LineCollisionModel"),
        "catheter_point_count": catheter_classes.count("PointCollisionModel"),
        "catheter_line_count": catheter_classes.count("LineCollisionModel"),
        "vessel_triangle_group": _data_scalar(
            vessel_env.TriangleCollisionModel, "group"
        ),
    }

    if hard is not None:
        hard_classes = [obj.getClassName() for obj in hard.collision_node.objects]
        result.update(
            {
                "hard_collision_classes": hard_classes,
                "hard_triangle_count": hard_classes.count("TriangleCollisionModel"),
                "hard_triangle_group": _data_scalar(hard.collision_model, "group"),
                "hard_triangle_both_side": _data_scalar(
                    hard.collision_model, "bothSide"
                ),
            }
        )
    return result


def _friction_contact_count(node) -> int:
    try:
        return sum(
            obj.getClassName() == "FrictionContact" for obj in node.objects
        )
    except Exception:
        return 0


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
    coll_pos = _as_array(collision_dofs.position)
    coll_free = _as_array(collision_dofs.free_position)

    catheter_contacts = _friction_contact_count(collision_node)
    hard_node_contacts = (
        _friction_contact_count(hard.collision_node) if hard is not None else 0
    )
    hard_diag = hard.get_diagnostics() if hard is not None else {}

    body_clearance_mm = float(
        env.current_sdf_body_min_surface_clearance * 1000.0
    )
    tip_clearance_mm = float(env.current_sdf_surface_clearance * 1000.0)

    beam_correction = np.linalg.norm(
        (beam_pos[:, :3] - beam_free[:, :3]) * 1000.0, axis=1
    )
    coll_correction = np.linalg.norm(
        (coll_pos[:, :3] - coll_free[:, :3]) * 1000.0, axis=1
    )

    return {
        "label": label,
        "root_time_s": float(env._sofa_root_node.getTime()),
        "root_dt_s": float(env._sofa_root_node.getDt()),
        "state_fingerprint_sha256": _state_fingerprint(env),
        "tip_mm": (tip * 1000.0).tolist(),
        "tip_clearance_mm": tip_clearance_mm,
        "body_clearance_mm": body_clearance_mm,
        "body_penetration_mm": max(0.0, -body_clearance_mm),
        "catheter_friction_contact_count": int(catheter_contacts),
        "hard_node_friction_contact_count": int(hard_node_contacts),
        "collision_constraint_rows": int(_count_constraint_rows(collision_dofs)),
        "beam_constraint_rows": int(_count_constraint_rows(beam_dofs)),
        "beam_max_correction_mm": float(np.max(beam_correction)),
        "beam_tip_correction_mm": float(beam_correction[-1]),
        "collision_max_correction_mm": float(np.max(coll_correction)),
        "collision_tip_correction_mm": float(coll_correction[-1]),
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
    controller = env.mcr_controller_sofa
    hard = env.scene_creation_result.get("sdf_hard_constraint_controller")
    if hard is None:
        raise RuntimeError("SDF hard constraint controller was not constructed")

    pre_enable_fingerprint = _state_fingerprint(env)
    if pre_enable_fingerprint != fork_fingerprint:
        raise RuntimeError(f"{branch_name}: fork state changed before branch")

    hard.set_enabled(bool(enable_hard_wall))

    post_enable_fingerprint = _state_fingerprint(env)
    if post_enable_fingerprint != fork_fingerprint:
        raise RuntimeError(
            f"{branch_name}: enabling hard wall changed catheter state before animate"
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
                if row["catheter_friction_contact_count"] == 0
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
                row["tip_clearance_mm"],
                row["body_clearance_mm"],
                row["beam_max_correction_mm"],
                row["collision_max_correction_mm"],
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
                int(r["catheter_friction_contact_count"]) for r in rows
            ],
            "substep_constraint_rows": [
                int(r["collision_constraint_rows"]) for r in rows
            ],
            "substep_body_clearance_mm": [
                float(r["body_clearance_mm"]) for r in rows
            ],
            "substep_tip_clearance_mm": [
                float(r["tip_clearance_mm"]) for r in rows
            ],
            "hard_active_patches": [
                int(r["hard_wall"].get("active_patches", 0)) for r in rows
            ],
            "hard_candidate_patches": [
                int(r["hard_wall"].get("candidate_patches", 0)) for r in rows
            ],
            "hard_wall_contacts": [
                r["hard_wall"].get("hard_wall_contact_count") for r in rows
            ],
            "max_body_penetration_mm": float(
                max(r["body_penetration_mm"] for r in rows)
            ),
            "max_contact_free_body_penetration_mm": float(
                max(r["contact_free_body_penetration_mm"] for r in rows)
            ),
            "max_beam_correction_mm": float(
                max(r["beam_max_correction_mm"] for r in rows)
            ),
            "finite": bool(np.all(np.isfinite(numeric))),
        },
    }


def _default_output(checkpoint: Path, seed: int, target_step: int) -> Path:
    run_dir = checkpoint.parent.parent
    return (
        run_dir
        / "diagnostics"
        / f"v15_2c_sdf_hard_same_state_step{target_step}_seed{seed}.json"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=15204)
    parser.add_argument("--target-step", type=int, default=1186)
    parser.add_argument("--wall-stiffness", type=float, default=10.0)
    args = parser.parse_args()

    if args.target_step < 2:
        raise ValueError("--target-step must be >= 2")

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
            # Preserve the current V15.2-C soft SDF wall in both branches.
            "sdf_physics_wall_enabled": True,
            "sdf_wall_stiffness_n_per_m": float(args.wall_stiffness),
            "diagnostic_intersection_method": "local_min_distance",
            # Explicitly exclude the previous vessel Point/Line experiment.
            "use_vessel_line_point_collision": False,
            "use_vessel_point_collision": False,
            "use_vessel_line_collision": False,
            # Construct the hard-wall geometry from the beginning, but keep it
            # parked/disabled through the shared baseline prefix.
            "sdf_hard_constraint_construct": True,
            "sdf_hard_constraint_enabled": False,
            "sdf_hard_constraint_max_active_patches": 12,
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
        or models.get("hard_triangle_count") != 1
    ):
        raise RuntimeError(
            "Expected vessel Triangle-only plus one separate SDF hard Triangle model; "
            f"got {models}"
        )

    if models["catheter_point_count"] < 1 or models["catheter_line_count"] < 1:
        raise RuntimeError(f"Original catheter Line+Point collision is missing: {models}")

    hard = env.scene_creation_result.get("sdf_hard_constraint_controller")
    if hard is None or hard.enabled:
        raise RuntimeError("Hard-wall controller must exist but be disabled before fork")

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

    fork_fingerprint = _state_fingerprint(env)
    fork_capture = _capture(env, f"fork_after_step_{prefix_steps}")
    raw_action, _ = model.predict(observation, deterministic=True)
    raw_action = np.asarray(raw_action, dtype=np.float32).reshape(3)

    child_pid = os.fork()
    if child_pid == 0:
        try:
            result_b = _run_target_action(
                env,
                raw_action,
                "B_sdf_hard",
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
        "A_baseline",
        False,
        fork_fingerprint,
    )
    output_a.write_text(json.dumps(result_a, indent=2, sort_keys=True) + "\n")

    _, child_status = os.waitpid(child_pid, 0)
    result_b = json.loads(output_b.read_text())

    comparison = None
    if child_status == 0 and "error" not in result_b:
        a = result_a["summary"]
        b = result_b["summary"]
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
            "A_substep_contacts": a["substep_contacts"],
            "B_substep_contacts": b["substep_contacts"],
            "A_substep_constraint_rows": a["substep_constraint_rows"],
            "B_substep_constraint_rows": b["substep_constraint_rows"],
            "A_body_clearance_mm": a["substep_body_clearance_mm"],
            "B_body_clearance_mm": b["substep_body_clearance_mm"],
            "B_hard_active_patches": b["hard_active_patches"],
            "B_hard_candidate_patches": b["hard_candidate_patches"],
            "B_hard_wall_contacts": b["hard_wall_contacts"],
            "A_max_body_penetration_mm": a["max_body_penetration_mm"],
            "B_max_body_penetration_mm": b["max_body_penetration_mm"],
            "penetration_reduction_mm": float(
                a["max_body_penetration_mm"] - b["max_body_penetration_mm"]
            ),
            "A_max_beam_correction_mm": a["max_beam_correction_mm"],
            "B_max_beam_correction_mm": b["max_beam_correction_mm"],
            "both_finite": bool(a["finite"] and b["finite"]),
            "known_baseline_reference": KNOWN_BASELINE,
            "A_first_substep_clearance_error_vs_known_mm": float(
                a["substep_body_clearance_mm"][0]
                - KNOWN_BASELINE["first_substep_body_clearance_mm"]
            ),
        }

    combined = {
        "test": "V15.2-C same-state SDF hard-contact causal fork",
        "diagnostic_only": True,
        "training_started": False,
        "production_default_changed": False,
        "checkpoint": str(checkpoint),
        "seed": int(args.seed),
        "target_step": int(args.target_step),
        "fork_after_step": prefix_steps,
        "physics_dt_s": 0.005,
        "physics_substeps": 2,
        "rl_control_period_s": 0.01,
        "collision_models": models,
        "fork_fingerprint_sha256": fork_fingerprint,
        "fork_capture": fork_capture,
        "A": result_a,
        "B": result_b,
        "comparison": comparison,
        "child_status": int(child_status),
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
            "SDF hard-contact branch failed; inspect the B JSON for details"
        )

    os._exit(0)


if __name__ == "__main__":
    main()
