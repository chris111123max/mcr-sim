"""Targeted step 775-776 diagnostic for SDF unilateral residual penetration.

Canonical diagnostic location:
    testing/py/diagnostics/

Protocol:
    1. Run GenericConstraintSolver + unilateral OFF from the checkpoint/seed
       to collect the deterministic raw-action prefix.
    2. Restart from the same seed with GenericConstraintSolver + unilateral ON.
    3. Replay exactly the same float32 raw actions.
    4. Capture only the requested RL-step window (default 775-776), both
       5 ms physics substeps.

The diagnostic compares the *actual unilateral rows used by the solver* with
the dense post-solve SDF worst point.  By default it preserves the production
node+midpoint sampler.  With --dense-adaptive it installs a test-only
voxel-derived edge sampler and a matching finer active-row dedup scale; reward,
observation, action, substep count, and production/training defaults remain
unchanged.

For every active row it records:
- indices/weights/sample kind;
- source clearance used when the row was created;
- stored inward normal and anchor;
- reconstructed pre-solve sample position;
- post-solve sample position;
- post-solve linear gap g = dot(sample-anchor, inward);
- post-solve true SDF clearance;
- angle between the stored inward normal and the post-solve SDF inward normal.

For the dense body worst point it records:
- position and true body clearance;
- distance to every active row sample;
- the nearest active row;
- the worst point's gap against every active tangent plane.

This separates:
- sample coverage mismatch;
- stale/local tangent-plane linearization;
- row dropping/capping;
- solver/mapping failures.
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

import numpy as np

PYTHON_ROOT = Path(__file__).resolve().parents[3]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from mcr_sim.distributed import DistributedPPO
from mcr_sim.mcr_rl_env import EnvType, MCREnv
from mcr_sim.rl_core.base import RenderMode
from mcr_sim.sdf_hard_constraint import sample_sdf_clearance_and_outward
from sdf_dense_adaptive_sampling import (
    dense_sampler_snapshot,
    install_dense_adaptive_sampling,
)


def _as_array(data, dtype=np.float64):
    try:
        return np.asarray(data.array(), dtype=dtype).copy()
    except Exception:
        return np.asarray(data.value, dtype=dtype).copy()


def _data_array(obj, name: str, dtype=np.float64):
    data = getattr(obj, name)
    try:
        return np.asarray(data.array(), dtype=dtype).copy()
    except Exception:
        return np.asarray(data.value, dtype=dtype).copy()


def _constraint_rows(dofs) -> int:
    try:
        value = dofs.constraint.value
    except Exception:
        return -1
    return len([line for line in str(value).splitlines() if line.strip()])


def _solver_object(env):
    for obj in env._sofa_root_node.objects:
        if obj.getClassName() in ("LCPConstraintSolver", "GenericConstraintSolver"):
            return obj
    return None


def _solver_snapshot(solver):
    if solver is None:
        return {"class": None}
    result = {"class": solver.getClassName()}
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
            if arr.size != 1:
                continue
            scalar = arr.reshape(-1)[0]
            if np.issubdtype(arr.dtype, np.number):
                scalar = float(scalar)
                result[name] = scalar if math.isfinite(scalar) else None
            else:
                result[name] = str(scalar)
        except Exception:
            pass
    return result


def _sha256_actions(actions) -> str:
    if not actions:
        return hashlib.sha256(b"").hexdigest()
    arr = np.asarray(actions, dtype=np.float32).reshape((-1, 3))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _create_env(*, unilateral_enabled: bool, max_steps: int, wall_stiffness: float):
    os.environ["MCR_CONSTRAINT_SOLVER"] = "generic"
    os.environ["MCR_SOFA_DT"] = "0.005"

    return MCREnv(
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
            "sdf_wall_stiffness_n_per_m": float(wall_stiffness),
            "diagnostic_intersection_method": "local_min_distance",
            "use_vessel_line_point_collision": False,
            "use_vessel_point_collision": False,
            "use_vessel_line_collision": False,
            "sdf_hard_constraint_construct": False,
            "sdf_hard_constraint_enabled": False,
            "sdf_unilateral_constraint_construct": True,
            "sdf_unilateral_constraint_enabled": bool(unilateral_enabled),
        },
        env_type=EnvType.AORTIC,
        render_mode=RenderMode.NONE,
        max_episode_steps=max(int(max_steps), 2048),
        time_step=0.005,
        frame_skip=1,
        physics_substeps=2,
    )


def _validate_env(env, expected_unilateral: bool):
    solver = _solver_object(env)
    if solver is None or solver.getClassName() != "GenericConstraintSolver":
        raise RuntimeError(
            "Expected GenericConstraintSolver, got "
            + repr(_solver_snapshot(solver))
        )

    unilateral = env.scene_creation_result.get(
        "sdf_unilateral_constraint_controller"
    )
    if unilateral is None:
        raise RuntimeError("SDF unilateral controller is missing")
    if bool(unilateral.enabled) != bool(expected_unilateral):
        raise RuntimeError(
            f"Unilateral enabled mismatch: controller={unilateral.enabled}, "
            f"expected={expected_unilateral}"
        )

    vessel_node = env.scene_creation_result["mcr_environment"].CollisionModel
    catheter_node = env.mcr_controller_sofa.instrument.InstrumentCombined.getChild(
        "mcr_collis"
    )
    vessel_classes = [obj.getClassName() for obj in vessel_node.objects]
    catheter_classes = [obj.getClassName() for obj in catheter_node.objects]
    if (
        vessel_classes.count("TriangleCollisionModel") != 1
        or vessel_classes.count("PointCollisionModel") != 0
        or vessel_classes.count("LineCollisionModel") != 0
        or catheter_classes.count("PointCollisionModel") < 1
        or catheter_classes.count("LineCollisionModel") < 1
    ):
        raise RuntimeError(
            "Unexpected collision configuration: "
            f"vessel={vessel_classes}, catheter={catheter_classes}"
        )

    return solver, unilateral


def _collect_reference_actions(
    *,
    model,
    seed: int,
    max_step: int,
    wall_stiffness: float,
    progress_every: int,
):
    env = _create_env(
        unilateral_enabled=False,
        max_steps=max_step,
        wall_stiffness=wall_stiffness,
    )
    observation, _ = env.reset(seed=int(seed))
    solver, _ = _validate_env(env, expected_unilateral=False)

    actions = []
    terminal_reason = None
    started = time.perf_counter()
    try:
        for step in range(1, int(max_step) + 1):
            raw_action, _ = model.predict(observation, deterministic=True)
            raw_action = np.asarray(raw_action, dtype=np.float32).reshape(3)
            actions.append(raw_action.copy())
            observation, _, terminated, truncated, info = env.step(raw_action)

            if progress_every > 0 and (
                step == 1 or step % progress_every == 0 or step == max_step
            ):
                print(
                    f"[TARGETED_A_PREFIX] step={step}/{max_step}",
                    flush=True,
                )

            if terminated or truncated:
                terminal_reason = str(
                    info.get("terminal_reason")
                    or ("terminated" if terminated else "truncated")
                )
                break
    finally:
        solver_final = _solver_snapshot(solver)
        try:
            env.close()
        except Exception:
            pass

    if len(actions) < int(max_step):
        raise RuntimeError(
            f"Reference A ended at {len(actions)} < requested {max_step}; "
            f"terminal_reason={terminal_reason}"
        )

    return {
        "actions": actions,
        "sha256": _sha256_actions(actions),
        "solver": solver_final,
        "wall_s": float(time.perf_counter() - started),
    }


def _query_clearance(unilateral, points):
    points = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    clearance, outward, valid = sample_sdf_clearance_and_outward(
        points,
        sdf_grid=unilateral.sdf_grid,
        asset_T_env_sim=unilateral.asset_T_env_sim,
        asset_offset_sim=unilateral.asset_offset_sim,
        asset_source_to_sim_scale=unilateral.asset_source_to_sim_scale,
        catheter_radius_m=unilateral.catheter_radius_m,
    )
    return (
        np.asarray(clearance, dtype=np.float64).reshape(-1),
        np.asarray(outward, dtype=np.float64).reshape((-1, 3)),
        np.asarray(valid, dtype=bool).reshape(-1),
    )


def _angle_deg(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(3)
    b = np.asarray(b, dtype=np.float64).reshape(3)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-12 or nb <= 1e-12:
        return None
    cosine = float(np.dot(a, b) / (na * nb))
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _active_row_snapshot(
    *,
    unilateral,
    collision_positions,
    dense_worst_point,
):
    constraint = unilateral.constraint

    indices0 = _data_array(constraint, "indices0", np.int64).reshape(-1)
    indices1 = _data_array(constraint, "indices1", np.int64).reshape(-1)
    weights0 = _data_array(constraint, "weights0", np.float64).reshape(-1)
    weights1 = _data_array(constraint, "weights1", np.float64).reshape(-1)
    normals = _data_array(constraint, "normals", np.float64).reshape((-1, 3))
    anchors = _data_array(constraint, "anchors", np.float64).reshape((-1, 3))
    source_clearances = _data_array(
        constraint, "sourceClearances", np.float64
    ).reshape(-1)

    count = min(
        len(indices0),
        len(indices1),
        len(weights0),
        len(weights1),
        len(normals),
        len(anchors),
        len(source_clearances),
    )
    diagnostics = unilateral.get_diagnostics()
    kinds = list(diagnostics.get("selected_sample_kinds", []))

    collision_positions = np.asarray(
        collision_positions, dtype=np.float64
    ).reshape((-1, 3))
    dense_worst_point = np.asarray(
        dense_worst_point, dtype=np.float64
    ).reshape(3)

    rows = []
    post_samples = []

    for row in range(count):
        i0 = int(indices0[row])
        i1 = int(indices1[row])
        w0 = float(weights0[row])
        w1 = float(weights1[row])
        normal = np.asarray(normals[row], dtype=np.float64).reshape(3)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm > 1e-12:
            normal = normal / normal_norm

        anchor = np.asarray(anchors[row], dtype=np.float64).reshape(3)
        source_clearance = float(source_clearances[row])

        pre_sample = anchor + normal * source_clearance
        post_sample = (
            collision_positions[i0] * w0
            + collision_positions[i1] * w1
        )
        post_samples.append(post_sample)

        true_clearance, outward_now, valid_now = _query_clearance(
            unilateral, post_sample.reshape(1, 3)
        )
        true_clearance_mm = (
            float(true_clearance[0] * 1000.0)
            if bool(valid_now[0]) and np.isfinite(true_clearance[0])
            else None
        )
        inward_now = -outward_now[0]
        linear_gap_m = float(np.dot(post_sample - anchor, normal))
        dense_gap_m = float(np.dot(dense_worst_point - anchor, normal))

        rows.append(
            {
                "row": int(row),
                "kind": str(kinds[row]) if row < len(kinds) else None,
                "index0": i0,
                "index1": i1,
                "weight0": w0,
                "weight1": w1,
                "source_clearance_mm": float(source_clearance * 1000.0),
                "normal_inward": normal.tolist(),
                "anchor_m": anchor.tolist(),
                "pre_sample_m": pre_sample.tolist(),
                "post_sample_m": post_sample.tolist(),
                "sample_displacement_mm": float(
                    np.linalg.norm(post_sample - pre_sample) * 1000.0
                ),
                "linear_gap_final_mm": float(linear_gap_m * 1000.0),
                "true_sdf_clearance_final_mm": true_clearance_mm,
                "post_sdf_inward": (
                    inward_now.tolist() if bool(valid_now[0]) else None
                ),
                "normal_change_deg": (
                    _angle_deg(normal, inward_now)
                    if bool(valid_now[0])
                    else None
                ),
                "distance_to_dense_worst_mm": float(
                    np.linalg.norm(post_sample - dense_worst_point) * 1000.0
                ),
                "dense_worst_gap_against_this_plane_mm": float(
                    dense_gap_m * 1000.0
                ),
            }
        )

    if post_samples:
        distances = np.linalg.norm(
            np.asarray(post_samples, dtype=np.float64)
            - dense_worst_point.reshape(1, 3),
            axis=1,
        )
        nearest_index = int(np.argmin(distances))
        nearest_distance_mm = float(distances[nearest_index] * 1000.0)
    else:
        nearest_index = None
        nearest_distance_mm = None

    return {
        "data_row_count": int(count),
        "cpp_active_count": int(
            diagnostics.get("cpp_active_count", -1)
        ),
        "python_active_count": int(
            diagnostics.get("active_constraints", -1)
        ),
        "sample_count": int(diagnostics.get("sample_count", -1)),
        "valid_sample_count": int(
            diagnostics.get("valid_samples", -1)
        ),
        "candidate_count": int(
            diagnostics.get("candidate_constraints", -1)
        ),
        "dropped_count": int(
            diagnostics.get("dropped_constraints", -1)
        ),
        "nearest_active_row_to_dense_worst": nearest_index,
        "nearest_active_distance_mm": nearest_distance_mm,
        "rows": rows,
    }


def _run_targeted_b(
    *,
    actions,
    reference_sha256: str,
    seed: int,
    capture_start: int,
    capture_end: int,
    wall_stiffness: float,
    progress_every: int,
    dense_adaptive: bool = False,
    dense_max_constraints: int = 128,
    dense_min_separation_fraction: float = 0.25,
):
    env = _create_env(
        unilateral_enabled=True,
        max_steps=capture_end,
        wall_stiffness=wall_stiffness,
    )
    observation, _ = env.reset(seed=int(seed))
    solver, unilateral = _validate_env(env, expected_unilateral=True)

    dense_config = None
    if dense_adaptive:
        dense_config = install_dense_adaptive_sampling(
            unilateral,
            sample_step_fraction=float(env.sdf_sample_step_fraction),
            max_constraints=int(dense_max_constraints),
            min_separation_fraction_of_step=float(
                dense_min_separation_fraction
            ),
        )

    controller = env.mcr_controller_sofa
    instrument = controller.instrument.InstrumentCombined
    collision_node = instrument.getChild("mcr_collis")
    collision_dofs = collision_node.getObject("CollisionDOFs")
    beam_dofs = instrument.getObject("DOFs")

    original_animate = env.sofa_simulation.animate
    current_rl_step = 0
    current_raw_action = np.zeros(3, dtype=np.float32)
    substep_counts = {}
    captures = []
    replayed_actions = []

    def traced_animate(root, dt):
        result = original_animate(root, dt)

        substep = int(substep_counts.get(current_rl_step, 0) + 1)
        substep_counts[current_rl_step] = substep

        if not (capture_start <= current_rl_step <= capture_end):
            return result

        tip = np.asarray(
            controller.get_pos_quat_catheter_tip()[:3],
            dtype=np.float64,
        ).copy()

        # Force a post-solve dense safety recomputation.
        env._sdf_geometry_cache_step = -1
        env._update_sdf_safety_state(
            tip,
            advance_failure_counters=False,
        )

        beam_pos = _as_array(beam_dofs.position)[:, :3]
        beam_free = _as_array(beam_dofs.free_position)[:, :3]
        collision_pos = _as_array(collision_dofs.position)[:, :3]
        collision_free = _as_array(collision_dofs.free_position)[:, :3]

        beam_correction_mm = float(
            np.max(
                np.linalg.norm(
                    (beam_pos - beam_free) * 1000.0,
                    axis=1,
                )
            )
        )
        collision_correction_mm = float(
            np.max(
                np.linalg.norm(
                    (collision_pos - collision_free) * 1000.0,
                    axis=1,
                )
            )
        )

        dense_worst_point = np.asarray(
            env.current_sdf_worst_point_sim,
            dtype=np.float64,
        ).reshape(3)
        dense_clearance_mm = float(
            env.current_sdf_body_min_surface_clearance * 1000.0
        )

        row_snapshot = _active_row_snapshot(
            unilateral=unilateral,
            collision_positions=collision_pos,
            dense_worst_point=dense_worst_point,
        )

        contacts = [
            obj
            for obj in collision_node.objects
            if obj.getClassName() == "FrictionContact"
        ]

        captures.append(
            {
                "rl_step": int(current_rl_step),
                "substep": substep,
                "root_time_s": float(root.getTime()),
                "raw_action": current_raw_action.astype(float).tolist(),
                "smoothed_action": np.asarray(
                    env._last_smoothed_action,
                    dtype=np.float64,
                ).reshape(-1).tolist(),
                "requested_insert_mm": float(
                    getattr(env, "current_raw_insert", 0.0) * 1000.0
                ),
                "effective_insert_mm": float(
                    getattr(env, "current_effective_insert", 0.0) * 1000.0
                ),
                "dense_worst_point_m": dense_worst_point.tolist(),
                "dense_body_clearance_mm": dense_clearance_mm,
                "dense_body_penetration_mm": max(0.0, -dense_clearance_mm),
                "tip_clearance_mm": float(
                    env.current_sdf_surface_clearance * 1000.0
                ),
                "friction_contact_count": len(contacts),
                "collision_constraint_rows": int(
                    _constraint_rows(collision_dofs)
                ),
                "beam_constraint_rows": int(_constraint_rows(beam_dofs)),
                "collision_max_correction_mm": collision_correction_mm,
                "beam_max_correction_mm": beam_correction_mm,
                "solver": _solver_snapshot(solver),
                "unilateral": row_snapshot,
            }
        )

        return result

    env.sofa_simulation.animate = traced_animate

    terminal_reason = None
    started = time.perf_counter()
    try:
        for step in range(1, int(capture_end) + 1):
            current_rl_step = step
            current_raw_action = np.asarray(
                actions[step - 1], dtype=np.float32
            ).reshape(3)
            replayed_actions.append(current_raw_action.copy())

            observation, _, terminated, truncated, info = env.step(
                current_raw_action
            )

            if progress_every > 0 and (
                step == 1
                or step % progress_every == 0
                or step == capture_end
            ):
                print(
                    f"[TARGETED_B_REPLAY] step={step}/{capture_end}",
                    flush=True,
                )

            if terminated or truncated:
                terminal_reason = str(
                    info.get("terminal_reason")
                    or ("terminated" if terminated else "truncated")
                )
                break
    finally:
        env.sofa_simulation.animate = original_animate
        solver_final = _solver_snapshot(solver)
        sampler_state = (
            dense_sampler_snapshot(
                unilateral,
                positions=_as_array(collision_dofs.position)[:, :3],
            )
            if dense_adaptive
            else {
                "installed": False,
                "config": None,
            }
        )
        try:
            env.close()
        except Exception:
            pass

    replay_sha256 = _sha256_actions(replayed_actions)
    if len(replayed_actions) < int(capture_end):
        raise RuntimeError(
            f"B replay ended at {len(replayed_actions)} < {capture_end}; "
            f"terminal_reason={terminal_reason}"
        )
    if replay_sha256 != reference_sha256:
        raise RuntimeError(
            "B replay action hash differs from A reference prefix: "
            f"A={reference_sha256}, B={replay_sha256}"
        )

    return {
        "solver": solver_final,
        "action_sha256": replay_sha256,
        "sampling_mode": (
            "dense_adaptive" if dense_adaptive else "node_plus_midpoint"
        ),
        "dense_sampling": (
            dense_config.to_dict() if dense_config is not None else None
        ),
        "dense_sampler_state": sampler_state,
        "captured_substeps": captures,
        "capture_count": len(captures),
        "terminal_reason": terminal_reason,
        "wall_s": float(time.perf_counter() - started),
    }


def _summarize(captures):
    summary = []
    for item in captures:
        u = item["unilateral"]
        nearest_index = u["nearest_active_row_to_dense_worst"]
        nearest = (
            u["rows"][nearest_index]
            if nearest_index is not None
            and nearest_index < len(u["rows"])
            else None
        )
        summary.append(
            {
                "rl_step": item["rl_step"],
                "substep": item["substep"],
                "dense_body_clearance_mm": item[
                    "dense_body_clearance_mm"
                ],
                "dense_body_penetration_mm": item[
                    "dense_body_penetration_mm"
                ],
                "python_active": u["python_active_count"],
                "cpp_active": u["cpp_active_count"],
                "sample_count": u["sample_count"],
                "valid_sample_count": u["valid_sample_count"],
                "candidate": u["candidate_count"],
                "dropped": u["dropped_count"],
                "nearest_active_distance_mm": u[
                    "nearest_active_distance_mm"
                ],
                "nearest_active_linear_gap_final_mm": (
                    nearest["linear_gap_final_mm"]
                    if nearest is not None
                    else None
                ),
                "nearest_active_true_sdf_clearance_final_mm": (
                    nearest["true_sdf_clearance_final_mm"]
                    if nearest is not None
                    else None
                ),
                "nearest_active_normal_change_deg": (
                    nearest["normal_change_deg"]
                    if nearest is not None
                    else None
                ),
                "dense_worst_gap_against_nearest_plane_mm": (
                    nearest["dense_worst_gap_against_this_plane_mm"]
                    if nearest is not None
                    else None
                ),
                "solver": item["solver"],
                "beam_max_correction_mm": item[
                    "beam_max_correction_mm"
                ],
                "collision_max_correction_mm": item[
                    "collision_max_correction_mm"
                ],
            }
        )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=15204)
    parser.add_argument("--capture-start", type=int, default=775)
    parser.add_argument("--capture-end", type=int, default=776)
    parser.add_argument("--wall-stiffness", type=float, default=10.0)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--dense-adaptive",
        action="store_true",
        help=(
            "Diagnostic-only: replace node+midpoint unilateral sampling with "
            "voxel-derived dense edge sampling."
        ),
    )
    parser.add_argument(
        "--dense-max-constraints",
        type=int,
        default=128,
        help="Diagnostic-only active-row cap for dense sampling.",
    )
    parser.add_argument(
        "--dense-min-separation-fraction",
        type=float,
        default=0.25,
        help=(
            "Diagnostic-only active-row spatial dedup distance as a fraction "
            "of the dense max sample step."
        ),
    )
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if args.capture_start < 1:
        raise ValueError("--capture-start must be >= 1")
    if args.capture_end < args.capture_start:
        raise ValueError("--capture-end must be >= --capture-start")

    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else (
            checkpoint.parent.parent
            / "diagnostics"
            / (
                "v15_2c_sdf_unilateral_targeted_"
                + ("dense_" if args.dense_adaptive else "")
                + f"step{args.capture_start}_{args.capture_end}_seed{args.seed}.json"
            )
        ).resolve()
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    model = DistributedPPO.load(str(checkpoint), device="cpu")
    model.policy.set_training_mode(False)

    print(
        "[TARGETED] Collecting Generic+OFF reference raw actions "
        f"through step {args.capture_end}",
        flush=True,
    )
    reference = _collect_reference_actions(
        model=model,
        seed=args.seed,
        max_step=args.capture_end,
        wall_stiffness=args.wall_stiffness,
        progress_every=args.progress_every,
    )

    print(
        "[TARGETED] Replaying identical actions with Generic+ON and "
        f"capturing steps {args.capture_start}-{args.capture_end}",
        flush=True,
    )
    result_b = _run_targeted_b(
        actions=reference["actions"],
        reference_sha256=reference["sha256"],
        seed=args.seed,
        capture_start=args.capture_start,
        capture_end=args.capture_end,
        wall_stiffness=args.wall_stiffness,
        progress_every=args.progress_every,
        dense_adaptive=bool(args.dense_adaptive),
        dense_max_constraints=int(args.dense_max_constraints),
        dense_min_separation_fraction=float(
            args.dense_min_separation_fraction
        ),
    )

    combined = {
        "test": (
            "SDF unilateral targeted residual-penetration diagnostic: "
            f"{result_b['sampling_mode']} active rows vs dense SDF worst point"
        ),
        "diagnostic_only": True,
        "training_started": False,
        "checkpoint": str(checkpoint),
        "seed": int(args.seed),
        "capture_start": int(args.capture_start),
        "capture_end": int(args.capture_end),
        "physics_dt_s": 0.005,
        "physics_substeps": 2,
        "rl_control_period_s": 0.01,
        "wall_stiffness_n_per_m": float(args.wall_stiffness),
        "sampling_mode": result_b["sampling_mode"],
        "dense_sampling": result_b["dense_sampling"],
        "A_reference": {
            "solver": reference["solver"],
            "action_count": len(reference["actions"]),
            "action_sha256": reference["sha256"],
            "wall_s": reference["wall_s"],
        },
        "B_targeted": result_b,
        "actions_identical": (
            reference["sha256"] == result_b["action_sha256"]
        ),
        "summary": _summarize(result_b["captured_substeps"]),
        "output": str(output),
    }

    output.write_text(
        json.dumps(combined, indent=2, sort_keys=True) + "\n"
    )

    print(
        json.dumps(
            {
                "test": combined["test"],
                "actions_identical": combined["actions_identical"],
                "summary": combined["summary"],
                "output": str(output),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
