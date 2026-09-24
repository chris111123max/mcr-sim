"""Full-horizon A/B diagnostic for the true SDF unilateral constraint.

Canonical location for new diagnostics:
    testing/py/diagnostics/

Experiment:
    A = GenericConstraintSolver + SDF unilateral OFF
    B = GenericConstraintSolver + SDF unilateral ON

Both use:
    - B02 / target_04 / seed 15204 by default
    - V15.2-C timing: 1 RL action = 2 x 5 ms SOFA substeps
    - catheter Line + Point collision
    - vessel Triangle-only collision
    - LocalMinDistance + FrictionContactConstraint
    - soft SDF physics wall preserved at 10 N/m by default
    - old tangent-triangle SDF hard wall disabled
    - no training, reward/observation/action shielding/projection changes

To isolate the physical unilateral constraint, A first generates the deterministic
raw-action sequence from the checkpoint. B starts from the same seed and replays
that exact float32 raw-action sequence open-loop.

Metrics are captured after every 5 ms physics substep.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
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


def _percentiles(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if not arr.size:
        return {"p50": None, "p95": None, "p99": None, "max": None}
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def _safe_ratio(num, den):
    try:
        num = float(num)
        den = float(den)
        if math.isfinite(num) and math.isfinite(den) and den != 0.0:
            return num / den
    except Exception:
        pass
    return None


def _action_sha256(actions) -> str:
    if not actions:
        return hashlib.sha256(b"").hexdigest()
    arr = np.asarray(actions, dtype=np.float32).reshape((-1, 3))
    return hashlib.sha256(arr.tobytes()).hexdigest()


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


def _collision_summary(env):
    vessel_node = env.scene_creation_result["mcr_environment"].CollisionModel
    catheter_node = env.mcr_controller_sofa.instrument.InstrumentCombined.getChild(
        "mcr_collis"
    )
    vessel = [obj.getClassName() for obj in vessel_node.objects]
    catheter = [obj.getClassName() for obj in catheter_node.objects]
    unilateral = env.scene_creation_result.get(
        "sdf_unilateral_constraint_controller"
    )
    hard = env.scene_creation_result.get("sdf_hard_constraint_controller")
    return {
        "vessel_classes": vessel,
        "catheter_classes": catheter,
        "vessel_triangle_count": vessel.count("TriangleCollisionModel"),
        "vessel_point_count": vessel.count("PointCollisionModel"),
        "vessel_line_count": vessel.count("LineCollisionModel"),
        "catheter_point_count": catheter.count("PointCollisionModel"),
        "catheter_line_count": catheter.count("LineCollisionModel"),
        "unilateral_controller_present": unilateral is not None,
        "unilateral_component_class": (
            unilateral.constraint.getClassName() if unilateral is not None else None
        ),
        "hard_controller_present": hard is not None,
    }


def _create_env(
    *,
    enable_unilateral: bool,
    max_steps: int,
    wall_stiffness: float,
):
    # GenericConstraintSolver must exist from scene creation. This avoids the
    # dynamic-solver-init path that was only needed for the earlier same-state test.
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
            # Construct in both branches so the only scene-level difference is
            # whether the component is enabled.
            "sdf_unilateral_constraint_construct": True,
            "sdf_unilateral_constraint_enabled": bool(enable_unilateral),
        },
        env_type=EnvType.AORTIC,
        render_mode=RenderMode.NONE,
        max_episode_steps=max(int(max_steps), 2048),
        time_step=0.005,
        frame_skip=1,
        physics_substeps=2,
    )


def _run_branch(
    *,
    branch_name: str,
    checkpoint: Path,
    seed: int,
    max_steps: int,
    wall_stiffness: float,
    enable_unilateral: bool,
    progress_every: int,
    model=None,
    replay_actions=None,
):
    scene_started = time.perf_counter()
    env = _create_env(
        enable_unilateral=enable_unilateral,
        max_steps=max_steps,
        wall_stiffness=wall_stiffness,
    )
    observation, info = env.reset(seed=int(seed))
    scene_init_s = time.perf_counter() - scene_started

    models = _collision_summary(env)
    if (
        models["vessel_triangle_count"] != 1
        or models["vessel_point_count"] != 0
        or models["vessel_line_count"] != 0
        or models["catheter_point_count"] < 1
        or models["catheter_line_count"] < 1
        or not models["unilateral_controller_present"]
        or models["hard_controller_present"]
    ):
        raise RuntimeError(
            f"{branch_name}: unexpected collision/constraint configuration: {models}"
        )

    solver = _solver_object(env)
    solver_initial = _solver_snapshot(solver)
    if solver_initial.get("class") != "GenericConstraintSolver":
        raise RuntimeError(
            f"{branch_name}: expected GenericConstraintSolver, got {solver_initial}"
        )

    unilateral = env.scene_creation_result["sdf_unilateral_constraint_controller"]
    if bool(unilateral.enabled) != bool(enable_unilateral):
        raise RuntimeError(
            f"{branch_name}: unilateral enabled mismatch: "
            f"controller={unilateral.enabled}, expected={enable_unilateral}"
        )

    controller = env.mcr_controller_sofa
    instrument = controller.instrument.InstrumentCombined
    collision_node = instrument.getChild("mcr_collis")
    collision_dofs = collision_node.getObject("CollisionDOFs")
    beam_dofs = instrument.getObject("DOFs")

    original_animate = env.sofa_simulation.animate
    substep_counter = defaultdict(int)
    substep_trace = []
    step_runtime_s = []
    policy_predict_runtime_s = []
    applied_actions = []
    current_rl_step = 0
    previous_tip = np.asarray(
        controller.get_pos_quat_catheter_tip()[:3], dtype=np.float64
    ).copy()
    capture_non_finite = False

    aggregates = {
        "substeps": 0,
        "contact_active_substeps": 0,
        "penetrating_substeps": 0,
        "penetration_gt_0_1mm_substeps": 0,
        "penetration_gt_0_2mm_substeps": 0,
        "penetration_gt_0_5mm_substeps": 0,
        "max_body_penetration_mm": 0.0,
        "worst_penetration_location": None,
        "min_body_clearance_mm": math.inf,
        "min_tip_clearance_mm": math.inf,
        "max_friction_contact_count": 0,
        "max_collision_constraint_rows": 0,
        "max_beam_constraint_rows": 0,
        "max_collision_correction_mm": 0.0,
        "max_beam_correction_mm": 0.0,
        "beam_correction_ge_0_5mm_substeps": 0,
        "beam_correction_ge_1_0mm_substeps": 0,
        "unilateral_active_substeps": 0,
        "max_unilateral_python_active": 0,
        "max_unilateral_cpp_active": 0,
        "max_unilateral_candidates": 0,
        "max_unilateral_dropped": 0,
        "max_tip_move_mm": 0.0,
    }

    traces = {
        "penetration_mm": [],
        "body_clearance_mm": [],
        "tip_clearance_mm": [],
        "constraint_rows": [],
        "collision_correction_mm": [],
        "beam_correction_mm": [],
        "unilateral_python_active": [],
        "unilateral_cpp_active": [],
        "unilateral_candidates": [],
    }

    def traced_animate(root, dt):
        nonlocal previous_tip, capture_non_finite

        result = original_animate(root, dt)
        substep_counter[current_rl_step] += 1
        substep = int(substep_counter[current_rl_step])

        tip = np.asarray(
            controller.get_pos_quat_catheter_tip()[:3], dtype=np.float64
        ).copy()

        env._sdf_geometry_cache_step = -1
        env._update_sdf_safety_state(tip, advance_failure_counters=False)

        beam_pos = _as_array(beam_dofs.position)
        beam_free = _as_array(beam_dofs.free_position)
        coll_pos = _as_array(collision_dofs.position)
        coll_free = _as_array(collision_dofs.free_position)

        beam_correction = np.linalg.norm(
            (beam_pos[:, :3] - beam_free[:, :3]) * 1000.0,
            axis=1,
        )
        collision_correction = np.linalg.norm(
            (coll_pos[:, :3] - coll_free[:, :3]) * 1000.0,
            axis=1,
        )
        max_beam_correction_mm = float(np.max(beam_correction))
        max_collision_correction_mm = float(np.max(collision_correction))

        contacts = [
            obj
            for obj in collision_node.objects
            if obj.getClassName() == "FrictionContact"
        ]
        collision_rows = int(_constraint_rows(collision_dofs))
        beam_rows = int(_constraint_rows(beam_dofs))

        body_clearance_mm = float(
            env.current_sdf_body_min_surface_clearance * 1000.0
        )
        tip_clearance_mm = float(env.current_sdf_surface_clearance * 1000.0)
        penetration_mm = max(0.0, -body_clearance_mm)
        tip_move_mm = float(np.linalg.norm(tip - previous_tip) * 1000.0)
        previous_tip = tip

        udiag = unilateral.get_diagnostics()
        python_active = int(udiag.get("active_constraints", 0))
        cpp_active = int(udiag.get("cpp_active_count", -1))
        candidates = int(udiag.get("candidate_constraints", 0))
        dropped = int(udiag.get("dropped_constraints", 0))

        finite = bool(
            np.all(np.isfinite(beam_pos))
            and np.all(np.isfinite(beam_free))
            and np.all(np.isfinite(coll_pos))
            and np.all(np.isfinite(coll_free))
            and np.all(np.isfinite(tip))
            and math.isfinite(body_clearance_mm)
            and math.isfinite(tip_clearance_mm)
            and math.isfinite(max_beam_correction_mm)
            and math.isfinite(max_collision_correction_mm)
        )
        capture_non_finite = capture_non_finite or (not finite)

        row = {
            "rl_step": int(current_rl_step),
            "substep": substep,
            "root_time_s": float(root.getTime()),
            "body_clearance_mm": body_clearance_mm,
            "tip_clearance_mm": tip_clearance_mm,
            "body_penetration_mm": penetration_mm,
            "friction_contact_count": len(contacts),
            "collision_constraint_rows": collision_rows,
            "beam_constraint_rows": beam_rows,
            "collision_max_correction_mm": max_collision_correction_mm,
            "beam_max_correction_mm": max_beam_correction_mm,
            "tip_move_mm": tip_move_mm,
            "unilateral_python_active": python_active,
            "unilateral_cpp_active": cpp_active,
            "unilateral_candidates": candidates,
            "unilateral_dropped": dropped,
            "unilateral_selected_sample_kinds": list(
                udiag.get("selected_sample_kinds", [])
            ),
            "solver": _solver_snapshot(solver),
            "finite": finite,
        }
        substep_trace.append(row)

        aggregates["substeps"] += 1
        aggregates["contact_active_substeps"] += int(len(contacts) > 0)
        aggregates["penetrating_substeps"] += int(penetration_mm > 0.0)
        aggregates["penetration_gt_0_1mm_substeps"] += int(penetration_mm > 0.1)
        aggregates["penetration_gt_0_2mm_substeps"] += int(penetration_mm > 0.2)
        aggregates["penetration_gt_0_5mm_substeps"] += int(penetration_mm > 0.5)
        aggregates["min_body_clearance_mm"] = min(
            aggregates["min_body_clearance_mm"], body_clearance_mm
        )
        aggregates["min_tip_clearance_mm"] = min(
            aggregates["min_tip_clearance_mm"], tip_clearance_mm
        )

        if penetration_mm > aggregates["max_body_penetration_mm"]:
            aggregates["max_body_penetration_mm"] = penetration_mm
            aggregates["worst_penetration_location"] = {
                "rl_step": int(current_rl_step),
                "substep": substep,
                "body_clearance_mm": body_clearance_mm,
                "friction_contact_count": len(contacts),
                "constraint_rows": collision_rows,
                "unilateral_python_active": python_active,
                "unilateral_cpp_active": cpp_active,
            }

        aggregates["max_friction_contact_count"] = max(
            aggregates["max_friction_contact_count"], len(contacts)
        )
        aggregates["max_collision_constraint_rows"] = max(
            aggregates["max_collision_constraint_rows"], collision_rows
        )
        aggregates["max_beam_constraint_rows"] = max(
            aggregates["max_beam_constraint_rows"], beam_rows
        )
        aggregates["max_collision_correction_mm"] = max(
            aggregates["max_collision_correction_mm"],
            max_collision_correction_mm,
        )
        aggregates["max_beam_correction_mm"] = max(
            aggregates["max_beam_correction_mm"], max_beam_correction_mm
        )
        aggregates["beam_correction_ge_0_5mm_substeps"] += int(
            max_beam_correction_mm >= 0.5
        )
        aggregates["beam_correction_ge_1_0mm_substeps"] += int(
            max_beam_correction_mm >= 1.0
        )
        aggregates["unilateral_active_substeps"] += int(
            python_active > 0 or cpp_active > 0
        )
        aggregates["max_unilateral_python_active"] = max(
            aggregates["max_unilateral_python_active"], python_active
        )
        aggregates["max_unilateral_cpp_active"] = max(
            aggregates["max_unilateral_cpp_active"], cpp_active
        )
        aggregates["max_unilateral_candidates"] = max(
            aggregates["max_unilateral_candidates"], candidates
        )
        aggregates["max_unilateral_dropped"] = max(
            aggregates["max_unilateral_dropped"], dropped
        )
        aggregates["max_tip_move_mm"] = max(
            aggregates["max_tip_move_mm"], tip_move_mm
        )

        traces["penetration_mm"].append(penetration_mm)
        traces["body_clearance_mm"].append(body_clearance_mm)
        traces["tip_clearance_mm"].append(tip_clearance_mm)
        traces["constraint_rows"].append(collision_rows)
        traces["collision_correction_mm"].append(max_collision_correction_mm)
        traces["beam_correction_mm"].append(max_beam_correction_mm)
        traces["unilateral_python_active"].append(python_active)
        traces["unilateral_cpp_active"].append(cpp_active)
        traces["unilateral_candidates"].append(candidates)

        return result

    env.sofa_simulation.animate = traced_animate

    completed_steps = 0
    terminal_reason = None
    non_finite_observation = False
    run_started = time.perf_counter()

    try:
        for step in range(1, int(max_steps) + 1):
            current_rl_step = step

            if replay_actions is None:
                if model is None:
                    raise RuntimeError(
                        f"{branch_name}: model required when replay_actions is None"
                    )
                predict_started = time.perf_counter()
                raw_action, _ = model.predict(observation, deterministic=True)
                policy_predict_runtime_s.append(
                    time.perf_counter() - predict_started
                )
                raw_action = np.asarray(raw_action, dtype=np.float32).reshape(3)
            else:
                if step > len(replay_actions):
                    terminal_reason = "replay_action_sequence_exhausted"
                    break
                raw_action = np.asarray(
                    replay_actions[step - 1], dtype=np.float32
                ).reshape(3)

            applied_actions.append(raw_action.copy())

            step_started = time.perf_counter()
            observation, _, terminated, truncated, info = env.step(raw_action)
            step_runtime_s.append(time.perf_counter() - step_started)
            completed_steps = step

            try:
                obs_arr = np.asarray(observation, dtype=np.float64)
                if not np.all(np.isfinite(obs_arr)):
                    non_finite_observation = True
            except Exception:
                pass

            if progress_every > 0 and (
                step == 1 or step % progress_every == 0
            ):
                print(
                    f"[{branch_name}] step={step} "
                    f"pen_max_mm={aggregates['max_body_penetration_mm']:.4f} "
                    f"pen_gt_0.5={aggregates['penetration_gt_0_5mm_substeps']} "
                    f"u_active_substeps={aggregates['unilateral_active_substeps']} "
                    f"beam_corr_max_mm={aggregates['max_beam_correction_mm']:.4f}",
                    flush=True,
                )

            if capture_non_finite or non_finite_observation:
                terminal_reason = "non_finite"
                break

            if terminated or truncated:
                terminal_reason = str(
                    info.get("terminal_reason")
                    or ("terminated" if terminated else "truncated")
                )
                break
    finally:
        env.sofa_simulation.animate = original_animate

    evaluation_wall_s = time.perf_counter() - run_started

    for key in ("min_body_clearance_mm", "min_tip_clearance_mm"):
        if not math.isfinite(float(aggregates[key])):
            aggregates[key] = None

    step_arr = np.asarray(step_runtime_s, dtype=np.float64)
    pred_arr = np.asarray(policy_predict_runtime_s, dtype=np.float64)

    result = {
        "branch": branch_name,
        "diagnostic_only": True,
        "training_started": False,
        "checkpoint": str(checkpoint),
        "seed": int(seed),
        "requested_max_steps": int(max_steps),
        "completed_steps": int(completed_steps),
        "physics_dt_s": 0.005,
        "physics_substeps": 2,
        "rl_control_period_s": 0.01,
        "solver": solver_initial,
        "unilateral_enabled": bool(enable_unilateral),
        "wall_stiffness_n_per_m": float(wall_stiffness),
        "collision_models": models,
        "terminal_reason": terminal_reason,
        "capture_non_finite": bool(capture_non_finite),
        "non_finite_observation": bool(non_finite_observation),
        "action_count": len(applied_actions),
        "action_sha256": _action_sha256(applied_actions),
        "aggregates": aggregates,
        "distributions": {
            "penetration_mm": _percentiles(traces["penetration_mm"]),
            "body_clearance_mm": _percentiles(traces["body_clearance_mm"]),
            "tip_clearance_mm": _percentiles(traces["tip_clearance_mm"]),
            "constraint_rows": _percentiles(traces["constraint_rows"]),
            "collision_correction_mm": _percentiles(
                traces["collision_correction_mm"]
            ),
            "beam_correction_mm": _percentiles(traces["beam_correction_mm"]),
            "unilateral_python_active": _percentiles(
                traces["unilateral_python_active"]
            ),
            "unilateral_cpp_active": _percentiles(
                traces["unilateral_cpp_active"]
            ),
            "unilateral_candidates": _percentiles(
                traces["unilateral_candidates"]
            ),
        },
        "speed": {
            "scene_init_s": float(scene_init_s),
            "evaluation_wall_s": float(evaluation_wall_s),
            "completed_rl_steps": int(completed_steps),
            "completed_physics_substeps": int(aggregates["substeps"]),
            "env_step_mean_s": (
                _finite_float(np.mean(step_arr)) if step_arr.size else None
            ),
            "env_step_p95_s": (
                _finite_float(np.percentile(step_arr, 95))
                if step_arr.size
                else None
            ),
            "env_step_p99_s": (
                _finite_float(np.percentile(step_arr, 99))
                if step_arr.size
                else None
            ),
            "env_step_max_s": (
                _finite_float(np.max(step_arr)) if step_arr.size else None
            ),
            "policy_predict_mean_s": (
                _finite_float(np.mean(pred_arr)) if pred_arr.size else None
            ),
        },
        "episode_info": {
            "episode_min_tip_clearance_m": _finite_float(
                info.get("episode_min_tip_clearance")
            ),
            "episode_min_body_clearance_m": _finite_float(
                info.get("episode_min_body_clearance")
            ),
            "successful_task": bool(info.get("successful_task", False)),
        },
        "substep_trace": substep_trace,
    }

    try:
        env.close()
    except Exception:
        pass

    return result, applied_actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=15204)
    parser.add_argument("--max-steps", type=int, default=1285)
    parser.add_argument("--wall-stiffness", type=float, default=10.0)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if args.max_steps < 1:
        raise ValueError("--max-steps must be >= 1")

    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else (
            checkpoint.parent.parent
            / "diagnostics"
            / (
                "v15_2c_sdf_unilateral_full_ab_"
                f"seed{args.seed}_steps{args.max_steps}.json"
            )
        ).resolve()
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output_a = output.with_name(output.stem + "_A_generic_off.json")
    output_b = output.with_name(output.stem + "_B_generic_on.json")

    model = DistributedPPO.load(str(checkpoint), device="cpu")
    model.policy.set_training_mode(False)

    print(
        "[FULL_AB] A: GenericConstraintSolver + unilateral OFF; "
        "collecting deterministic raw actions",
        flush=True,
    )
    result_a, actions_a = _run_branch(
        branch_name="A_generic_off",
        checkpoint=checkpoint,
        seed=args.seed,
        max_steps=args.max_steps,
        wall_stiffness=args.wall_stiffness,
        enable_unilateral=False,
        progress_every=args.progress_every,
        model=model,
        replay_actions=None,
    )
    output_a.write_text(json.dumps(result_a, indent=2, sort_keys=True) + "\n")

    print(
        "[FULL_AB] B: GenericConstraintSolver + unilateral ON; "
        "replaying A raw actions exactly",
        flush=True,
    )
    result_b, actions_b = _run_branch(
        branch_name="B_generic_on",
        checkpoint=checkpoint,
        seed=args.seed,
        max_steps=args.max_steps,
        wall_stiffness=args.wall_stiffness,
        enable_unilateral=True,
        progress_every=args.progress_every,
        model=None,
        replay_actions=actions_a,
    )
    output_b.write_text(json.dumps(result_b, indent=2, sort_keys=True) + "\n")

    a = result_a["aggregates"]
    b = result_b["aggregates"]
    a_dist = result_a["distributions"]
    b_dist = result_b["distributions"]
    a_speed = result_a["speed"]
    b_speed = result_b["speed"]

    common_action_count = min(len(actions_a), len(actions_b))
    actions_identical = bool(
        common_action_count == len(actions_b)
        and all(
            np.array_equal(
                np.asarray(actions_a[i], dtype=np.float32),
                np.asarray(actions_b[i], dtype=np.float32),
            )
            for i in range(common_action_count)
        )
    )

    comparison = {
        "same_checkpoint": result_a["checkpoint"] == result_b["checkpoint"],
        "same_seed": result_a["seed"] == result_b["seed"],
        "same_requested_horizon": (
            result_a["requested_max_steps"] == result_b["requested_max_steps"]
        ),
        "A_solver": result_a["solver"].get("class"),
        "B_solver": result_b["solver"].get("class"),
        "A_unilateral_enabled": result_a["unilateral_enabled"],
        "B_unilateral_enabled": result_b["unilateral_enabled"],
        "A_completed_steps": result_a["completed_steps"],
        "B_completed_steps": result_b["completed_steps"],
        "A_terminal_reason": result_a["terminal_reason"],
        "B_terminal_reason": result_b["terminal_reason"],
        "A_action_sha256": result_a["action_sha256"],
        "B_action_sha256": result_b["action_sha256"],
        "B_replayed_A_actions_exactly_for_completed_steps": actions_identical,
        "A_max_penetration_mm": a["max_body_penetration_mm"],
        "B_max_penetration_mm": b["max_body_penetration_mm"],
        "max_penetration_reduction_mm": (
            a["max_body_penetration_mm"] - b["max_body_penetration_mm"]
        ),
        "A_penetration_p95_mm": a_dist["penetration_mm"]["p95"],
        "B_penetration_p95_mm": b_dist["penetration_mm"]["p95"],
        "A_penetration_p99_mm": a_dist["penetration_mm"]["p99"],
        "B_penetration_p99_mm": b_dist["penetration_mm"]["p99"],
        "A_penetrating_substeps": a["penetrating_substeps"],
        "B_penetrating_substeps": b["penetrating_substeps"],
        "A_pen_gt_0_1mm": a["penetration_gt_0_1mm_substeps"],
        "B_pen_gt_0_1mm": b["penetration_gt_0_1mm_substeps"],
        "A_pen_gt_0_2mm": a["penetration_gt_0_2mm_substeps"],
        "B_pen_gt_0_2mm": b["penetration_gt_0_2mm_substeps"],
        "A_pen_gt_0_5mm": a["penetration_gt_0_5mm_substeps"],
        "B_pen_gt_0_5mm": b["penetration_gt_0_5mm_substeps"],
        "A_worst_location": a["worst_penetration_location"],
        "B_worst_location": b["worst_penetration_location"],
        "B_unilateral_active_substeps": b["unilateral_active_substeps"],
        "B_max_unilateral_python_active": b["max_unilateral_python_active"],
        "B_max_unilateral_cpp_active": b["max_unilateral_cpp_active"],
        "A_constraint_rows_p95": a_dist["constraint_rows"]["p95"],
        "B_constraint_rows_p95": b_dist["constraint_rows"]["p95"],
        "A_constraint_rows_max": a["max_collision_constraint_rows"],
        "B_constraint_rows_max": b["max_collision_constraint_rows"],
        "A_collision_correction_p95_mm": a_dist[
            "collision_correction_mm"
        ]["p95"],
        "B_collision_correction_p95_mm": b_dist[
            "collision_correction_mm"
        ]["p95"],
        "A_beam_correction_p95_mm": a_dist["beam_correction_mm"]["p95"],
        "B_beam_correction_p95_mm": b_dist["beam_correction_mm"]["p95"],
        "A_beam_correction_max_mm": a["max_beam_correction_mm"],
        "B_beam_correction_max_mm": b["max_beam_correction_mm"],
        "A_beam_corr_ge_0_5mm": a[
            "beam_correction_ge_0_5mm_substeps"
        ],
        "B_beam_corr_ge_0_5mm": b[
            "beam_correction_ge_0_5mm_substeps"
        ],
        "A_beam_corr_ge_1_0mm": a[
            "beam_correction_ge_1_0mm_substeps"
        ],
        "B_beam_corr_ge_1_0mm": b[
            "beam_correction_ge_1_0mm_substeps"
        ],
        "A_env_step_mean_s": a_speed["env_step_mean_s"],
        "B_env_step_mean_s": b_speed["env_step_mean_s"],
        "env_step_mean_ratio_B_over_A": _safe_ratio(
            b_speed["env_step_mean_s"], a_speed["env_step_mean_s"]
        ),
        "both_finite": bool(
            not result_a["capture_non_finite"]
            and not result_b["capture_non_finite"]
            and not result_a["non_finite_observation"]
            and not result_b["non_finite_observation"]
        ),
    }

    combined = {
        "test": (
            "V15.2-C full replay: GenericConstraintSolver unilateral OFF/ON "
            "with identical raw actions"
        ),
        "diagnostic_only": True,
        "training_started": False,
        "checkpoint": str(checkpoint),
        "seed": int(args.seed),
        "requested_max_steps": int(args.max_steps),
        "A_generic_off": result_a,
        "B_generic_on": result_b,
        "comparison": comparison,
        "files": {
            "combined": str(output),
            "A_generic_off": str(output_a),
            "B_generic_on": str(output_b),
        },
    }
    output.write_text(json.dumps(combined, indent=2, sort_keys=True) + "\n")

    print(
        json.dumps(
            {
                "test": combined["test"],
                "comparison": comparison,
                "files": combined["files"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
