"""Bounded A/B stability + speed diagnostic for SDF hard contact.

Runs two independent deterministic evaluation episodes with the same
checkpoint/seed/horizon:

A baseline:
    catheter Line + Point
    vessel Triangle only
    existing soft SDF physics wall
    SDF hard contact NOT constructed

B hard:
    same baseline collision configuration
    + SDF hard contact constructed and enabled from scene creation

The previous vessel Point/Line experiment is explicitly disabled in both runs.
No training is started. No reward/observation/action safety logic is changed.

Metrics:
- body/tip clearance every 5 ms physics substep;
- penetration count/depth regardless of whether some other contact exists;
- FrictionContact count and constraint rows;
- beam correction distribution and >0.5 / >1.0 mm counts;
- SDF hard candidate/active/dropped patches;
- finite/NaN status;
- env.step wall time, policy predict time, total evaluation speed;
- direct B/A runtime ratios.
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


def _solver_object(env):
    for obj in env._sofa_root_node.objects:
        if obj.getClassName() in ("LCPConstraintSolver", "GenericConstraintSolver"):
            return obj
    return None


def _solver_snapshot(solver):
    if solver is None:
        return {}
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


def _collision_summary(env):
    vessel_env = env.scene_creation_result["mcr_environment"]
    vessel_node = vessel_env.CollisionModel
    catheter_node = env.mcr_controller_sofa.instrument.InstrumentCombined.getChild(
        "mcr_collis"
    )
    vessel = [obj.getClassName() for obj in vessel_node.objects]
    catheter = [obj.getClassName() for obj in catheter_node.objects]
    hard = env.scene_creation_result.get("sdf_hard_constraint_controller")
    hard_classes = (
        [obj.getClassName() for obj in hard.collision_node.objects]
        if hard is not None
        else []
    )
    return {
        "vessel_classes": vessel,
        "catheter_classes": catheter,
        "hard_classes": hard_classes,
        "vessel_triangle_count": vessel.count("TriangleCollisionModel"),
        "vessel_point_count": vessel.count("PointCollisionModel"),
        "vessel_line_count": vessel.count("LineCollisionModel"),
        "catheter_point_count": catheter.count("PointCollisionModel"),
        "catheter_line_count": catheter.count("LineCollisionModel"),
        "hard_triangle_count": hard_classes.count("TriangleCollisionModel"),
    }


def _create_env(enable_hard: bool, max_steps: int, wall_stiffness: float):
    kwargs = {
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
        # Explicitly exclude previous vessel Point/Line experiment.
        "use_vessel_line_point_collision": False,
        "use_vessel_point_collision": False,
        "use_vessel_line_collision": False,
    }
    if enable_hard:
        kwargs.update(
            {
                "sdf_hard_constraint_construct": True,
                "sdf_hard_constraint_enabled": True,
                "sdf_hard_constraint_max_active_patches": 12,
            }
        )
    else:
        # Real baseline: do not even construct parked hard-wall geometry.
        kwargs.update(
            {
                "sdf_hard_constraint_construct": False,
                "sdf_hard_constraint_enabled": False,
            }
        )

    return MCREnv(
        create_scene_kwargs=kwargs,
        env_type=EnvType.AORTIC,
        render_mode=RenderMode.NONE,
        max_episode_steps=max(int(max_steps), 2048),
        time_step=0.005,
        frame_skip=1,
        physics_substeps=2,
    )


def _run_branch(
    model,
    checkpoint: Path,
    seed: int,
    max_steps: int,
    wall_stiffness: float,
    enable_hard: bool,
    branch_name: str,
    progress_every: int,
):
    scene_started = time.perf_counter()
    env = _create_env(enable_hard, max_steps, wall_stiffness)
    observation, info = env.reset(seed=int(seed))
    scene_init_s = time.perf_counter() - scene_started

    models = _collision_summary(env)
    if (
        models["vessel_triangle_count"] != 1
        or models["vessel_point_count"] != 0
        or models["vessel_line_count"] != 0
        or models["catheter_point_count"] < 1
        or models["catheter_line_count"] < 1
    ):
        raise RuntimeError(
            f"{branch_name}: expected catheter Line+Point and vessel Triangle-only; "
            f"got {models}"
        )
    if enable_hard and models["hard_triangle_count"] != 1:
        raise RuntimeError(f"{branch_name}: expected one hard Triangle model: {models}")
    if (not enable_hard) and models["hard_triangle_count"] != 0:
        raise RuntimeError(
            f"{branch_name}: baseline must not construct hard wall: {models}"
        )

    hard = env.scene_creation_result.get("sdf_hard_constraint_controller")
    if enable_hard and (hard is None or not hard.enabled):
        raise RuntimeError(f"{branch_name}: hard wall must be enabled from scene start")
    if (not enable_hard) and hard is not None:
        raise RuntimeError(f"{branch_name}: baseline unexpectedly has hard controller")

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
        "penetrating_substeps": 0,
        "penetrating_with_contact_substeps": 0,
        "penetrating_without_contact_substeps": 0,
        "max_body_penetration_mm": 0.0,
        "max_contact_free_body_penetration_mm": 0.0,
        "min_body_clearance_mm": math.inf,
        "min_tip_clearance_mm": math.inf,
        "max_friction_contact_count": 0,
        "max_collision_constraint_rows": 0,
        "max_beam_constraint_rows": 0,
        "max_beam_correction_mm": 0.0,
        "beam_correction_ge_0_5mm_substeps": 0,
        "beam_correction_ge_1_0mm_substeps": 0,
        "max_tip_move_mm": 0.0,
        "max_hard_candidate_patches": 0,
        "max_hard_active_patches": 0,
        "max_hard_dropped_patches": 0,
    }

    trace_lists = {
        "constraint_rows": [],
        "beam_correction_mm": [],
        "body_clearance_mm": [],
        "tip_clearance_mm": [],
        "hard_candidate_patches": [],
        "hard_active_patches": [],
        "hard_dropped_patches": [],
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

        beam_correction = np.linalg.norm(
            (beam_pos[:, :3] - beam_free[:, :3]) * 1000.0,
            axis=1,
        )
        max_beam_correction_mm = float(np.max(beam_correction))
        tip_move_mm = float(np.linalg.norm(tip - previous_tip) * 1000.0)
        previous_tip = tip

        collision_rows = int(_constraint_rows(collision_dofs))
        beam_rows = int(_constraint_rows(beam_dofs))

        hard_diag = hard.get_diagnostics() if hard is not None else {}
        hard_candidate = int(hard_diag.get("candidate_patches", 0))
        hard_active = int(hard_diag.get("active_patches", 0))
        hard_dropped = int(hard_diag.get("dropped_patches", 0))

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

        penetrating = body_clearance_mm < 0.0
        contact_active = len(contacts) > 0

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
            "beam_max_correction_mm": max_beam_correction_mm,
            "tip_move_mm": tip_move_mm,
            "hard_candidate_patches": hard_candidate,
            "hard_active_patches": hard_active,
            "hard_dropped_patches": hard_dropped,
            "hard_selected_indices": list(
                hard_diag.get("selected_sample_indices", [])
            ),
            "solver": _solver_snapshot(solver),
            "finite": finite,
        }
        substep_trace.append(row)

        aggregates["substeps"] += 1
        aggregates["contact_active_substeps"] += int(contact_active)
        aggregates["penetrating_substeps"] += int(penetrating)
        aggregates["penetrating_with_contact_substeps"] += int(
            penetrating and contact_active
        )
        aggregates["penetrating_without_contact_substeps"] += int(
            penetrating and not contact_active
        )
        aggregates["max_body_penetration_mm"] = max(
            aggregates["max_body_penetration_mm"],
            body_penetration_mm,
        )
        if not contact_active:
            aggregates["max_contact_free_body_penetration_mm"] = max(
                aggregates["max_contact_free_body_penetration_mm"],
                body_penetration_mm,
            )
        aggregates["min_body_clearance_mm"] = min(
            aggregates["min_body_clearance_mm"], body_clearance_mm
        )
        aggregates["min_tip_clearance_mm"] = min(
            aggregates["min_tip_clearance_mm"], tip_clearance_mm
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
        aggregates["beam_correction_ge_0_5mm_substeps"] += int(
            max_beam_correction_mm >= 0.5
        )
        aggregates["beam_correction_ge_1_0mm_substeps"] += int(
            max_beam_correction_mm >= 1.0
        )
        aggregates["max_tip_move_mm"] = max(
            aggregates["max_tip_move_mm"], tip_move_mm
        )
        aggregates["max_hard_candidate_patches"] = max(
            aggregates["max_hard_candidate_patches"], hard_candidate
        )
        aggregates["max_hard_active_patches"] = max(
            aggregates["max_hard_active_patches"], hard_active
        )
        aggregates["max_hard_dropped_patches"] = max(
            aggregates["max_hard_dropped_patches"], hard_dropped
        )

        trace_lists["constraint_rows"].append(collision_rows)
        trace_lists["beam_correction_mm"].append(max_beam_correction_mm)
        trace_lists["body_clearance_mm"].append(body_clearance_mm)
        trace_lists["tip_clearance_mm"].append(tip_clearance_mm)
        trace_lists["hard_candidate_patches"].append(hard_candidate)
        trace_lists["hard_active_patches"].append(hard_active)
        trace_lists["hard_dropped_patches"].append(hard_dropped)

        return result

    env.sofa_simulation.animate = traced_animate

    step_runtime_s = []
    predict_runtime_s = []
    completed_steps = 0
    terminal_reason = "diagnostic_limit"
    non_finite_observation = False

    run_started = time.perf_counter()
    try:
        for step in range(1, int(max_steps) + 1):
            t0 = time.perf_counter()
            raw_action, _ = model.predict(observation, deterministic=True)
            predict_runtime_s.append(time.perf_counter() - t0)

            t1 = time.perf_counter()
            observation, reward, terminated, truncated, info = env.step(raw_action)
            step_runtime_s.append(time.perf_counter() - t1)
            completed_steps = step

            try:
                obs_arr = np.asarray(observation)
                if not np.all(np.isfinite(obs_arr)):
                    non_finite_observation = True
            except Exception:
                if isinstance(observation, dict):
                    non_finite_observation = any(
                        not np.all(np.isfinite(np.asarray(v)))
                        for v in observation.values()
                    )

            if progress_every > 0 and step % int(progress_every) == 0:
                elapsed = time.perf_counter() - run_started
                print(
                    f"[SDF_HARD_AB][{branch_name}]",
                    "step=", step,
                    "body_min_mm=", aggregates["min_body_clearance_mm"],
                    "penetrating_substeps=", aggregates["penetrating_substeps"],
                    "rows_max=", aggregates["max_collision_constraint_rows"],
                    "beam_corr_max_mm=", aggregates["max_beam_correction_mm"],
                    "hard_active_max=", aggregates["max_hard_active_patches"],
                    "env_step_mean_s=", float(np.mean(step_runtime_s)),
                    "eval_steps_per_s=", step / max(elapsed, 1e-12),
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

    run_total_s = time.perf_counter() - run_started
    step_arr = np.asarray(step_runtime_s, dtype=np.float64)
    pred_arr = np.asarray(predict_runtime_s, dtype=np.float64)

    for key in ("min_body_clearance_mm", "min_tip_clearance_mm"):
        if not math.isfinite(float(aggregates[key])):
            aggregates[key] = None

    distribution = {
        "constraint_rows": _percentiles(trace_lists["constraint_rows"]),
        "beam_correction_mm": _percentiles(trace_lists["beam_correction_mm"]),
        "body_clearance_mm": _percentiles(trace_lists["body_clearance_mm"]),
        "tip_clearance_mm": _percentiles(trace_lists["tip_clearance_mm"]),
        "hard_candidate_patches": _percentiles(
            trace_lists["hard_candidate_patches"]
        ),
        "hard_active_patches": _percentiles(trace_lists["hard_active_patches"]),
        "hard_dropped_patches": _percentiles(
            trace_lists["hard_dropped_patches"]
        ),
    }

    speed = {
        "scene_init_s": float(scene_init_s),
        "evaluation_wall_s": float(run_total_s),
        "completed_rl_steps": int(completed_steps),
        "completed_physics_substeps": int(aggregates["substeps"]),
        "rl_steps_per_s": (
            float(completed_steps / run_total_s) if run_total_s > 0 else None
        ),
        "physics_substeps_per_s": (
            float(aggregates["substeps"] / run_total_s)
            if run_total_s > 0
            else None
        ),
        "env_step_mean_s": _finite_float(np.mean(step_arr))
        if step_arr.size
        else None,
        "env_step_p50_s": _finite_float(np.percentile(step_arr, 50))
        if step_arr.size
        else None,
        "env_step_p95_s": _finite_float(np.percentile(step_arr, 95))
        if step_arr.size
        else None,
        "env_step_p99_s": _finite_float(np.percentile(step_arr, 99))
        if step_arr.size
        else None,
        "env_step_max_s": _finite_float(np.max(step_arr))
        if step_arr.size
        else None,
        "policy_predict_mean_s": _finite_float(np.mean(pred_arr))
        if pred_arr.size
        else None,
        "policy_predict_p95_s": _finite_float(np.percentile(pred_arr, 95))
        if pred_arr.size
        else None,
    }

    result = {
        "branch": branch_name,
        "hard_enabled_from_scene_start": bool(enable_hard),
        "diagnostic_only": True,
        "training_started": False,
        "checkpoint": str(checkpoint),
        "seed": int(seed),
        "requested_max_steps": int(max_steps),
        "completed_steps": int(completed_steps),
        "physics_dt_s": 0.005,
        "physics_substeps": 2,
        "rl_control_period_s": 0.01,
        "wall_stiffness_n_per_m": float(wall_stiffness),
        "collision_models": models,
        "solver_initial": _solver_snapshot(solver),
        "terminal_reason": terminal_reason,
        "capture_non_finite": bool(capture_non_finite),
        "non_finite_observation": bool(non_finite_observation),
        "aggregates": aggregates,
        "distributions": distribution,
        "speed": speed,
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
        "substep_trace": substep_trace,
    }

    try:
        env.close()
    except Exception:
        pass

    return result


def _safe_ratio(num, den):
    try:
        num = float(num)
        den = float(den)
        if math.isfinite(num) and math.isfinite(den) and den != 0.0:
            return num / den
    except Exception:
        pass
    return None


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
            / f"v15_2c_sdf_hard_stability_ab_seed{args.seed}_steps{args.max_steps}.json"
        ).resolve()
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output_a = output.with_name(output.stem + "_A_baseline.json")
    output_b = output.with_name(output.stem + "_B_sdf_hard.json")

    os.environ["MCR_SOFA_DT"] = "0.005"

    model = DistributedPPO.load(str(checkpoint), device="cpu")
    model.policy.set_training_mode(False)

    result_a = _run_branch(
        model=model,
        checkpoint=checkpoint,
        seed=args.seed,
        max_steps=args.max_steps,
        wall_stiffness=args.wall_stiffness,
        enable_hard=False,
        branch_name="A_baseline",
        progress_every=args.progress_every,
    )
    output_a.write_text(json.dumps(result_a, indent=2, sort_keys=True) + "\n")

    result_b = _run_branch(
        model=model,
        checkpoint=checkpoint,
        seed=args.seed,
        max_steps=args.max_steps,
        wall_stiffness=args.wall_stiffness,
        enable_hard=True,
        branch_name="B_sdf_hard",
        progress_every=args.progress_every,
    )
    output_b.write_text(json.dumps(result_b, indent=2, sort_keys=True) + "\n")

    a_speed = result_a["speed"]
    b_speed = result_b["speed"]
    a_agg = result_a["aggregates"]
    b_agg = result_b["aggregates"]

    comparison = {
        "same_checkpoint": result_a["checkpoint"] == result_b["checkpoint"],
        "same_seed": result_a["seed"] == result_b["seed"],
        "same_requested_horizon": (
            result_a["requested_max_steps"] == result_b["requested_max_steps"]
        ),
        "A_completed_steps": result_a["completed_steps"],
        "B_completed_steps": result_b["completed_steps"],
        "A_terminal_reason": result_a["terminal_reason"],
        "B_terminal_reason": result_b["terminal_reason"],
        "A_min_body_clearance_mm": a_agg["min_body_clearance_mm"],
        "B_min_body_clearance_mm": b_agg["min_body_clearance_mm"],
        "A_penetrating_substeps": a_agg["penetrating_substeps"],
        "B_penetrating_substeps": b_agg["penetrating_substeps"],
        "A_penetrating_with_contact_substeps": a_agg[
            "penetrating_with_contact_substeps"
        ],
        "B_penetrating_with_contact_substeps": b_agg[
            "penetrating_with_contact_substeps"
        ],
        "A_penetrating_without_contact_substeps": a_agg[
            "penetrating_without_contact_substeps"
        ],
        "B_penetrating_without_contact_substeps": b_agg[
            "penetrating_without_contact_substeps"
        ],
        "A_constraint_rows_p95": result_a["distributions"]["constraint_rows"]["p95"],
        "B_constraint_rows_p95": result_b["distributions"]["constraint_rows"]["p95"],
        "A_constraint_rows_max": a_agg["max_collision_constraint_rows"],
        "B_constraint_rows_max": b_agg["max_collision_constraint_rows"],
        "A_beam_correction_p95_mm": result_a["distributions"][
            "beam_correction_mm"
        ]["p95"],
        "B_beam_correction_p95_mm": result_b["distributions"][
            "beam_correction_mm"
        ]["p95"],
        "A_beam_correction_max_mm": a_agg["max_beam_correction_mm"],
        "B_beam_correction_max_mm": b_agg["max_beam_correction_mm"],
        "A_beam_correction_ge_0_5mm": a_agg[
            "beam_correction_ge_0_5mm_substeps"
        ],
        "B_beam_correction_ge_0_5mm": b_agg[
            "beam_correction_ge_0_5mm_substeps"
        ],
        "A_beam_correction_ge_1_0mm": a_agg[
            "beam_correction_ge_1_0mm_substeps"
        ],
        "B_beam_correction_ge_1_0mm": b_agg[
            "beam_correction_ge_1_0mm_substeps"
        ],
        "B_hard_active_patches_p95": result_b["distributions"][
            "hard_active_patches"
        ]["p95"],
        "B_hard_active_patches_max": b_agg["max_hard_active_patches"],
        "A_env_step_mean_s": a_speed["env_step_mean_s"],
        "B_env_step_mean_s": b_speed["env_step_mean_s"],
        "A_env_step_p95_s": a_speed["env_step_p95_s"],
        "B_env_step_p95_s": b_speed["env_step_p95_s"],
        "A_rl_steps_per_s": a_speed["rl_steps_per_s"],
        "B_rl_steps_per_s": b_speed["rl_steps_per_s"],
        "env_step_mean_s_ratio_B_over_A": _safe_ratio(
            b_speed["env_step_mean_s"], a_speed["env_step_mean_s"]
        ),
        "env_step_p95_s_ratio_B_over_A": _safe_ratio(
            b_speed["env_step_p95_s"], a_speed["env_step_p95_s"]
        ),
        "throughput_ratio_B_over_A": _safe_ratio(
            b_speed["rl_steps_per_s"], a_speed["rl_steps_per_s"]
        ),
        "hard_runtime_overhead_fraction": (
            _safe_ratio(
                b_speed["env_step_mean_s"] - a_speed["env_step_mean_s"],
                a_speed["env_step_mean_s"],
            )
            if a_speed["env_step_mean_s"] is not None
            and b_speed["env_step_mean_s"] is not None
            else None
        ),
        "both_finite": bool(
            not result_a["capture_non_finite"]
            and not result_b["capture_non_finite"]
            and not result_a["non_finite_observation"]
            and not result_b["non_finite_observation"]
        ),
    }

    combined = {
        "test": "V15.2-C baseline vs scene-start SDF hard-contact stability/speed A-B",
        "diagnostic_only": True,
        "training_started": False,
        "checkpoint": str(checkpoint),
        "seed": int(args.seed),
        "requested_max_steps": int(args.max_steps),
        "A": result_a,
        "B": result_b,
        "comparison": comparison,
        "files": {
            "combined": str(output),
            "A": str(output_a),
            "B": str(output_b),
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

    os._exit(0)


if __name__ == "__main__":
    main()
