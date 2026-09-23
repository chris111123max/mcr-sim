"""Deterministic B02/target04 physics-wall preflight (no training)."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

TEST_DIR = Path(__file__).resolve().parent
PYTHON_ROOT = TEST_DIR.parents[1]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from mcr_sim.distributed import DistributedPPO
from mcr_sim.mcr_rl_env import EnvType, MCREnv
from mcr_sim.rl_core.base import RenderMode
from mcr_sim.training_config import PHYSICS_SUBSTEPS, SOFA_TIME_STEP_S


def _finite(value, default=math.nan):
    try:
        result = float(value)
    except Exception:
        return float(default)
    return result if math.isfinite(result) else float(default)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--wall", choices=("off", "on"), required=True)
    parser.add_argument("--stiffness", type=float, default=10.0)
    parser.add_argument("--max-steps", type=int, default=2048)
    parser.add_argument("--physics-dt", type=float, default=SOFA_TIME_STEP_S)
    parser.add_argument("--physics-substeps", type=int, default=PHYSICS_SUBSTEPS)
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--seed", type=int, default=15204)
    parser.add_argument(
        "--intersection-method", choices=("local_min_distance", "min_proximity"),
        default="local_min_distance",
    )
    parser.add_argument("--contact-trace-start", type=int, default=1087)
    parser.add_argument("--contact-trace-end", type=int, default=1099)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    enabled = args.wall == "on"
    os.environ["MCR_SOFA_DT"] = str(float(args.physics_dt))
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    model = DistributedPPO.load(str(checkpoint), device="cpu")
    model.policy.set_training_mode(False)

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
        "sdf_physics_wall_enabled": enabled,
        "sdf_wall_stiffness_n_per_m": float(args.stiffness),
        "diagnostic_intersection_method": args.intersection_method,
    }
    env = MCREnv(
        create_scene_kwargs=kwargs,
        env_type=EnvType.AORTIC,
        render_mode=RenderMode.NONE,
        max_episode_steps=int(args.max_steps),
        time_step=float(args.physics_dt),
        frame_skip=int(args.frame_skip),
        physics_substeps=int(args.physics_substeps),
    )
    observation, info = env.reset(seed=int(args.seed))
    intersection_objects = [
        obj.getClassName() for obj in env._sofa_root_node.objects
        if obj.getClassName() in ("LocalMinDistance", "MinProximityIntersection")
    ]
    expected_intersection = (
        "LocalMinDistance" if args.intersection_method == "local_min_distance"
        else "MinProximityIntersection"
    )
    if intersection_objects != [expected_intersection]:
        raise RuntimeError(
            f"Expected only {expected_intersection}, got {intersection_objects}"
        )

    started = time.perf_counter()
    max_route_potential = 0.0
    max_wall_force = 0.0
    max_wall_total_force = 0.0
    max_wall_active_nodes = 0
    max_tip_step_mm = 0.0
    non_finite = False
    previous_tip = np.asarray(
        env.mcr_controller_sofa.get_pos_quat_catheter_tip()[:3], dtype=np.float64
    ).copy()
    terminal_reason = "evaluation_limit"
    reward_total = 0.0
    contact_trace = []
    min_tip_clearance_step = None
    min_body_clearance_step = None
    max_contact_free_penetration_step = None
    min_tip_clearance_seen = math.inf
    min_body_clearance_seen = math.inf
    max_contact_free_penetration_seen = 0.0
    collision_node = env.mcr_controller_sofa.instrument.InstrumentCombined.getChild("mcr_collis")

    for step_index in range(int(args.max_steps)):
        action, _ = model.predict(observation, deterministic=True)
        action = np.asarray(action, dtype=np.float32).reshape(3)
        observation, reward, terminated, truncated, info = env.step(action)
        reward_total += float(reward)
        episode_tip_clearance = _finite(info.get("episode_min_tip_clearance"), math.inf)
        episode_body_clearance = _finite(info.get("episode_min_body_clearance"), math.inf)
        contact_free_penetration = _finite(
            info.get("episode_max_contact_free_penetration"), 0.0
        )
        if episode_tip_clearance < min_tip_clearance_seen:
            min_tip_clearance_seen = episode_tip_clearance
            min_tip_clearance_step = step_index + 1
        if episode_body_clearance < min_body_clearance_seen:
            min_body_clearance_seen = episode_body_clearance
            min_body_clearance_step = step_index + 1
        if contact_free_penetration > max_contact_free_penetration_seen:
            max_contact_free_penetration_seen = contact_free_penetration
            max_contact_free_penetration_step = step_index + 1
        if args.contact_trace_start <= step_index + 1 <= args.contact_trace_end:
            contacts = [obj for obj in collision_node.objects
                        if obj.getClassName() == "FrictionContact"]
            contact_trace.append({
                "step": step_index + 1,
                "friction_contact_count": len(contacts),
                "route_potential": _finite(info.get("route_potential")),
                "tip_surface_clearance_m": _finite(info.get("sdf_surface_clearance")),
            })
        tip = np.asarray(
            env.mcr_controller_sofa.get_pos_quat_catheter_tip()[:3],
            dtype=np.float64,
        ).copy()
        max_tip_step_mm = max(
            max_tip_step_mm, float(np.linalg.norm(tip - previous_tip) * 1000.0)
        )
        previous_tip = tip
        max_route_potential = max(
            max_route_potential, _finite(info.get("route_potential"), 0.0)
        )
        max_wall_force = max(
            max_wall_force, _finite(info.get("sdf_wall_max_force_N"), 0.0)
        )
        max_wall_total_force = max(
            max_wall_total_force,
            _finite(info.get("sdf_wall_total_force_N"), 0.0),
        )
        max_wall_active_nodes = max(
            max_wall_active_nodes, int(info.get("sdf_wall_active_nodes", 0))
        )
        if not (
            np.all(np.isfinite(observation))
            and np.all(np.isfinite(action))
            and math.isfinite(float(reward))
            and np.all(np.isfinite(tip))
        ):
            non_finite = True
            terminal_reason = "non_finite_preflight"
            break
        if terminated or truncated:
            terminal_reason = str(info.get("terminal_reason", "unknown"))
            break
    else:
        step_index = int(args.max_steps) - 1

    elapsed = time.perf_counter() - started
    steps = int(step_index + 1)
    result = {
        "wall": args.wall,
        "intersection_method": args.intersection_method,
        "intersection_objects": intersection_objects,
        "contact_trace": contact_trace,
        "wall_enabled": bool(info.get("sdf_physics_wall_enabled", False)),
        "stiffness_n_per_m": float(args.stiffness),
        "checkpoint": str(checkpoint),
        "seed": int(args.seed),
        "steps": steps,
        "physics_dt_s": float(args.physics_dt),
        "frame_skip": int(args.frame_skip),
        "physics_steps_per_rl_action": int(args.physics_substeps),
        "physics_substeps": int(args.physics_substeps),
        "rl_control_period_s": float(args.physics_dt * args.physics_substeps),
        "episode_min_tip_clearance_m": _finite(
            info.get("episode_min_tip_clearance")
        ),
        "episode_min_body_clearance_m": _finite(
            info.get("episode_min_body_clearance")
        ),
        "episode_max_contact_free_penetration_m": _finite(
            info.get("episode_max_contact_free_penetration"), 0.0
        ),
        "contact_active_substeps": int(info.get("contact_active_substeps", 0)),
        "episode_min_tip_clearance_step": min_tip_clearance_step,
        "episode_min_body_clearance_step": min_body_clearance_step,
        "episode_max_contact_free_penetration_step": max_contact_free_penetration_step,
        "runtime_s": float(elapsed),
        "steps_per_second": float(steps / max(elapsed, 1e-9)),
        "terminal_reason": terminal_reason,
        "success": bool(info.get("successful_task", False)),
        "reward_total": float(reward_total),
        "route_potential_final": _finite(info.get("route_potential")),
        "route_potential_max": float(max_route_potential),
        "body_min_surface_clearance_m": _finite(
            info.get("sdf_body_surface_clearance_min_episode")
        ),
        "tip_min_surface_clearance_m": _finite(
            info.get("sdf_surface_clearance_min_episode")
        ),
        "wall_min_clearance_m": _finite(
            info.get("sdf_wall_min_clearance_episode_m")
        ),
        "wall_max_force_N": float(max_wall_force),
        "wall_max_total_force_N": float(max_wall_total_force),
        "wall_max_active_nodes": int(max_wall_active_nodes),
        "max_tip_step_mm": float(max_tip_step_mm),
        "non_finite": bool(non_finite),
        "abnormal_bounce": bool(max_tip_step_mm > 5.0),
        "observation_dim": int(np.asarray(observation).size),
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
