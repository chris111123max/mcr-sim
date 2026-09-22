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
    parser.add_argument("--seed", type=int, default=15204)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    enabled = args.wall == "on"
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
    }
    env = MCREnv(
        create_scene_kwargs=kwargs,
        env_type=EnvType.AORTIC,
        render_mode=RenderMode.NONE,
        max_episode_steps=int(args.max_steps),
    )
    observation, info = env.reset(seed=int(args.seed))

    started = time.perf_counter()
    max_route_potential = 0.0
    max_wall_force = 0.0
    max_wall_total_force = 0.0
    max_wall_active_nodes = 0
    max_tip_step_mm = 0.0
    non_finite = False
    previous_tip = np.asarray(
        env.mcr_controller_sofa.get_pos_quat_catheter_tip()[:3], dtype=np.float64
    )
    terminal_reason = "evaluation_limit"
    reward_total = 0.0

    for step_index in range(int(args.max_steps)):
        action, _ = model.predict(observation, deterministic=True)
        action = np.asarray(action, dtype=np.float32).reshape(3)
        observation, reward, terminated, truncated, info = env.step(action)
        reward_total += float(reward)
        tip = np.asarray(
            env.mcr_controller_sofa.get_pos_quat_catheter_tip()[:3],
            dtype=np.float64,
        )
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
        "wall_enabled": bool(info.get("sdf_physics_wall_enabled", False)),
        "stiffness_n_per_m": float(args.stiffness),
        "checkpoint": str(checkpoint),
        "seed": int(args.seed),
        "steps": steps,
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
