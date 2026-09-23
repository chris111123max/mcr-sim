"""Fork one identical SOFA state at RL step 1095 for contact diagnosis."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np

PYTHON_ROOT = Path(__file__).resolve().parents[2]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from mcr_sim.distributed import DistributedPPO
from mcr_sim.mcr_rl_env import EnvType, MCREnv
from mcr_sim.rl_core.base import RenderMode


def as_array(data):
    return np.asarray(data.value, dtype=np.float64).copy()


def count_constraint_rows(dofs):
    value = dofs.constraint.value
    return len([line for line in str(value).splitlines() if line.strip()])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=15204)
    parser.add_argument("--fork-step", type=int, default=1095)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ["MCR_SOFA_DT"] = "0.01"
    checkpoint = Path(args.checkpoint).resolve()
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
        },
        env_type=EnvType.AORTIC,
        render_mode=RenderMode.NONE,
        max_episode_steps=2048,
        time_step=0.01,
        frame_skip=1,
        physics_substeps=1,
    )
    observation, info = env.reset(seed=args.seed)
    for step in range(1, args.fork_step + 1):
        raw_action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = env.step(raw_action)
        if step % 250 == 0:
            print("baseline_step", step, flush=True)
        if terminated or truncated:
            raise RuntimeError(f"Baseline stopped at step {step}: {info.get('terminal_reason')}")

    root = env._sofa_root_node
    controller = env.mcr_controller_sofa
    instrument = controller.instrument.InstrumentCombined
    collision_node = instrument.getChild("mcr_collis")
    collision_dofs = collision_node.getObject("CollisionDOFs")
    beam_dofs = instrument.getObject("DOFs")
    fingerprint = hashlib.sha256(
        as_array(collision_dofs.position).tobytes()
        + as_array(beam_dofs.position).tobytes()
        + as_array(beam_dofs.velocity).tobytes()
    ).hexdigest()
    original_tip = np.asarray(controller.get_pos_quat_catheter_tip()[:3], dtype=np.float64).copy()
    original_tip_mm = [3.992588756316794, -450.1349417194569, -44.56123628899209]
    baseline_tip_error_mm = float(np.linalg.norm(original_tip * 1000 - original_tip_mm))
    if baseline_tip_error_mm > 1e-4:
        raise RuntimeError(f"Baseline 1095 state did not reproduce: {baseline_tip_error_mm} mm")
    raw_action, _ = model.predict(observation, deterministic=True)
    raw_action = np.asarray(raw_action, dtype=np.float32).reshape(3)

    def capture(label):
        tip = np.asarray(controller.get_pos_quat_catheter_tip()[:3], dtype=np.float64).copy()
        env._update_sdf_safety_state(tip, advance_failure_counters=False)
        beam_pos = as_array(beam_dofs.position)
        beam_free = as_array(beam_dofs.free_position)
        coll_pos = as_array(collision_dofs.position)
        coll_free = as_array(collision_dofs.free_position)
        contacts = [obj for obj in collision_node.objects if obj.getClassName() == "FrictionContact"]
        return {
            "label": label,
            "root_time_s": float(root.getTime()),
            "root_dt_s": float(root.getDt()),
            "tip_mm": (tip * 1000).tolist(),
            "tip_surface_clearance_mm": float(env.current_sdf_surface_clearance * 1000),
            "body_min_surface_clearance_mm": float(env.current_sdf_body_min_surface_clearance * 1000),
            "friction_contact_count": len(contacts),
            "collision_constraint_rows": count_constraint_rows(collision_dofs),
            "beam_constraint_rows": count_constraint_rows(beam_dofs),
            "beam_tip_corrected_mm": float(np.linalg.norm((beam_pos[-1, :3] - beam_free[-1, :3]) * 1000)),
            "beam_max_corrected_mm": float(np.max(np.linalg.norm((beam_pos[:, :3] - beam_free[:, :3]) * 1000, axis=1))),
            "collision_tip_corrected_mm": float(np.linalg.norm((coll_pos[-1, :3] - coll_free[-1, :3]) * 1000)),
            "xtip_m": float(controller._getXTipValue()),
            "pending_insert_m": float(controller.pending_insert_delta),
        }

    def branch(name, dt, physics_steps):
        root.dt.value = dt
        env.time_step = dt
        env.frame_skip = physics_steps
        before = capture("fork_state")
        action = np.nan_to_num(raw_action, nan=0.0, posinf=1.0, neginf=-1.0)
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        previous = env._last_smoothed_action.copy()
        delta = np.clip(action - previous, -float(env.max_action_delta), float(env.max_action_delta))
        smoothed = np.clip(previous + delta, -1.0, 1.0).astype(np.float32)
        env._prev_smoothed_action = previous
        env._last_smoothed_action = smoothed.copy()
        env._do_action(smoothed)
        requested_insert_m = float(controller.pending_insert_delta)
        if physics_steps == 2 and abs(requested_insert_m) > 1e-12:
            controller.insert_substep_max = abs(requested_insert_m) / 2.0
        rows = []
        previous_tip = np.asarray(controller.get_pos_quat_catheter_tip()[:3], dtype=np.float64).copy()
        for substep in range(1, physics_steps + 1):
            env.sofa_simulation.animate(root, root.getDt())
            row = capture(f"physics_{substep}")
            tip = np.asarray(row["tip_mm"], dtype=np.float64) / 1000
            row["tip_move_mm"] = float(np.linalg.norm(tip - previous_tip) * 1000)
            previous_tip = tip
            rows.append(row)
        return {
            "branch": name,
            "fork_step": args.fork_step,
            "fork_fingerprint_sha256": fingerprint,
            "baseline_tip_error_mm": baseline_tip_error_mm,
            "raw_action": raw_action.tolist(),
            "smoothed_action": smoothed.tolist(),
            "dt_s": dt,
            "physics_steps": physics_steps,
            "requested_insert_m": requested_insert_m,
            "before": before,
            "rows": rows,
        }

    child_path = output.with_name(output.stem + "_B.json")
    child_pid = os.fork()
    if child_pid == 0:
        try:
            child_path.write_text(json.dumps(branch("B", 0.005, 2), indent=2) + "\n")
            os._exit(0)
        except BaseException:
            child_path.write_text(json.dumps({"error": traceback.format_exc()}, indent=2) + "\n")
            os._exit(1)
    result_a = branch("A", 0.01, 1)
    _, status = os.waitpid(child_pid, 0)
    result_b = json.loads(child_path.read_text())
    combined = {"A": result_a, "B": result_b, "child_status": status, "checkpoint": str(checkpoint), "seed": args.seed}
    output.write_text(json.dumps(combined, indent=2) + "\n")
    print(json.dumps(combined, sort_keys=True), flush=True)
    if status != 0 or "error" in result_b:
        raise RuntimeError("Forked branch B failed")
    os._exit(0)


if __name__ == "__main__":
    main()
