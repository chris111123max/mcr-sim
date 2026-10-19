#!/usr/bin/env python3
"""Test-only 32-worker Production Beam safety AUDIT throughput benchmark."""
from __future__ import annotations

import json
import os
from pathlib import Path
import random
import sys
import time
from collections import Counter

import numpy as np

THIS = Path(__file__).resolve()
PYTHON_ROOT = THIS.parents[3]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from mcr_sim.beam_safety_env import BeamSafetyMCREnv
from mcr_sim.beam_sdf_unilateral_fast import (
    ACTIVE_BAND_M,
    DENSE_SPACING_M,
    MARGIN_M,
    resolve_plugin_path,
)
from mcr_sim.mcr_rl_env import EnvType
from mcr_sim.rl_core.base import RenderMode


CHECKPOINT = Path(
    "/data/home/3220251075/mcr_sim/mcr_project/mCR_simulator-master/"
    "training_runs/ppo_v15_2b_physics_wall_10Nm_20260922_215738/models/"
    "ppo_base_epoch_074_episodes_07400.zip"
)
SEED = 15204
N_ENVS = 32
WARMUP_VECTOR_STEPS = 100
MEASURED_VECTOR_STEPS = 600
DT = 0.005
PHYSICS_SUBSTEPS = 2
MARGIN_M = 1.0e-4
DENSE_SPACING_M = 1.0e-5
ACTIVE_BAND_M = 2.0e-4
NEAR_M = MARGIN_M + ACTIVE_BAND_M
COMMITTED_LIMIT_M = 1.0e-6
BASELINE_ENV_STEPS_PER_S = 171.69


def _percentiles(values):
    arr = np.asarray(values, dtype=np.float64)
    if not arr.size:
        return {"mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


class TrackedProductionBeamSafetyEnv(BeamSafetyMCREnv):
    """Collect read-only per-worker safety telemetry around production behavior."""

    def __init__(self, *args, **kwargs):
        self._bench_record_counts = Counter()
        self._bench_original_animate = None
        super().__init__(*args, **kwargs)

    def reset(self, *args, **kwargs):
        observation, info = super().reset(*args, **kwargs)
        self._bench_record_counts = Counter()
        self._bench_record_counts["reset_audit"] = int(
            self._beam_audit_count > 0
        )
        self._bench_original_animate = self.sofa_simulation.animate
        base_animate = self._bench_original_animate

        def capture(root, dt):
            result = base_animate(root, dt)
            record = self._beam_last_substep_record
            if not isinstance(record, dict):
                self._bench_record_counts["record_missing"] += 1
                return result
            self._bench_record_counts["physics_substeps"] += 1
            if bool(record.get("rows_required")):
                self._bench_record_counts["active_substeps"] += 1
                self._bench_record_counts["row_build_events"] += 1
                self._bench_record_counts[
                    "total_rows"
                ] += int(record.get("row_count") or 0)
            free = record.get("q_free_clearance_m")
            if free is None or not np.isfinite(float(free)):
                self._bench_record_counts["nonfinite_count"] += 1
            elif float(free) <= NEAR_M:
                self._bench_record_counts["near_wall_substeps"] += 1
            if bool(record.get("committed_audited")):
                self._bench_record_counts["committed_audits_step"] += 1
                clearance = record.get("committed_audited_clearance_m")
                if clearance is not None and float(clearance) < -COMMITTED_LIMIT_M:
                    self._bench_record_counts[
                        "audited_penetration_violations"
                    ] += 1
                reason = str(record.get("committed_audit_reason"))
                self._bench_record_counts["audit_reason:" + reason] += 1
            else:
                self._bench_record_counts["committed_skips_step"] += 1
            if record.get("planning_status") == "FAIL":
                self._bench_record_counts["planning_failures"] += 1
            if bool(record.get("constraint_active_mismatch")):
                self._bench_record_counts["active_count_mismatches"] += 1
            if record.get("reason") == "FAST_BUILDER_INVARIANT_VIOLATION":
                self._bench_record_counts["builder_invariant_failures"] += 1
            return result

        self.sofa_simulation.animate = capture
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        info = dict(info) if isinstance(info, dict) else {}
        info["beam_safety_benchmark"] = dict(self._bench_record_counts)
        return observation, reward, terminated, truncated, info


def _env_thunk(rank: int):
    def make():
        np.random.seed(SEED)
        random.seed(SEED)
        os.environ["MCR_CONSTRAINT_SOLVER"] = "generic"
        os.environ["MCR_SOFA_DT"] = str(DT)
        return TrackedProductionBeamSafetyEnv(
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
                "sdf_unilateral_constraint_construct": False,
                "sdf_unilateral_constraint_enabled": False,
                "sampling_slot": int(rank),
            },
            env_type=EnvType.AORTIC,
            render_mode=RenderMode.NONE,
            max_episode_steps=2048,
            time_step=DT,
            frame_skip=1,
            physics_substeps=PHYSICS_SUBSTEPS,
            beam_safety_plugin=resolve_plugin_path(),
            beam_safety_validation_mode="audit",
            beam_safety_audit_interval_substeps=16,
            beam_safety_audit_near_clearance_m=NEAR_M,
            beam_safety_committed_penetration_limit_m=COMMITTED_LIMIT_M,
        )
    return make


def main():
    import torch
    import torch_npu
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from mcr_sim.distributed import DistributedPPO, resolve_device

    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--checkpoint", default=str(CHECKPOINT))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = output.with_suffix(".report.md")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    device = resolve_device("npu", distributed=False, local_rank=0)
    npu_preflight = {
        "resolved": str(device.resolved),
        "accelerator": str(device.accelerator),
        "available": bool(torch.npu.is_available()),
        "device_count": int(torch.npu.device_count()),
        "torch": str(torch.__version__),
        "torch_npu": str(getattr(torch_npu, "__version__", "unknown")),
    }
    if (
        not npu_preflight["available"]
        or npu_preflight["resolved"] != "npu:0"
        or npu_preflight["accelerator"] != "npu"
    ):
        payload = {
            "decision": "INCONCLUSIVE",
            "reason": "NPU_PREFLIGHT_FAILED_NO_WORKERS_STARTED",
            "npu_preflight": npu_preflight,
            "workers_started": False,
        }
        output.write_text(json.dumps(payload, indent=2) + "\n")
        report.write_text(
            "FINAL = INCONCLUSIVE\nNPU preflight failed; no SOFA workers started.\n"
        )
        print(json.dumps(payload, sort_keys=True), flush=True)
        return

    model = DistributedPPO.load(str(checkpoint), device="npu")
    model.policy.set_training_mode(False)
    try:
        policy_device = str(next(model.policy.parameters()).device)
    except Exception:
        policy_device = "unavailable"
    if not policy_device.startswith("npu"):
        payload = {
            "decision": "INCONCLUSIVE",
            "reason": "POLICY_NOT_ON_NPU_NO_WORKERS_STARTED",
            "npu_preflight": npu_preflight,
            "policy_device": policy_device,
            "workers_started": False,
        }
        output.write_text(json.dumps(payload, indent=2) + "\n")
        report.write_text(
            "FINAL = INCONCLUSIVE\nPolicy did not resolve to NPU; no SOFA workers started.\n"
        )
        print(json.dumps(payload, sort_keys=True), flush=True)
        return

    envs = None
    failure = None
    first_failure = None
    vector_latencies = []
    vec_step_latencies = []
    policy_latencies = []
    dones_seen = 0
    last_infos = []
    measured_started = None
    measured_finished = None
    wall_started = time.perf_counter()
    try:
        envs = SubprocVecEnv(
            [_env_thunk(i) for i in range(N_ENVS)],
            start_method="spawn",
        )
        envs.seed(SEED)
        observations = envs.reset()
        measured_wall_start = None

        total_vector_steps = WARMUP_VECTOR_STEPS + MEASURED_VECTOR_STEPS
        for vector_index in range(total_vector_steps):
            if vector_index == WARMUP_VECTOR_STEPS:
                measured_wall_start = time.perf_counter()
                measured_started = measured_wall_start

            iteration_start = time.perf_counter()
            if vector_index >= WARMUP_VECTOR_STEPS:
                try:
                    torch.npu.synchronize()
                except Exception:
                    pass
            policy_start = time.perf_counter()
            actions, _ = model.predict(observations, deterministic=True)
            if vector_index >= WARMUP_VECTOR_STEPS:
                try:
                    torch.npu.synchronize()
                except Exception:
                    pass
            policy_end = time.perf_counter()
            observations, rewards, dones, infos = envs.step(actions)
            iteration_end = time.perf_counter()
            last_infos = list(infos)
            if np.any(dones):
                dones_seen += int(np.sum(dones))
                first_failure = first_failure or {
                    "vector_step": int(vector_index + 1),
                    "reason": "UNEXPECTED_EPISODE_TERMINATION_BEFORE_700_STEPS",
                    "done_envs": np.where(dones)[0].astype(int).tolist(),
                }
                failure = "UNEXPECTED_EPISODE_TERMINATION"
                break
            if vector_index >= WARMUP_VECTOR_STEPS:
                policy_latencies.append(policy_end - policy_start)
                vec_step_latencies.append(iteration_end - policy_end)
                vector_latencies.append(iteration_end - iteration_start)

        measured_finished = time.perf_counter()
    except Exception as exc:
        failure = f"{type(exc).__name__}:{exc}"
        first_failure = first_failure or {
            "vector_step": len(vector_latencies) + WARMUP_VECTOR_STEPS + 1,
            "reason": failure,
        }
    finally:
        if envs is not None:
            try:
                envs.close()
            except Exception as exc:
                failure = failure or f"VECENV_CLOSE:{type(exc).__name__}:{exc}"

    measured_wall = (
        max(0.0, measured_finished - measured_started)
        if measured_started is not None and measured_finished is not None
        else 0.0
    )
    measured_vectors = len(vector_latencies)
    env_steps_measured = measured_vectors * N_ENVS
    physics_measured = env_steps_measured * PHYSICS_SUBSTEPS
    vector_sps = measured_vectors / measured_wall if measured_wall > 0 else 0.0
    env_sps = env_steps_measured / measured_wall if measured_wall > 0 else 0.0
    physics_sps = physics_measured / measured_wall if measured_wall > 0 else 0.0

    aggregate = Counter()
    final_summaries = []
    for info in last_infos:
        if not isinstance(info, dict):
            continue
        beam = info.get("beam_safety") or {}
        tracked = info.get("beam_safety_benchmark") or {}
        final_summaries.append({"beam_safety": beam, "tracked": tracked})
        for key in (
            "physics_substeps",
            "active_substeps",
            "row_build_events",
            "total_rows",
            "near_wall_substeps",
            "committed_audits_step",
            "committed_skips_step",
            "audited_penetration_violations",
            "active_count_mismatches",
            "builder_invariant_failures",
            "planning_failures",
            "nonfinite_count",
            "record_missing",
            "reset_audit",
        ):
            aggregate[key] += int(tracked.get(key, 0) or 0)
        aggregate["audit_total_including_reset"] += int(
            beam.get("committed_audits", 0) or 0
        )
        aggregate["audit_skips_from_production"] += int(
            beam.get("committed_audit_skips", 0) or 0
        )
        aggregate["production_active_substeps"] += int(
            beam.get("active_substeps", 0) or 0
        )
        aggregate["production_near_wall_substeps"] += int(
            beam.get("near_wall_substeps", 0) or 0
        )
        aggregate["reset_audits_from_production"] += int(
            tracked.get("reset_audit", 0) or 0
        )
        for key, value in tracked.items():
            if str(key).startswith("audit_reason:"):
                aggregate[str(key)] += int(value or 0)

    expected_env_steps_total = total_vector_steps * N_ENVS
    expected_substeps_total = expected_env_steps_total * PHYSICS_SUBSTEPS
    safety_failure_total = sum(
        int(aggregate[key])
        for key in (
            "audited_penetration_violations",
            "active_count_mismatches",
            "builder_invariant_failures",
            "planning_failures",
            "nonfinite_count",
            "record_missing",
        )
    )
    if failure is None and measured_vectors != MEASURED_VECTOR_STEPS:
        failure = "MEASURED_VECTOR_STEP_COUNT_MISMATCH"
    if failure is None and aggregate["physics_substeps"] != expected_substeps_total:
        failure = "PHYSICS_SUBSTEP_COUNT_MISMATCH"
    if failure is None and aggregate["reset_audits_from_production"] != N_ENVS:
        failure = "RESET_AUDIT_MISSING"
    if failure is None and safety_failure_total:
        failure = "SAFETY_TELEMETRY_FAILURE"
    if failure is None and aggregate["active_substeps"] != aggregate["production_active_substeps"]:
        failure = "ACTIVE_TELEMETRY_MISMATCH"
    if failure is None and aggregate["near_wall_substeps"] != aggregate["production_near_wall_substeps"]:
        failure = "NEAR_WALL_TELEMETRY_MISMATCH"

    speedup = (
        env_sps / BASELINE_ENV_STEPS_PER_S
        if BASELINE_ENV_STEPS_PER_S > 0
        else None
    )
    decision = "PASS" if failure is None else "FAIL"
    payload = {
        "decision": decision,
        "reason": failure or "32_ENV_AUDIT_THROUGHPUT_COMPLETED",
        "workers": N_ENVS,
        "start_method": "spawn",
        "vector_steps_total": WARMUP_VECTOR_STEPS + MEASURED_VECTOR_STEPS,
        "warmup_vector_steps": WARMUP_VECTOR_STEPS,
        "measured_vector_steps": measured_vectors,
        "env_steps_total": expected_env_steps_total,
        "env_steps_measured": env_steps_measured,
        "physics_substeps_total": aggregate["physics_substeps"],
        "physics_substeps_measured": physics_measured,
        "checkpoint": str(checkpoint),
        "seed": SEED,
        "policy_device": policy_device,
        "npu_preflight": npu_preflight,
        "measured_wall_s": measured_wall,
        "vector_steps_per_s": vector_sps,
        "env_steps_per_s": env_sps,
        "physics_substeps_per_s": physics_sps,
        "vector_iteration_latency_s": _percentiles(vector_latencies),
        "subproc_vecenv_step_latency_s": _percentiles(vec_step_latencies),
        "policy_inference_latency_s": _percentiles(policy_latencies),
        "baseline_v2_debug_every_substep_env_steps_per_s": BASELINE_ENV_STEPS_PER_S,
        "audit_vs_v2_env_steps_speedup_x": speedup,
        "audit_vs_v2_env_steps_improvement_percent": (
            (speedup - 1.0) * 100.0 if speedup is not None else None
        ),
        "telemetry": dict(aggregate),
        "safety_failures": safety_failure_total,
        "dones_seen": dones_seen,
        "first_failure": first_failure,
        "workers_final_summaries": final_summaries,
        "training_started": False,
        "optimizer_updates": 0,
        "production_files_modified": False,
        "audit_mode": {
            "requested_margin_m": MARGIN_M,
            "dense_spacing_m": DENSE_SPACING_M,
            "physics_substeps": PHYSICS_SUBSTEPS,
            "dt_s": DT,
            "audit_interval_substeps": 16,
            "near_clearance_m": NEAR_M,
            "committed_penetration_limit_m": COMMITTED_LIMIT_M,
        },
        "wall_total_s": time.perf_counter() - wall_started,
    }
    output.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "PRODUCTION BEAM SAFETY AUDIT 32-ENV THROUGHPUT",
        "FINAL = " + decision,
        "Workers = " + str(N_ENVS) + " (spawn)",
        "Vector steps = " + str(WARMUP_VECTOR_STEPS + measured_vectors),
        "Warmup/measured = "
        + str(WARMUP_VECTOR_STEPS)
        + "/"
        + str(measured_vectors),
        "Measured wall (s) = " + str(measured_wall),
        "Vector steps/s = " + str(vector_sps),
        "Env steps/s = " + str(env_sps),
        "Physics substeps/s = " + str(physics_sps),
        "Vector latency = " + str(payload["vector_iteration_latency_s"]),
        "SubprocVecEnv step latency = " + str(payload["subproc_vecenv_step_latency_s"]),
        "Policy latency = " + str(payload["policy_inference_latency_s"]),
        "V2 debug reference env steps/s = " + str(BASELINE_ENV_STEPS_PER_S),
        "AUDIT/V2 speedup = " + str(speedup),
        "Physics substeps total = " + str(aggregate["physics_substeps"]),
        "Active substeps = " + str(aggregate["active_substeps"]),
        "Near-wall substeps = " + str(aggregate["near_wall_substeps"]),
        "Committed audits incl. reset = " + str(aggregate["audit_total_including_reset"]),
        "Committed skips = " + str(aggregate["audit_skips_from_production"]),
        "Safety failures = " + str(safety_failure_total),
        "Training started = NO",
        "Production files modified = NO",
    ]
    report.write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "decision": decision,
        "reason": payload["reason"],
        "workers": N_ENVS,
        "measured_vectors": measured_vectors,
        "measured_wall_s": measured_wall,
        "env_steps_per_s": env_sps,
        "physics_substeps_per_s": physics_sps,
        "audit_skips": aggregate["audit_skips_from_production"],
        "safety_failures": safety_failure_total,
        "output": str(output),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
