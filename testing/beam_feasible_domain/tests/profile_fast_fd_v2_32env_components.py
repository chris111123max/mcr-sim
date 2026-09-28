#!/usr/bin/env python3
"""32-env Fast FD + Generic throughput benchmark.

Real execution model:
- 32 independent SOFA21 workers via SubprocVecEnv(start_method="spawn");
- fixed B02 / target_04 environment in every worker;
- epoch74 PPO policy predicts batched actions;
- Fast local-point FD Beam unilateral safety runs inside every worker;
- total 1024 vector steps, first 100 excluded from throughput statistics.

This is a throughput/stress benchmark, not a training run.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv

THIS = Path(__file__).resolve()
TEST_DIR = THIS.parent
BEAM_ROOT = THIS.parents[1]
PYTHON_ROOT = THIS.parents[3]
UNILATERAL_DIR = BEAM_ROOT / "beam_unilateral_lcp"

for p in (TEST_DIR, UNILATERAL_DIR, PYTHON_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from b02_step776_precommit_integration import CHECKPOINT
from beam_linearized_unilateral import find_plugin
from fast_fd_v2_32env_profile_worker import (
    INFO_KEY,
    make_fast_fd_v2_profile_env,
)
from mcr_sim.distributed import (
    DistributedPPO,
    configure_npu_execution,
    resolve_device,
)


N_ENVS = 32
TOTAL_VECTOR_STEPS = 700
WARMUP_VECTOR_STEPS = 100
PHYSICS_SUBSTEPS_PER_ENV_STEP = 2
BASE_SEED = 15204

RUNTIME = (
    BEAM_ROOT
    / "_runtime"
    / "beam_unilateral_fast_fd_v2/profile_32env"
)
DEFAULT_BUILD_DIRS = (
    BEAM_ROOT
    / "_runtime"
    / "beam_unilateral_lcp_step665"
    / "native_build",
    BEAM_ROOT
    / "_runtime"
    / "beam_unilateral_generic_full_episode"
    / "native_build",
)


def _find_default_plugin() -> Path | None:
    for build_dir in DEFAULT_BUILD_DIRS:
        plugin = find_plugin(build_dir)
        if plugin is not None:
            return plugin
    return None


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, Counter):
        return dict(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def _percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "mean": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.quantile(arr, 0.50)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": float(np.max(arr)),
    }


def _read_cpu_quota() -> dict[str, Any]:
    result: dict[str, Any] = {
        "os_cpu_count": os.cpu_count(),
        "cpu_max_raw": None,
        "effective_cpu_quota_cores": None,
    }
    for path in (Path("/sys/fs/cgroup/cpu.max"),):
        try:
            raw = path.read_text().strip()
        except Exception:
            continue
        result["cpu_max_raw"] = raw
        fields = raw.split()
        if len(fields) == 2 and fields[0] != "max":
            try:
                result["effective_cpu_quota_cores"] = (
                    float(fields[0]) / float(fields[1])
                )
            except Exception:
                pass
        break
    return result


def _new_aggregate() -> dict[str, Any]:
    return {
        "env_steps": 0,
        "physics_substeps": 0,
        "row_build_substeps": 0,
        "negative_free_substeps": 0,
        "below_minus_0p001mm_free_substeps": 0,
        "row_count_total": 0,
        "committed_violation_count": 0,
        "active_count_mismatch_count": 0,
        "fast_builder_invariant_failure_count": 0,
        "non_finite_count": 0,
        "planning_failure_count": 0,
        "batched_fd_point_count": 0,
        "batched_fd_sdf_query_count": 0,
        "post_selection_full_beam_profile_count": 0,
        "min_free_clearance_m": float("inf"),
        "min_committed_clearance_m": float("inf"),
        "max_committed_penetration_m": 0.0,
        "row_build_runtimes_s": [],
        "row_counts": [],
        "support_mode_counts": Counter(),
        "worker_step_cpu_s": [],
        "worker_step_wall_s": [],
        "done_count": 0,
    }


def _accumulate(
    aggregate: dict[str, Any],
    bench: dict[str, Any],
    done: bool,
) -> None:
    aggregate["env_steps"] += 1
    aggregate["physics_substeps"] += int(
        bench.get("physics_substeps", 0)
    )
    aggregate["row_build_substeps"] += int(
        bench.get("row_build_substeps", 0)
    )
    aggregate["negative_free_substeps"] += int(
        bench.get("negative_free_substeps", 0)
    )
    aggregate["below_minus_0p001mm_free_substeps"] += int(
        bench.get("below_minus_0p001mm_free_substeps", 0)
    )
    aggregate["row_count_total"] += int(
        bench.get("row_count_total", 0)
    )
    aggregate["committed_violation_count"] += int(
        bench.get("committed_violation_count", 0)
    )
    aggregate["active_count_mismatch_count"] += int(
        bench.get("active_count_mismatch_count", 0)
    )
    aggregate["fast_builder_invariant_failure_count"] += int(
        bench.get("fast_builder_invariant_failure_count", 0)
    )
    aggregate["non_finite_count"] += int(
        bench.get("non_finite_count", 0)
    )
    aggregate["planning_failure_count"] += int(
        bench.get("planning_failure_count", 0)
    )
    aggregate["batched_fd_point_count"] += int(
        bench.get("batched_fd_point_count", 0)
    )
    aggregate["batched_fd_sdf_query_count"] += int(
        bench.get("batched_fd_sdf_query_count", 0)
    )
    aggregate["post_selection_full_beam_profile_count"] += int(
        bench.get("post_selection_full_beam_profile_count", 0)
    )

    free = bench.get("min_free_clearance_m")
    if free is not None and np.isfinite(float(free)):
        aggregate["min_free_clearance_m"] = min(
            float(aggregate["min_free_clearance_m"]),
            float(free),
        )

    committed = bench.get("min_committed_clearance_m")
    if committed is not None and np.isfinite(float(committed)):
        aggregate["min_committed_clearance_m"] = min(
            float(aggregate["min_committed_clearance_m"]),
            float(committed),
        )

    max_pen = bench.get("max_committed_penetration_m")
    if max_pen is not None and np.isfinite(float(max_pen)):
        aggregate["max_committed_penetration_m"] = max(
            float(aggregate["max_committed_penetration_m"]),
            float(max_pen),
        )

    aggregate["row_build_runtimes_s"].extend(
        float(v)
        for v in bench.get("row_build_runtimes_s", [])
    )
    aggregate["row_counts"].extend(
        int(v) for v in bench.get("row_counts", [])
    )
    for mode in bench.get("support_modes", []):
        aggregate["support_mode_counts"][str(mode)] += 1

    aggregate["worker_step_cpu_s"].append(
        float(bench.get("step_cpu_s", 0.0))
    )
    aggregate["worker_step_wall_s"].append(
        float(bench.get("step_wall_s", 0.0))
    )
    if done:
        aggregate["done_count"] += 1


def _finalize_aggregate(aggregate: dict[str, Any]) -> dict[str, Any]:
    out = dict(aggregate)
    if not np.isfinite(float(out["min_free_clearance_m"])):
        out["min_free_clearance_m"] = None
    if not np.isfinite(float(out["min_committed_clearance_m"])):
        out["min_committed_clearance_m"] = None

    out["min_free_clearance_mm"] = (
        None
        if out["min_free_clearance_m"] is None
        else float(out["min_free_clearance_m"]) * 1000.0
    )
    out["min_committed_clearance_mm"] = (
        None
        if out["min_committed_clearance_m"] is None
        else float(out["min_committed_clearance_m"]) * 1000.0
    )
    out["max_committed_penetration_mm"] = (
        float(out["max_committed_penetration_m"]) * 1000.0
    )

    out["row_build_latency_s"] = _percentiles(
        list(out["row_build_runtimes_s"])
    )
    out["worker_step_cpu_s_stats"] = _percentiles(
        list(out["worker_step_cpu_s"])
    )
    out["worker_step_wall_s_stats"] = _percentiles(
        list(out["worker_step_wall_s"])
    )
    out["row_count_distribution"] = dict(
        Counter(str(v) for v in out["row_counts"])
    )
    out["support_mode_counts"] = dict(out["support_mode_counts"])

    # Avoid bloating final JSON with tens of thousands of raw samples.
    out.pop("row_build_runtimes_s", None)
    out.pop("row_counts", None)
    out.pop("worker_step_cpu_s", None)
    out.pop("worker_step_wall_s", None)
    return out


def _failure_from_bench(bench: dict[str, Any]) -> str | None:
    failures = list(bench.get("failure_reasons", []))
    if failures:
        return ";".join(str(v) for v in failures)
    if bool(bench.get("physics_substep_count_mismatch", False)):
        return "PHYSICS_SUBSTEP_COUNT_MISMATCH"
    return None


def _write_report(payload: dict[str, Any], report_path: Path) -> None:
    perf = payload["measured_performance"]
    profile = payload["component_profile"]
    lines = ["32-ENV FAST FD V2 COMPONENT PROFILER", f"FINAL: {payload['decision']}", f"Reason: {payload['reason']}", f"Measured wall: {perf['measured_wall_s']:.3f} s", f"Env steps/s: {perf['env_steps_per_s']:.3f}", ""]
    for group in ("safe_free", "active"):
        item = profile["groups"][group]
        lines += [f"{group.upper()} SUBSTEP count={item['count']}", "phase | mean ms | p50 | p95 | p99 | max | fraction"]
        for key, metrics in item["phases"].items():
            stat = metrics["wall_s"]
            if stat["mean"] is None: continue
            lines.append(f"{key} | {stat['mean']*1000:.4f} | {stat['p50']*1000:.4f} | {stat['p95']*1000:.4f} | {stat['p99']*1000:.4f} | {stat['max']*1000:.4f} | {metrics['fraction_of_total']*100:.2f}%")
        lines.append("")
    lines += ["VECTOR CONCURRENCY", "active envs | count | mean ms | p95 ms"]
    for key, item in profile["concurrency_groups"].items():
        stat=item["latency_s"]
        lines.append(f"{key} | {item['count']} | {stat['mean']*1000 if stat['mean'] is not None else 'N/A'} | {stat['p95']*1000 if stat['p95'] is not None else 'N/A'}")
    lines += ["", "V2 speedups vs V1: " + str(profile["v2_speedups_vs_v1"]), "", "Duplicate dense work: " + str(profile["duplicate_dense_work"]), "", "Safety: " + str({k: payload["all_steps_aggregate"][k] for k in ("committed_violation_count", "active_count_mismatch_count", "fast_builder_invariant_failure_count", "non_finite_count", "planning_failure_count")}), "Production modified: NO; training: NO; optimizer updates: 0."]
    report_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-envs", type=int, default=N_ENVS)
    parser.add_argument(
        "--vector-steps",
        type=int,
        default=TOTAL_VECTOR_STEPS,
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=WARMUP_VECTOR_STEPS,
    )
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--checkpoint", default=str(CHECKPOINT))
    parser.add_argument("--device", default="npu")
    parser.add_argument("--plugin-lib", default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--output",
        default=str(
            RUNTIME / "results" / "component_profile_v2.json"
        ),
    )
    args = parser.parse_args()

    if int(args.n_envs) != N_ENVS:
        raise ValueError("this benchmark is fixed to exactly 32 envs")
    if int(args.vector_steps) != TOTAL_VECTOR_STEPS:
        raise ValueError("this profiler is fixed to exactly 700 vector steps")
    if int(args.warmup_steps) != WARMUP_VECTOR_STEPS:
        raise ValueError("this benchmark is fixed to exactly 100 warmup steps")

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path = output.with_name("component_profile_v2_report.md")
    trace_path = output.with_name("vector_profile_v2_trace.jsonl")

    plugin = (
        Path(args.plugin_lib).expanduser().resolve()
        if args.plugin_lib
        else _find_default_plugin()
    )
    if plugin is None or not plugin.is_file():
        raise RuntimeError(
            "libMCRBeamLinearizedUnilateral.so not found"
        )

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    env_fns = [
        make_fast_fd_v2_profile_env(
            rank=i,
            plugin_path=str(plugin),
            base_seed=int(args.seed),
        )
        for i in range(N_ENVS)
    ]

    vec_env = None
    trace_file = None
    first_failure = None
    decision = "INCONCLUSIVE"
    reason = "NOT_RUN"

    all_aggregate = _new_aggregate()
    measured_aggregate = _new_aggregate()
    policy_latencies: list[float] = []
    vec_step_latencies: list[float] = []
    vector_iteration_latencies: list[float] = []
    measured_vector_wall_start = None
    measured_vector_wall_s = 0.0

    profile_records = {"safe_free": [], "active": []}
    env_postphysics_wall = []
    env_postphysics_cpu = []
    concurrency_groups = {k: [] for k in ("0", "1-8", "9-16", "17-24", "25-31", "32")}
    envs_with_rows_distribution = Counter()
    envs_with_negative_distribution = Counter()
    max_envs_with_rows = 0

    cpu_info = _read_cpu_quota()

    # Match the production PPO device path.  resolve_device imports torch_npu
    # when requested, validates torch.npu availability, activates device 0,
    # and returns the concrete SB3/PyTorch device string (normally "npu:0").
    device_selection = resolve_device(
        requested=str(args.device),
        distributed=False,
        local_rank=0,
    )
    npu_execution = configure_npu_execution(
        device_selection.accelerator,
        enabled=True,
    )
    resolved_device = str(device_selection.resolved)

    print(
        "[FAST_FD_32ENV][DEVICE] "
        f"requested={args.device} resolved={resolved_device} "
        f"accelerator={device_selection.accelerator} "
        f"npu_execution={npu_execution}",
        flush=True,
    )

    try:
        vec_env = SubprocVecEnv(
            env_fns,
            start_method="spawn",
        )
        vec_env.seed(int(args.seed))
        observations = vec_env.reset()

        model = DistributedPPO.load(
            str(checkpoint),
            device=resolved_device,
        )
        model.policy.set_training_mode(False)

        # One dry prediction before vector-step warm-up so lazy accelerator
        # initialization/graph setup does not pollute the first timing.
        model.predict(observations, deterministic=True)

        trace_file = trace_path.open("w", buffering=1)

        for vector_step in range(1, TOTAL_VECTOR_STEPS + 1):
            iteration0 = time.perf_counter()

            infer0 = time.perf_counter()
            actions, _ = model.predict(
                observations,
                deterministic=True,
            )
            infer_s = float(time.perf_counter() - infer0)

            env0 = time.perf_counter()
            observations, rewards, dones, infos = vec_env.step(actions)
            env_s = float(time.perf_counter() - env0)
            iteration_s = float(time.perf_counter() - iteration0)

            measured = vector_step > WARMUP_VECTOR_STEPS
            if vector_step == WARMUP_VECTOR_STEPS + 1:
                measured_vector_wall_start = iteration0

            envs_with_rows = 0
            envs_with_negative = 0
            vector_failure = None

            for env_index, (done, info) in enumerate(zip(dones, infos)):
                bench = (
                    info.get(INFO_KEY)
                    if isinstance(info, dict)
                    else None
                )
                if not isinstance(bench, dict):
                    vector_failure = (
                        f"WORKER_{env_index}_BENCH_INFO_MISSING"
                    )
                    continue

                _accumulate(all_aggregate, bench, bool(done))
                if measured:
                    _accumulate(measured_aggregate, bench, bool(done))
                    env_postphysics_wall.append(float(bench.get("env_postphysics_remainder_s", 0.0)))
                    env_postphysics_cpu.append(float(bench.get("env_postphysics_remainder_cpu_s", 0.0)))
                    subprofiles = bench.get("profile_substeps", [])
                    if len(subprofiles) != PHYSICS_SUBSTEPS_PER_ENV_STEP:
                        vector_failure = vector_failure or f"WORKER_{env_index}_TIMING_TELEMETRY_MISSING"
                    for rec in subprofiles:
                        group = "active" if bool(rec.get("rows_required", False)) else "safe_free"
                        required = ["q_prev_measure_s", "q_free_measure_s", "gate_s", "post_planner_sofa_remainder_s", "committed_dense_validation_s", "total_physics_substep_s", "other_remainder_s"]
                        if group == "active":
                            required += ["fast_builder_base_profile_s", "support_mapping_s", "fd_point_generation_s", "fd_batched_sdf_s", "jacobian_assembly_s", "constraint_write_s"]
                        if any(k not in rec for k in required):
                            vector_failure = vector_failure or f"WORKER_{env_index}_TIMING_TELEMETRY_MISSING"
                        profile_records[group].append(rec)

                if int(bench.get("row_build_substeps", 0)) > 0:
                    envs_with_rows += 1
                if int(bench.get("negative_free_substeps", 0)) > 0:
                    envs_with_negative += 1

                worker_failure = _failure_from_bench(bench)
                if worker_failure and vector_failure is None:
                    vector_failure = (
                        f"WORKER_{env_index}:{worker_failure}"
                    )

            if measured:
                policy_latencies.append(infer_s)
                vec_step_latencies.append(env_s)
                vector_iteration_latencies.append(iteration_s)
                envs_with_rows_distribution[str(envs_with_rows)] += 1
                envs_with_negative_distribution[
                    str(envs_with_negative)
                ] += 1
                max_envs_with_rows = max(max_envs_with_rows, envs_with_rows)
                group_key = ("0" if envs_with_rows == 0 else "1-8" if envs_with_rows <= 8 else "9-16" if envs_with_rows <= 16 else "17-24" if envs_with_rows <= 24 else "25-31" if envs_with_rows <= 31 else "32")
                concurrency_groups[group_key].append(iteration_s)

            trace_rec = {
                "vector_step": vector_step,
                "measured": measured,
                "policy_inference_s": infer_s,
                "vec_env_step_s": env_s,
                "vector_iteration_s": iteration_s,
                "envs_with_rows": envs_with_rows,
                "envs_with_negative_free": envs_with_negative,
                "done_envs": int(np.sum(dones)),
                "failure": vector_failure,
            }
            trace_file.write(
                json.dumps(_json_safe(trace_rec), sort_keys=True)
                + "\n"
            )
            trace_file.flush()

            if args.progress_every > 0 and (
                vector_step == 1
                or vector_step % int(args.progress_every) == 0
                or vector_failure is not None
            ):
                measured_so_far = max(
                    0,
                    vector_step - WARMUP_VECTOR_STEPS,
                )
                print(
                    "[FAST_FD_32ENV] "
                    f"vector_step={vector_step}/{TOTAL_VECTOR_STEPS} "
                    f"measured={measured_so_far}/"
                    f"{TOTAL_VECTOR_STEPS-WARMUP_VECTOR_STEPS} "
                    f"env_step_s={env_s:.4f} "
                    f"rows_envs={envs_with_rows}/32 "
                    f"negative_envs={envs_with_negative}/32 "
                    f"failure={vector_failure}",
                    flush=True,
                )

            if vector_failure is not None:
                first_failure = {
                    "vector_step": vector_step,
                    "reason": vector_failure,
                }
                reason = (
                    f"VECTOR_STEP_{vector_step}_{vector_failure}"
                )
                decision = "FAIL"
                break

        if (
            measured_vector_wall_start is not None
            and decision != "FAIL"
        ):
            measured_vector_wall_s = float(
                time.perf_counter() - measured_vector_wall_start
            )
        elif measured_vector_wall_start is not None:
            measured_vector_wall_s = float(
                sum(vector_iteration_latencies)
            )

        all_final = _finalize_aggregate(all_aggregate)
        measured_final = _finalize_aggregate(measured_aggregate)

        measured_vector_steps = len(vector_iteration_latencies)
        measured_env_steps = measured_vector_steps * N_ENVS
        measured_physics_substeps = (
            measured_env_steps * PHYSICS_SUBSTEPS_PER_ENV_STEP
        )

        perf = {
            "measured_wall_s": measured_vector_wall_s,
            "vector_steps": measured_vector_steps,
            "env_steps": measured_env_steps,
            "physics_substeps": measured_physics_substeps,
            "vector_steps_per_s": (
                measured_vector_steps / measured_vector_wall_s
                if measured_vector_wall_s > 0
                else None
            ),
            "env_steps_per_s": (
                measured_env_steps / measured_vector_wall_s
                if measured_vector_wall_s > 0
                else None
            ),
            "physics_substeps_per_s": (
                measured_physics_substeps / measured_vector_wall_s
                if measured_vector_wall_s > 0
                else None
            ),
            "policy_inference_latency_s": _percentiles(
                policy_latencies
            ),
            "vector_env_step_latency_s": _percentiles(
                vec_step_latencies
            ),
            "vector_iteration_latency_s": _percentiles(
                vector_iteration_latencies
            ),
        }

        if decision != "FAIL":
            if measured_vector_steps != (
                TOTAL_VECTOR_STEPS - WARMUP_VECTOR_STEPS
            ):
                decision = "INCONCLUSIVE"
                reason = "MEASURED_VECTOR_STEP_COUNT_MISMATCH"
            elif all_final["committed_violation_count"] != 0:
                decision = "FAIL"
                reason = "COMMITTED_VIOLATION_OBSERVED"
            elif all_final["active_count_mismatch_count"] != 0:
                decision = "FAIL"
                reason = "ACTIVE_COUNT_MISMATCH_OBSERVED"
            elif (
                all_final["fast_builder_invariant_failure_count"]
                != 0
            ):
                decision = "FAIL"
                reason = "FAST_BUILDER_INVARIANT_FAILURE_OBSERVED"
            elif all_final["non_finite_count"] != 0:
                decision = "FAIL"
                reason = "NON_FINITE_STATE_OBSERVED"
            elif all_final["planning_failure_count"] != 0:
                decision = "FAIL"
                reason = "ROW_BUILD_OR_PLANNING_FAILURE_OBSERVED"
            else:
                decision = "PASS"
                reason = "32ENV_FAST_FD_V2_COMPONENT_PROFILE_COMPLETED"

        phase_keys = ("q_prev_measure", "q_free_measure", "gate", "fast_builder_base_profile", "selected_dense_indices", "support_mapping", "fd_point_generation", "fd_batched_sdf", "jacobian_assembly", "constraint_write", "post_planner_sofa_remainder", "committed_dense_validation", "other_remainder", "total_physics_substep")
        phase_summary = {}
        for group_name, records in profile_records.items():
            total_mean = _percentiles([float(r.get("total_physics_substep_s", 0.0)) for r in records])["mean"]
            phases = {}
            for key in phase_keys:
                wall = _percentiles([float(r.get(key + "_s", 0.0)) for r in records])
                cpu = _percentiles([float(r.get(key + "_cpu_s", 0.0)) for r in records])
                phases[key] = {"wall_s": wall, "cpu_s": cpu, "fraction_of_total": (wall["mean"] / total_mean if total_mean else None)}
            phase_summary[group_name] = {"count": len(records), "phases": phases}
        active_records = profile_records["active"]
        duplicate = {"q_prev_count_distribution": dict(Counter(str(r.get("q_prev_full_dense_profile_count", -1)) for r in profile_records["active"] + profile_records["safe_free"])), "external_q_free_count_distribution": dict(Counter(str(r.get("q_free_external_dense_count", 0)) for r in active_records)), "internal_q_free_count_distribution": dict(Counter(str(r.get("q_free_internal_dense_count", 0)) for r in active_records)), "full_q_free_count_distribution": dict(Counter(str(r.get("full_q_free_dense_profile_count", 0)) for r in active_records)), "q_prev_required_by_safety_math": False, "q_prev_reason": "Builder uses q_prev only for shape and finite validation; rows are derived from q_free and SDF."}
        v1_ref = {"safe_free_total_mean_s": 0.0852169, "active_total_mean_s": 0.1680443, "zero_active_vector_mean_s": 0.21185767717709367, "thirtytwo_active_vector_mean_s": 0.37785727481823417, "env_steps_per_s": 142.29818203721095}
        safe_mean = phase_summary["safe_free"]["phases"]["total_physics_substep"]["wall_s"]["mean"]
        active_mean = phase_summary["active"]["phases"]["total_physics_substep"]["wall_s"]["mean"]
        zero_vals = concurrency_groups["0"]
        full_vals = concurrency_groups["32"]
        speedups = {"safe_free_substep_x": v1_ref["safe_free_total_mean_s"] / safe_mean if safe_mean else None, "active_substep_x": v1_ref["active_total_mean_s"] / active_mean if active_mean else None, "zero_active_vector_x": v1_ref["zero_active_vector_mean_s"] / float(np.mean(zero_vals)) if zero_vals else None, "thirtytwo_active_vector_x": v1_ref["thirtytwo_active_vector_mean_s"] / float(np.mean(full_vals)) if full_vals else None, "throughput_x": (measured_env_steps / measured_vector_wall_s) / v1_ref["env_steps_per_s"] if measured_vector_wall_s else None}
        profile_result = {"v1_reference": v1_ref, "v2_speedups_vs_v1": speedups, "groups": phase_summary, "env_postphysics_remainder_s": _percentiles(env_postphysics_wall), "env_postphysics_remainder_cpu_s": _percentiles(env_postphysics_cpu), "duplicate_dense_work": duplicate, "concurrency_groups": {k: {"count": len(v), "latency_s": _percentiles(v)} for k, v in concurrency_groups.items()}}

        payload = {
            "component_profile": profile_result,
            "test": "32-env Fast FD V2 + Generic component profiler",
            "decision": decision,
            "reason": reason,
            "n_envs": N_ENVS,
            "total_vector_steps": TOTAL_VECTOR_STEPS,
            "warmup_vector_steps": WARMUP_VECTOR_STEPS,
            "measured_vector_steps": (
                TOTAL_VECTOR_STEPS - WARMUP_VECTOR_STEPS
            ),
            "total_env_steps_requested": N_ENVS
            * TOTAL_VECTOR_STEPS,
            "measured_env_steps_requested": N_ENVS
            * (TOTAL_VECTOR_STEPS - WARMUP_VECTOR_STEPS),
            "physics_substeps_per_env_step": (
                PHYSICS_SUBSTEPS_PER_ENV_STEP
            ),
            "policy_device_requested": str(args.device),
            "policy_device": resolved_device,
            "policy_accelerator": str(device_selection.accelerator),
            "npu_execution": _json_safe(npu_execution),
            "checkpoint": str(checkpoint),
            "plugin": str(plugin),
            "start_method": "spawn",
            "seed_base": int(args.seed),
            "cpu_info": cpu_info,
            "measured_performance": perf,
            "measured_aggregate": measured_final,
            "all_steps_aggregate": all_final,
            "concurrency": {
                "envs_with_rows_distribution": dict(
                    envs_with_rows_distribution
                ),
                "envs_with_negative_free_distribution": dict(
                    envs_with_negative_distribution
                ),
                "max_envs_with_rows": max_envs_with_rows,
            },
            "first_failure": first_failure,
            "production_files_modified": False,
            "training_started": False,
            "optimizer_updates": 0,
            "engineering_interpretation": (
                "This measures 32 spawned SOFA workers plus batched PPO "
                "inference, but performs no PPO optimizer/training update."
            ),
            "trace_file": str(trace_path),
        }

        output.write_text(
            json.dumps(
                _json_safe(payload),
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        _write_report(payload, report_path)

        print(
            json.dumps(
                {
                    "decision": decision,
                    "reason": reason,
                    "measured_wall_s": perf["measured_wall_s"],
                    "vector_steps_per_s": perf["vector_steps_per_s"],
                    "env_steps_per_s": perf["env_steps_per_s"],
                    "physics_substeps_per_s": perf[
                        "physics_substeps_per_s"
                    ],
                    "vec_step_latency": perf[
                        "vector_env_step_latency_s"
                    ],
                    "row_build_latency": measured_final[
                        "row_build_latency_s"
                    ],
                    "measured_row_build_substeps": measured_final[
                        "row_build_substeps"
                    ],
                    "max_envs_with_rows": max_envs_with_rows,
                    "committed_violations": all_final[
                        "committed_violation_count"
                    ],
                    "output": str(output),
                    "report": str(report_path),
                    "trace": str(trace_path),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    finally:
        if trace_file is not None:
            trace_file.close()
        if vec_env is not None:
            vec_env.close()


if __name__ == "__main__":
    main()
