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
from fast_fd_32env_worker import (
    INFO_KEY,
    make_fast_fd_benchmark_env,
)
from mcr_sim.distributed import DistributedPPO


N_ENVS = 32
TOTAL_VECTOR_STEPS = 1024
WARMUP_VECTOR_STEPS = 100
PHYSICS_SUBSTEPS_PER_ENV_STEP = 2
BASE_SEED = 15204

RUNTIME = (
    BEAM_ROOT
    / "_runtime"
    / "beam_unilateral_fast_fd_32env_benchmark"
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
    agg = payload["measured_aggregate"]
    safety = payload["all_steps_aggregate"]
    row = agg["row_build_latency_s"]
    vec = perf["vector_env_step_latency_s"]
    total = perf["vector_iteration_latency_s"]

    lines = [
        "32-ENV FAST FD + GENERIC THROUGHPUT BENCHMARK",
        "=============================================",
        "",
        f"FINAL: {payload['decision']}",
        f"Reason: {payload['reason']}",
        "",
        "Configuration",
        "-------------",
        f"n_envs: {payload['n_envs']}",
        f"total vector steps: {payload['total_vector_steps']}",
        f"warmup vector steps: {payload['warmup_vector_steps']}",
        f"measured vector steps: {payload['measured_vector_steps']}",
        f"policy device: {payload['policy_device']}",
        f"SubprocVecEnv start method: {payload['start_method']}",
        f"CPU quota: {payload['cpu_info']}",
        "",
        "Measured throughput",
        "-------------------",
        f"Measured wall time: {perf['measured_wall_s']} s",
        f"Vector steps/s: {perf['vector_steps_per_s']}",
        f"Env steps/s: {perf['env_steps_per_s']}",
        f"Physics substeps/s: {perf['physics_substeps_per_s']}",
        f"Policy inference latency mean/p95/max: "
        f"{perf['policy_inference_latency_s']['mean']} / "
        f"{perf['policy_inference_latency_s']['p95']} / "
        f"{perf['policy_inference_latency_s']['max']} s",
        f"VecEnv step latency mean/p95/max: "
        f"{vec['mean']} / {vec['p95']} / {vec['max']} s",
        f"Full vector iteration mean/p95/max: "
        f"{total['mean']} / {total['p95']} / {total['max']} s",
        "",
        "Fast FD under 32-env contention",
        "-------------------------------",
        f"Measured row-build substeps: {agg['row_build_substeps']}",
        f"Row-build latency mean/p50/p95/p99/max: "
        f"{row['mean']} / {row['p50']} / {row['p95']} / "
        f"{row['p99']} / {row['max']} s",
        f"Rows total: {agg['row_count_total']}",
        f"Row distribution: {agg['row_count_distribution']}",
        f"Support modes: {agg['support_mode_counts']}",
        "",
        "Concurrency",
        "-----------",
        f"Env-with-rows count distribution per vector step: "
        f"{payload['concurrency']['envs_with_rows_distribution']}",
        f"Env-with-negative-free count distribution per vector step: "
        f"{payload['concurrency']['envs_with_negative_free_distribution']}",
        f"Max envs simultaneously building rows: "
        f"{payload['concurrency']['max_envs_with_rows']}",
        "",
        "Safety over all 1024 vector steps",
        "---------------------------------",
        f"Env steps: {safety['env_steps']}",
        f"Physics substeps: {safety['physics_substeps']}",
        f"q_free < 0 substeps: {safety['negative_free_substeps']}",
        f"Worst free clearance: {safety['min_free_clearance_mm']} mm",
        f"Worst committed clearance: {safety['min_committed_clearance_mm']} mm",
        f"Max committed penetration: {safety['max_committed_penetration_mm']} mm",
        f"Committed violations: {safety['committed_violation_count']}",
        f"ActiveCount mismatches: {safety['active_count_mismatch_count']}",
        f"Fast-builder invariant failures: "
        f"{safety['fast_builder_invariant_failure_count']}",
        f"NaN/Inf: {safety['non_finite_count']}",
        f"Planning failures: {safety['planning_failure_count']}",
        f"First failure: {payload['first_failure']}",
        "",
        "Interpretation",
        "--------------",
        "This benchmark measures 32 spawned SOFA workers with the validated",
        "Fast FD safety layer. It is not PPO training because no optimizer",
        "updates are performed.",
        "",
    ]
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
            RUNTIME / "results" / "fast_fd_32env_benchmark.json"
        ),
    )
    args = parser.parse_args()

    if int(args.n_envs) != N_ENVS:
        raise ValueError("this benchmark is fixed to exactly 32 envs")
    if int(args.vector_steps) != TOTAL_VECTOR_STEPS:
        raise ValueError("this benchmark is fixed to exactly 1024 vector steps")
    if int(args.warmup_steps) != WARMUP_VECTOR_STEPS:
        raise ValueError("this benchmark is fixed to exactly 100 warmup steps")

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path = output.with_name("fast_fd_32env_benchmark_report.md")
    trace_path = output.with_name("fast_fd_32env_vector_trace.jsonl")

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
        make_fast_fd_benchmark_env(
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

    envs_with_rows_distribution = Counter()
    envs_with_negative_distribution = Counter()
    max_envs_with_rows = 0

    cpu_info = _read_cpu_quota()

    try:
        vec_env = SubprocVecEnv(
            env_fns,
            start_method="spawn",
        )
        vec_env.seed(int(args.seed))
        observations = vec_env.reset()

        model = DistributedPPO.load(
            str(checkpoint),
            device=str(args.device),
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
                    _accumulate(
                        measured_aggregate,
                        bench,
                        bool(done),
                    )

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
                max_envs_with_rows = max(
                    max_envs_with_rows,
                    envs_with_rows,
                )

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
                reason = "32ENV_FAST_FD_THROUGHPUT_BENCHMARK_COMPLETED"

        payload = {
            "test": "32-env Fast FD + Generic throughput benchmark",
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
            "policy_device": str(args.device),
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
