#!/usr/bin/env python3
"""Test-only production Beam safety DEBUG/AUDIT full-episode acceptance."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from collections import Counter
from typing import Any

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
from mcr_sim.distributed import DistributedPPO


CHECKPOINT = Path(
    "/data/home/3220251075/mcr_sim/mcr_project/mCR_simulator-master/"
    "training_runs/ppo_v15_2b_physics_wall_10Nm_20260922_215738/models/"
    "ppo_base_epoch_074_episodes_07400.zip"
)
BASELINE_ACTION_SHA256 = (
    "e6854f5496eb2d43eb913d1116238d831666f7589ada051325bfdf14d294900e"
)
EXPECTED_PHYSICS_SUBSTEPS = 2
DT = 0.005
MARGIN_M = 1.0e-4
DENSE_SPACING_M = 1.0e-5
ACTIVE_BAND_M = 2.0e-4
NEAR_M = MARGIN_M + ACTIVE_BAND_M
COMMITTED_LIMIT_M = 1.0e-6


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


def _make_env(mode: str):
    os.environ["MCR_CONSTRAINT_SOLVER"] = "generic"
    os.environ["MCR_SOFA_DT"] = str(DT)
    return BeamSafetyMCREnv(
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
        },
        env_type=EnvType.AORTIC,
        render_mode=RenderMode.NONE,
        max_episode_steps=2048,
        time_step=DT,
        frame_skip=1,
        physics_substeps=EXPECTED_PHYSICS_SUBSTEPS,
        beam_safety_plugin=resolve_plugin_path(),
        beam_safety_validation_mode=mode,
        beam_safety_audit_interval_substeps=16,
        beam_safety_audit_near_clearance_m=NEAR_M,
        beam_safety_committed_penetration_limit_m=COMMITTED_LIMIT_M,
    )


def _solver_classes(env):
    return [
        obj.getClassName()
        for obj in env._sofa_root_node.objects
        if obj.getClassName()
        in ("GenericConstraintSolver", "LCPConstraintSolver")
    ]


def _trace_record(record: dict[str, Any], index: int) -> dict[str, Any]:
    keys = (
        "rl_step",
        "substep",
        "event_source",
        "planning_status",
        "reason",
        "rows_required",
        "rows_armed",
        "row_count",
        "q_free_clearance_m",
        "q_free_worst_index",
        "dense_sample_count",
        "selected_dense_indices",
        "selected_element_indices",
        "row_support_nodes",
        "support_mode",
        "constraint_active_count",
        "committed_audited",
        "committed_audit_reason",
        "committed_audited_clearance_m",
        "terminal_committed_audited_clearance_m",
        "planner_runtime_s",
        "q_free_external_dense_count",
        "q_free_internal_dense_count",
        "total_q_free_full_profile_count",
        "batched_fd_sdf_query_count",
        "full_beam_profile_evaluations_after_selection",
        "selected_point_reconstruction_max_error_m",
    )
    out = {key: record.get(key) for key in keys}
    out["physics_index"] = int(index)
    return out


def _load_jsonl(path: Path):
    with path.open("r") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _compare_debug_trace(path: Path, audit_trace: Path):
    debug = list(_load_jsonl(path))
    audit = list(_load_jsonl(audit_trace))
    fields = (
        "physics_index",
        "rl_step",
        "substep",
        "planning_status",
        "rows_required",
        "rows_armed",
        "row_count",
        "q_free_clearance_m",
        "q_free_worst_index",
        "dense_sample_count",
        "selected_dense_indices",
        "selected_element_indices",
        "row_support_nodes",
        "support_mode",
        "constraint_active_count",
    )
    first = None
    if len(debug) != len(audit):
        first = {
            "reason": "TRACE_LENGTH_MISMATCH",
            "debug": len(debug),
            "audit": len(audit),
        }
    else:
        for i, (left, right) in enumerate(zip(debug, audit)):
            for field in fields:
                if left.get(field) != right.get(field):
                    first = {
                        "trace_index": i,
                        "physics_index": left.get("physics_index"),
                        "field": field,
                        "debug": left.get(field),
                        "audit": right.get(field),
                    }
                    break
            if first:
                break
    return {
        "equal": first is None,
        "debug_trace_count": len(debug),
        "audit_trace_count": len(audit),
        "compared_fields": list(fields),
        "first_difference": first,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("debug", "audit"), required=True)
    parser.add_argument("--checkpoint", default=str(CHECKPOINT))
    parser.add_argument("--seed", type=int, default=15204)
    parser.add_argument("--max-rl-steps", type=int, default=2048)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--output", required=True)
    parser.add_argument("--debug-trace", default=None)
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    trace_path = output.with_suffix(".trace.jsonl")
    report_path = output.with_suffix(".report.md")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if args.max_rl_steps != 2048:
        raise ValueError("acceptance run requires max_rl_steps=2048")

    env = _make_env(args.mode)
    observation, reset_info = env.reset(seed=int(args.seed))
    if int(env.physics_substeps) != EXPECTED_PHYSICS_SUBSTEPS:
        raise RuntimeError(
            "wrong physics substeps: "
            f"{env.physics_substeps} != {EXPECTED_PHYSICS_SUBSTEPS}"
        )
    solvers = _solver_classes(env)
    if solvers != ["GenericConstraintSolver"]:
        raise RuntimeError(f"wrong constraint solver: {solvers}")
    model = DistributedPPO.load(str(checkpoint), device="cpu")
    model.policy.set_training_mode(False)

    original_animate = env.sofa_simulation.animate
    trace_stream = trace_path.open("w+", buffering=1)
    action_hasher = hashlib.sha256()
    records = []
    fatal_reason = None
    first_failure = None
    terminated = False
    truncated = False
    terminal_reason = None
    rl_steps_executed = 0
    info = {}
    started = time.perf_counter()

    def capture_animate(root, dt):
        result = original_animate(root, dt)
        record = env._beam_last_substep_record
        if record is None:
            raise RuntimeError("PRODUCTION_BEAM_SAFETY_RECORD_MISSING")
        item = _trace_record(
            dict(record), int(env._beam_physics_index)
        )
        records.append(item)
        trace_stream.write(
            json.dumps(_json_safe(item), sort_keys=True) + "\n"
        )
        return result

    env.sofa_simulation.animate = capture_animate
    try:
        for step in range(1, int(args.max_rl_steps) + 1):
            raw_action, _ = model.predict(observation, deterministic=True)
            raw_action = np.asarray(raw_action, dtype=np.float32).reshape(3)
            action_hasher.update(
                np.asarray(raw_action, dtype="<f4").tobytes(order="C")
            )
            try:
                observation, _reward, terminated, truncated, info = env.step(
                    raw_action
                )
            except Exception as exc:
                fatal_reason = f"{type(exc).__name__}:{exc}"
                record = env._beam_last_substep_record
                first_failure = {
                    "rl_step": int(step),
                    "physics_substep": int(env._beam_substep),
                    "reason": fatal_reason,
                    "last_record": (
                        _json_safe(dict(record))
                        if isinstance(record, dict)
                        else None
                    ),
                }
                rl_steps_executed = int(step)
                break
            rl_steps_executed = int(step)
            if terminated or truncated:
                terminal_reason = None
                for key in (
                    "terminal_reason",
                    "termination_reason",
                    "done_reason",
                    "reason",
                ):
                    if isinstance(info, dict) and info.get(key):
                        terminal_reason = str(info[key])
                        break
                if terminal_reason is None:
                    terminal_reason = (
                        "terminated" if terminated else "truncated"
                    )
                last = env._beam_last_substep_record
                if (
                    isinstance(last, dict)
                    and records
                    and last.get("terminal_committed_audited_clearance_m")
                    is not None
                ):
                    records[-1]["terminal_committed_audited_clearance_m"] = (
                        float(last["terminal_committed_audited_clearance_m"])
                    )
                    trace_stream.seek(0)
                    lines = trace_stream.readlines()
                    trace_stream.seek(0, 2)
                    lines[-1] = json.dumps(
                        _json_safe(records[-1]), sort_keys=True
                    ) + "\n"
                    trace_stream.seek(0)
                    trace_stream.writelines(lines)
                    trace_stream.truncate()
                break
            if args.progress_every > 0 and (
                step == 1 or step % int(args.progress_every) == 0
            ):
                safety = info.get("beam_safety", {}) if isinstance(info, dict) else {}
                print(
                    "[PRODUCTION_BEAM_" + args.mode.upper() + "] "
                    f"step={step}/2048 physics={safety.get('physics_substeps_seen')} "
                    f"active={safety.get('active_substeps')} "
                    f"audits={safety.get('committed_audits')} "
                    f"skips={safety.get('committed_audit_skips')}",
                    flush=True,
                )
    finally:
        trace_stream.close()
        env.sofa_simulation.animate = original_animate

    action_sha = action_hasher.hexdigest()
    mode_info = info.get("beam_safety", {}) if isinstance(info, dict) else {}
    physical_substeps = len(records)
    free_values = [
        float(rec["q_free_clearance_m"])
        for rec in records
        if rec.get("q_free_clearance_m") is not None
        and np.isfinite(float(rec["q_free_clearance_m"]))
    ]
    committed_values = [
        float(rec["committed_audited_clearance_m"])
        for rec in records
        if rec.get("committed_audited")
        and rec.get("committed_audited_clearance_m") is not None
        and np.isfinite(float(rec["committed_audited_clearance_m"]))
    ]
    row_records = [rec for rec in records if rec.get("rows_required")]
    active_count_mismatches = [
        rec
        for rec in row_records
        if int(rec.get("row_count") or 0) <= 0
        or int(rec.get("constraint_active_count") or -1)
        != int(rec.get("row_count") or 0)
        or not rec.get("rows_armed")
    ]
    qfree_negative = sum(value < 0.0 for value in free_values)
    qfree_deeper = sum(value < -1.0e-6 for value in free_values)
    row_dist = Counter(int(rec.get("row_count") or 0) for rec in row_records)
    audit_reasons = Counter(
        str(rec.get("committed_audit_reason"))
        for rec in records
        if rec.get("committed_audited")
    )
    audit_skips = sum(not rec.get("committed_audited") for rec in records)
    active_not_audited = [
        rec for rec in row_records if not rec.get("committed_audited")
    ]
    near_not_audited = [
        rec
        for rec in records
        if rec.get("q_free_clearance_m") is not None
        and float(rec["q_free_clearance_m"]) <= NEAR_M
        and not rec.get("committed_audited")
    ]
    far_safe_skips = sum(
        rec.get("q_free_clearance_m") is not None
        and float(rec["q_free_clearance_m"]) > NEAR_M
        and not rec.get("committed_audited")
        for rec in records
    )
    periodic_far_safe = sum(
        rec.get("committed_audit_reason") == "periodic"
        and rec.get("q_free_clearance_m") is not None
        and float(rec["q_free_clearance_m"]) > NEAR_M
        for rec in records
    )
    violations = sum(value < -COMMITTED_LIMIT_M for value in committed_values)
    nonfinite = sum(
        not np.isfinite(float(rec.get("q_free_clearance_m", np.nan)))
        for rec in records
    )
    if mode_info.get("physics_substeps_seen") != physical_substeps:
        fatal_reason = fatal_reason or "PHYSICS_SUBSTEP_COUNT_MISMATCH"
    if active_count_mismatches:
        fatal_reason = fatal_reason or "ACTIVE_COUNT_MISMATCH"
    if active_not_audited and args.mode == "audit":
        fatal_reason = fatal_reason or "ACTIVE_SUBSTEP_NOT_AUDITED"
    if near_not_audited and args.mode == "audit":
        fatal_reason = fatal_reason or "NEAR_WALL_SUBSTEP_NOT_AUDITED"
    if violations:
        fatal_reason = fatal_reason or "COMMITTED_PENETRATION_VIOLATION"
    if nonfinite:
        fatal_reason = fatal_reason or "NONFINITE_FREE_CLEARANCE"
    if action_sha != BASELINE_ACTION_SHA256:
        fatal_reason = fatal_reason or "ACTION_SHA_MISMATCH"
    if physical_substeps != rl_steps_executed * EXPECTED_PHYSICS_SUBSTEPS:
        fatal_reason = fatal_reason or "WRONG_PHYSICS_SUBSTEP_COUNT"
    if args.mode == "debug" and int(mode_info.get("committed_audit_skips", -1)) != 0:
        fatal_reason = fatal_reason or "DEBUG_AUDIT_SKIPS_NONZERO"
    if args.mode == "debug" and int(mode_info.get("committed_audits", -1)) != physical_substeps + 1:
        fatal_reason = fatal_reason or "DEBUG_NOT_AUDITING_EVERY_SUBSTEP"
    if args.mode == "audit":
        if not far_safe_skips:
            fatal_reason = fatal_reason or "AUDIT_DID_NOT_SKIP_FAR_SAFE"
        if not periodic_far_safe:
            fatal_reason = fatal_reason or "NO_PERIODIC_FAR_SAFE_AUDIT"

    trace_comparison = None
    if args.mode == "audit" and args.debug_trace:
        trace_comparison = _compare_debug_trace(
            Path(args.debug_trace).expanduser().resolve(), trace_path
        )
        if not trace_comparison["equal"]:
            fatal_reason = fatal_reason or "DEBUG_AUDIT_PHYSICAL_TRACE_MISMATCH"

    stats = {
        "physics_substeps": physical_substeps,
        "safe_free_substeps": int(physical_substeps - len(row_records)),
        "active_substeps": len(row_records),
        "total_rows": sum(int(rec.get("row_count") or 0) for rec in row_records),
        "row_count_distribution": dict(row_dist),
        "q_free_negative_substeps": int(qfree_negative),
        "q_free_below_minus_0p001mm_substeps": int(qfree_deeper),
        "worst_free_clearance_m": min(free_values) if free_values else None,
        "worst_free_clearance_mm": min(free_values) * 1000.0 if free_values else None,
        "worst_audited_committed_clearance_m": (
            min(committed_values) if committed_values else None
        ),
        "worst_audited_committed_clearance_mm": (
            min(committed_values) * 1000.0 if committed_values else None
        ),
        "max_audited_committed_penetration_mm": (
            max((max(0.0, -value) for value in committed_values), default=0.0)
            * 1000.0
        ),
        "committed_penetration_violations": int(violations),
        "active_count_mismatches": len(active_count_mismatches),
        "nonfinite_count": int(nonfinite),
        "audit_reasons": dict(audit_reasons),
        "committed_audits_total_including_reset": int(
            mode_info.get("committed_audits", 0)
        ),
        "committed_audit_skips": int(mode_info.get("committed_audit_skips", 0)),
        "active_substeps_not_audited": len(active_not_audited),
        "near_wall_substeps": int(mode_info.get("near_wall_substeps", 0)),
        "near_wall_substeps_not_audited": len(near_not_audited),
        "far_safe_skips": int(far_safe_skips),
        "periodic_far_safe_audits": int(periodic_far_safe),
        "reset_audit": bool(
            args.mode != "off"
            and int((reset_info.get("beam_safety") or {}).get("committed_audits", 0))
            >= 1
        ),
        "terminal_audit": bool(
            any(
                rec.get("terminal_committed_audited_clearance_m") is not None
                for rec in records
            )
        ),
        "q_prev_full_dense_profile_total": sum(
            int(rec.get("q_prev_full_dense_profile_count", 0) or 0)
            for rec in records
        ),
        "q_free_full_dense_profile_total": sum(
            int(rec.get("q_free_external_dense_count", 0) or 0)
            for rec in records
        ),
        "builder_internal_q_free_profile_total": sum(
            int(rec.get("q_free_internal_dense_count", 0) or 0)
            for rec in records
        ),
        "planning_status_failures": sum(
            rec.get("planning_status") == "FAIL" for rec in records
        ),
        "builder_invariant_failures": sum(
            rec.get("reason") == "FAST_BUILDER_INVARIANT_VIOLATION"
            for rec in records
        ),
    }

    step_count_ok = (
        physical_substeps == rl_steps_executed * EXPECTED_PHYSICS_SUBSTEPS
        and (physical_substeps == 4096 or terminated or truncated or fatal_reason)
    )
    if stats["q_free_full_dense_profile_total"] != physical_substeps:
        fatal_reason = fatal_reason or "Q_FREE_PROFILE_MISSING"
    if stats["q_prev_full_dense_profile_total"] != 0:
        fatal_reason = fatal_reason or "Q_PREV_PROFILE_INVARIANT"
    if stats["builder_internal_q_free_profile_total"] != 0:
        fatal_reason = fatal_reason or "BUILDER_INTERNAL_PROFILE_INVARIANT"
    if not step_count_ok:
        fatal_reason = fatal_reason or "INCOMPLETE_OR_INVALID_STEP_COUNT"

    decision = "PASS" if fatal_reason is None else "FAIL"
    payload = {
        "test": "Production Beam safety " + args.mode.upper() + " full-episode acceptance",
        "mode": args.mode,
        "decision": decision,
        "reason": fatal_reason or "FIXED_EPISODE_COMPLETED_WITHIN_SAFETY_LIMITS",
        "seed": int(args.seed),
        "checkpoint": str(checkpoint),
        "device": "cpu",
        "max_rl_steps": int(args.max_rl_steps),
        "rl_steps_executed": int(rl_steps_executed),
        "physics_substeps_per_rl_step": EXPECTED_PHYSICS_SUBSTEPS,
        "physics_dt_s": DT,
        "physics_substeps_observed": int(physical_substeps),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "terminal_reason": terminal_reason or (
            "max_rl_steps_reached" if rl_steps_executed == args.max_rl_steps else None
        ),
        "action_sha256": action_sha,
        "baseline_action_sha256": BASELINE_ACTION_SHA256,
        "action_sha_matches": action_sha == BASELINE_ACTION_SHA256,
        "solver": solvers,
        "validation": {
            "requested_margin_m": MARGIN_M,
            "dense_spacing_m": DENSE_SPACING_M,
            "translation_fd_m": 1.0e-6,
            "rotation_fd_rad": 1.0e-4,
            "max_rows": 12,
            "active_band_m": ACTIVE_BAND_M,
            "audit_interval_substeps": 16,
            "audit_near_clearance_m": NEAR_M,
            "committed_penetration_limit_m": COMMITTED_LIMIT_M,
        },
        "production_plugin": str(env._beam_plugin_path),
        "beam_safety_summary": _json_safe(mode_info),
        "reset_info": _json_safe(reset_info),
        "stats": stats,
        "first_failure": first_failure,
        "trace_comparison": trace_comparison,
        "wall_seconds": time.perf_counter() - started,
        "training_started": False,
        "optimizer_updates": 0,
        "production_files_modified": False,
        "trace_path": str(trace_path),
    }
    output.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n"
    )
    report_lines = [
        "PRODUCTION BEAM SAFETY " + args.mode.upper() + " FULL-EPISODE ACCEPTANCE",
        "FINAL = " + decision,
        "Reason = " + payload["reason"],
        "RL steps = " + str(rl_steps_executed),
        "Physics substeps = " + str(physical_substeps),
        "Action SHA256 = " + action_sha,
        "Action SHA match = " + str(payload["action_sha_matches"]),
        "Solver = " + str(solvers),
        "Worst free clearance (mm) = " + str(stats["worst_free_clearance_mm"]),
        "Worst audited committed clearance (mm) = "
        + str(stats["worst_audited_committed_clearance_mm"]),
        "Max audited committed penetration (mm) = "
        + str(stats["max_audited_committed_penetration_mm"]),
        "Committed violations = " + str(violations),
        "Active substeps = " + str(stats["active_substeps"]),
        "Total rows = " + str(stats["total_rows"]),
        "Committed audits including reset = "
        + str(stats["committed_audits_total_including_reset"]),
        "Committed audit skips = " + str(stats["committed_audit_skips"]),
        "Near-wall unaudited = " + str(stats["near_wall_substeps_not_audited"]),
        "Far-safe skips = " + str(stats["far_safe_skips"]),
        "Periodic far-safe audits = " + str(stats["periodic_far_safe_audits"]),
        "First failure = " + str(first_failure),
        "Training started = NO",
        "Production files modified = NO",
    ]
    if trace_comparison is not None:
        report_lines.extend([
            "DEBUG/AUDIT physical trace equal = "
            + str(trace_comparison["equal"]),
            "First trace difference = "
            + str(trace_comparison["first_difference"]),
        ])
    report_path.write_text("\n".join(report_lines) + "\n")
    print(json.dumps({
        "decision": decision,
        "reason": payload["reason"],
        "steps": rl_steps_executed,
        "physics_substeps": physical_substeps,
        "action_sha256": action_sha,
        "active_substeps": stats["active_substeps"],
        "total_rows": stats["total_rows"],
        "worst_free_mm": stats["worst_free_clearance_mm"],
        "worst_committed_mm": stats["worst_audited_committed_clearance_mm"],
        "committed_violations": violations,
        "audit_count": stats["committed_audits_total_including_reset"],
        "audit_skips": stats["committed_audit_skips"],
        "output": str(output),
        "trace": str(trace_path),
    }, sort_keys=True), flush=True)
    env.close()


if __name__ == "__main__":
    main()
