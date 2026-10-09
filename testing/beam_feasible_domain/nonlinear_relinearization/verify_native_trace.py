#!/usr/bin/env python3
"""Test-only auditor for genuine SOFA same-substep relinearization traces.

This tool DOES NOT perform a native solve. It refuses to infer one from numeric
NPZ edits or an offline Newton result. PASS means trace consistency, conditional
on the test-only capture's truthful native-solver instrumentation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

EXPECTED_SUBSTEP = 1
EXPECTED_MARGIN_M = 1e-4
EXPECTED_BASELINE_M = 0.082681e-3
MAX_FRAME_MISMATCH_M = 1e-10
MAX_QUAT_MISMATCH = 1e-10
MAX_BASELINE_MISMATCH_M = 1e-8


def _load_arrays(path):
    path = Path(path).expanduser().resolve()
    with np.load(path, allow_pickle=False) as z:
        d = {k: np.array(z[k]) for k in z.files}
    for key in ("q_prev", "q_free", "q_committed", "q_committed_dense_clearance"):
        if key not in d:
            raise ValueError(f"{path}: missing {key}")
    for k in ("q_prev", "q_free", "q_committed"):
        q = np.asarray(d[k], dtype=float)
        if q.ndim != 2 or q.shape[1] != 7 or not np.isfinite(q).all():
            raise ValueError(f"{path}: invalid {k}")
        d[k] = q
    if d["q_prev"].shape != d["q_free"].shape or d["q_free"].shape != d["q_committed"].shape:
        raise ValueError(f"{path}: Rigid3 shapes differ")
    profile = np.asarray(d["q_committed_dense_clearance"], dtype=float).reshape(-1)
    if profile.size == 0 or not np.isfinite(profile).all():
        raise ValueError(f"{path}: invalid committed dense profile")
    d["q_committed_dense_clearance"] = profile
    return d


def _sha(q):
    return hashlib.sha256(np.asarray(q, dtype="<f8").tobytes()).hexdigest()


def _read_trace(path):
    trace = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(trace, dict):
        raise ValueError("Native trace JSON must be an object")
    required = (
        "capture_kind", "physics_substeps_per_action",
        "physics_dt_s", "target_rl_step", "target_substep",
        "physics_substeps_observed_in_fork", "passes",
    )
    for key in required:
        if key not in trace:
            raise ValueError("NATIVE_TRACE_FIELD_MISSING:" + key)
    if trace["capture_kind"] != "REAL_SOFA_NATIVE_GENERIC_CONSTRAINT_SOLVER":
        raise ValueError("No attested native GenericConstraintSolver instrumentation")
    if trace["physics_substeps_per_action"] != 2 or abs(trace["physics_dt_s"]-0.005)>1e-12:
        raise ValueError("PHYSICS_CONFIGURATION_NOT_2x5MS")
    if trace["target_substep"] != EXPECTED_SUBSTEP:
        raise ValueError("WRONG_TARGET_SUBSTEP")
    if trace["physics_substeps_observed_in_fork"] != 1:
        raise ValueError("ADDITIONAL_PHYSICS_SUBSTEPS_EXECUTED")
    passes = trace["passes"]
    if not isinstance(passes, list) or not (1 <= len(passes) <= 3):
        raise ValueError("Expected 1..3 native same-substep solver passes")
    return trace


def verify(trace_path, baseline_path, *, expected_mm=EXPECTED_BASELINE_M*1000):
    trace_path = Path(trace_path).expanduser().resolve()
    trace = _read_trace(trace_path)
    baseline = _load_arrays(baseline_path)
    initial = {
        "q_prev_sha256": _sha(baseline["q_prev"]),
        "q_free_sha256": _sha(baseline["q_free"]),
    }
    records = []
    previous_committed = None
    margin = EXPECTED_MARGIN_M
    for idx, entry in enumerate(trace["passes"]):
        if not isinstance(entry, dict):
            raise ValueError("Each pass must be an object")
        for key in ("npz", "native_solver_invocations_cumulative", "native_solver",
                    "rows_rebuilt_from_native_state", "direct_position_write",
                    "projection_used", "rollback_used", "action_shielding_used"):
            if key not in entry:
                raise ValueError(f"NATIVE_PASS_FIELD_MISSING:{idx}:{key}")
        if entry["native_solver"] != "GenericConstraintSolver":
            raise ValueError(f"NATIVE_SOLVER_MISMATCH:pass{idx}")
        if entry["direct_position_write"] or entry["projection_used"] or entry["rollback_used"] or entry["action_shielding_used"]:
            raise ValueError(f"PROHIBITED_STATE_MANIPULATION:pass{idx}")
        if idx > 0 and not entry["rows_rebuilt_from_native_state"]:
            raise ValueError(f"NO_RELINEARIZATION_AT_PASS:{idx}")
        if int(entry["native_solver_invocations_cumulative"]) < idx+1:
            raise ValueError(f"NO_ADDITIONAL_NATIVE_SOLVE:pass{idx}")
        local = (trace_path.parent / entry["npz"]).resolve()
        arrays = _load_arrays(local)
        for k in ("q_prev","q_free"):
            p = float(np.max(np.abs(arrays[k][:, :3] - baseline[k][:, :3])))
            r = float(np.max(np.abs(arrays[k][:, 3:7] - baseline[k][:, 3:7])))
            if p > MAX_FRAME_MISMATCH_M or r > MAX_QUAT_MISMATCH:
                raise ValueError(f"INCONSISTENT_SAME_FRAME_INPUT:pass{idx}:{k}:{p}:{r}")
        if idx > 0:
            expected = entry.get("relinearized_about_committed_state_sha256")
            if not expected or expected != _sha(previous_committed):
                raise ValueError(f"NOT_RELINEARIZED_AT_PRIOR_NATIVE_COMMITTED_STATE:pass{idx}")
        val = float(np.min(arrays["q_committed_dense_clearance"]))
        if idx == 0 and abs(val - expected_mm * 1e-3) > MAX_BASELINE_MISMATCH_M:
            raise ValueError(
                f"BASELINE_MISMATCH:measured_mm={val*1000},expected_mm={expected_mm}"
            )
        records.append({
            "pass_index": idx,
            "native_solver_invocations_cumulative":int(entry["native_solver_invocations_cumulative"]),
            "committed_state_sha256":_sha(arrays["q_committed"]),
            "dense_min_committed_clearance_mm":val*1000,
            "dense_worst_index":int(np.argmin(arrays["q_committed_dense_clearance"])),
            "relinearized_from_previous_committed":bool(idx>0),
        })
        previous_committed = arrays["q_committed"]
    margin_met = bool(records[-1]["dense_min_committed_clearance_mm"] >= margin*1000)
    return {
        "trace_consistency":"PASS",
        "native_solve_instrumentation":"REQUIRED_AND_ATTESTED_BY_TEST_HARNESS_NOT_INDEPENDENTLY_OBSERVABLE_HERE",
        "input_state_sha256":initial,
        "target_rl_step":trace["target_rl_step"],
        "target_substep":trace["target_substep"],
        "physics_substeps_observed_in_fork":1,
        "passes":records,
        "margin_target_mm":margin*1000,
        "native_relinearization_attempted":len(records)>1,
        "recorded_native_final_margin_met":margin_met,
        "historical_deep_penetration_root_cause_proven":False,
        "warning":"This validates recorded trace consistency only, not the truthfulness of native solver instrumentation.",
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace",type=Path,required=True)
    p.add_argument("--baseline",type=Path,required=True)
    p.add_argument("--expected-baseline-mm",type=float,default=EXPECTED_BASELINE_M*1000)
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    report=verify(args.trace,args.baseline,expected_mm=args.expected_baseline_mm)
    payload=json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)
    print(payload)
    if args.output:
        dest=args.output.expanduser().resolve()
        for source in (args.trace,args.baseline):
            if dest==source.expanduser().resolve():
                raise ValueError("Refuse to overwrite source trace or snapshot")
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(payload+"\n",encoding="utf-8")
    return 0


if __name__=="__main__":
    try:
        sys.exit(main())
    except (OSError,ValueError,KeyError,TypeError) as exc:
        print(f"NATIVE_RELINEARIZATION_INCONCLUSIVE:{type(exc).__name__}:{exc}",file=sys.stderr)
        sys.exit(2)
