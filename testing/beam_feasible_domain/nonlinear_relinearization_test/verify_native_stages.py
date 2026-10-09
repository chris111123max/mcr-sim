#!/usr/bin/env python3
"""Single-frame nonlinear-relinearization acceptance gate (TEST ONLY).

This script does NOT do a fake SOFA solve.  It validates native SOFA fork output
from repeated constraint rebuilds and diagnoses how linear and nonlinear row
predictions diverge.  Without independently produced native stage snapshots it
returns INCONCLUSIVE, never claims that a repeated solve is physically possible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

THIS = Path(__file__).resolve()
THREE_FACTOR_DIR = THIS.parent.parent / "three_factor_validation"
sys.path.insert(0, str(THREE_FACTOR_DIR))
from validate_three_factors import load_snapshot, linearized_g, nonlinear_case  # noqa: E402

TARGET_MM = 0.100
BASELINE_MM = 0.082681
BASELINE_TOL_MM = 0.00001
STATE_TOL_M = 1.0e-10
QUAT_TOL = 1.0e-10


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_metadata(data, meta_path: Path | None):
    metadata = data.get("metadata") or {}
    if meta_path is not None:
        external = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(external, dict):
            raise ValueError("METADATA_NOT_OBJECT")
        metadata = external
    return metadata


def require_context_match(baseline_meta, variant_meta):
    """Use explicit keys where provided; do not invent absent identifiers."""
    keys = ("vessel", "target", "seed", "rl_step", "substep", "action_prefix_sha256",
            "geometry_sha256", "sdf_sha256", "topology_sha256")
    missing = [k for k in keys if k not in baseline_meta or k not in variant_meta]
    different = {k:[baseline_meta[k], variant_meta[k]]
                 for k in keys if k in baseline_meta and k in variant_meta
                 and baseline_meta[k] != variant_meta[k]}
    if different:
        raise ValueError("CROSS_FRAME_METADATA_MISMATCH:" + json.dumps(different))
    return missing


def diff_init_states(base, test):
    if base["q_prev"] is None or test["q_prev"] is None:
        raise ValueError("MISSING_Q_PREV_FOR_A_B")
    errs = {}
    for key in ("q_prev", "q_free"):
        a, b = base[key], test[key]
        if a.shape != b.shape:
            raise ValueError("INITIAL_STATE_SHAPE_MISMATCH:" + key)
        errs[key+"_position_m"] = float(np.max(np.abs(a[:, :3] - b[:, :3])))
        errs[key+"_quaternion"] = float(np.max(np.abs(a[:, 3:] - b[:, 3:])))
    if max(errs["q_prev_position_m"], errs["q_free_position_m"]) > STATE_TOL_M:
        raise ValueError("STATE_TRANSLATION_MISMATCH")
    if max(errs["q_prev_quaternion"], errs["q_free_quaternion"]) > QUAT_TOL:
        raise ValueError("STATE_QUATERNION_MISMATCH")
    return errs


def analyze(data):
    summary = nonlinear_case(data, tol_m=1e-9)
    gap_raw, _ = linearized_g(data, convert_angular=False)
    gap_world, _ = linearized_g(data, convert_angular=True)
    cc = np.asarray(data["q_committed_dense_clearance"], dtype=float)
    selected = np.asarray(data["selected_dense_indices"], dtype=int)
    actual_g = cc[selected] - TARGET_MM / 1000
    return {
        "committed_min_mm":float(np.min(cc)*1000),
        "margin_shortfall_mm":float(max(0, TARGET_MM - np.min(cc)*1000)),
        "worst_dense_index":int(np.argmin(cc)),
        "worst_index_selected":bool(np.argmin(cc) in set(selected.tolist())),
        "row_count":int(len(gap_raw)),
        "min_linear_gap_original_mm":float(np.min(gap_raw)*1000),
        "min_linear_gap_world_transform_mm":float(np.min(gap_world)*1000),
        "selected_max_abs_nonlinear_error_original_mm":
            float(np.max(np.abs(actual_g-gap_raw))*1000),
        "selected_max_abs_nonlinear_error_world_transform_mm":
            float(np.max(np.abs(actual_g-gap_world))*1000),
        "all_dense_above_margin_1nm":bool(np.min(cc) >= TARGET_MM/1000-1e-9),
        "reported_status":summary["status"],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", type=Path, required=True,
                    help="Original one-native-solve capture at B02/target_04 step2009/substep1")
    ap.add_argument("--baseline-meta", type=Path)
    ap.add_argument("--stage", type=Path, action="append", default=[],
                    help="Complete capture after another *genuine* native SOFA stage; repeat up to twice")
    ap.add_argument("--stage-meta", type=Path, action="append", default=[])
    ap.add_argument("--output", type=Path)
    ap.add_argument("--expect-baseline-mm", type=float, default=BASELINE_MM)
    ap.add_argument("--skip-baseline-value-check", action="store_true",
                    help="Only for a different frame; cannot make historical B02 claims")
    args = ap.parse_args()
    if len(args.stage) > 2 or len(args.stage_meta) > len(args.stage):
        ap.error("Maximum 2 subsequent native stages and matching metadata")
    result = {
        "experiment":"B02 target04 single-substep nonlinear relinearization",
        "mode":"READ_ONLY_POSTHOC_NATIVE_CAPTURE_VALIDATION",
        "native_solve_executed_by_this_script":False,
        "production_modified":False,
        "historical_minus_0_766410143_mm_cause":"UNPROVEN",
        "stages":[],
        "root_cause":"INCONCLUSIVE",
    }
    try:
        base = load_snapshot(args.baseline)
        baseline_meta = clean_metadata(base, args.baseline_meta)
        baseline_report = analyze(base)
        result["stages"].append({
            "kind":"baseline", "path":str(args.baseline), "sha256":hash_file(args.baseline),
            "measurement":baseline_report, "metadata":baseline_meta})
        if not args.skip_baseline_value_check and abs(
            baseline_report["committed_min_mm"]-args.expect_baseline_mm
        ) > BASELINE_TOL_MM:
            raise ValueError("BASELINE_MISMATCH: observed_mm="
                             +str(baseline_report["committed_min_mm"]))
        if not args.stage:
            result["status"]="INCONCLUSIVE_NO_NATIVE_RELINEARIZATION_CAPTURES"
            result["reason"]=(
                "The existing capture proves a nonlinear prediction discrepancy; "
                "it cannot establish native iterative relinearization by itself. "
                "Do not project/overwrite positions or call extra animate substeps.")
        else:
            for i,path in enumerate(args.stage):
                current=load_snapshot(path)
                meta_path=args.stage_meta[i] if i < len(args.stage_meta) else None
                metadata=clean_metadata(current,meta_path)
                missing=require_context_match(baseline_meta,metadata)
                deltas=diff_init_states(base,current)
                measure=analyze(current)
                result["stages"].append({
                    "kind":f"native_stage_{i+2}", "path":str(path),
                    "sha256":hash_file(path), "measurement":measure,
                    "initial_state_max_differences":deltas,
                    "context_unverified_keys":missing, "metadata":metadata})
            # Can't attest a stage is a *native same-substep* solve from NPZ alone.
            # Require explicit external evidence and avoid assigning a causal PASS
            # automatically just because the numbers improved.
            result["status"]="NUMERIC_A_B_MEASURED_NATIVE_PROVENANCE_REQUIRED"
            result["margin_met_in_last_capture"]=result["stages"][-1]["measurement"][
                "all_dense_above_margin_1nm"]
            result["improvement_mm"]=(
                result["stages"][-1]["measurement"]["committed_min_mm"]-
                baseline_report["committed_min_mm"]
            )
            result["reason"]=(
                "A/B initial states matched. An external fork trace must prove "
                "these are sequential inner solves of ONE SOFA physical substep, "
                "not extra physics steps, edited positions, or separate states. "
                "Check native solver execution trace and mapping coherence.")
    except (OSError,ValueError,TypeError,KeyError) as exc:
        result["status"]="FAIL_OR_INCONCLUSIVE_INVALID_CAPTURE"
        result["reason"]=str(exc)
    outfile=args.output or args.baseline.with_name("nonlinear_relinearization_gate.json")
    outfile.parent.mkdir(parents=True,exist_ok=True)
    outfile.write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str),
                       encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2,default=str))
    print("REPORT="+str(outfile))
    return 1 if result["status"]=="FAIL_OR_INCONCLUSIVE_INVALID_CAPTURE" else 0


if __name__=="__main__":
    raise SystemExit(main())
