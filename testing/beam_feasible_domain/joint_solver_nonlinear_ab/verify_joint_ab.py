#!/usr/bin/env python3
"""Test-only joint tolerance × native relinearization A/B acceptance.

This is a strict, read-only native capture auditor; it does not perform an extra
SOFA solve or assert that a file proves native execution. Native branch generation
must reuse the server's existing fork-at-CollisionBeginEvent harness.

Arms:
 A baseline one native solve, tolerance=1e-6
 B three native solves, tolerance=1e-6
 C one native solve, tolerance=1e-9
 D three native solves, tolerance=1e-9
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "nonlinear_relinearization_test"))
from verify_native_stages import load_snapshot, analyze, diff_init_states, require_context_match, clean_metadata, hash_file

EXPECTED = {
    "A": {"solves": 1, "tol": 1e-6, "expected_clearance_mm": 0.082681},
    "B": {"solves": 3, "tol": 1e-6, "expected_clearance_mm": 0.096711},
    "C": {"solves": 1, "tol": 1e-9, "expected_clearance_mm": 0.085538},
    "D": {"solves": 3, "tol": 1e-9, "expected_clearance_mm": None},
}
REQUIRED_PROVENANCE = (
    "native_solve_count", "native_tolerance", "native_solver_iterations",
    "native_solver_error", "native_relinearization_count", "native_mapping_refresh_count",
    "native_substeps_executed", "native_extra_animate_count", "native_position_overwrite_count",
)
TARGET_MM=0.100
CLEARANCE_BAND_MM=0.00001
GAP_TOL_MM=0.000001

def get_number(meta, key):
    if key not in meta:
        raise ValueError("PROVENANCE_MISSING:"+key)
    x=meta[key]
    if isinstance(x,(bool,dict,list)):
        raise ValueError("PROVENANCE_NON_NUMERIC:"+key)
    result=float(x)
    if not np.isfinite(result):
        raise ValueError("PROVENANCE_NONFINITE:"+key)
    return result

def norm_legacy(meta):
    # A source capture often records solver provenance in a nested dict.
    # This function deliberately performs no inference from file names.
    if not isinstance(meta,dict):
        raise ValueError("Invalid metadata object")
    return meta

def evaluate_arm(label,path,meta_path,base,base_meta,allow_missing_provenance):
    data=load_snapshot(path)
    metadata=norm_legacy(clean_metadata(data,meta_path))
    numeric=analyze(data)
    state=diff_init_states(base,data)
    context_missing=require_context_match(base_meta,metadata)
    expected=EXPECTED[label]
    audit={
        "arm":label, "path":str(path.resolve()), "sha256":hash_file(path),
        "measurement":numeric,
        "initial_state_differences":state,
        "context_unverified_keys":context_missing,
        "claimed_native_provenance":"NOT_VERIFIED",
    }
    expected_val=expected["expected_clearance_mm"]
    if expected_val is not None:
        audit["reference_match"]=abs(numeric["committed_min_mm"]-expected_val)<=CLEARANCE_BAND_MM
        if not audit["reference_match"]:
            raise ValueError("KNOWN_REFERENCE_MISMATCH:"+label)
    try:
        props={k:get_number(metadata,k) for k in REQUIRED_PROVENANCE}
        wrong=[]
        if round(props["native_solve_count"]) != expected["solves"]:wrong.append("solve_count")
        if not np.isclose(props["native_tolerance"],expected["tol"],rtol=1e-6,atol=0):
            wrong.append("tolerance")
        if round(props["native_relinearization_count"]) != expected["solves"]-1:
            wrong.append("relinearization_count")
        if round(props["native_mapping_refresh_count"]) != expected["solves"]-1:
            wrong.append("mapping_refresh_count")
        if round(props["native_substeps_executed"])!=1:wrong.append("substeps")
        if round(props["native_extra_animate_count"])!=0:wrong.append("extra_animate")
        if round(props["native_position_overwrite_count"])!=0:wrong.append("position_overwrite")
        if wrong:
            audit["provenance_violations"]=wrong
            audit["claimed_native_provenance"]="INVALID"
        else:
            audit["claimed_native_provenance"]="SELF_REPORTED_NEEDS_NATIVE_LOG_AUDIT"
        audit["native"]=props
        audit["solver_tolerance_reported_met"]=bool(props["native_solver_error"] <= props["native_tolerance"])
    except ValueError as exc:
        audit["provenance_missing_or_invalid"]=str(exc)
        if not allow_missing_provenance:
            audit["claimed_native_provenance"]="INCONCLUSIVE"
    # Actual row count must agree with activeCount for THIS stage, not an old fixed 6.
    if "active_count" in metadata and int(metadata["active_count"]) != numeric["row_count"]:
        audit["active_count_mismatch"]=True
        audit["claimed_native_provenance"]="INVALID"
    else:
        audit["active_count_mismatch"]=False if "active_count" in metadata else None
    return audit

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for label in EXPECTED:
        p.add_argument("--"+label.lower(),type=Path,required=label=="A")
        p.add_argument("--"+label.lower()+"-meta",type=Path)
    p.add_argument("--output",type=Path)
    p.add_argument("--allow-missing-provenance",action="store_true", help="Calculate numerics but keep provenance INCONCLUSIVE")
    opts=p.parse_args(argv)
    result={"experiment":"native same-frame 3× relinearization × tolerance 1e-6/1e-9",
            "method":"READ_ONLY_EXTERNAL_CAPTURE_VALIDATION",
            "native_solver_invoked_by_verifier":False,
            "production_modified_by_verifier":False,
            "historical_deep_penetration_root_cause":"UNPROVEN",
            "arms":{},"status":"INCONCLUSIVE"}
    try:
        paths={k:getattr(opts,k.lower()) for k in EXPECTED}
        bas=load_snapshot(paths["A"])
        basemeta=norm_legacy(clean_metadata(bas,opts.a_meta))
        for k in EXPECTED:
            path=paths[k]
            if path is None:
                result["arms"][k]={"status":"MISSING_NATIVE_CAPTURE"}
                continue
            meta=getattr(opts,k.lower()+"_meta")
            result["arms"][k]=evaluate_arm(k,path,meta,bas,basemeta,opts.allow_missing_provenance)
        all_complete=all(k in result["arms"] and "measurement" in result["arms"][k] for k in EXPECTED)
        if all_complete:
            a=result["arms"]["A"]["measurement"]
            d=result["arms"]["D"]["measurement"]
            result["joint_delta_mm"]=d["committed_min_mm"]-a["committed_min_mm"]
            result["joint_reaches_margin"]=bool(d["all_dense_above_margin_1nm"])
            result["joint_linear_row_gap_mm"]=d["min_linear_gap_original_mm"]
            result["joint_selected_nonlinear_error_mm"]=d["selected_max_abs_nonlinear_error_original_mm"]
            statuses={k:result["arms"][k]["claimed_native_provenance"] for k in EXPECTED}
            if any(x=="INVALID" for x in statuses.values()):
                result["status"]="FAIL_INVALID_NATIVE_PROVENANCE"
            else:
                result["status"]="PASS_NUMERIC_MARGIN_REACHED_PENDING_NATIVE_LOG_AUDIT" if result["joint_reaches_margin"] else "FAIL_MARGIN_UNREACHED_PENDING_NATIVE_LOG_AUDIT"
                if any(x=="INCONCLUSIVE" for x in statuses.values()):
                    result["status"]="INCONCLUSIVE_MISSING_NATIVE_PROVENANCE"
        else:
            result["status"]="INCONCLUSIVE_MISSING_SAME_FRAME_NATIVE_ARMS"
        result["required_final_verification"]=[
            "Inspect genuine SOFA native solver trace for each arm",
            "Ensure q_prev/q_free match and same raw action prefix hash",
            "Verify exactly one 5ms target substep and no additional animate",
            "Use per-stage rebuilt CSR row_count/activeCount, not initial row count",
            "Report measured solver error > configured tolerance without calling it converged",
        ]
    except (OSError,ValueError,KeyError,TypeError) as exc:
        result["status"]="INVALID_CAPTURE_OR_METADATA"
        result["error"]=str(exc)
    dest=opts.output or HERE/"results"/"joint_native_ab.json"
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,indent=2,ensure_ascii=False,default=str),encoding="utf-8")
    print(json.dumps(result,indent=2,ensure_ascii=False,default=str))
    return 0 if result["status"] not in ("INVALID_CAPTURE_OR_METADATA","FAIL_INVALID_NATIVE_PROVENANCE") else 2

if __name__=="__main__":
    raise SystemExit(main())
