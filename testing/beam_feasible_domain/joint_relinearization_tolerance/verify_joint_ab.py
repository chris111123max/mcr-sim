#!/usr/bin/env python3
"""TEST ONLY: strict native 3x-relinearization / solver-tolerance joint A/B gate.

This script reads saved SOFA-native test captures; it never calls animate,
projects/overwrites Beam DOFs, performs a physical solve, or starts PPO.
It rejects falsely comparable states, non-native / extra-substep claims,
stale activeCount expectations and mismatched relinearization provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "three_factor_validation"))
from validate_three_factors import load_snapshot, nonlinear_case  # noqa: E402

ACTION_SHA = "6f534ad5155bfa635666f9d3237783681e65dc378a3d5608706db30bb00e181c"
MARGIN_MM = 0.100
BASELINE_MM = 0.082681
NOMINAL_THIRD_MM = 0.096711
MEASURE_TOL_MM = 0.00005
STATE_TOL_M = 1e-10
ROTATION_TOL_RAD = 1e-10
MARGIN_GATE_M = 1e-9

class CaptureError(ValueError):
    pass

def must(cond: bool, code: str) -> None:
    if not bool(cond):
        raise CaptureError(code)

def numeric(x: Any, name: str) -> float:
    must(type(x) in (int, float) and np.isfinite(x), "INVALID_NUMBER:" + name)
    return float(x)

def state_diff(a: np.ndarray, b: np.ndarray) -> dict:
    must(a.shape == b.shape and a.ndim == 2 and a.shape[1] == 7, "INITIAL_STATE_SHAPE_MISMATCH")
    pos = float(np.max(np.abs(a[:, :3] - b[:, :3])))
    r0 = Rotation.from_quat(a[:, 3:7])
    r1 = Rotation.from_quat(b[:, 3:7])
    rot = float(np.max((r1 * r0.inv()).magnitude()))
    return {"max_translation_component_m":pos,"max_rotation_geodesic_rad":rot,
            "match":pos <= STATE_TOL_M and rot <= ROTATION_TOL_RAD}

def hash_committed(q: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(q, dtype="<f8").tobytes(order="C")).hexdigest()

def resolve_snapshot(manifest: Path, item: dict) -> Path:
    must(isinstance(item.get("npz"), str), "MISSING_STAGE_NPZ")
    path = (manifest.parent / item["npz"]).resolve()
    must(path.is_file(), "STAGE_NPZ_NOT_FOUND:" + str(path))
    return path

def check_flags(record: dict, scope: str):
    for key in ("direct_position_write", "projection_used", "rollback_used",
                "action_shielding_used", "modified_production"):
        must(record.get(key) is False, f"PROHIBITED_OR_UNKNOWN_{scope}:{key}")
    must(record.get("native_solver") == "GenericConstraintSolver",
         "NATIVE_SOLVER_MISMATCH:"+scope)

def analyze(manifest: Path, d: dict) -> dict:
    must(d.get("capture_kind") == "REAL_SOFA_NATIVE_GENERIC_CONSTRAINT_SOLVER",
         "CAPTURE_NOT_NATIVE")
    frame = d.get("frame", {})
    must(frame.get("vessel") == "B02" and frame.get("target") == "target_04"
         and frame.get("rl_step") == 2009 and frame.get("substep") == 1,
         "FRAME_ID_MISMATCH")
    must(frame.get("action_prefix_sha256") == ACTION_SHA,
         "ACTION_PREFIX_MISMATCH")
    must(d.get("physics_dt_s") == 0.005 and d.get("physics_substeps_per_action") == 2,
         "PHYSICS_CONFIGURATION_MISMATCH")
    arms = d.get("arms", {})
    must(isinstance(arms, dict) and set(arms) == {"nominal", "tight"},
         "EXPECTED_TWO_ARMS")
    result = {
        "test":"three_native_linearizations_plus_tight_solver_tolerance",
        "target":"B02/target_04 step2009 substep1",
        "action_sha256":ACTION_SHA,
        "historical_deep_penetration_cause":"UNPROVEN",
        "production_code_modified_by_verifier":False,
        "physical_solves_executed_by_verifier":0,
        "arms":{},
    }
    base_snapshot = None
    init_by_arm = {}
    for arm in ("nominal", "tight"):
        cfg = arms[arm]
        target_tol = 1e-6 if arm == "nominal" else 1e-9
        got_tol = numeric(cfg.get("requested_tolerance"), "requested_tolerance")
        must(got_tol == target_tol, "SOLVER_TOLERANCE_MISMATCH:" + arm)
        must(cfg.get("physics_substeps_observed_in_fork") == 1,
             "EXTRA_PHYSICS_SUBSTEP:"+arm)
        must(cfg.get("target_animate_calls") == 1, "EXTRA_ANIMATE:"+arm)
        must(cfg.get("native_solver_invocations") == 3, "NATIVE_INVOCATIONS_NOT_THREE:"+arm)
        must(cfg.get("mapping_refresh_count") == 2, "MAPPING_REFRESH_COUNT_MISMATCH:"+arm)
        check_flags(cfg, arm)
        must(cfg.get("solver_tolerance_applied_on_all_stages") is True,
             "TOLERANCE_NOT_APPLIED_ON_ALL_STAGES:"+arm)
        stages = cfg.get("stages")
        must(isinstance(stages,list) and len(stages) == 3,
             "EXPECTED_THREE_STAGES:"+arm)
        stage_results = []
        prior = None
        q_start = None
        for number, item in enumerate(stages, start=1):
            must(isinstance(item,dict),"INVALID_STAGE_OBJECT")
            check_flags(item,arm+f"_pass{number}")
            must(item.get("native_solver_invocations_cumulative") == number,
                 "NATIVE_INVOCATION_COUNTER_MISMATCH:"+arm+str(number))
            must(item.get("pass_index") == number, "STAGE_INDEX_MISMATCH")
            must(item.get("solver_tolerance") == target_tol,
                 "PER_STAGE_TOLERANCE_MISMATCH")
            path = resolve_snapshot(manifest, item)
            snap = load_snapshot(path)
            must(snap["q_prev"] is not None, "MISSING_Q_PREV:"+arm)
            # Every stage must carry same initial q_prev/q_free. Subsequent
            # row linearization base is recorded separately by the native hook.
            if q_start is None:
                q_start = snap
            else:
                for key in ("q_prev", "q_free"):
                    diff=state_diff(q_start[key], snap[key])
                    must(diff["match"],"STAGE_INITIAL_"+key.upper()+"_MISMATCH:"+arm)
            if number == 1:
                init_by_arm[arm] = snap
                if base_snapshot is None:
                    base_snapshot = snap
                else:
                    for key in ("q_prev", "q_free"):
                        must(state_diff(base_snapshot[key],snap[key])["match"],
                             "ARM_INITIAL_"+key.upper()+"_MISMATCH")
            else:
                must(item.get("rows_rebuilt_from_native_state") is True,
                     "ROWS_NOT_NATIVE_REBUILT")
                must(isinstance(item.get("relinearized_about_committed_state_sha256"),str),
                     "MISSING_RELINEARIZATION_PARENT_HASH")
                must(item["relinearized_about_committed_state_sha256"]
                     == hash_committed(prior["q_committed"]),
                     "RELINEARIZATION_PARENT_HASH_MISMATCH")
            rows=int(len(snap["free_violations"]))
            must(item.get("rebuilt_row_count") == rows,
                 "REBUILT_ROW_COUNT_MISMATCH:"+arm+str(number))
            must(item.get("active_count") == rows,
                 "ACTIVE_COUNT_MISMATCH:"+arm+str(number))
            must(item.get("native_solver_iterations") is not None,
                 "MISSING_NATIVE_SOLVER_ITERATIONS")
            iterations=int(item["native_solver_iterations"])
            must(iterations>=1,"INVALID_SOLVER_ITERATIONS")
            solver_error=numeric(item.get("solver_error"),"solver_error")
            must(solver_error >= 0,"NEGATIVE_SOLVER_ERROR")
            mm=float(np.min(snap["q_committed_dense_clearance"])*1000)
            last_sdf=nonlinear_case(snap,tol_m=MARGIN_GATE_M)
            stage_results.append({
                "pass":number,"capture_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                "npz":str(path), "row_count_actual":rows,
                "active_count":item["active_count"],"native_solver_iterations":iterations,
                "solver_error":solver_error,"requested_tolerance":target_tol,
                "solver_error_above_requested_tolerance":solver_error>target_tol,
                "committed_min_clearance_mm":mm,
                "margin_shortfall_mm":max(0.0,MARGIN_MM-mm),
                "nonlinear_max_prediction_error_raw_mm":
                    last_sdf["max_abs_nonlinear_mismatch_original_mm"],
                "worst_dense_index":last_sdf["committed_worst_index"],
                "worst_point_selected":last_sdf["committed_worst_selected"],
                "reported_linear_gap_mm":item.get("native_linear_row_gap_min_mm"),
                "native_solver_wall_ms":item.get("native_solver_wall_ms"),
            })
            if item.get("native_linear_row_gap_min_mm") is not None:
                numeric(item["native_linear_row_gap_min_mm"],"native_linear_row_gap_min_mm")
            if item.get("native_solver_wall_ms") is not None:
                must(numeric(item["native_solver_wall_ms"],"native_solver_wall_ms")>=0,
                     "NEGATIVE_NATIVE_SOLVE_TIME")
            prior=snap
        result["arms"][arm]={"requested_tolerance":target_tol,"passes":stage_results}
        final=stage_results[-1]
        result["arms"][arm]["final_clearance_mm"]=final["committed_min_clearance_mm"]
        result["arms"][arm]["margin_met"]=(
            final["committed_min_clearance_mm"] >= MARGIN_MM - MARGIN_GATE_M*1000)
        durations=[s["native_solver_wall_ms"] for s in stage_results]
        result["arms"][arm]["solver_wall_ms_total"]=(
            float(sum(durations)) if all(v is not None for v in durations) else None
        )
    baseline=result["arms"]["nominal"]["passes"][0]["committed_min_clearance_mm"]
    third=result["arms"]["nominal"]["passes"][2]["committed_min_clearance_mm"]
    must(abs(baseline-BASELINE_MM) <= MEASURE_TOL_MM,
         "BASELINE_NOT_REPRODUCED:"+str(baseline))
    must(abs(third-NOMINAL_THIRD_MM) <= MEASURE_TOL_MM,
         "NOMINAL_3PASS_NOT_REPRODUCED:"+str(third))
    a=result["arms"]["nominal"]
    b=result["arms"]["tight"]
    result["delta_final_clearance_tight_minus_nominal_mm"]=(
        b["final_clearance_mm"]-a["final_clearance_mm"])
    result["tight_margin_met"]=b["margin_met"]
    if a["solver_wall_ms_total"] and b["solver_wall_ms_total"] is not None:
        result["solver_wall_cost_ratio_tight_over_nominal"]=(
            b["solver_wall_ms_total"]/a["solver_wall_ms_total"])
    else:
        result["solver_wall_cost_ratio_tight_over_nominal"]=None
    result["numeric_outcome"]=(
        "PASS_MARGIN" if b["margin_met"] else "FAIL_MARGIN")
    result["native_execution_verification"]=(
        "TRACE_CONSISTENCY_ONLY_AUDIT_NATIVE_LOGS")
    result["explanation"]=(
        "Both arms must be independently backed by actual native fork logs. "
        "Manifest counters alone cannot prove C++ invocation or solver convergence. "
        "A PASS_MARGIN establishes only this safe test frame, not historical deep-penetration fix."
    )
    return result

def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    manifest=args.manifest.resolve()
    dest=args.output.resolve()
    if manifest==dest:
        p.error("Refusing to overwrite source manifest")
    try:
        d=json.loads(manifest.read_text(encoding="utf-8"))
        must(isinstance(d,dict),"MANIFEST_NOT_OBJECT")
        report=analyze(manifest,d)
        print("RESULT="+report["numeric_outcome"])
        code=0 if report["tight_margin_met"] else 1
    except (OSError,ValueError,KeyError,TypeError,IndexError) as exc:
        report={"numeric_outcome":"INCONCLUSIVE_INVALID_CAPTURE",
                "reason":str(exc),"historical_deep_penetration_cause":"UNPROVEN",
                "production_code_modified_by_verifier":False}
        print("RESULT=INCONCLUSIVE_INVALID_CAPTURE",file=sys.stderr)
        code=2
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+"\n",
                    encoding="utf-8")
    print(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False))
    print("REPORT="+str(dest))
    return code

if __name__=="__main__":
    raise SystemExit(main())
