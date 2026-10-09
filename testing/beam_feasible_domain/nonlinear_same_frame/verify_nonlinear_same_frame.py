#!/usr/bin/env python3
"""TEST ONLY: nonlinear Beam/SDF same-frame diagnostic + native-AB evidence gate.

No SOFA stepping, no state writes, no training.  A second native solve cannot be
inferred from existing q_free/q_committed endpoints: it must be supplied by an
independent, actually-executed native replay capture. This verifier makes that
distinction machine-checkable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

MARGIN_M = 0.0001
MATCH_TRANSLATION_M = 1e-10
MATCH_ANGLE_RAD = 1e-10
BASELINE_TARGET_MM = 0.082681
BASELINE_TOL_MM = 0.00005

def fetch(arr, name, required=True):
    if name not in arr:
        if required:
            raise ValueError(f"Missing {name}; provide real normalized NPZ data")
        return None
    out = np.asarray(arr[name])
    if not np.issubdtype(out.dtype, np.number) or not np.isfinite(out).all():
        raise ValueError(f"{name} must contain finite numeric values")
    return out

def compare_states(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 7:
        raise ValueError("Rigid3 states must have matching (N,7) shape")
    t = float(np.max(np.abs(x[:, :3] - y[:, :3])))
    rx = Rotation.from_quat(x[:, 3:7])
    ry = Rotation.from_quat(y[:, 3:7])
    angle = float(np.max((ry * rx.inv()).magnitude()))
    return {"translation_max_abs_m": t, "quaternion_geodesic_max_rad": angle,
            "match":t <= MATCH_TRANSLATION_M and angle <= MATCH_ANGLE_RAD}

def load(path):
    if path is None:
        return None
    with np.load(path, allow_pickle=False) as data:
        return {key:np.asarray(data[key]).copy() for key in data.files}

def inspect_snapshot(d):
    qf=fetch(d,"q_free")
    qc=fetch(d,"q_committed")
    free=fetch(d,"q_free_dense_clearance").reshape(-1)
    committed=fetch(d,"q_committed_dense_clearance").reshape(-1)
    selected=fetch(d,"selected_dense_indices").astype(int).reshape(-1)
    gf=fetch(d,"free_violations").reshape(-1)
    offs=fetch(d,"row_offsets").astype(int).reshape(-1)
    dofs=fetch(d,"dof_indices").astype(int).reshape(-1)
    jl=fetch(d,"linear_jacobian").reshape(-1,3)
    ja=fetch(d,"angular_jacobian").reshape(-1,3)
    if not (len(free) == len(committed) > 0):
        raise ValueError("Free and committed dense profiles must have same sample count")
    if not (len(selected)==len(gf)==len(offs)-1 and offs[0]==0
            and offs[-1]==len(dofs)==len(jl)==len(ja)):
        raise ValueError("CSR row sizes inconsistent")
    if np.any(np.diff(offs)<0) or np.any(dofs<0) or np.any(dofs>=len(qf)):
        raise ValueError("CSR row indices invalid")
    if np.any(selected<0) or np.any(selected>=len(free)):
        raise ValueError("Dense row indices invalid")
    compare_states(qf,qc)
    dtrans=qc[:,:3]-qf[:,:3]
    drot=(Rotation.from_quat(qc[:,3:7])*Rotation.from_quat(qf[:,3:7]).inv()).as_rotvec()
    predicted=gf.copy()
    for r in range(len(gf)):
        for j in range(int(offs[r]),int(offs[r+1])):
            n=dofs[j]
            predicted[r]+=float(jl[j]@dtrans[n]+ja[j]@drot[n])
    margin_a=fetch(d,"requested_margin_m",required=False)
    margin=float(margin_a.reshape(-1)[0]) if margin_a is not None else MARGIN_M
    actual=committed[selected]-margin
    errs=actual-predicted
    worst_free=int(np.argmin(free))
    worst_commit=int(np.argmin(committed))
    return {
        "status":"MEASURED_OFFLINE",
        "free_min_clearance_mm":float(np.min(free)*1000),
        "committed_min_clearance_mm":float(np.min(committed)*1000),
        "margin_mm":margin*1000,
        "margin_shortfall_mm":float((margin-np.min(committed))*1000),
        "selected_row_count":len(selected),
        "linear_min_row_gap_mm":float(np.min(predicted)*1000),
        "actual_nonlinear_selected_min_gap_mm":float(np.min(actual)*1000),
        "max_abs_nonlinear_prediction_error_mm":float(np.max(np.abs(errs))*1000),
        "mean_abs_nonlinear_prediction_error_mm":float(np.mean(np.abs(errs))*1000),
        "selected_error_mm":(errs*1000).tolist(),
        "selected_linear_gap_mm":(predicted*1000).tolist(),
        "selected_actual_gap_mm":(actual*1000).tolist(),
        "free_worst_index":worst_free,
        "committed_worst_index":worst_commit,
        "minimum_migrated":worst_free!=worst_commit,
        "committed_worst_selected":worst_commit in set(selected.tolist()),
        "all_linear_rows_satisfied_within_1e-12m":bool(np.all(predicted>=-1e-12)),
        "global_margin_reached_within_1e-12m":bool(np.min(committed)>=margin-1e-12),
        "note":"Only captured endpoints; errors show nonlinearity, not native causal improvement."
    }

def evidence(base, candidate, *, label):
    if candidate is None:
        return {"status":"INCONCLUSIVE","reason":"No independent native branch NPZ was provided"}
    qf_b=fetch(base,"q_free")
    qf_c=fetch(candidate,"q_free")
    qprev_b=fetch(base,"q_prev")
    qprev_c=fetch(candidate,"q_prev")
    initial_free=compare_states(qf_b,qf_c)
    initial_prev=compare_states(qprev_b,qprev_c)
    if not initial_free["match"] or not initial_prev["match"]:
        return {"status":"INVALID_COMPARISON","reason":"q_prev or q_free differs; not the same frame",
                "q_free":initial_free,"q_prev":initial_prev}
    req={"native_branch_kind","native_solve_count","native_substeps_executed","native_position_overwrite_count",
         "native_solver_used","native_same_frame","native_relinearization_count"}
    missing=sorted(req-set(candidate))
    if missing:
        return {"status":"INCONCLUSIVE","reason":"Missing native execution evidence",
                "missing":missing}
    kind=str(np.asarray(candidate["native_branch_kind"]).reshape(-1)[0])
    solves=int(np.asarray(candidate["native_solve_count"]).reshape(-1)[0])
    substeps=int(np.asarray(candidate["native_substeps_executed"]).reshape(-1)[0])
    writes=int(np.asarray(candidate["native_position_overwrite_count"]).reshape(-1)[0])
    used=bool(np.asarray(candidate["native_solver_used"]).reshape(-1)[0])
    same=bool(np.asarray(candidate["native_same_frame"]).reshape(-1)[0])
    rounds=int(np.asarray(candidate["native_relinearization_count"]).reshape(-1)[0])
    if (kind!="SOFA_NATIVE" or not used or not same or substeps!=1 or writes!=0
            or solves<1 or rounds<1 or rounds>3):
        return {"status":"INVALID_COMPARISON",
                "reason":"Evidence flags violate native single-frame no-write contract",
                "kind":kind,"solves":solves,"substeps":substeps,
                "position_overwrites":writes,"relinearizations":rounds}
    # Flags are assertions from the recording harness, not independently
    # verified by NPZ; raw SOFA logs are mandatory provenance.
    metrics=inspect_snapshot(candidate)
    return {"status":"EVIDENCE_REQUIRES_LOG_AUDIT",
            "label":label,"native_asserted":True,
            "baseline_committed_mm":inspect_snapshot(base)["committed_min_clearance_mm"],
            "candidate_committed_mm":metrics["committed_min_clearance_mm"],
            "improvement_mm":metrics["committed_min_clearance_mm"]-inspect_snapshot(base)["committed_min_clearance_mm"],
            "candidate_margin_reached":metrics["global_margin_reached_within_1e-12m"],
            "native_solve_count":solves,"relinearization_count":rounds,
            "q_free_match":initial_free,"q_prev_match":initial_prev,
            "note":"Not PASS until native branch logs verify repeated solves, updated rows, and zero state overwrite."}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline",type=Path,required=True)
    p.add_argument("--native-round2",type=Path)
    p.add_argument("--native-round3",type=Path)
    p.add_argument("--output",type=Path)
    p.add_argument("--require-baseline-match",action="store_true",
                   help="Require captured committed minimum to match +0.082681 mm")
    ns=p.parse_args(argv)
    try:
        base=load(ns.baseline)
        baseline=inspect_snapshot(base)
        match=abs(baseline["committed_min_clearance_mm"]-BASELINE_TARGET_MM)<=BASELINE_TOL_MM
        result={
            "baseline":baseline,"baseline_expected_mm":BASELINE_TARGET_MM,
            "baseline_match_0p00005mm":match,
            "round2":evidence(base,load(ns.native_round2),label="two native linearizations"),
            "round3":evidence(base,load(ns.native_round3),label="three native linearizations"),
            "historical_minus_0p766410143mm_cause":"UNPROVEN",
            "production_modified_by_this_script":False,
            "simulation_stepped_by_this_script":False,
        }
        if ns.output:
            ns.output.parent.mkdir(parents=True,exist_ok=True)
            ns.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
        print(json.dumps(result,ensure_ascii=False,indent=2))
        if ns.require_baseline_match and not match:
            return 3
        return 0
    except (ValueError,KeyError,OSError) as exc:
        print("INCONCLUSIVE: "+str(exc),file=sys.stderr)
        return 2

if __name__=="__main__":
    raise SystemExit(main())
