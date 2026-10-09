#!/usr/bin/env python3
"""TEST ONLY: paired three-pass native relinearization × solver-tolerance gate.

This is a *native capture runner/validator*, not an alternative physics solver.
Run an existing test-only native replay through --command-template; it must emit
real same-substep SOFA stage NPZ/trace files for both arms. Never alters production.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import numpy as np

EXPECTED_BASELINE_MM = 0.082681
MARGIN_MM = 0.100
PREFIX_SHA256 = "6f534ad5155bfa635666f9d3237783681e65dc378a3d5608706db30bb00e181c"
ARMS = (("three_default", 1e-6), ("three_tight", 1e-9))

def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k:np.array(z[k]) for k in z.files}

def maximum_difference(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=float)-np.asarray(b, dtype=float))))

def captured_fields(d):
    names = ("q_prev","q_free","q_committed","q_committed_dense_clearance")
    absent = [name for name in names if name not in d]
    if absent:
        raise ValueError(f"Missing actual SOFA capture fields: {absent}")
    for n in names[:3]:
        if d[n].ndim!=2 or d[n].shape[1]!=7 or not np.isfinite(d[n]).all():
            raise ValueError(f"Invalid Rigid3 capture: {n}")
    v=np.asarray(d["q_committed_dense_clearance"],dtype=float).reshape(-1)
    if len(v)==0 or not np.isfinite(v).all():
        raise ValueError("Invalid full dense committed SDF capture")
    return float(np.min(v)*1e3)

def inspect_arm(root, label, target_tol, baseline):
    trace_path=root/"native_trace.json"
    if not trace_path.is_file():
        return {"status":"INCONCLUSIVE","reason":"Missing native_trace.json"}
    trace=json.loads(trace_path.read_text(encoding="utf-8"))
    if trace.get("capture_kind")!="REAL_SOFA_NATIVE_GENERIC_CONSTRAINT_SOLVER":
        raise ValueError("Not a genuine SOFA native trace")
    if trace.get("target_rl_step")!=2009 or trace.get("target_substep")!=1:
        raise ValueError("Wrong RL step/substep")
    if trace.get("physics_substeps_per_action")!=2 or trace.get("physics_substeps_observed_in_fork")!=1:
        raise ValueError("Extra physical substep or invalid physics")
    if abs(float(trace.get("physics_dt_s",0))-.005)>1e-12:
        raise ValueError("Wrong physical timestep")
    if trace.get("action_prefix_sha256")!=PREFIX_SHA256:
        return {"status":"INCONCLUSIVE","reason":"Trace missing or wrong verified raw action prefix"}
    if abs(float(trace.get("solver_tolerance",float("nan")))-target_tol)>target_tol*1e-6:
        raise ValueError("Requested native solver tolerance not verified by trace")
    passes=trace.get("passes",[])
    if not isinstance(passes,list) or len(passes)!=3:
        raise ValueError("Expected exactly three genuine same-substep native solver passes")
    records=[]
    previous=None
    for i,record in enumerate(passes):
        if record.get("native_solver")!="GenericConstraintSolver":
            raise ValueError(f"Wrong solver pass {i}")
        if int(record.get("native_solver_invocations_cumulative",-1))!=i+1:
            raise ValueError(f"Native solver count wrong at pass {i}")
        if i>0 and record.get("rows_rebuilt_from_native_state") is not True:
            raise ValueError(f"Native relinearization not shown for pass {i}")
        if any(record.get(k) is not False for k in
                ("direct_position_write","projection_used","rollback_used","action_shielding_used")):
            raise ValueError(f"Forbidden DOF manipulation at pass {i}")
        npz=(root / record["npz"]).resolve()
        if root.resolve() not in npz.parents:
            raise ValueError("Native evidence path escapes arm results directory")
        arr=load_npz(npz)
        clearance=captured_fields(arr)
        for field in ("q_prev","q_free"):
            if baseline[field].shape != arr[field].shape or maximum_difference(baseline[field],arr[field])>1e-10:
                raise ValueError(f"Input {field} mismatch in {label} pass{i}")
        if i==0 and abs(clearance-EXPECTED_BASELINE_MM)>0.00001:
            raise ValueError(f"Baseline +0.082681mm not reproduced: {clearance}")
        if i>0:
            sha=hashlib.sha256(np.asarray(previous,dtype="<f8").tobytes()).hexdigest()
            if record.get("relinearized_about_committed_state_sha256")!=sha:
                raise ValueError(f"Pass {i} not relinearized at previous committed state")
        row_count=record.get("rebuilt_row_count")
        active=record.get("native_active_count")
        if row_count is None or active is None or int(row_count)!=int(active):
            raise ValueError(f"Pass{i}: actual rebuilt rows / activeCount mismatch")
        iterations=record.get("solver_iterations")
        error=record.get("solver_error")
        if iterations is None or error is None or not np.isfinite(float(error)):
            raise ValueError(f"Pass{i}: missing solver iterations/error")
        free_g=arr.get("free_violations")
        if free_g is None:
            raise ValueError(f"Pass{i}: missing actual row violations")
        record_out={
            "pass":i+1,"committed_clearance_mm":clearance,
            "row_count":int(row_count),"active_count":int(active),
            "solver_iterations":int(iterations),"solver_error":float(error),
            "solver_tolerance":target_tol,
            "solver_met_tolerance":float(error)<=target_tol,
        }
        if "min_linear_row_gap_mm" in record:
            record_out["min_linear_row_gap_mm"]=float(record["min_linear_row_gap_mm"])
        if "max_nonlinear_sdf_error_mm" in record:
            record_out["max_nonlinear_sdf_error_mm"]=float(record["max_nonlinear_sdf_error_mm"])
        records.append(record_out)
        previous=arr["q_committed"]
    return {"status":"VALIDATED_TRACE_WITH_NATIVE_LOG_AUDIT_REQUIRED",
            "arm":label,"tolerance":target_tol,"passes":records,
            "final_clearance_mm":records[-1]["committed_clearance_mm"],
            "reaches_margin":records[-1]["committed_clearance_mm"]>=MARGIN_MM,
            "all_solver_errors_meet_tolerance":all(r["solver_met_tolerance"] for r in records)}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline",type=Path,required=True)
    ap.add_argument("--results",type=Path,required=True)
    ap.add_argument("--command-template",help=(
        "Test-only REAL SOFA runner command with placeholders {tolerance}, {output_dir}, "
        "{baseline}. Must produce output_dir/native_trace.json + stage NPZ. "
        "Omit for audit-only mode; command runs once per arm."))
    args=ap.parse_args()
    base=load_npz(args.baseline.resolve())
    clearance=captured_fields(base)
    if abs(clearance-EXPECTED_BASELINE_MM)>0.00001:
        raise ValueError(f"Reference committed clearance mismatch: {clearance} mm")
    root=args.results.resolve()
    root.mkdir(parents=True,exist_ok=True)
    findings={}
    for label,tol in ARMS:
        dst=root/label
        dst.mkdir(parents=True,exist_ok=True)
        started=time.perf_counter()
        if args.command_template:
            # Never shell=True. Each arm must use the server's test-only harness
            # and be inspected for no PPO / no production writes before execution.
            cmd=shlex.split(args.command_template.format(
                tolerance=f"{tol:.0e}",output_dir=str(dst),baseline=str(args.baseline.resolve())))
            if not cmd:
                raise ValueError("Empty native test command")
            completed=subprocess.run(cmd,cwd=Path.cwd(),check=False,capture_output=True,text=True)
            (dst/"runner.stdout.log").write_text(completed.stdout,encoding="utf-8")
            (dst/"runner.stderr.log").write_text(completed.stderr,encoding="utf-8")
            if completed.returncode:
                findings[label]={"status":"INCONCLUSIVE","reason":f"Native runner exit {completed.returncode}"}
                continue
        try:
            findings[label]=inspect_arm(dst,label,tol,base)
        except (ValueError,KeyError,OSError,TypeError) as exc:
            findings[label]={"status":"INVALID_EVIDENCE","reason":str(exc)}
        findings[label]["elapsed_s"]=time.perf_counter()-started
    a=findings["three_default"]
    b=findings["three_tight"]
    if a.get("status","").startswith("VALIDATED") and b.get("status","").startswith("VALIDATED"):
        findings["comparison"]={
            "tight_minus_default_clearance_mm":b["final_clearance_mm"]-a["final_clearance_mm"],
            "tight_margin_pass":b["reaches_margin"],
            "tight_all_solver_errors_within_tolerance":b["all_solver_errors_meet_tolerance"],
            "same_frame_but_original_deep_penetration_cause":"UNPROVEN"}
    else:
        findings["comparison"]={"status":"INCONCLUSIVE","reason":"Both genuine native A/B captures required"}
    report={"test":"three-native-solve tolerance joint AB","source_baseline":str(args.baseline.resolve()),
            "action_prefix_sha256":PREFIX_SHA256,"results":findings,"production_edited":False,
            "original_minus_0p766410143mm_failure_cause":"UNPROVEN"}
    out=root/"joint_ab_report.json"
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False))
    print("REPORT="+str(out))
    return 0 if "tight_minus_default_clearance_mm" in findings.get("comparison",{}) else 2

if __name__=="__main__":
    try: sys.exit(main())
    except (OSError,ValueError,KeyError) as exc:
        print("INCONCLUSIVE: "+str(exc),file=sys.stderr)
        sys.exit(2)
