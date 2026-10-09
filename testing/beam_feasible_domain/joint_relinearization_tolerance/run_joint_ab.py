#!/usr/bin/env python3
"""Test-only two-arm launcher for an EXISTING same-substep native replay harness.

The harness is provided by the server's previously validated SOFA test fork.
This launcher invokes it twice with independently fixed tolerance settings,
checks outputs and calls verify_joint_ab.py. It does not implement or fake a
native solve and rejects production codepaths as harness entrypoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

HERE=Path(__file__).resolve().parent
TESTING=HERE.parents[1].resolve()
EXPECTED_SHA="6f534ad515bfa635666f9d3237783681e65dc378a3d5608706db30bb00e181c"

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--native-runner",type=Path,required=True,
                    help="Existing validated test-only native replay .py, inside testing/")
    ap.add_argument("--runner-args",required=True,
                    help="Runner CLI argument template, shlex-parsed (no shell): "
                         "use {arm}, {tolerance}, {passes}, {output_dir}, {action_sha256}")
    ap.add_argument("--python",type=Path,default=Path(sys.executable))
    ap.add_argument("--output-dir",type=Path,default=HERE/"results"/"joint_ab")
    ap.add_argument("--max-seconds-per-arm",type=int,default=1800)
    args=ap.parse_args()
    runner=args.native_runner.expanduser().resolve()
    output=args.output_dir.expanduser().resolve()
    try:
        runner.relative_to(TESTING)
    except ValueError:
        ap.error("Native runner must live under repository testing/")
    if not runner.is_file() or runner.suffix!=".py":
        ap.error("Native runner must be an existing test-only Python file")
    if runner==Path(__file__).resolve():
        ap.error("Cannot recursively invoke the launcher")
    if args.max_seconds_per_arm<=0:
        ap.error("Timeout must be positive")
    if output==runner or runner in output.parents:
        ap.error("Refuse to write output into existing runner source tree")
    output.mkdir(parents=True,exist_ok=True)
    compiled=[]
    for arm,tol in (("nominal",1e-6),("tight",1e-9)):
        arm_dir=output/arm
        arm_dir.mkdir(parents=True,exist_ok=True)
        fmt={"arm":arm,"tolerance":format(tol,".12g"),"passes":"3",
             "output_dir":str(arm_dir),"action_sha256":EXPECTED_SHA}
        try:
            runner_args=[item.format_map(fmt) for item in shlex.split(args.runner_args)]
        except (ValueError,KeyError) as exc:
            ap.error(f"Invalid --runner-args: {exc}")
        # Runner accepts exact args only; these deny some obvious unrelated
        # training commands, but the native implementation must still be audited.
        forbidden=("--epochs","--episodes-per-epoch","--train","--n-envs")
        if any(x in forbidden for x in runner_args):
            ap.error("Training-oriented options are forbidden")
        command=[str(args.python),str(runner),*runner_args]
        log_path=arm_dir/"native_runner.log"
        with log_path.open("w",encoding="utf-8") as log:
            completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,
                                     cwd=Path.cwd(),timeout=args.max_seconds_per_arm,
                                     check=False)
        if completed.returncode:
            print(f"INCONCLUSIVE: {arm} native runner exit {completed.returncode}; {log_path}",
                  file=sys.stderr)
            return 2
        fragment=arm_dir/"arm_manifest.json"
        if not fragment.is_file():
            print(f"INCONCLUSIVE: missing REAL native output {fragment}",file=sys.stderr)
            return 2
        with fragment.open(encoding="utf-8") as fh:
            arm_data=json.load(fh)
        if not isinstance(arm_data,dict) or "stages" not in arm_data:
            print(f"INCONCLUSIVE: incomplete arm fragment {fragment}",file=sys.stderr)
            return 2
        if arm_data.get("requested_tolerance")!=tol:
            print(f"INCONCLUSIVE: actual solver tolerance not attested: {arm}",file=sys.stderr)
            return 2
        for st in arm_data["stages"]:
            if not isinstance(st,dict) or "npz" not in st:
                print(f"INCONCLUSIVE: missing stage path {arm}",file=sys.stderr)
                return 2
            snap=(fragment.parent/st["npz"]).resolve()
            if not snap.is_file():
                print(f"INCONCLUSIVE: missing actual snapshot {snap}",file=sys.stderr)
                return 2
            # Compose only paths; never duplicate, interpolate or invent
            # the physical capture's numeric values.
            st["npz"]=str(snap.relative_to(output))
        compiled.append(arm_data)
    manifest={
        "capture_kind":"REAL_SOFA_NATIVE_GENERIC_CONSTRAINT_SOLVER",
        "frame":{"vessel":"B02","target":"target_04","rl_step":2009,
                 "substep":1,"action_prefix_sha256":EXPECTED_SHA},
        "physics_dt_s":0.005,"physics_substeps_per_action":2,
        "arms":{"nominal":compiled[0],"tight":compiled[1]},
    }
    path=output/"joint_ab_manifest.json"
    path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    verify=HERE/"verify_joint_ab.py"
    cmd=[str(args.python),str(verify),"--manifest",str(path),
         "--output",str(output/"joint_ab_report.json")]
    process=subprocess.run(cmd,cwd=Path.cwd(),check=False)
    if process.returncode==0:
        print("PASS_MARGIN_NUMERIC — native logs require manual audit")
    elif process.returncode==1:
        print("FAIL_MARGIN_NUMERIC — native logs require manual audit")
    else:
        print("INCONCLUSIVE — see native logs and verification report")
    return process.returncode

if __name__=="__main__":
    raise SystemExit(main())
