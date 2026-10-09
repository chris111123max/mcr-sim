#!/usr/bin/env python3
"""Synthetic contract tests only. NO SOFA / no physics / no PPO."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from verify_joint_ab import analyze, hash_committed, CaptureError  # noqa: E402

def q0():
    q=np.zeros((2,7),dtype=np.float64)
    q[:,0]=[0,0.01]
    q[:,6]=1.0
    return q

def make_manifest(directory:Path, tight_final_mm=0.10005):
    baseline=q0()
    arms={}
    defaults={"native_solver":"GenericConstraintSolver",
              "direct_position_write":False,"projection_used":False,
              "rollback_used":False,"action_shielding_used":False,
              "modified_production":False}
    for arm,tol,values in (
        ("nominal",1e-6,[0.082681,0.095125,0.096711]),
        ("tight",1e-9,[0.085538,0.0990,tight_final_mm])
    ):
        arm_dir=directory/arm
        arm_dir.mkdir(parents=True,exist_ok=True)
        stages=[]
        prior=None
        for i,clearance in enumerate(values,1):
            nrows=6 if i==1 else 5
            qc=baseline.copy()
            qc[0,1]=i*1e-5+(0 if arm=="nominal" else 1e-6)
            path=arm_dir/f"pass{i}.npz"
            sample_count=20
            freeprofile=np.full(sample_count,0.00015)
            freeprofile[0]=0.00008
            profile=np.full(sample_count,0.0002)
            profile[0]=clearance*1e-3
            np.savez_compressed(
                path,q_prev=baseline,q_free=baseline,q_committed=qc,
                row_offsets=np.arange(nrows+1),
                dof_indices=np.zeros(nrows,dtype=int),
                linear_jacobian=np.zeros((nrows,3)),
                angular_jacobian=np.zeros((nrows,3)),
                free_violations=np.zeros(nrows),
                selected_dense_indices=np.arange(nrows,dtype=int),
                q_free_dense_clearance=freeprofile,
                q_committed_dense_clearance=profile,
            )
            meta=dict(defaults,
                      npz=str(path.relative_to(directory)),
                      pass_index=i,solver_tolerance=tol,
                      native_solver_invocations_cumulative=i,
                      native_solver_iterations=5+i,
                      solver_error=tol*0.5,rebuilt_row_count=nrows,
                      active_count=nrows,native_solver_wall_ms=2.0*i)
            if i>1:
                meta["rows_rebuilt_from_native_state"]=True
                meta["relinearized_about_committed_state_sha256"]=hash_committed(prior)
            stages.append(meta)
            prior=qc
        arms[arm]=dict(defaults,requested_tolerance=tol,
                       physics_substeps_observed_in_fork=1,target_animate_calls=1,
                       native_solver_invocations=3,mapping_refresh_count=2,
                       solver_tolerance_applied_on_all_stages=True,stages=stages)
    return {"capture_kind":"REAL_SOFA_NATIVE_GENERIC_CONSTRAINT_SOLVER",
            "frame":{"vessel":"B02","target":"target_04","rl_step":2009,
                     "substep":1,
                     "action_prefix_sha256":
                         "6f534ad515bfa635666f9d3237783681e65dc378a3d5608706db30bb00e181c"},
            "physics_dt_s":0.005,"physics_substeps_per_action":2,
            "arms":arms}

class ContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.dir=Path(self.tmp.name)
        self.manifest=self.dir/"joint_ab_manifest.json"
        self.data=make_manifest(self.dir)
    def tearDown(self):
        self.tmp.cleanup()
    def run_validation(self):
        return analyze(self.manifest,self.data)
    def test_synthetic_success_requires_tight_margin(self):
        r=self.run_validation()
        self.assertEqual(r["numeric_outcome"],"PASS_MARGIN")
        self.assertEqual([x["active_count"] for x in r["arms"]["nominal"]["passes"]],[6,5,5])
        self.assertGreater(r["delta_final_clearance_tight_minus_nominal_mm"],0)
    def test_synthetic_margin_fail(self):
        self.data=make_manifest(self.dir,tight_final_mm=0.099)
        self.assertEqual(self.run_validation()["numeric_outcome"],"FAIL_MARGIN")
    def test_stale_expected_six_rows_cannot_pass(self):
        self.data["arms"]["tight"]["stages"][1]["active_count"]=6
        with self.assertRaisesRegex(CaptureError,"ACTIVE_COUNT_MISMATCH"):
            self.run_validation()
    def test_no_fake_native_second_solve(self):
        self.data["arms"]["tight"]["stages"][1]["native_solver_invocations_cumulative"]=1
        with self.assertRaisesRegex(CaptureError,"NATIVE_INVOCATION_COUNTER_MISMATCH"):
            self.run_validation()
    def test_mismatched_parent_sha_rejected(self):
        self.data["arms"]["tight"]["stages"][2]["relinearized_about_committed_state_sha256"]="dummy"
        with self.assertRaisesRegex(CaptureError,"RELINEARIZATION_PARENT_HASH_MISMATCH"):
            self.run_validation()
    def test_extra_animate_rejected(self):
        self.data["arms"]["tight"]["target_animate_calls"]=2
        with self.assertRaisesRegex(CaptureError,"EXTRA_ANIMATE"):
            self.run_validation()
    def test_position_overwrite_rejected(self):
        self.data["arms"]["tight"]["stages"][1]["direct_position_write"]=True
        with self.assertRaisesRegex(CaptureError,"PROHIBITED_OR_UNKNOWN"):
            self.run_validation()
    def test_initial_state_mismatch_rejected(self):
        stage_path=self.dir/self.data["arms"]["tight"]["stages"][0]["npz"]
        with np.load(stage_path,allow_pickle=False) as z:
            contents={k:np.array(z[k]) for k in z.files}
        contents["q_free"][0,0]+=1e-6
        np.savez_compressed(stage_path,**contents)
        with self.assertRaisesRegex(CaptureError,"ARM_INITIAL_Q_FREE_MISMATCH"):
            self.run_validation()

if __name__=="__main__":
    unittest.main()
