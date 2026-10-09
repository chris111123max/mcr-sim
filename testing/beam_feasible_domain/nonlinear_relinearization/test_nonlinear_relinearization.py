#!/usr/bin/env python3
"""Synthetic smoke tests for read-only nonlinear probe and native trace gate.

Run: python -m unittest discover -s testing/beam_feasible_domain/nonlinear_relinearization
These tests do NOT establish SOFA native feasibility or solve anything physical.
"""
from __future__ import annotations
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from offline_directional_probe import (
    _selftest, interpolate_correction, probe, _validate_q
)
from verify_native_trace import verify


def _beam(z=0.0):
    return np.array([
        [0.0, 0.0, z, 0.0, 0.0, 0.0, 1.0],
        [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)


class FakeSDF:
    """Numerical fixture only. Does not impersonate production SDF."""
    def __init__(self):
        self.count = 0

    def profile(self, q):
        self.count += 1
        return np.array([6e-5 + .3 * q[0, 2], 2e-4], dtype=float)


class ProbeTests(unittest.TestCase):
    def test_internal_synthetic_selftest(self):
        self.assertEqual(_selftest()["selftest"], "PASS")

    def test_directional_can_reach_margin_only_in_numpy(self):
        qf = _beam()
        qc = _beam(1e-4)
        sdf = FakeSDF()
        arrays = {
            "q_free_dense_clearance": sdf.profile(qf),
            "q_committed_dense_clearance": sdf.profile(qc),
        }
        result = probe(arrays, qf, qc, sdf, max_newton_rounds=3)
        self.assertEqual(result["baseline_capture_gate"], "PASS")
        self.assertEqual(result["directional_outcome"], "DIRECTIONAL_MARGIN_MET")
        self.assertEqual(result["native_physics_solve_count"], 0)
        self.assertFalse(result["native_relinearization_verified"])
        self.assertAlmostEqual(result["baseline_clearance_mm"], .09)

    def test_capture_mismatch_is_not_counted_as_success(self):
        qf, qc, sdf = _beam(), _beam(1e-4), FakeSDF()
        arrays = {
            "q_free_dense_clearance": sdf.profile(qf)+.00001,
            "q_committed_dense_clearance": sdf.profile(qc),
        }
        with self.assertRaisesRegex(ValueError, "CAPTURE_GEOMETRY_MISMATCH"):
            probe(arrays, qf, qc, sdf)

    def test_rotation_interpolation_endpoint(self):
        qf, qc = _beam(), _beam(1e-4)
        qc[0,3:7] = Rotation.from_euler("x",23,degrees=True).as_quat()
        np.testing.assert_allclose(interpolate_correction(qf,qc,1.),qc,atol=1e-14)
        np.testing.assert_allclose(interpolate_correction(qf,qc,0.),qf,atol=1e-14)

    def test_nan_is_rejected(self):
        q=_beam()
        q[0,0]=float("nan")
        with self.assertRaises(ValueError):
            _validate_q(q,"test")


class NativeGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.qf = _beam()
        self.qc = _beam(.00001)
        self.arrays = self.root/"frame.npz"
        self._save(self.arrays,self.qc,0.082681e-3)
        self._save(self.root/"pass1.npz",_beam(.00002),0.1001e-3)
        self.trace=self.root/"native.json"
        import hashlib
        self.previous_sha = hashlib.sha256(np.asarray(self.qc,dtype="<f8").tobytes()).hexdigest()
        self.payload={
            "capture_kind":"REAL_SOFA_NATIVE_GENERIC_CONSTRAINT_SOLVER",
            "physics_substeps_per_action":2,
            "physics_dt_s":0.005,
            "target_rl_step":2009,
            "target_substep":1,
            "physics_substeps_observed_in_fork":1,
            "passes":[
                {"npz":"frame.npz", "native_solver":"GenericConstraintSolver",
                 "native_solver_invocations_cumulative":1, "rows_rebuilt_from_native_state":False,
                 "direct_position_write":False,"projection_used":False,
                 "rollback_used":False,"action_shielding_used":False},
                {"npz":"pass1.npz", "native_solver":"GenericConstraintSolver",
                 "native_solver_invocations_cumulative":2, "rows_rebuilt_from_native_state":True,
                 "relinearized_about_committed_state_sha256":self.previous_sha,
                 "direct_position_write":False,"projection_used":False,
                 "rollback_used":False,"action_shielding_used":False},
            ]
        }
        self.trace.write_text(json.dumps(self.payload))

    def _save(self,path,qc,clearance):
        np.savez_compressed(
            path,q_prev=self.qf,q_free=self.qf,q_committed=qc,
            q_committed_dense_clearance=np.array([clearance,2e-4])
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_native_trace_consistency(self):
        report=verify(self.trace,self.arrays)
        self.assertEqual(report["trace_consistency"],"PASS")
        self.assertTrue(report["recorded_native_final_margin_met"])
        self.assertFalse(report["historical_deep_penetration_root_cause_proven"])

    def test_no_second_native_call_rejected(self):
        self.payload["passes"][1]["native_solver_invocations_cumulative"]=1
        self.trace.write_text(json.dumps(self.payload))
        with self.assertRaisesRegex(ValueError,"NO_ADDITIONAL_NATIVE_SOLVE"):
            verify(self.trace,self.arrays)

    def test_extra_substep_rejected(self):
        self.payload["physics_substeps_observed_in_fork"]=2
        self.trace.write_text(json.dumps(self.payload))
        with self.assertRaisesRegex(ValueError,"ADDITIONAL_PHYSICS_SUBSTEPS_EXECUTED"):
            verify(self.trace,self.arrays)

    def test_position_write_rejected(self):
        self.payload["passes"][1]["direct_position_write"]=True
        self.trace.write_text(json.dumps(self.payload))
        with self.assertRaisesRegex(ValueError,"PROHIBITED_STATE_MANIPULATION"):
            verify(self.trace,self.arrays)

    def test_relinearize_source_hash_rejected(self):
        self.payload["passes"][1]["relinearized_about_committed_state_sha256"]="fake"
        self.trace.write_text(json.dumps(self.payload))
        with self.assertRaisesRegex(ValueError,"NOT_RELINEARIZED_AT_PRIOR_NATIVE_COMMITTED_STATE"):
            verify(self.trace,self.arrays)


if __name__=="__main__":
    unittest.main()
