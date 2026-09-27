#!/usr/bin/env python3
"""Final B02 full-episode acceptance using the dense-aware strict 0.100 mm solver.

This is intentionally a thin, immutable test-only entry point around
b02_full_episode_strict_margin_0p100_acceptance.py.  It locks the final
acceptance parameters and writes into a fresh runtime directory so earlier
historical FAIL/PASS artifacts are never overwritten.

Required safety formulation:
  solver requested margin      = 0.100 mm
  independent dense gate       = 0.100 mm (10 um Beam sampling)
  committed penetration limit  = 0.001 mm
  physics                      = 2 x 5 ms per RL step

The strict solver must include the step662 CASE-B fix: the independent dense
worst-clearance finite-difference gradient is an active SLSQP inequality.
"""
from __future__ import annotations

import inspect
from pathlib import Path
import sys

THIS = Path(__file__).resolve()
TEST_DIR = THIS.parent
BEAM_ROOT = THIS.parents[1]

if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

import b02_full_episode_strict_margin_0p100_acceptance as base


FINAL_RUNTIME = BEAM_ROOT / "_runtime" / "dense_aware_strict_margin_0p100_final"

# Exact engineering acceptance parameters.  Do not loosen these in this runner.
EXPECTED_MARGIN_M = 1e-4
EXPECTED_COMMITTED_LIMIT_M = 1e-6
EXPECTED_DENSE_SPACING_M = 1e-5
EXPECTED_SUBSTEPS = 2

# Structural preflight: make sure the solver being executed is the dense-aware
# version that fixed step662, not an older strict-margin file left on server.
_REQUIRED_SOLVER_SOURCE_MARKERS = (
    "dense_row=np.zeros(dim)",
    "STRICT_DENSE_CHECKER(state_from(yp))",
    "STRICT_DENSE_CHECKER(state_from(ym))",
    "J=np.vstack((J,dense_row))",
)


def _preflight() -> None:
    if "--step949-validation" in sys.argv:
        raise RuntimeError(
            "FINAL acceptance must run the complete episode; "
            "--step949-validation is forbidden here"
        )

    if abs(float(base.CANDIDATE_TARGET_MARGIN_M) - EXPECTED_MARGIN_M) > 1e-15:
        raise RuntimeError("candidate requested margin is not locked to 0.100 mm")
    if abs(float(base.CANDIDATE_DENSE_GATE_M) - EXPECTED_MARGIN_M) > 1e-15:
        raise RuntimeError("candidate dense gate is not locked to 0.100 mm")
    if abs(float(base.ACCEPT_MAX_PENETRATION_M) - EXPECTED_COMMITTED_LIMIT_M) > 1e-15:
        raise RuntimeError("committed penetration limit is not locked to 0.001 mm")
    if abs(float(base.DENSE_SPACING_M) - EXPECTED_DENSE_SPACING_M) > 1e-15:
        raise RuntimeError("independent dense spacing is not 10 um")
    if int(base.EXPECTED_SUBSTEPS) != EXPECTED_SUBSTEPS:
        raise RuntimeError("physics substep count is not 2")

    solver_source = inspect.getsource(base.strict_solver)
    missing = [
        marker for marker in _REQUIRED_SOLVER_SOURCE_MARKERS
        if marker not in solver_source
    ]
    if missing:
        raise RuntimeError(
            "DENSE_AWARE_STEP662_FIX_NOT_PRESENT: missing markers " + repr(missing)
        )

    # Fresh result namespace: preserve all earlier strict-margin runs.
    base.RESULTS_DIR = FINAL_RUNTIME / "results"
    base.LOG_DIR = FINAL_RUNTIME / "logs"

    # Re-lock globals explicitly in case this module is imported from an
    # interactive/debug environment that changed them.
    base.CANDIDATE_TARGET_MARGIN_M = EXPECTED_MARGIN_M
    base.CANDIDATE_DENSE_GATE_M = EXPECTED_MARGIN_M
    base.ACCEPT_MAX_PENETRATION_M = EXPECTED_COMMITTED_LIMIT_M
    base.DENSE_SPACING_M = EXPECTED_DENSE_SPACING_M
    base.STEP949_VALIDATION_MODE = False


def main() -> None:
    _preflight()
    base.main()


if __name__ == "__main__":
    main()
