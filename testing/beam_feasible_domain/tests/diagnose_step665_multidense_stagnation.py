#!/usr/bin/env python3
"""B02 step665/substep1 multi-dense active-set stagnation diagnosis.

This is the final engineering acceptance gate for the current penetration
problem, not a mathematical proof of all future trajectories.

Configuration:
  - B02 / target_04
  - seed 15204 by default
  - fixed epoch-74 PPO checkpoint
  - deterministic policy
  - production execution: 1 RL step = 2 x 5 ms physics substeps
  - safety layer active from the FIRST physics substep
  - run until the episode terminates/truncates or --max-rl-steps (default 2048)

At every CollisionBeginEvent:
  committed Beam state
      -> real Beam free_position
      -> production BeamAdapter + B02 SDF dense check
      -> if unsafe: validated real-Beam feasible solver
      -> native freePosition/freeVelocity write
      -> native MultiAdaptiveBeamMapping propagation
      -> native collision/constraint completion
      -> committed real-Beam dense check

Hard acceptance criterion:
  max committed penetration <= 1e-6 m == 0.001 mm
  solver failures == 0
  native failures == 0
  NaN/Inf == 0
  forbidden fallbacks == 0

CollisionDOFs are mapping diagnostics only.
No rollback, projection, action shielding, committed-position write, or training.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import Sofa

THIS = Path(__file__).resolve()
TEST_DIR = THIS.parent
BEAM_ROOT = THIS.parents[1]
RUNTIME_DIR = BEAM_ROOT / "_runtime"
RESULTS_DIR = RUNTIME_DIR / "strict_margin_0p100" / "step665_multidense_diagnosis" / "results"
LOG_DIR = RUNTIME_DIR / "strict_margin_0p100" / "step665_multidense_diagnosis" / "logs"
DEFAULT_BUILD_DIR = RUNTIME_DIR / "native_build"

if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from b02_online_feasible_solver_bridge import (
    ValidatedSolverBridgeUnavailable,
    _extract,
)
import b02_step776_feasible_solve as validated_solver
STRICT_SOLVER_DIR = BEAM_ROOT / "strict_margin_solver"
if str(STRICT_SOLVER_DIR) not in sys.path:
    sys.path.insert(0, str(STRICT_SOLVER_DIR))
import strict_b02_feasible_solve as strict_solver
import strict_b02_feasible_solve_multidense as multidense_solver

STEP665_BASELINE_DIAGNOSTIC = {}
from b02_step776_precommit_integration import (
    AdapterCompat,
    CHECKPOINT,
    DT,
    NUM_TOL_M,
    _as_array,
    _create_env,
    _finite,
)
from b02_step776_two_substep_invariance import (
    _as_bool,
    _as_float,
    _as_int,
    _as_str,
    _find_plugin,
    _json_safe,
    _load_plugin,
    _measure,
    _native_snapshot,
)
from mcr_sim.distributed import DistributedPPO


EXPECTED_SUBSTEPS = 2
DEFAULT_MAX_RL_STEPS = 2048
DENSE_SPACING_M = 0.00001
ACCEPT_MAX_PENETRATION_M = 1e-6  # 0.001 mm
CANDIDATE_TARGET_MARGIN_M = 1e-4  # strict 0.100 mm requested margin
STEP665_VALIDATION_MODE = False
STEP665_TARGET = (665, 1)
STEP665_EXPECTED_ACTION_SHA256 = None  # report realized prefix; state match is the hard replay gate
STEP665_EXPECTED_QPREV_MM = 0.468945
STEP665_EXPECTED_QFREE_MM = -0.032075
STEP665_STATE_TOL_MM = 0.002
CANDIDATE_DENSE_GATE_M = 1e-4  # independent 0.100 mm gate
CANDIDATE_FLOAT_TOL_M = 1e-12  # pure roundoff only: 0.000000001 mm
PARENT_WRITE_ERROR_LIMIT_MM = 1e-6
MAPPED_CHANGE_EPS_MM = 1e-7

SAFETY_TERMINAL_KEYWORDS = (
    "out_of_bounds",
    "out-of-bounds",
    "outside",
    "body_out",
    "oob",
    "penetr",
)


def _sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            data = f.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def _clearance_m(measurement: dict[str, Any] | None) -> float:
    if not measurement:
        return float("nan")
    return float(measurement.get("min_clearance_m", np.nan))


def _penetration_m(clearance_m: float) -> float:
    if not np.isfinite(clearance_m):
        return float("inf")
    return float(max(0.0, -clearance_m))


def _mm(m: float | None) -> float | None:
    if m is None:
        return None
    value = float(m)
    if not np.isfinite(value):
        return value
    return value * 1000.0


def _terminal_reason(info: dict[str, Any], terminated: bool, truncated: bool) -> str | None:
    if not (terminated or truncated):
        return None
    for key in (
        "terminal_reason",
        "termination_reason",
        "done_reason",
        "reason",
    ):
        value = info.get(key)
        if value not in (None, ""):
            return str(value)
    if terminated:
        return "terminated"
    if truncated:
        return "truncated"
    return None


def _looks_like_safety_terminal(reason: str | None) -> bool:
    if not reason:
        return False
    text = reason.lower()
    return any(token in text for token in SAFETY_TERMINAL_KEYWORDS)


class _FailFastStop(Exception):
    """Stop before another physics substep after the first failed substep."""


class _TargetReached(Exception):
    """Stop the validation after the committed state of step665/substep1."""


def _solve_current_frame_robust(*, q_prev, q_free, adapter, context):
    """Use baseline strict solver on prefix; compare baseline vs multi-dense at step665."""
    q_prev = np.asarray(q_prev, dtype=np.float64)
    q_free = np.asarray(q_free, dtype=np.float64)
    if q_prev.shape != q_free.shape or q_prev.ndim != 2 or q_prev.shape[1] != 7:
        raise ValueError("invalid current-frame Beam state shape")
    if not (_finite(q_prev) and _finite(q_free)):
        raise ValueError("non-finite current-frame Beam state")

    key = (int(context["step"]), int(context["substep"]))
    if not (STEP665_VALIDATION_MODE and key == STEP665_TARGET):
        raw = strict_solver.solve_feasible_state(
            q_prev=q_prev,
            q_free=q_free,
            adapter=adapter,
            context=context,
            requested_margin_m=CANDIDATE_TARGET_MARGIN_M,
        )
        solved = _extract(raw, "strict_0p100_prefix")
        solved.metadata["requested_margin_m"] = CANDIDATE_TARGET_MARGIN_M
        return solved

    baseline_trace = RESULTS_DIR.parent / "step665_single_argmin_baseline_trace.jsonl"
    multi_trace = RESULTS_DIR.parent / "step665_multidense_trace.jsonl"
    for path in (baseline_trace, multi_trace):
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    STEP665_BASELINE_DIAGNOSTIC.clear()
    baseline_context = dict(context)
    baseline_context["solver_trace_path"] = str(baseline_trace)
    t0 = time.perf_counter()
    try:
        raw_baseline = strict_solver.solve_feasible_state(
            q_prev=q_prev,
            q_free=q_free,
            adapter=adapter,
            context=baseline_context,
            requested_margin_m=CANDIDATE_TARGET_MARGIN_M,
        )
        STEP665_BASELINE_DIAGNOSTIC.update({
            "status": "UNEXPECTED_PASS",
            "runtime_s": float(time.perf_counter() - t0),
            "solver_status": raw_baseline.get("solver_status"),
            "dense_clearance_m": raw_baseline.get(
                "accepted_independent_dense_min_clearance_m"
            ),
            "trace_path": str(baseline_trace),
        })
    except Exception as exc:
        STEP665_BASELINE_DIAGNOSTIC.update({
            "status": "EXPECTED_STAGNATION_OR_FAIL",
            "runtime_s": float(time.perf_counter() - t0),
            "error": f"{type(exc).__name__}: {exc}",
            "trace_path": str(baseline_trace),
        })

    multi_context = dict(context)
    multi_context["solver_trace_path"] = str(multi_trace)
    raw = multidense_solver.solve_feasible_state(
        q_prev=q_prev,
        q_free=q_free,
        adapter=adapter,
        context=multi_context,
        requested_margin_m=CANDIDATE_TARGET_MARGIN_M,
    )
    solved = _extract(raw, "strict_0p100_multidense_step665")
    solved.metadata["requested_margin_m"] = CANDIDATE_TARGET_MARGIN_M
    solved.metadata["step665_baseline_diagnostic"] = dict(
        STEP665_BASELINE_DIAGNOSTIC
    )
    solved.metadata["step665_multidense_trace_path"] = str(multi_trace)
    return solved


class FullEpisodeFeasibleController(Sofa.Core.Controller):
    """Online feasible planner for every physics substep of one episode."""

    def __init__(
        self,
        *,
        beam_dofs,
        adapter: AdapterCompat,
        dt: float,
        **kwargs,
    ):
        Sofa.Core.Controller.__init__(self, **kwargs)
        self.beam_dofs = beam_dofs
        self.adapter = adapter
        self.dt = float(dt)
        self.native = None

        self.current_rl_step = 0
        self.current_substep = 0
        self.enabled = True
        self._last_processed_key: tuple[int, int] | None = None
        self._record: dict[str, Any] | None = None

    def onEvent(self, event):
        name = None
        if isinstance(event, dict):
            for key in ("type", "Type", "name", "event", "className"):
                if key in event:
                    name = str(event[key])
                    break
        if name is None:
            name = type(event).__name__
        if "CollisionBegin" in name:
            self._collision_begin("onEvent:" + name)

    def onCollisionBeginEvent(self, event):
        self._collision_begin("onCollisionBeginEvent")

    def take_record(self) -> dict[str, Any] | None:
        rec = self._record
        self._record = None
        return rec

    def _collision_begin(self, source: str) -> None:
        if not self.enabled:
            return
        if self.current_rl_step <= 0:
            return
        if self.current_substep not in (1, 2):
            return

        key = (int(self.current_rl_step), int(self.current_substep))
        if self._last_processed_key == key:
            return
        self._last_processed_key = key

        rec: dict[str, Any] = {
            "rl_step": key[0],
            "substep": key[1],
            "physics_index": (key[0] - 1) * EXPECTED_SUBSTEPS + key[1],
            "event_source": source,
            "solver_required": False,
            "solver_called": False,
            "native_arm_requested": False,
            "uses_collision_dofs_as_constraint_source": False,
            "used_native_post_contact_state_as_solver_input": False,
            "rollback_used": False,
            "projection_used": False,
        }
        self._record = rec

        q_prev = _as_array(self.beam_dofs.position)
        q_free = _as_array(self.beam_dofs.free_position)

        rec["q_prev_finite"] = _finite(q_prev)
        rec["q_free_finite"] = _finite(q_free)

        if not (_finite(q_prev) and _finite(q_free)):
            rec["planning_status"] = "FAIL"
            rec["reason"] = "NON_FINITE_PRECOMMIT_STATE"
            return

        prev_measure = _measure(self.adapter, q_prev, DENSE_SPACING_M)
        free_measure = _measure(self.adapter, q_free, DENSE_SPACING_M)
        prev_clearance = _clearance_m(prev_measure)
        free_clearance = _clearance_m(free_measure)

        rec["previous_committed_clearance_m"] = prev_clearance
        rec["free_clearance_m"] = free_clearance
        rec["previous_committed_penetration_m"] = _penetration_m(prev_clearance)
        rec["free_penetration_m"] = _penetration_m(free_clearance)

        if STEP665_VALIDATION_MODE and key == STEP665_TARGET and (
            abs(prev_clearance * 1000.0 - STEP665_EXPECTED_QPREV_MM) > STEP665_STATE_TOL_MM
            or abs(free_clearance * 1000.0 - STEP665_EXPECTED_QFREE_MM) > STEP665_STATE_TOL_MM
        ):
            rec["planning_status"] = "FAIL"
            rec["reason"] = "STEP665_PROTECTED_STATE_MISMATCH"
            return

        if not (np.isfinite(prev_clearance) and np.isfinite(free_clearance)):
            rec["planning_status"] = "FAIL"
            rec["reason"] = "NON_FINITE_CLEARANCE_MEASUREMENT"
            return

        if free_clearance >= -ACCEPT_MAX_PENETRATION_M:
            if self.native is not None:
                self.native.armed.value = False
            rec["planning_status"] = "SAFE_FREE_NO_INTERVENTION"
            return

        rec["solver_required"] = True
        rec["solver_called"] = True
        started = time.perf_counter()

        context = {
            "step": key[0],
            "substep": key[1],
            "dt": self.dt,
            "beam_dofs": self.beam_dofs,
            "adapter": self.adapter,
        }

        try:
            solved = _solve_current_frame_robust(
                q_prev=q_prev,
                q_free=q_free,
                adapter=self.adapter,
                context=context,
            )
        except ValidatedSolverBridgeUnavailable as exc:
            rec["solver_runtime_s"] = float(time.perf_counter() - started)
            rec["planning_status"] = "INCONCLUSIVE"
            rec["reason"] = "VALIDATED_SOLVER_BRIDGE_UNAVAILABLE"
            rec["bridge_error"] = str(exc)
            return
        except Exception as exc:
            rec["solver_runtime_s"] = float(time.perf_counter() - started)
            rec["planning_status"] = "FAIL"
            rec["reason"] = (
                f"ONLINE_FEASIBLE_SOLVE_FAILED: {type(exc).__name__}: {exc}"
            )
            return

        rec["solver_runtime_s"] = float(time.perf_counter() - started)
        rec["solver_source"] = str(solved.source)
        rec["solver_status"] = solved.metadata.get("solver_status")
        rec["solver_iterations"] = solved.metadata.get("nonlinear_iterations")
        rec["dense_active_set_mode"] = solved.metadata.get("dense_active_set_mode")
        rec["dense_active_max"] = solved.metadata.get("dense_active_max")
        rec["step665_baseline_diagnostic"] = solved.metadata.get(
            "step665_baseline_diagnostic"
        )
        rec["step665_multidense_trace_path"] = solved.metadata.get(
            "step665_multidense_trace_path"
        )
        rec["candidate_max_translation_correction_mm"] = solved.metadata.get("max_translation_correction_mm")
        rec["candidate_rms_translation_correction_mm"] = solved.metadata.get("rms_translation_correction_mm")
        rec["candidate_max_rotation_correction_deg"] = solved.metadata.get("max_rotation_correction_deg")

        q_acc = np.asarray(solved.accepted, dtype=np.float64)
        if q_acc.shape != q_free.shape or not _finite(q_acc):
            rec["planning_status"] = "FAIL"
            rec["reason"] = "INVALID_ACCEPTED_STATE"
            return

        md = solved.metadata
        rec["uses_collision_dofs_as_constraint_source"] = bool(
            md.get("uses_collision_dofs_as_constraint_source", False)
        )
        rec["used_native_post_contact_state_as_solver_input"] = bool(
            md.get("used_native_post_contact_state_as_solver_input", False)
        )
        rec["rollback_used"] = bool(md.get("rollback_used", False))
        rec["projection_used"] = bool(md.get("projection_used", False))

        if (
            rec["uses_collision_dofs_as_constraint_source"]
            or rec["used_native_post_contact_state_as_solver_input"]
            or rec["rollback_used"]
            or rec["projection_used"]
        ):
            rec["planning_status"] = "FAIL"
            rec["reason"] = "FORBIDDEN_SOLVER_CONTAMINATION_OR_FALLBACK"
            return

        accepted = _measure(self.adapter, q_acc, DENSE_SPACING_M)
        accepted_clearance = _clearance_m(accepted)
        rec["accepted_clearance_m"] = accepted_clearance
        rec["accepted_penetration_m"] = _penetration_m(accepted_clearance)
        rec["accepted_worst_point_m"] = accepted.get("worst_point_m")
        rec["requested_candidate_margin_mm"] = float(md.get("requested_margin_m", CANDIDATE_TARGET_MARGIN_M)) * 1000.0
        rec["solver_internal_clearance_m"] = md.get("accepted_solver_min_clearance_m")

        if (
            not np.isfinite(accepted_clearance)
            or accepted_clearance + CANDIDATE_FLOAT_TOL_M
                < CANDIDATE_DENSE_GATE_M
        ):
            rec["planning_status"] = "FAIL"
            rec["reason"] = "CANDIDATE_DENSE_GATE_NOT_ACHIEVED"
            return

        if self.native is None:
            rec["planning_status"] = "INCONCLUSIVE"
            rec["reason"] = "NATIVE_HOOK_NOT_ATTACHED"
            return

        self.native.candidateFreePosition.value = q_acc.tolist()
        self.native.armed.value = True
        rec["native_arm_requested"] = True
        rec["planning_status"] = "CANDIDATE_ARMED_FOR_NATIVE_HOOK"


def _new_stats() -> dict[str, Any]:
    return {
        "physics_substeps": 0,
        "unsafe_free_count": 0,
        "safe_free_count": 0,
        "solver_call_count": 0,
        "solver_candidate_safe_count": 0,
        "candidate_certification_failure_count": 0,
        "min_solver_internal_clearance_m": float("inf"),
        "solver_failure_count": 0,
        "native_injection_count": 0,
        "native_failure_count": 0,
        "mapping_propagation_failure_count": 0,
        "velocity_correction_failure_count": 0,
        "committed_violation_count": 0,
        "non_finite_count": 0,
        "forbidden_fallback_count": 0,
        "min_previous_committed_clearance_m": float("inf"),
        "min_free_clearance_m": float("inf"),
        "min_accepted_clearance_m": float("inf"),
        "min_committed_clearance_m": float("inf"),
        "max_free_penetration_m": 0.0,
        "max_committed_penetration_m": 0.0,
        "solver_runtime_total_s": 0.0,
        "solver_runtime_max_s": 0.0,
        "solver_runtime_count": 0,
        "first_unsafe_free": None,
        "last_unsafe_free": None,
        "worst_free": None,
        "worst_committed": None,
        "first_committed_violation": None,
    }


def _update_stats(stats: dict[str, Any], rec: dict[str, Any]) -> None:
    stats["physics_substeps"] += 1

    prev_c = float(rec.get("previous_committed_clearance_m", np.nan))
    free_c = float(rec.get("free_clearance_m", np.nan))
    accepted_c = float(rec.get("accepted_clearance_m", np.nan))
    committed_c = float(rec.get("committed_clearance_m", np.nan))

    if np.isfinite(prev_c):
        stats["min_previous_committed_clearance_m"] = min(
            stats["min_previous_committed_clearance_m"], prev_c
        )
    if np.isfinite(free_c):
        stats["min_free_clearance_m"] = min(stats["min_free_clearance_m"], free_c)
        stats["max_free_penetration_m"] = max(
            stats["max_free_penetration_m"], _penetration_m(free_c)
        )
        if (
            stats["worst_free"] is None
            or free_c < float(stats["worst_free"]["clearance_m"])
        ):
            stats["worst_free"] = {
                "rl_step": rec.get("rl_step"),
                "substep": rec.get("substep"),
                "clearance_m": free_c,
                "clearance_mm": _mm(free_c),
            }
    internal_c = float(rec.get("solver_internal_clearance_m", np.nan))
    if np.isfinite(internal_c):
        stats["min_solver_internal_clearance_m"] = min(
            stats["min_solver_internal_clearance_m"], internal_c
        )
    if np.isfinite(accepted_c):
        stats["min_accepted_clearance_m"] = min(
            stats["min_accepted_clearance_m"], accepted_c
        )
    if np.isfinite(committed_c):
        stats["min_committed_clearance_m"] = min(
            stats["min_committed_clearance_m"], committed_c
        )
        stats["max_committed_penetration_m"] = max(
            stats["max_committed_penetration_m"], _penetration_m(committed_c)
        )
        if (
            stats["worst_committed"] is None
            or committed_c < float(stats["worst_committed"]["clearance_m"])
        ):
            stats["worst_committed"] = {
                "rl_step": rec.get("rl_step"),
                "substep": rec.get("substep"),
                "clearance_m": committed_c,
                "clearance_mm": _mm(committed_c),
            }

    if rec.get("solver_required"):
        stats["unsafe_free_count"] += 1
        where = {
            "rl_step": rec.get("rl_step"),
            "substep": rec.get("substep"),
            "free_clearance_mm": _mm(free_c),
        }
        if stats["first_unsafe_free"] is None:
            stats["first_unsafe_free"] = where
        stats["last_unsafe_free"] = where
    else:
        stats["safe_free_count"] += 1

    if rec.get("solver_called"):
        stats["solver_call_count"] += 1

    runtime = rec.get("solver_runtime_s")
    if runtime is not None and np.isfinite(float(runtime)):
        runtime = float(runtime)
        stats["solver_runtime_total_s"] += runtime
        stats["solver_runtime_max_s"] = max(
            stats["solver_runtime_max_s"], runtime
        )
        stats["solver_runtime_count"] += 1

    if rec.get("candidate_safe"):
        stats["solver_candidate_safe_count"] += 1

    if rec.get("substep_reason") == "CANDIDATE_DENSE_GATE_NOT_ACHIEVED":
        stats["candidate_certification_failure_count"] += 1
    elif rec.get("solver_failure"):
        stats["solver_failure_count"] += 1

    if int(rec.get("native_fire_delta", 0)) == 1:
        stats["native_injection_count"] += 1

    if rec.get("native_failure"):
        stats["native_failure_count"] += 1
    if (
        rec.get("solver_required")
        and int(rec.get("native_fire_delta", 0)) == 1
        and not rec.get("native_propagated", False)
    ):
        stats["mapping_propagation_failure_count"] += 1
    if (
        rec.get("solver_required")
        and int(rec.get("native_fire_delta", 0)) == 1
        and not rec.get("native_velocity_corrected", False)
    ):
        stats["velocity_correction_failure_count"] += 1

    if rec.get("committed_violation"):
        stats["committed_violation_count"] += 1
        if stats["first_committed_violation"] is None:
            stats["first_committed_violation"] = {
                "rl_step": rec.get("rl_step"),
                "substep": rec.get("substep"),
                "committed_clearance_mm": _mm(committed_c),
            }

    if rec.get("non_finite"):
        stats["non_finite_count"] += 1

    if (
        rec.get("uses_collision_dofs_as_constraint_source")
        or rec.get("used_native_post_contact_state_as_solver_input")
        or rec.get("rollback_used")
        or rec.get("projection_used")
    ):
        stats["forbidden_fallback_count"] += 1


def _finalize_stats(stats: dict[str, Any]) -> dict[str, Any]:
    out = dict(stats)
    for key in (
        "min_previous_committed_clearance_m",
        "min_solver_internal_clearance_m",
        "min_free_clearance_m",
        "min_accepted_clearance_m",
        "min_committed_clearance_m",
    ):
        value = float(out[key])
        if not np.isfinite(value):
            out[key] = None

    n = int(out["solver_runtime_count"])
    out["solver_runtime_mean_s"] = (
        float(out["solver_runtime_total_s"]) / n if n else None
    )

    out["min_previous_committed_clearance_mm"] = _mm(
        out["min_previous_committed_clearance_m"]
    )
    out["min_free_clearance_mm"] = _mm(out["min_free_clearance_m"])
    out["min_solver_internal_clearance_mm"] = _mm(
        out["min_solver_internal_clearance_m"]
    )
    out["min_accepted_clearance_mm"] = _mm(out["min_accepted_clearance_m"])
    out["min_committed_clearance_mm"] = _mm(out["min_committed_clearance_m"])
    out["max_free_penetration_mm"] = _mm(out["max_free_penetration_m"])
    out["max_committed_penetration_mm"] = _mm(
        out["max_committed_penetration_m"]
    )
    return out


def _write_summary(
    *,
    output: Path,
    report_path: Path,
    payload: dict[str, Any],
) -> None:
    safe = _json_safe(payload)
    output.write_text(json.dumps(safe, indent=2, sort_keys=True) + "\n")

    stats = safe.get("stats", {})
    lines = [
        "B02 STRICT 0.100MM MARGIN SOLVER ACCEPTANCE",
        "==================================================",
        "",
        f"Decision: {safe.get('decision')}",
        f"Reason: {safe.get('reason')}",
        f"Checkpoint SHA256: {safe.get('checkpoint_sha256')}",
        f"Seed: {safe.get('seed')}",
        f"RL steps executed: {safe.get('rl_steps_executed')}",
        f"Physics substeps observed: {stats.get('physics_substeps')}",
        f"Terminal reason: {safe.get('terminal_reason')}",
        "",
        f"Unsafe free states: {stats.get('unsafe_free_count')}",
        f"Solver calls: {stats.get('solver_call_count')}",
        f"Certified candidates (independent dense >= 0.100 mm): {stats.get('solver_candidate_safe_count')}",
        f"Candidate certification failures: {stats.get('candidate_certification_failure_count')}",
        f"Solver failures: {stats.get('solver_failure_count')}",
        f"Native injections: {stats.get('native_injection_count')}",
        f"Native failures: {stats.get('native_failure_count')}",
        f"Mapping propagation failures: {stats.get('mapping_propagation_failure_count')}",
        f"Velocity correction failures: {stats.get('velocity_correction_failure_count')}",
        f"Final native fireCount: {safe.get('native_fire_count_final')}",
        f"Forbidden fallback count: {stats.get('forbidden_fallback_count')}",
        f"NaN/Inf count: {stats.get('non_finite_count')}",
        "",
        f"Min free clearance: {stats.get('min_free_clearance_mm')} mm",
        f"Max free penetration: {stats.get('max_free_penetration_mm')} mm",
        f"Worst free state: {stats.get('worst_free')}",
        f"Candidate target margin: {CANDIDATE_TARGET_MARGIN_M * 1000.0} mm",
        f"Independent dense candidate gate: {CANDIDATE_DENSE_GATE_M * 1000.0} mm (roundoff tolerance {CANDIDATE_FLOAT_TOL_M * 1000.0} mm)",
        f"Minimum solver-internal clearance: {stats.get('min_solver_internal_clearance_mm')} mm",
        f"Min accepted clearance: {stats.get('min_accepted_clearance_mm')} mm",
        f"Min committed clearance: {stats.get('min_committed_clearance_mm')} mm",
        f"Max committed penetration: {stats.get('max_committed_penetration_mm')} mm",
        f"Worst committed state: {stats.get('worst_committed')}",
        f"Acceptance limit: {ACCEPT_MAX_PENETRATION_M * 1000.0} mm",
        "",
        f"Solver total runtime: {stats.get('solver_runtime_total_s')} s",
        f"Solver mean runtime: {stats.get('solver_runtime_mean_s')} s",
        f"Solver max runtime: {stats.get('solver_runtime_max_s')} s",
        "",
        "PASS here is an engineering acceptance for this fixed worst-case episode.",
        "It is not a mathematical proof of global nonpenetration.",
        "",
    ]
    report_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(CHECKPOINT))
    parser.add_argument("--seed", type=int, default=15204)
    parser.add_argument("--max-rl-steps", type=int, default=DEFAULT_MAX_RL_STEPS)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--step665-validation", action="store_true")
    parser.add_argument("--build-dir", default=str(DEFAULT_BUILD_DIR))
    parser.add_argument("--plugin-lib", default=None)
    parser.add_argument(
        "--output",
        default=str(RESULTS_DIR / "step665_multidense_result.json"),
    )
    args = parser.parse_args()
    global STEP665_VALIDATION_MODE
    STEP665_VALIDATION_MODE = bool(args.step665_validation)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path = output.with_name(output.stem + "_report.md")
    trace_path = output.with_name(output.stem + "_trace.jsonl")

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_sha = _sha256_file(checkpoint)

    plugin = (
        Path(args.plugin_lib).expanduser().resolve()
        if args.plugin_lib
        else _find_plugin(Path(args.build_dir).expanduser().resolve())
    )

    base_payload: dict[str, Any] = {
        "test": "B02 step665/substep1 single-argmin vs multi-dense active-set diagnosis",
        "step665_validation_mode": bool(args.step665_validation),
        "seed": int(args.seed),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "max_rl_steps": int(args.max_rl_steps),
        "physics_dt_s": float(DT),
        "expected_physics_substeps": EXPECTED_SUBSTEPS,
        "dense_spacing_m": DENSE_SPACING_M,
        "accept_max_penetration_m": ACCEPT_MAX_PENETRATION_M,
        "accept_max_penetration_mm": ACCEPT_MAX_PENETRATION_M * 1000.0,
        "candidate_target_margin_m": CANDIDATE_TARGET_MARGIN_M,
        "candidate_target_margin_mm": CANDIDATE_TARGET_MARGIN_M * 1000.0,
        "candidate_dense_gate_m": CANDIDATE_DENSE_GATE_M,
        "candidate_dense_gate_mm": CANDIDATE_DENSE_GATE_M * 1000.0,
        "candidate_float_tolerance_m": CANDIDATE_FLOAT_TOL_M,
        "training_started": False,
        "production_files_modified": False,
        "collision_dofs_constraint_source": False,
        "native_post_contact_solver_input": False,
        "rollback_used": False,
        "projection_used": False,
        "action_shielding_used": False,
        "direct_committed_position_repair_used": False,
    }

    if plugin is None:
        payload = {
            **base_payload,
            "decision": "INCONCLUSIVE",
            "reason": "NATIVE_PLUGIN_NOT_BUILT",
        }
        _write_summary(output=output, report_path=report_path, payload=payload)
        print(json.dumps(payload, sort_keys=True))
        return

    base_payload["plugin"] = str(plugin)

    try:
        _plugin_handle = _load_plugin(plugin)
    except Exception as exc:
        payload = {
            **base_payload,
            "decision": "INCONCLUSIVE",
            "reason": (
                f"NATIVE_PLUGIN_LOAD_FAILED: {type(exc).__name__}: {exc}"
            ),
        }
        _write_summary(output=output, report_path=report_path, payload=payload)
        print(json.dumps(payload, sort_keys=True))
        return

    env = _create_env()
    observation, reset_info = env.reset(seed=int(args.seed))

    if int(env.physics_substeps) != EXPECTED_SUBSTEPS:
        payload = {
            **base_payload,
            "configured_physics_substeps": int(env.physics_substeps),
            "decision": "FAIL",
            "reason": "PHYSICS_SUBSTEP_CONFIGURATION_MISMATCH",
        }
        _write_summary(output=output, report_path=report_path, payload=payload)
        env.close()
        print(json.dumps(payload, sort_keys=True))
        return

    base_payload["configured_physics_substeps"] = int(env.physics_substeps)
    base_payload["reset_info"] = _json_safe(reset_info)

    controller = env.mcr_controller_sofa
    instrument = controller.instrument.InstrumentCombined
    beam_dofs = instrument.getObject("DOFs")
    collision_dofs = instrument.getChild("mcr_collis").getObject(
        "CollisionDOFs"
    )
    adapter = AdapterCompat(env, instrument)

    initial_q = _as_array(beam_dofs.position)
    initial_measure = _measure(adapter, initial_q, DENSE_SPACING_M)
    initial_clearance = _clearance_m(initial_measure)
    base_payload["initial_committed_clearance_m"] = initial_clearance
    base_payload["initial_committed_clearance_mm"] = _mm(initial_clearance)

    if (
        not _finite(initial_q)
        or not np.isfinite(initial_clearance)
        or initial_clearance < -ACCEPT_MAX_PENETRATION_M
    ):
        payload = {
            **base_payload,
            "decision": "FAIL",
            "reason": "RESET_STATE_ALREADY_OUTSIDE_ACCEPTED_FEASIBLE_DOMAIN",
        }
        _write_summary(output=output, report_path=report_path, payload=payload)
        env.close()
        print(json.dumps(payload, sort_keys=True))
        return

    planner = FullEpisodeFeasibleController(
        name="B02FullEpisodeFeasiblePlanner",
        beam_dofs=beam_dofs,
        adapter=adapter,
        dt=DT,
    )
    instrument.addObject(planner)

    native = instrument.addObject(
        "BeamFeasibleNativePrecommitHook",
        name="BeamFeasibleNativePrecommitHookFullEpisode",
        beamState="@DOFs",
        collisionState="@mcr_collis/CollisionDOFs",
        dt=float(DT),
        armed=False,
    )
    planner.native = native

    model = DistributedPPO.load(str(checkpoint), device="cpu")
    model.policy.set_training_mode(False)

    original_animate = env.sofa_simulation.animate
    current_step = 0
    substep_in_current_step = 0
    fatal_reason: str | None = None
    fatal_kind: str | None = None
    first_failure: dict[str, Any] | None = None
    target_record: dict[str, Any] | None = None
    target_completed = False
    partial_last_rl_step = False
    stats = _new_stats()
    action_hasher = hashlib.sha256()
    terminal_reason: str | None = None
    terminal_info: dict[str, Any] | None = None
    terminated_flag = False
    truncated_flag = False
    rl_steps_executed = 0
    started = time.perf_counter()

    trace_file = trace_path.open("w", buffering=1)

    def record_trace(rec: dict[str, Any]) -> None:
        trace_file.write(
            json.dumps(_json_safe(rec), sort_keys=True) + "\n"
        )
        trace_file.flush()

    def traced_animate(root, dt):
        nonlocal substep_in_current_step, fatal_reason, fatal_kind, first_failure, target_record

        substep_in_current_step += 1
        sub = int(substep_in_current_step)
        planner.current_rl_step = int(current_step)
        planner.current_substep = sub

        fire_before = _as_int(native.fireCount)
        result = original_animate(root, dt)
        fire_after = _as_int(native.fireCount)

        rec = planner.take_record() or {
            "rl_step": int(current_step),
            "substep": sub,
            "physics_index": (int(current_step) - 1) * EXPECTED_SUBSTEPS + sub,
            "planning_status": "INCONCLUSIVE",
            "reason": "COLLISION_BEGIN_EVENT_NOT_OBSERVED",
            "solver_required": False,
            "solver_called": False,
            "uses_collision_dofs_as_constraint_source": False,
            "used_native_post_contact_state_as_solver_input": False,
            "rollback_used": False,
            "projection_used": False,
        }

        rec["native_fire_before"] = int(fire_before)
        rec["native_fire_after"] = int(fire_after)
        rec["native_fire_delta"] = int(fire_after - fire_before)

        q_committed = _as_array(beam_dofs.position)
        committed_finite = _finite(q_committed)
        rec["committed_finite"] = bool(committed_finite)

        if committed_finite:
            committed_measure = _measure(
                adapter, q_committed, DENSE_SPACING_M
            )
            committed_clearance = _clearance_m(committed_measure)
            rec["committed_worst_point_m"] = committed_measure.get("worst_point_m")
        else:
            committed_clearance = float("nan")
            rec["committed_worst_point_m"] = None

        rec["committed_clearance_m"] = committed_clearance
        rec["committed_penetration_m"] = _penetration_m(
            committed_clearance
        )
        accepted_clearance = float(rec.get("accepted_clearance_m", np.nan))
        rec["candidate_to_committed_clearance_loss_mm"] = (
            (accepted_clearance - committed_clearance) * 1000.0
            if int(rec["native_fire_delta"]) == 1
            and np.isfinite(accepted_clearance)
            and np.isfinite(committed_clearance)
            else None
        )

        collision_free = _as_array(collision_dofs.free_position)
        rec["collision_free_finite_diagnostic"] = bool(
            _finite(collision_free)
        )

        if rec["native_fire_delta"] > 0:
            snap = _native_snapshot(native)
            rec["native_status"] = snap["status"]
            rec["native_propagated"] = snap["propagated"]
            rec["native_velocity_corrected"] = snap[
                "velocity_corrected"
            ]
            rec["mapped_child_max_change_mm"] = snap[
                "mapped_child_max_change_mm"
            ]
            rec["parent_write_max_error_mm"] = snap[
                "parent_write_max_error_mm"
            ]
        else:
            rec["native_status"] = "NOT_USED"
            rec["native_propagated"] = False
            rec["native_velocity_corrected"] = False
            rec["mapped_child_max_change_mm"] = None
            rec["parent_write_max_error_mm"] = None

        planning = str(rec.get("planning_status", "MISSING"))
        solver_required = bool(rec.get("solver_required", False))

        rec["candidate_safe"] = bool(
            solver_required
            and planning == "CANDIDATE_ARMED_FOR_NATIVE_HOOK"
            and np.isfinite(float(rec.get("accepted_clearance_m", np.nan)))
            and float(rec.get("accepted_clearance_m")) + CANDIDATE_FLOAT_TOL_M
                >= CANDIDATE_DENSE_GATE_M
        )
        rec["solver_failure"] = False
        rec["native_failure"] = False
        rec["committed_violation"] = False
        rec["non_finite"] = False
        rec["substep_status"] = "PASS"
        rec["substep_reason"] = "SAFE_FREE_NO_INTERVENTION"

        forbidden = (
            bool(rec.get("uses_collision_dofs_as_constraint_source"))
            or bool(rec.get("used_native_post_contact_state_as_solver_input"))
            or bool(rec.get("rollback_used"))
            or bool(rec.get("projection_used"))
        )

        if planning == "INCONCLUSIVE":
            rec["substep_status"] = "INCONCLUSIVE"
            rec["substep_reason"] = str(
                rec.get("reason", "PLANNER_INCONCLUSIVE")
            )
            rec["solver_failure"] = bool(solver_required)
        elif planning == "FAIL":
            rec["substep_status"] = "FAIL"
            rec["substep_reason"] = str(
                rec.get("reason", "PLANNER_FAILED")
            )
            rec["solver_failure"] = bool(solver_required)
        elif forbidden:
            rec["substep_status"] = "FAIL"
            rec["substep_reason"] = "FORBIDDEN_FALLBACK_OR_CONSTRAINT_SOURCE"
            rec["solver_failure"] = bool(solver_required)
        elif solver_required:
            rec["substep_reason"] = "UNSAFE_FREE_CORRECTED"
            if planning != "CANDIDATE_ARMED_FOR_NATIVE_HOOK":
                rec["substep_status"] = "FAIL"
                rec["substep_reason"] = "SOLVER_DID_NOT_ARM_SAFE_CANDIDATE"
                rec["solver_failure"] = True
            elif rec["native_fire_delta"] != 1:
                rec["substep_status"] = "FAIL"
                rec["substep_reason"] = "NATIVE_HOOK_DID_NOT_FIRE_EXACTLY_ONCE"
                rec["native_failure"] = True
            elif rec.get("native_status") != "PASS_NATIVE_WRITE_AND_PROPAGATE":
                rec["substep_status"] = "FAIL"
                rec["substep_reason"] = "NATIVE_HOOK_STATUS_FAILED"
                rec["native_failure"] = True
            elif not rec.get("native_propagated", False):
                rec["substep_status"] = "FAIL"
                rec["substep_reason"] = "NATIVE_MAPPING_PROPAGATION_NOT_RUN"
                rec["native_failure"] = True
            elif not rec.get("native_velocity_corrected", False):
                rec["substep_status"] = "FAIL"
                rec["substep_reason"] = "FREE_VELOCITY_CORRECTION_NOT_RUN"
                rec["native_failure"] = True
            elif float(
                rec.get("parent_write_max_error_mm", np.inf)
            ) > PARENT_WRITE_ERROR_LIMIT_MM:
                rec["substep_status"] = "FAIL"
                rec["substep_reason"] = "PARENT_CANDIDATE_WRITE_MISMATCH"
                rec["native_failure"] = True
            elif float(
                rec.get("mapped_child_max_change_mm", 0.0)
            ) <= MAPPED_CHANGE_EPS_MM:
                rec["substep_status"] = "FAIL"
                rec["substep_reason"] = "MAPPED_COLLISION_FREE_STATE_DID_NOT_CHANGE"
                rec["native_failure"] = True
            elif not rec["candidate_safe"]:
                rec["substep_status"] = "FAIL"
                rec["substep_reason"] = "ACCEPTED_CANDIDATE_NOT_SAFE"
                rec["solver_failure"] = True
        else:
            if planning != "SAFE_FREE_NO_INTERVENTION":
                rec["substep_status"] = "FAIL"
                rec["substep_reason"] = "SAFE_FREE_CLASSIFICATION_INCONSISTENT"
            elif rec["native_fire_delta"] != 0:
                rec["substep_status"] = "FAIL"
                rec["substep_reason"] = "UNEXPECTED_NATIVE_INTERVENTION"
                rec["native_failure"] = True

        if (
            not committed_finite
            or not np.isfinite(committed_clearance)
            or not rec["collision_free_finite_diagnostic"]
        ):
            rec["substep_status"] = "FAIL"
            rec["substep_reason"] = "NON_FINITE_POST_SUBSTEP_STATE"
            rec["non_finite"] = True

        if (
            np.isfinite(committed_clearance)
            and committed_clearance < -ACCEPT_MAX_PENETRATION_M
        ):
            rec["committed_violation"] = True
            if rec["substep_status"] == "PASS":
                rec["substep_status"] = "FAIL"
                rec["substep_reason"] = "COMMITTED_DENSE_CLEARANCE_VIOLATION"

        _update_stats(stats, rec)
        record_trace(rec)
        if args.step665_validation and (int(current_step), sub) == STEP665_TARGET:
            target_record = dict(rec)

        if rec["substep_status"] != "PASS" and fatal_reason is None:
            fatal_kind = rec["substep_status"]
            fatal_reason = (
                f"STEP_{current_step}_SUBSTEP_{sub}_"
                f"{rec['substep_reason']}"
            )
            first_failure = dict(rec)
            raise _FailFastStop(fatal_reason)

        if args.step665_validation and (int(current_step), sub) == STEP665_TARGET:
            raise _TargetReached()

        return result

    env.sofa_simulation.animate = traced_animate

    try:
        for step in range(1, int(args.max_rl_steps) + 1):
            current_step = int(step)
            substep_in_current_step = 0

            raw_action, _ = model.predict(observation, deterministic=True)
            raw_action = np.asarray(raw_action, dtype=np.float32).reshape(3)
            action_hasher.update(
                np.asarray(raw_action, dtype="<f4").tobytes(order="C")
            )

            try:
                observation, _, terminated, truncated, info = env.step(
                    raw_action
                )
            except _FailFastStop:
                rl_steps_executed = step
                partial_last_rl_step = substep_in_current_step < EXPECTED_SUBSTEPS
                break
            except _TargetReached:
                rl_steps_executed = step
                partial_last_rl_step = True
                target_completed = True
                break
            except Exception as exc:
                if fatal_reason is None:
                    fatal_kind = "FAIL"
                    fatal_reason = (
                        f"ENV_STEP_EXCEPTION_STEP_{step}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                break

            rl_steps_executed = step

            if substep_in_current_step != EXPECTED_SUBSTEPS:
                if fatal_reason is None:
                    fatal_kind = "FAIL"
                    fatal_reason = (
                        f"STEP_{step}_PHYSICS_SUBSTEP_COUNT_"
                        f"{substep_in_current_step}_EXPECTED_{EXPECTED_SUBSTEPS}"
                    )

            if args.progress_every > 0 and (
                step == 1
                or step % int(args.progress_every) == 0
                or fatal_reason is not None
                or terminated
                or truncated
            ):
                current_stats = _finalize_stats(stats)
                print(
                    "[FULL_EPISODE_SAFETY] "
                    f"step={step}/{args.max_rl_steps} "
                    f"substeps={current_stats['physics_substeps']} "
                    f"unsafe_free={current_stats['unsafe_free_count']} "
                    f"solver_fail={current_stats['solver_failure_count']} "
                    f"native_fail={current_stats['native_failure_count']} "
                    f"max_committed_pen_mm="
                    f"{current_stats['max_committed_penetration_mm']:.6f} "
                    f"fatal={fatal_reason}",
                    flush=True,
                )

            if fatal_reason is not None:
                break

            if terminated or truncated:
                terminated_flag = bool(terminated)
                truncated_flag = bool(truncated)
                terminal_info = _json_safe(info)
                terminal_reason = _terminal_reason(
                    info, bool(terminated), bool(truncated)
                )
                break

    finally:
        planner.enabled = False
        env.sofa_simulation.animate = original_animate
        trace_file.close()

    elapsed = float(time.perf_counter() - started)
    final_stats = _finalize_stats(stats)

    if fatal_reason is not None:
        decision = (
            "INCONCLUSIVE"
            if fatal_kind == "INCONCLUSIVE"
            else "FAIL"
        )
        reason = fatal_reason
    elif args.step665_validation:
        if not target_completed:
            decision = "INCONCLUSIVE"
            reason = "STEP665_TARGET_NOT_COMPLETED"
        else:
            decision = "PASS"
            reason = "STEP665_MULTIDENSE_STRICT_MARGIN_AND_NATIVE_COMMIT_PASS"
    elif final_stats["non_finite_count"] != 0:
        decision = "FAIL"
        reason = "NON_FINITE_STATE_OBSERVED"
    elif final_stats["forbidden_fallback_count"] != 0:
        decision = "FAIL"
        reason = "FORBIDDEN_FALLBACK_OR_CONSTRAINT_SOURCE_OBSERVED"
    elif final_stats["candidate_certification_failure_count"] != 0:
        decision = "FAIL"
        reason = "CANDIDATE_DENSE_GATE_FAILURE_OBSERVED"
    elif final_stats["solver_failure_count"] != 0:
        decision = "FAIL"
        reason = "SOLVER_FAILURE_OBSERVED"
    elif final_stats["native_failure_count"] != 0:
        decision = "FAIL"
        reason = "NATIVE_INJECTION_FAILURE_OBSERVED"
    elif final_stats["committed_violation_count"] != 0:
        decision = "FAIL"
        reason = "COMMITTED_DENSE_CLEARANCE_VIOLATION_OBSERVED"
    elif (
        final_stats["max_committed_penetration_m"]
        > ACCEPT_MAX_PENETRATION_M
    ):
        decision = "FAIL"
        reason = "MAX_COMMITTED_PENETRATION_EXCEEDS_ACCEPTANCE_LIMIT"
    elif final_stats["solver_call_count"] == 0:
        decision = "INCONCLUSIVE"
        reason = "EPISODE_NEVER_CHALLENGED_FEASIBLE_SOLVER"
    elif _looks_like_safety_terminal(terminal_reason):
        decision = "FAIL"
        reason = "ENVIRONMENT_REPORTED_SAFETY_RELATED_TERMINATION"
    elif (
        not (terminated_flag or truncated_flag)
        and rl_steps_executed < int(args.max_rl_steps)
    ):
        decision = "INCONCLUSIVE"
        reason = "EPISODE_DID_NOT_REACH_TERMINATION_OR_MAX_HORIZON"
    else:
        decision = "PASS"
        reason = "FULL_EPISODE_SAFETY_ACCEPTANCE_HELD"

    payload = {
        **base_payload,
        "action_stream_sha256": action_hasher.hexdigest(),
        "rl_steps_executed": int(rl_steps_executed),
        "partial_last_rl_step": bool(partial_last_rl_step),
        "first_failure": first_failure,
        "target_record": target_record,
        "target_completed": bool(target_completed),
        "target_expected_action_sha256": None,
        "terminated": bool(terminated_flag),
        "truncated": bool(truncated_flag),
        "terminal_reason": terminal_reason,
        "terminal_info": terminal_info,
        "stats": final_stats,
        "native_fire_count_final": _as_int(native.fireCount),
        "trace_file": str(trace_path),
        "wall_s": elapsed,
        "decision": decision,
        "reason": reason,
        "engineering_interpretation": (
            "PASS validates this fixed worst-case episode end-to-end; "
            "it does not mathematically prove all reachable states."
        ),
    }

    _write_summary(
        output=output,
        report_path=report_path,
        payload=payload,
    )

    print(
        json.dumps(
            {
                "decision": decision,
                "reason": reason,
                "rl_steps_executed": rl_steps_executed,
                "terminal_reason": terminal_reason,
                "physics_substeps": final_stats["physics_substeps"],
                "unsafe_free_count": final_stats["unsafe_free_count"],
                "solver_call_count": final_stats["solver_call_count"],
                "solver_failure_count": final_stats["solver_failure_count"],
                "native_injection_count": final_stats[
                    "native_injection_count"
                ],
                "native_failure_count": final_stats["native_failure_count"],
                "min_free_clearance_mm": final_stats[
                    "min_free_clearance_mm"
                ],
                "min_committed_clearance_mm": final_stats[
                    "min_committed_clearance_mm"
                ],
                "max_committed_penetration_mm": final_stats[
                    "max_committed_penetration_mm"
                ],
                "solver_runtime_total_s": final_stats[
                    "solver_runtime_total_s"
                ],
                "solver_runtime_mean_s": final_stats[
                    "solver_runtime_mean_s"
                ],
                "solver_runtime_max_s": final_stats[
                    "solver_runtime_max_s"
                ],
                "output": str(output),
                "report": str(report_path),
                "trace": str(trace_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
