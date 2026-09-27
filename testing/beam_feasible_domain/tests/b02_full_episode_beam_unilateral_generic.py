#!/usr/bin/env python3
"""2048-step B02 full-episode safety acceptance for Beam unilateral + Generic.

Safety route under test, from physics substep 1 onward:

    real Beam Rigid3 free state
      -> production BeamAdapter 10 um dense geometry + B02 SDF
      -> linearized Beam-level unilateral rows, g = clearance - 0.100 mm >= 0
      -> GenericConstraintSolver
      -> committed Beam state
      -> independent 10 um dense committed check

No SLSQP q_candidate solver, native candidate injection, direct free-position
write, committed-position projection, rollback, or action shielding is used.

This is an engineering acceptance for one fixed B02/target04/seed15204 episode,
not a mathematical proof for all reachable states.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

THIS = Path(__file__).resolve()
TEST_DIR = THIS.parent
BEAM_ROOT = THIS.parents[1]
PYTHON_ROOT = THIS.parents[3]
UNILATERAL_DIR = BEAM_ROOT / "beam_unilateral_lcp"

for p in (TEST_DIR, UNILATERAL_DIR, PYTHON_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from b02_step776_precommit_integration import (
    AdapterCompat,
    CHECKPOINT,
    DT,
    _as_array,
    _create_env,
    _finite,
)
from beam_linearized_unilateral import (
    DENSE_SPACING_M,
    MARGIN_M,
    find_plugin,
    load_plugin,
)
from beam_unilateral_full_episode import (
    FullEpisodeBeamUnilateralController,
)
from mcr_sim.distributed import DistributedPPO


DEFAULT_MAX_RL_STEPS = 2048
EXPECTED_SUBSTEPS = 2
COMMITTED_PENETRATION_LIMIT_M = 1.0e-6
FLOAT_TOL_M = 1.0e-12

RUNTIME = BEAM_ROOT / "_runtime" / "beam_unilateral_generic_full_episode"
DEFAULT_BUILD_DIRS = (
    RUNTIME / "native_build",
    BEAM_ROOT / "_runtime" / "beam_unilateral_lcp_step665" / "native_build",
)

SAFETY_TERMINAL_TOKENS = (
    "out_of_bounds",
    "out-of-bounds",
    "out of bounds",
    "outside",
    "penetr",
    "collision_failure",
    "safety",
)


class _FailFastStop(Exception):
    pass


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def _find_default_plugin() -> Path | None:
    for build_dir in DEFAULT_BUILD_DIRS:
        plugin = find_plugin(build_dir)
        if plugin is not None:
            return plugin
    return None


def _terminal_reason(info, terminated: bool, truncated: bool) -> str | None:
    if not (terminated or truncated):
        return None
    if isinstance(info, dict):
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
    return any(token in text for token in SAFETY_TERMINAL_TOKENS)


def _new_stats() -> dict[str, Any]:
    return {
        "physics_substeps": 0,
        "safe_free_substeps": 0,
        "row_build_substeps": 0,
        "penetrating_free_substeps": 0,
        "total_rows_built": 0,
        "max_rows_in_substep": 0,
        "row_build_runtime_total_s": 0.0,
        "row_build_runtime_max_s": 0.0,
        "min_free_clearance_m": float("inf"),
        "min_committed_clearance_m": float("inf"),
        "max_committed_penetration_m": 0.0,
        "committed_violation_count": 0,
        "committed_below_margin_count": 0,
        "non_finite_count": 0,
        "constraint_active_mismatch_count": 0,
        "planning_failure_count": 0,
    }


def _finalize_stats(stats: dict[str, Any]) -> dict[str, Any]:
    out = dict(stats)
    calls = int(out["row_build_substeps"])
    out["row_build_runtime_mean_s"] = (
        float(out["row_build_runtime_total_s"]) / calls
        if calls
        else 0.0
    )
    for key in ("min_free_clearance_m", "min_committed_clearance_m"):
        if not np.isfinite(float(out[key])):
            out[key] = None
    out["min_free_clearance_mm"] = (
        None
        if out["min_free_clearance_m"] is None
        else float(out["min_free_clearance_m"]) * 1000.0
    )
    out["min_committed_clearance_mm"] = (
        None
        if out["min_committed_clearance_m"] is None
        else float(out["min_committed_clearance_m"]) * 1000.0
    )
    out["max_committed_penetration_mm"] = (
        float(out["max_committed_penetration_m"]) * 1000.0
    )
    return out


def _update_stats(stats: dict[str, Any], rec: dict[str, Any]) -> None:
    stats["physics_substeps"] += 1

    free_clearance = float(rec.get("q_free_clearance_m", np.nan))
    committed_clearance = float(rec.get("committed_clearance_m", np.nan))
    if np.isfinite(free_clearance):
        stats["min_free_clearance_m"] = min(
            float(stats["min_free_clearance_m"]), free_clearance
        )
        if free_clearance < -COMMITTED_PENETRATION_LIMIT_M:
            stats["penetrating_free_substeps"] += 1

    if bool(rec.get("rows_required", False)):
        stats["row_build_substeps"] += 1
        rows = int(rec.get("row_count", 0))
        stats["total_rows_built"] += rows
        stats["max_rows_in_substep"] = max(
            int(stats["max_rows_in_substep"]), rows
        )
        runtime = float(rec.get("row_build_runtime_s", 0.0))
        stats["row_build_runtime_total_s"] += runtime
        stats["row_build_runtime_max_s"] = max(
            float(stats["row_build_runtime_max_s"]), runtime
        )
    else:
        stats["safe_free_substeps"] += 1

    if np.isfinite(committed_clearance):
        stats["min_committed_clearance_m"] = min(
            float(stats["min_committed_clearance_m"]),
            committed_clearance,
        )
        penetration = max(0.0, -committed_clearance)
        stats["max_committed_penetration_m"] = max(
            float(stats["max_committed_penetration_m"]), penetration
        )
        if committed_clearance < -COMMITTED_PENETRATION_LIMIT_M:
            stats["committed_violation_count"] += 1
        if committed_clearance + FLOAT_TOL_M < MARGIN_M:
            stats["committed_below_margin_count"] += 1

    if bool(rec.get("non_finite", False)):
        stats["non_finite_count"] += 1
    if bool(rec.get("constraint_active_mismatch", False)):
        stats["constraint_active_mismatch_count"] += 1
    if str(rec.get("planning_status")) == "FAIL":
        stats["planning_failure_count"] += 1


def _write_report(payload: dict[str, Any], report: Path) -> None:
    stats = payload["stats"]
    first = payload.get("first_failure")
    lines = [
        "B02 BEAM-LEVEL SDF UNILATERAL + GENERIC FULL-EPISODE ACCEPTANCE",
        "================================================================",
        "",
        f"FINAL: {payload['decision']}",
        f"Reason: {payload['reason']}",
        "",
        "Configuration",
        "-------------",
        f"Seed: {payload['seed']}",
        f"Max RL steps: {payload['max_rl_steps']}",
        f"Physics: {payload['expected_physics_substeps']} x {payload['physics_dt_s']} s",
        f"Constraint solver: {payload['constraint_solver']}",
        f"Requested unilateral margin: {payload['requested_margin_mm']} mm",
        f"Dense Beam spacing: {payload['dense_spacing_mm']} mm",
        f"Committed penetration limit: {payload['committed_penetration_limit_mm']} mm",
        f"Production modified: {payload['production_files_modified']}",
        f"Training started: {payload['training_started']}",
        f"Target q_candidate SLSQP used: {payload['q_candidate_solver_used']}",
        "",
        "Run",
        "---",
        f"RL steps executed: {payload['rl_steps_executed']}",
        f"Physics substeps: {stats['physics_substeps']}",
        f"Terminated: {payload['terminated']}",
        f"Truncated: {payload['truncated']}",
        f"Terminal reason: {payload['terminal_reason']}",
        "",
        "Unilateral rows",
        "----------------",
        f"Safe-free substeps: {stats['safe_free_substeps']}",
        f"Row-build substeps: {stats['row_build_substeps']}",
        f"Penetrating free substeps: {stats['penetrating_free_substeps']}",
        f"Total rows built: {stats['total_rows_built']}",
        f"Max rows/substep: {stats['max_rows_in_substep']}",
        f"Row build runtime total/mean/max: "
        f"{stats['row_build_runtime_total_s']} / "
        f"{stats['row_build_runtime_mean_s']} / "
        f"{stats['row_build_runtime_max_s']} s",
        f"Constraint active mismatches: {stats['constraint_active_mismatch_count']}",
        "",
        "Clearance",
        "---------",
        f"Worst free clearance: {stats['min_free_clearance_mm']} mm",
        f"Worst committed clearance: {stats['min_committed_clearance_mm']} mm",
        f"Max committed penetration: {stats['max_committed_penetration_mm']} mm",
        f"Committed violations: {stats['committed_violation_count']}",
        f"Committed states below +0.100 mm margin: {stats['committed_below_margin_count']}",
        f"Non-finite substeps: {stats['non_finite_count']}",
        f"Planning failures: {stats['planning_failure_count']}",
        "",
        f"First failure: {first}",
        "",
        "Interpretation",
        "--------------",
        "PASS means this one fixed B02/target04/seed15204 episode completed or "
        "ended normally without any committed Beam penetration beyond 0.001 mm.",
        "It does not mathematically prove global nonpenetration for every future "
        "policy, vessel, action, or state.",
        "",
    ]
    report.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(CHECKPOINT))
    parser.add_argument("--seed", type=int, default=15204)
    parser.add_argument("--max-rl-steps", type=int, default=DEFAULT_MAX_RL_STEPS)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--plugin-lib", default=None)
    parser.add_argument(
        "--output",
        default=str(
            RUNTIME
            / "results"
            / "b02_beam_unilateral_generic_full_episode.json"
        ),
    )
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path = output.with_name(
        "b02_beam_unilateral_generic_full_episode_report.md"
    )
    trace_path = output.with_name(
        "b02_beam_unilateral_generic_full_episode_trace.jsonl"
    )

    plugin = (
        Path(args.plugin_lib).expanduser().resolve()
        if args.plugin_lib
        else _find_default_plugin()
    )
    if plugin is None or not plugin.is_file():
        raise RuntimeError(
            "libMCRBeamLinearizedUnilateral.so not found; build the "
            "test-only plugin first"
        )
    _plugin_handle = load_plugin(plugin)

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    env = _create_env()
    observation, reset_info = env.reset(seed=int(args.seed))

    solver_classes = [
        obj.getClassName() for obj in env._sofa_root_node.objects
        if obj.getClassName() in (
            "GenericConstraintSolver",
            "LCPConstraintSolver",
        )
    ]
    if solver_classes != ["GenericConstraintSolver"]:
        raise RuntimeError(
            f"expected exactly GenericConstraintSolver, got {solver_classes}"
        )
    if int(env.physics_substeps) != EXPECTED_SUBSTEPS:
        raise RuntimeError(
            f"physics_substeps={env.physics_substeps}, "
            f"expected {EXPECTED_SUBSTEPS}"
        )

    instrument = env.mcr_controller_sofa.instrument.InstrumentCombined
    beam_dofs = instrument.getObject("DOFs")
    adapter = AdapterCompat(env, instrument)

    initial_q = _as_array(beam_dofs.position)
    initial_measure = adapter.measure(
        initial_q, spacing_m=DENSE_SPACING_M
    )
    initial_clearance = float(initial_measure["min_clearance_m"])
    if (
        not _finite(initial_q)
        or not np.isfinite(initial_clearance)
        or initial_clearance < -COMMITTED_PENETRATION_LIMIT_M
    ):
        raise RuntimeError(
            f"reset state violates committed safety: "
            f"{initial_clearance * 1000.0} mm"
        )

    constraint = instrument.addObject(
        "BeamLinearizedUnilateralConstraint",
        name="B02FullEpisodeBeamLinearizedUnilateralConstraint",
        enabled=False,
        rowOffsets=[],
        dofIndices=[],
        linearJacobian=[],
        angularJacobian=[],
        freeViolations=[],
        sourceClearances=[],
    )
    constraint.init()

    planner = FullEpisodeBeamUnilateralController(
        name="B02FullEpisodeBeamUnilateralController",
        beam_dofs=beam_dofs,
        adapter=adapter,
        constraint=constraint,
        requested_margin_m=MARGIN_M,
    )
    instrument.addObject(planner)

    model = DistributedPPO.load(str(checkpoint), device="cpu")
    model.policy.set_training_mode(False)

    original_animate = env.sofa_simulation.animate
    current_step = 0
    substep_in_step = 0
    fatal_reason: str | None = None
    first_failure: dict[str, Any] | None = None
    stats = _new_stats()
    action_hasher = hashlib.sha256()
    rl_steps_executed = 0
    partial_last_rl_step = False
    terminated_flag = False
    truncated_flag = False
    terminal_reason = None
    terminal_info = None
    started = time.perf_counter()

    trace_file = trace_path.open("w", buffering=1)

    def write_trace(record):
        trace_file.write(
            json.dumps(_json_safe(record), sort_keys=True) + "\n"
        )
        trace_file.flush()

    def traced_animate(root, dt):
        nonlocal substep_in_step, fatal_reason, first_failure

        substep_in_step += 1
        sub = int(substep_in_step)
        planner.current_rl_step = int(current_step)
        planner.current_substep = sub

        result = original_animate(root, dt)

        rec = planner.take_record() or {
            "rl_step": int(current_step),
            "substep": sub,
            "planning_status": "FAIL",
            "reason": "COLLISION_BEGIN_EVENT_NOT_OBSERVED",
            "rows_required": False,
            "rows_armed": False,
            "row_count": 0,
            "row_build_runtime_s": 0.0,
            "uses_collision_dofs_as_constraint_source": False,
            "q_candidate_solver_used": False,
            "writes_free_position": False,
            "writes_committed_position": False,
            "projection_used": False,
            "rollback_used": False,
        }
        rec["physics_index"] = (
            (int(current_step) - 1) * EXPECTED_SUBSTEPS + sub
        )

        try:
            active_count = int(constraint.activeCount.value)
        except Exception:
            active_count = -1
        rec["constraint_active_count"] = active_count

        q_committed = _as_array(beam_dofs.position)
        committed_finite = _finite(q_committed)
        rec["committed_finite"] = bool(committed_finite)

        if committed_finite:
            committed_measure = adapter.measure(
                q_committed, spacing_m=DENSE_SPACING_M
            )
            committed_clearance = float(
                committed_measure["min_clearance_m"]
            )
            rec["committed_worst_point_m"] = committed_measure.get(
                "worst_point_m"
            )
        else:
            committed_clearance = float("nan")
            rec["committed_worst_point_m"] = None

        rec["committed_clearance_m"] = committed_clearance
        rec["committed_clearance_mm"] = (
            committed_clearance * 1000.0
            if np.isfinite(committed_clearance)
            else None
        )
        rec["committed_penetration_m"] = (
            max(0.0, -committed_clearance)
            if np.isfinite(committed_clearance)
            else float("nan")
        )
        rec["committed_penetration_mm"] = (
            rec["committed_penetration_m"] * 1000.0
            if np.isfinite(float(rec["committed_penetration_m"]))
            else None
        )
        rec["committed_meets_requested_margin"] = bool(
            np.isfinite(committed_clearance)
            and committed_clearance + FLOAT_TOL_M >= MARGIN_M
        )

        rows_required = bool(rec.get("rows_required", False))
        row_count = int(rec.get("row_count", 0))
        if rows_required:
            mismatch = (
                row_count <= 0
                or active_count != row_count
                or not bool(rec.get("rows_armed", False))
            )
        else:
            mismatch = active_count not in (0, -1)
        rec["constraint_active_mismatch"] = bool(mismatch)

        forbidden = bool(
            rec.get("uses_collision_dofs_as_constraint_source")
            or rec.get("q_candidate_solver_used")
            or rec.get("writes_free_position")
            or rec.get("writes_committed_position")
            or rec.get("projection_used")
            or rec.get("rollback_used")
        )
        rec["forbidden_fallback_or_write"] = forbidden

        rec["non_finite"] = bool(
            not committed_finite
            or not np.isfinite(committed_clearance)
            or not np.isfinite(
                float(rec.get("q_free_clearance_m", np.nan))
            )
        )

        rec["substep_status"] = "PASS"
        rec["substep_reason"] = "COMMITTED_BEAM_SAFE"

        if str(rec.get("planning_status")) == "FAIL":
            rec["substep_status"] = "FAIL"
            rec["substep_reason"] = str(
                rec.get("reason", "BEAM_UNILATERAL_PLANNER_FAILED")
            )
        elif forbidden:
            rec["substep_status"] = "FAIL"
            rec["substep_reason"] = "FORBIDDEN_FALLBACK_OR_POSITION_WRITE"
        elif mismatch:
            rec["substep_status"] = "FAIL"
            rec["substep_reason"] = "CONSTRAINT_ACTIVE_COUNT_MISMATCH"
        elif rec["non_finite"]:
            rec["substep_status"] = "FAIL"
            rec["substep_reason"] = "NON_FINITE_POST_SUBSTEP_STATE"
        elif committed_clearance < -COMMITTED_PENETRATION_LIMIT_M:
            rec["substep_status"] = "FAIL"
            rec["substep_reason"] = "COMMITTED_DENSE_PENETRATION_VIOLATION"

        _update_stats(stats, rec)
        write_trace(rec)

        if rec["substep_status"] != "PASS" and fatal_reason is None:
            fatal_reason = (
                f"STEP_{current_step}_SUBSTEP_{sub}_"
                f"{rec['substep_reason']}"
            )
            first_failure = dict(rec)
            raise _FailFastStop(fatal_reason)

        return result

    env.sofa_simulation.animate = traced_animate

    try:
        for step in range(1, int(args.max_rl_steps) + 1):
            current_step = int(step)
            substep_in_step = 0

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
                rl_steps_executed = int(step)
                partial_last_rl_step = (
                    substep_in_step < EXPECTED_SUBSTEPS
                )
                break
            except Exception as exc:
                fatal_reason = (
                    f"ENV_STEP_EXCEPTION_STEP_{step}: "
                    f"{type(exc).__name__}: {exc}"
                )
                first_failure = {
                    "rl_step": int(step),
                    "substep": int(substep_in_step),
                    "exception": fatal_reason,
                }
                rl_steps_executed = int(step)
                partial_last_rl_step = True
                break

            rl_steps_executed = int(step)

            if substep_in_step != EXPECTED_SUBSTEPS:
                fatal_reason = (
                    f"STEP_{step}_PHYSICS_SUBSTEP_COUNT_"
                    f"{substep_in_step}_EXPECTED_{EXPECTED_SUBSTEPS}"
                )
                first_failure = {
                    "rl_step": int(step),
                    "substeps_observed": int(substep_in_step),
                }
                break

            if args.progress_every > 0 and (
                step == 1
                or step % int(args.progress_every) == 0
                or terminated
                or truncated
            ):
                current_stats = _finalize_stats(stats)
                print(
                    "[BEAM_UNILATERAL_FULL] "
                    f"step={step}/{args.max_rl_steps} "
                    f"substeps={current_stats['physics_substeps']} "
                    f"row_builds={current_stats['row_build_substeps']} "
                    f"free_pen={current_stats['penetrating_free_substeps']} "
                    f"worst_committed_mm="
                    f"{current_stats['min_committed_clearance_mm']} "
                    f"max_pen_mm="
                    f"{current_stats['max_committed_penetration_mm']:.6f}",
                    flush=True,
                )

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
        constraint.enabled.value = False
        env.sofa_simulation.animate = original_animate
        trace_file.close()

    final_stats = _finalize_stats(stats)

    if fatal_reason is not None:
        decision = "FAIL"
        reason = fatal_reason
    elif final_stats["committed_violation_count"] != 0:
        decision = "FAIL"
        reason = "COMMITTED_DENSE_PENETRATION_VIOLATION_OBSERVED"
    elif final_stats["non_finite_count"] != 0:
        decision = "FAIL"
        reason = "NON_FINITE_STATE_OBSERVED"
    elif final_stats["constraint_active_mismatch_count"] != 0:
        decision = "FAIL"
        reason = "CONSTRAINT_ACTIVE_COUNT_MISMATCH_OBSERVED"
    elif final_stats["planning_failure_count"] != 0:
        decision = "FAIL"
        reason = "BEAM_UNILATERAL_ROW_BUILD_FAILURE_OBSERVED"
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
        reason = "BEAM_UNILATERAL_FULL_EPISODE_COMMITTED_SAFETY_HELD"

    payload = {
        "test": (
            "B02 full-episode Beam-level SDF unilateral + "
            "GenericConstraintSolver safety acceptance"
        ),
        "decision": decision,
        "reason": reason,
        "seed": int(args.seed),
        "checkpoint": str(checkpoint),
        "plugin": str(plugin),
        "max_rl_steps": int(args.max_rl_steps),
        "rl_steps_executed": int(rl_steps_executed),
        "partial_last_rl_step": bool(partial_last_rl_step),
        "physics_dt_s": float(DT),
        "expected_physics_substeps": EXPECTED_SUBSTEPS,
        "constraint_solver": "GenericConstraintSolver",
        "requested_margin_m": MARGIN_M,
        "requested_margin_mm": MARGIN_M * 1000.0,
        "dense_spacing_m": DENSE_SPACING_M,
        "dense_spacing_mm": DENSE_SPACING_M * 1000.0,
        "committed_penetration_limit_m": COMMITTED_PENETRATION_LIMIT_M,
        "committed_penetration_limit_mm": (
            COMMITTED_PENETRATION_LIMIT_M * 1000.0
        ),
        "initial_committed_clearance_m": initial_clearance,
        "initial_committed_clearance_mm": initial_clearance * 1000.0,
        "action_stream_sha256": action_hasher.hexdigest(),
        "terminated": bool(terminated_flag),
        "truncated": bool(truncated_flag),
        "terminal_reason": terminal_reason,
        "terminal_info": terminal_info,
        "first_failure": first_failure,
        "stats": final_stats,
        "trace_file": str(trace_path),
        "wall_s": float(time.perf_counter() - started),
        "production_files_modified": False,
        "training_started": False,
        "q_candidate_solver_used": False,
        "native_candidate_injection_used": False,
        "collision_dofs_used_as_safety_constraint_source": False,
        "direct_free_position_write_used": False,
        "direct_committed_position_write_used": False,
        "rollback_used": False,
        "projection_used": False,
        "reset_info": _json_safe(reset_info),
        "engineering_interpretation": (
            "PASS is an engineering acceptance for this one fixed "
            "B02/target04/seed15204 episode only; it is not a global proof."
        ),
    }

    output.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n"
    )
    _write_report(payload, report_path)

    print(
        json.dumps(
            {
                "decision": decision,
                "reason": reason,
                "rl_steps_executed": rl_steps_executed,
                "physics_substeps": final_stats["physics_substeps"],
                "row_build_substeps": final_stats["row_build_substeps"],
                "penetrating_free_substeps": final_stats[
                    "penetrating_free_substeps"
                ],
                "min_free_clearance_mm": final_stats[
                    "min_free_clearance_mm"
                ],
                "min_committed_clearance_mm": final_stats[
                    "min_committed_clearance_mm"
                ],
                "max_committed_penetration_mm": final_stats[
                    "max_committed_penetration_mm"
                ],
                "committed_violations": final_stats[
                    "committed_violation_count"
                ],
                "committed_below_margin_count": final_stats[
                    "committed_below_margin_count"
                ],
                "row_build_runtime_total_s": final_stats[
                    "row_build_runtime_total_s"
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
