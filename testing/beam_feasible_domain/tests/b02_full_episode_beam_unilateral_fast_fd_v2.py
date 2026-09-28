#!/usr/bin/env python3
"""2048-step B02 acceptance for fast local-point FD Beam unilateral + Generic.

This is the full-episode safety/performance validation of the fast builder that
matched the validated baseline Jacobian exactly at step665/substep1.

Safety formulation is unchanged:
    clearance(q) - 0.100 mm >= 0

Optimization under test:
    one full 10 um Beam profile for row selection
    + selected-point local Rigid3 finite differences
    + one batched SDF query for all FD points

No SLSQP q_candidate route, native candidate injection, direct Beam position
write, projection, rollback, action shielding, or CollisionDOFs safety source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
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
from beam_unilateral_fast_full_episode_v2 import (
    FastFullEpisodeBeamUnilateralControllerV2,
)
from b02_full_episode_beam_unilateral_generic import (
    _FailFastStop,
    _finalize_stats,
    _json_safe,
    _looks_like_safety_terminal,
    _new_stats,
    _terminal_reason,
    _update_stats,
)
from mcr_sim.distributed import DistributedPPO


DEFAULT_MAX_RL_STEPS = 2048
EXPECTED_SUBSTEPS = 2
COMMITTED_PENETRATION_LIMIT_M = 1.0e-6
FLOAT_TOL_M = 1.0e-12
BASELINE_ACTION_SHA256 = (
    "e6854f5496eb2d43eb913d1116238d831666f7589ada051325bfdf14d294900e"
)
BASELINE_ROW_BUILD_SUBSTEPS = 1436
BASELINE_ROW_BUILD_TOTAL_S = 1410.52
BASELINE_ROW_BUILD_MEAN_S = 0.9823

RUNTIME = (
    BEAM_ROOT
    / "_runtime"
    / "beam_unilateral_fast_fd_v2/full_episode"
)
DEFAULT_BUILD_DIRS = (
    BEAM_ROOT
    / "_runtime"
    / "beam_unilateral_lcp_step665"
    / "native_build",
    BEAM_ROOT
    / "_runtime"
    / "beam_unilateral_generic_full_episode"
    / "native_build",
)


def _find_default_plugin() -> Path | None:
    for build_dir in DEFAULT_BUILD_DIRS:
        plugin = find_plugin(build_dir)
        if plugin is not None:
            return plugin
    return None


def _new_fast_stats() -> dict[str, Any]:
    stats = _new_stats()
    stats.update(
        {
            "negative_free_substeps": 0,
            "support_mode_counts": Counter(),
            "batched_fd_point_total": 0,
            "batched_fd_sdf_query_total": 0,
            "post_selection_full_beam_profile_total": 0,
            "max_selected_point_reconstruction_error_m": 0.0,
            "fast_builder_invariant_failure_count": 0,
            "row_count_distribution": Counter(),
            "q_prev_full_dense_profile_total": 0,
            "q_free_full_dense_profile_total": 0,
            "builder_internal_q_free_profile_total": 0,
            "v2_invariant_failure_count": 0,
            "q_free_profile_runtimes_s": [],
            "active_planner_runtimes_s": [],
        }
    )
    return stats


def _update_fast_stats(
    stats: dict[str, Any],
    rec: dict[str, Any],
) -> None:
    free_clearance = float(rec.get("q_free_clearance_m", np.nan))
    if np.isfinite(free_clearance) and free_clearance < 0.0:
        stats["negative_free_substeps"] += 1

    stats["q_prev_full_dense_profile_total"] += int(rec.get("q_prev_full_dense_profile_count", -999))
    stats["q_free_full_dense_profile_total"] += int(rec.get("q_free_external_dense_count", -999))
    stats["builder_internal_q_free_profile_total"] += int(rec.get("q_free_internal_dense_count", 0))
    stats["q_free_profile_runtimes_s"].append(float(rec.get("q_free_measure_s", 0.0)))
    if not bool(rec.get("rows_required", False)):
        return
    stats["active_planner_runtimes_s"].append(float(rec.get("planner_total_s", 0.0)))

    support_mode = str(rec.get("support_mode", "MISSING"))
    stats["support_mode_counts"][support_mode] += 1
    stats["row_count_distribution"][
        str(int(rec.get("row_count", 0)))
    ] += 1
    stats["batched_fd_point_total"] += int(
        rec.get("batched_fd_point_count", 0)
    )
    stats["batched_fd_sdf_query_total"] += int(
        rec.get("batched_fd_sdf_query_count", 0)
    )
    stats["post_selection_full_beam_profile_total"] += int(
        rec.get("full_beam_profile_evaluations_after_selection", 0)
    )
    recon = float(
        rec.get("selected_point_reconstruction_max_error_m", 0.0)
    )
    stats["max_selected_point_reconstruction_error_m"] = max(
        float(stats["max_selected_point_reconstruction_error_m"]),
        recon,
    )

    if (
        support_mode != "exact_selected_point_local_fd"
        or int(rec.get("batched_fd_sdf_query_count", 0)) != 1
        or int(
            rec.get(
                "full_beam_profile_evaluations_after_selection",
                -1,
            )
        )
        != 0
        or recon > 1.0e-10
    ):
        stats["fast_builder_invariant_failure_count"] += 1


def _finalize_fast_stats(
    stats: dict[str, Any],
) -> dict[str, Any]:
    out = _finalize_stats(stats)
    out["support_mode_counts"] = dict(stats["support_mode_counts"])
    out["row_count_distribution"] = dict(
        stats["row_count_distribution"]
    )
    out["negative_free_substeps"] = int(
        stats["negative_free_substeps"]
    )
    out["batched_fd_point_total"] = int(
        stats["batched_fd_point_total"]
    )
    out["batched_fd_sdf_query_total"] = int(
        stats["batched_fd_sdf_query_total"]
    )
    out["post_selection_full_beam_profile_total"] = int(
        stats["post_selection_full_beam_profile_total"]
    )
    out["max_selected_point_reconstruction_error_m"] = float(
        stats["max_selected_point_reconstruction_error_m"]
    )
    out["q_prev_full_dense_profile_total"] = int(stats["q_prev_full_dense_profile_total"])
    out["q_free_full_dense_profile_total"] = int(stats["q_free_full_dense_profile_total"])
    out["builder_internal_q_free_profile_total"] = int(stats["builder_internal_q_free_profile_total"])
    out["v2_invariant_failure_count"] = int(stats["v2_invariant_failure_count"])
    out["q_free_profile_runtime_mean_s"] = float(np.mean(stats["q_free_profile_runtimes_s"])) if stats["q_free_profile_runtimes_s"] else None
    out["active_planner_runtime_mean_s"] = float(np.mean(stats["active_planner_runtimes_s"])) if stats["active_planner_runtimes_s"] else None
    out["fast_builder_invariant_failure_count"] = int(
        stats["fast_builder_invariant_failure_count"]
    )
    return out


def _write_report(payload: dict[str, Any], report: Path) -> None:
    stats = payload["stats"]
    baseline_total_s = float(payload["baseline_row_build_total_s"])
    fast_total_s = float(stats["row_build_runtime_total_s"])
    measured_speedup = payload["full_episode_row_build_speedup_vs_baseline"]
    mean_speedup = payload["row_build_mean_speedup_vs_baseline"]

    lines = [
        "B02 FAST LOCAL-POINT FD BEAM UNILATERAL + GENERIC FULL-EPISODE ACCEPTANCE",
        "============================================================================",
        "",
        f"FINAL: {payload['decision']}",
        f"Reason: {payload['reason']}",
        "",
        "Configuration",
        "-------------",
        f"Seed: {payload['seed']}",
        f"Max RL steps: {payload['max_rl_steps']}",
        f"Physics: {payload['expected_physics_substeps']} x {payload['physics_dt_s']} s",
        f"Solver: {payload['constraint_solver']}",
        f"Requested margin: {payload['requested_margin_mm']} mm",
        f"Dense spacing: {payload['dense_spacing_mm']} mm",
        f"Committed penetration limit: {payload['committed_penetration_limit_mm']} mm",
        "",
        "Run",
        "---",
        f"RL steps executed: {payload['rl_steps_executed']}",
        f"Physics substeps: {stats['physics_substeps']}",
        f"Terminated / truncated: {payload['terminated']} / {payload['truncated']}",
        f"Terminal reason: {payload['terminal_reason']}",
        f"Action SHA256: {payload['action_stream_sha256']}",
        f"Baseline action SHA256 match: {payload['baseline_action_sha256_match']}",
        "",
        "Constraint activity",
        "-------------------",
        f"Safe-free substeps: {stats['safe_free_substeps']}",
        f"Row-build substeps: {stats['row_build_substeps']}",
        f"q_free < 0 substeps: {stats['negative_free_substeps']}",
        f"q_free < -0.001 mm substeps: {stats['penetrating_free_substeps']}",
        f"Total rows: {stats['total_rows_built']}",
        f"Max rows/substep: {stats['max_rows_in_substep']}",
        f"Row distribution: {stats['row_count_distribution']}",
        f"Support modes: {stats['support_mode_counts']}",
        f"ActiveCount mismatches: {stats['constraint_active_mismatch_count']}",
        "",
        "Fast builder",
        "------------",
        f"Row-build runtime total/mean/max: "
        f"{stats['row_build_runtime_total_s']} / "
        f"{stats['row_build_runtime_mean_s']} / "
        f"{stats['row_build_runtime_max_s']} s",
        f"Baseline validated row-build total: {baseline_total_s} s",
        f"Full-episode row-build speedup vs baseline: {measured_speedup}",
        f"Mean row-build speedup vs baseline: {mean_speedup}",
        f"Batched FD points total: {stats['batched_fd_point_total']}",
        f"Batched FD SDF calls total: {stats['batched_fd_sdf_query_total']}",
        f"Post-selection full-Beam profiles: {stats['post_selection_full_beam_profile_total']}",
        f"Max selected-point reconstruction error: "
        f"{stats['max_selected_point_reconstruction_error_m']} m",
        f"Fast-builder invariant failures: "
        f"{stats['fast_builder_invariant_failure_count']}",
        "",
        "Clearance",
        "---------",
        f"Worst free clearance: {stats['min_free_clearance_mm']} mm",
        f"Worst committed clearance: {stats['min_committed_clearance_mm']} mm",
        f"Max committed penetration: {stats['max_committed_penetration_mm']} mm",
        f"Committed violations: {stats['committed_violation_count']}",
        f"Committed states below +0.100 mm margin: "
        f"{stats['committed_below_margin_count']}",
        "",
        "Numerics",
        "--------",
        f"NaN/Inf substeps: {stats['non_finite_count']}",
        f"Planning/row-build failures: {stats['planning_failure_count']}",
        f"First failure: {payload['first_failure']}",
        "",
        "Interpretation",
        "--------------",
        "PASS means the fast local-point FD implementation preserved the fixed",
        "B02/target04/seed15204 full-episode committed nonpenetration acceptance.",
        "It does not prove all future policies, states, seeds, or vessels.",
        "",
    ]
    report.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(CHECKPOINT))
    parser.add_argument("--seed", type=int, default=15204)
    parser.add_argument(
        "--max-rl-steps",
        type=int,
        default=DEFAULT_MAX_RL_STEPS,
    )
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--plugin-lib", default=None)
    parser.add_argument(
        "--output",
        default=str(
            RUNTIME
            / "results"
            / "b02_fast_fd_v2_full_episode.json"
        ),
    )
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path = output.with_name(
        "b02_fast_fd_v2_full_episode_report.md"
    )
    trace_path = output.with_name(
        "b02_fast_fd_v2_full_episode_trace.jsonl"
    )

    plugin = (
        Path(args.plugin_lib).expanduser().resolve()
        if args.plugin_lib
        else _find_default_plugin()
    )
    if plugin is None or not plugin.is_file():
        raise RuntimeError(
            "libMCRBeamLinearizedUnilateral.so not found"
        )
    _plugin_handle = load_plugin(plugin)

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    env = _create_env()
    observation, reset_info = env.reset(seed=int(args.seed))

    solver_classes = [
        obj.getClassName()
        for obj in env._sofa_root_node.objects
        if obj.getClassName()
        in ("GenericConstraintSolver", "LCPConstraintSolver")
    ]
    if solver_classes != ["GenericConstraintSolver"]:
        raise RuntimeError(
            f"expected GenericConstraintSolver only, got {solver_classes}"
        )
    if int(env.physics_substeps) != EXPECTED_SUBSTEPS:
        raise RuntimeError(
            f"physics_substeps={env.physics_substeps}, "
            f"expected={EXPECTED_SUBSTEPS}"
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
        name="B02FastFDFullEpisodeBeamUnilateralConstraint",
        enabled=False,
        rowOffsets=[],
        dofIndices=[],
        linearJacobian=[],
        angularJacobian=[],
        freeViolations=[],
        sourceClearances=[],
    )
    constraint.init()

    planner = FastFullEpisodeBeamUnilateralControllerV2(
        name="B02FastFDFullEpisodeController",
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
    stats = _new_fast_stats()
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
            "native_candidate_injection_used": False,
            "writes_free_position": False,
            "writes_committed_position": False,
            "projection_used": False,
            "rollback_used": False,
            "action_shielding_used": False,
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
            or rec.get("native_candidate_injection_used")
            or rec.get("writes_free_position")
            or rec.get("writes_committed_position")
            or rec.get("projection_used")
            or rec.get("rollback_used")
            or rec.get("action_shielding_used")
        )
        rec["forbidden_fallback_or_write"] = forbidden

        rec["non_finite"] = bool(
            not committed_finite
            or not np.isfinite(committed_clearance)
            or not np.isfinite(
                float(rec.get("q_free_clearance_m", np.nan))
            )
        )

        v2_invariant_failure = bool(
            int(rec.get("q_prev_full_dense_profile_count", -1)) != 0
            or int(rec.get("q_free_external_dense_count", -1)) != 1
            or int(rec.get("q_free_internal_dense_count", -1)) != 0
            or int(rec.get("total_q_free_full_profile_count", -1)) != 1
            or (bool(rec.get("rows_required", False)) and (int(rec.get("batched_fd_sdf_query_count", 0)) != 1 or int(rec.get("full_beam_profile_evaluations_after_selection", -1)) != 0 or str(rec.get("support_mode")) != "exact_selected_point_local_fd"))
        )
        rec["v2_invariant_failure"] = v2_invariant_failure
        stats["v2_invariant_failure_count"] += int(v2_invariant_failure)
        rec["substep_status"] = "PASS"
        rec["substep_reason"] = "COMMITTED_BEAM_SAFE"

        if str(rec.get("planning_status")) == "FAIL":
            rec["substep_status"] = "FAIL"
            rec["substep_reason"] = str(
                rec.get("reason", "FAST_BEAM_UNILATERAL_PLANNER_FAILED")
            )
        elif forbidden:
            rec["substep_status"] = "FAIL"
            rec["substep_reason"] = "FORBIDDEN_FALLBACK_OR_POSITION_WRITE"
        elif v2_invariant_failure:
            rec["substep_status"] = "FAIL"
            rec["substep_reason"] = "FAST_FD_V2_PROFILE_OR_ROW_INVARIANT_FAILURE"
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
        _update_fast_stats(stats, rec)
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

            raw_action, _ = model.predict(
                observation, deterministic=True
            )
            raw_action = np.asarray(
                raw_action, dtype=np.float32
            ).reshape(3)
            action_hasher.update(
                np.asarray(
                    raw_action, dtype="<f4"
                ).tobytes(order="C")
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
                current_stats = _finalize_fast_stats(stats)
                print(
                    "[FAST_FD_FULL] "
                    f"step={step}/{args.max_rl_steps} "
                    f"substeps={current_stats['physics_substeps']} "
                    f"row_builds={current_stats['row_build_substeps']} "
                    f"qfree_neg={current_stats['negative_free_substeps']} "
                    f"row_mean_s="
                    f"{current_stats['row_build_runtime_mean_s']:.6f} "
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

    final_stats = _finalize_fast_stats(stats)
    action_sha = action_hasher.hexdigest()

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
        reason = "FAST_BEAM_UNILATERAL_ROW_BUILD_FAILURE_OBSERVED"
    elif final_stats["fast_builder_invariant_failure_count"] != 0:
        decision = "FAIL"
        reason = "FAST_BUILDER_INVARIANT_FAILURE_OBSERVED"
    elif final_stats["v2_invariant_failure_count"] != 0:
        decision = "FAIL"
        reason = "FAST_FD_V2_INVARIANT_FAILURE_OBSERVED"
    elif final_stats["q_prev_full_dense_profile_total"] != 0 or final_stats["q_free_full_dense_profile_total"] != final_stats["physics_substeps"] or final_stats["builder_internal_q_free_profile_total"] != 0:
        decision = "FAIL"
        reason = "FAST_FD_V2_DENSE_PROFILE_COUNT_MISMATCH"
    elif final_stats["post_selection_full_beam_profile_total"] != 0:
        decision = "FAIL"
        reason = "FAST_BUILDER_RECOMPUTED_FULL_BEAM_AFTER_SELECTION"
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
        reason = "FAST_FD_V2_FULL_EPISODE_COMMITTED_SAFETY_HELD"

    baseline_row_build_total_s = BASELINE_ROW_BUILD_TOTAL_S
    full_episode_row_build_speedup = (
        baseline_row_build_total_s
        / float(final_stats["row_build_runtime_total_s"])
        if float(final_stats["row_build_runtime_total_s"]) > 0.0
        else None
    )
    row_build_mean_speedup = (
        BASELINE_ROW_BUILD_MEAN_S
        / float(final_stats["row_build_runtime_mean_s"])
        if float(final_stats["row_build_runtime_mean_s"]) > 0.0
        else None
    )

    payload = {
        "test": (
            "B02 2048-step fast local-point FD Beam-level SDF "
            "unilateral + GenericConstraintSolver acceptance"
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
        "builder": "fast_local_point_fd_v2",
        "requested_margin_m": MARGIN_M,
        "requested_margin_mm": MARGIN_M * 1000.0,
        "dense_spacing_m": DENSE_SPACING_M,
        "dense_spacing_mm": DENSE_SPACING_M * 1000.0,
        "committed_penetration_limit_m": (
            COMMITTED_PENETRATION_LIMIT_M
        ),
        "committed_penetration_limit_mm": (
            COMMITTED_PENETRATION_LIMIT_M * 1000.0
        ),
        "initial_committed_clearance_m": initial_clearance,
        "initial_committed_clearance_mm": initial_clearance * 1000.0,
        "action_stream_sha256": action_sha,
        "baseline_action_sha256": BASELINE_ACTION_SHA256,
        "baseline_action_sha256_match": (
            action_sha == BASELINE_ACTION_SHA256
        ),
        "terminated": bool(terminated_flag),
        "truncated": bool(truncated_flag),
        "terminal_reason": terminal_reason,
        "terminal_info": terminal_info,
        "first_failure": first_failure,
        "stats": final_stats,
        "baseline_row_build_substeps": BASELINE_ROW_BUILD_SUBSTEPS,
        "baseline_row_build_total_s": baseline_row_build_total_s,
        "baseline_row_build_mean_s": BASELINE_ROW_BUILD_MEAN_S,
        "full_episode_row_build_speedup_vs_baseline": (
            full_episode_row_build_speedup
        ),
        "row_build_mean_speedup_vs_baseline": row_build_mean_speedup,
        "trace_file": str(trace_path),
        "wall_s": float(time.perf_counter() - started),
        "production_files_modified": False,
        "existing_production_files_touched": False,
        "training_started": False,
        "v2_q_prev_dense_removed": True,
        "v2_q_free_profile_reused": True,
        "committed_independent_dense_validator_retained": True,
        "q_candidate_solver_used": False,
        "native_candidate_injection_used": False,
        "collision_dofs_used_as_safety_constraint_source": False,
        "direct_free_position_write_used": False,
        "direct_committed_position_write_used": False,
        "rollback_used": False,
        "projection_used": False,
        "action_shielding_used": False,
        "reset_info": _json_safe(reset_info),
        "engineering_interpretation": (
            "PASS validates the fast local-point FD implementation on the "
            "same fixed B02/target04/seed15204 2048-step engineering case. "
            "It does not prove global nonpenetration."
        ),
    }

    output.write_text(
        json.dumps(
            _json_safe(payload),
            indent=2,
            sort_keys=True,
        )
        + "\n"
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
                "q_free_negative_substeps": final_stats[
                    "negative_free_substeps"
                ],
                "q_free_below_minus_0p001mm_substeps": final_stats[
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
                "row_build_runtime_total_s": final_stats[
                    "row_build_runtime_total_s"
                ],
                "row_build_runtime_mean_s": final_stats[
                    "row_build_runtime_mean_s"
                ],
                "row_build_runtime_max_s": final_stats[
                    "row_build_runtime_max_s"
                ],
                "full_episode_row_build_speedup_vs_baseline": (
                    full_episode_row_build_speedup
                ),
                "row_build_mean_speedup_vs_baseline": (
                    row_build_mean_speedup
                ),
                "support_mode_counts": final_stats[
                    "support_mode_counts"
                ],
                "action_sha256": action_sha,
                "baseline_action_sha256_match": (
                    action_sha == BASELINE_ACTION_SHA256
                ),
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
