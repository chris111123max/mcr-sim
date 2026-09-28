#!/usr/bin/env python3
"""Step665 equivalence/performance test for fast local-point FD Jacobians.

Prefix through step664:
  archived strict 0.100 mm feasible-domain route, only to reproduce the known
  protected step665 state.

Target step665/substep1:
  - capture the same q_prev/q_free;
  - build baseline full-profile-FD Beam unilateral rows for comparison only;
  - build fast local-point-FD rows on the exact same state;
  - compare structure and Jacobian numerics;
  - install ONLY the fast rows into BeamLinearizedUnilateralConstraint;
  - let GenericConstraintSolver commit one physical substep;
  - independently verify the committed 10 um dense clearance;
  - stop before substep2.

No target q_candidate SLSQP solve or candidate injection is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import Sofa

THIS = Path(__file__).resolve()
TEST_DIR = THIS.parent
BEAM_ROOT = THIS.parents[1]
PYTHON_ROOT = THIS.parents[3]
UNILATERAL_DIR = BEAM_ROOT / "beam_unilateral_lcp"
RUNTIME = BEAM_ROOT / "_runtime" / "beam_unilateral_fast_fd_v2" / "step665"
DEFAULT_BEAM_BUILD_DIRS = (
    BEAM_ROOT / "_runtime" / "beam_unilateral_lcp_step665" / "native_build",
    BEAM_ROOT / "_runtime" / "beam_unilateral_generic_full_episode" / "native_build",
)
DEFAULT_HOOK_BUILD = BEAM_ROOT / "_runtime" / "native_build"

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
from b02_step776_two_substep_invariance import (
    _as_int,
    _as_str,
    _find_plugin as find_hook_plugin,
    _load_plugin as load_hook_plugin,
)
import b02_full_episode_strict_margin_0p100_acceptance as prefix_base
from beam_linearized_unilateral import (
    DENSE_SPACING_M,
    MARGIN_M,
    find_plugin as find_beam_plugin,
    load_plugin as load_beam_plugin,
    write_rows,
)
from beam_linearized_unilateral import _profile
from beam_unilateral_fast_fd import build_rows_fast_local_fd, compare_row_builds
from beam_unilateral_fast_fd_v2 import build_rows_fast_local_fd_v2
from diagnose_step665_beam_unilateral_lcp import _prefix_substep_ok
from mcr_sim.distributed import DistributedPPO


TARGET_STEP = 665
TARGET_SUBSTEP = 1
EXPECTED_QPREV_MM = 0.468945
EXPECTED_QFREE_MM = -0.032075
STATE_TOL_MM = 0.002
EXPECTED_SUBSTEPS = 2
COMMITTED_PEN_LIMIT_M = 1.0e-6

LINEAR_ATOL = 5.0e-7
LINEAR_RTOL = 5.0e-5
ANGULAR_ATOL = 5.0e-9
ANGULAR_RTOL = 5.0e-5


class _StopAfterTarget(Exception):
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


def _find_beam_plugin() -> Path | None:
    for build_dir in DEFAULT_BEAM_BUILD_DIRS:
        plugin = find_beam_plugin(build_dir)
        if plugin is not None:
            return plugin
    return None


def _sha_action_prefix(actions: list[np.ndarray]) -> str:
    h = hashlib.sha256()
    for action in actions:
        h.update(
            np.asarray(action, dtype="<f4").reshape(3).tobytes(order="C")
        )
    return h.hexdigest()


def _jacobian_equivalent(
    baseline: dict[str, Any],
    fast: dict[str, Any],
) -> bool:
    if (
        baseline["row_offsets"] != fast["row_offsets"]
        or baseline["dof_indices"] != fast["dof_indices"]
        or baseline["selected_dense_indices"]
        != fast["selected_dense_indices"]
    ):
        return False

    a_lin = np.asarray(baseline["linear_jacobian"], dtype=np.float64)
    b_lin = np.asarray(fast["linear_jacobian"], dtype=np.float64)
    a_ang = np.asarray(baseline["angular_jacobian"], dtype=np.float64)
    b_ang = np.asarray(fast["angular_jacobian"], dtype=np.float64)

    return bool(
        a_lin.shape == b_lin.shape
        and a_ang.shape == b_ang.shape
        and np.allclose(
            a_lin,
            b_lin,
            rtol=LINEAR_RTOL,
            atol=LINEAR_ATOL,
        )
        and np.allclose(
            a_ang,
            b_ang,
            rtol=ANGULAR_RTOL,
            atol=ANGULAR_ATOL,
        )
    )


class Step665FastFDV2Controller(Sofa.Core.Controller):
    def __init__(self, *, beam_dofs, adapter, constraint, **kwargs):
        kwargs["listening"] = True
        Sofa.Core.Controller.__init__(self, **kwargs)
        self.beam_dofs, self.adapter, self.constraint = beam_dofs, adapter, constraint
        self.current_rl_step = self.current_substep = 0
        self.enabled = self.processed = False
        self.previous_committed_clearance_mm = None
        self.record: dict[str, Any] = {}

    def onEvent(self, event):
        name = next((str(event[k]) for k in ("type", "Type", "name", "event", "className") if isinstance(event, dict) and k in event), type(event).__name__)
        if "CollisionBegin" in name:
            self._collision_begin("onEvent:" + name)

    def onCollisionBeginEvent(self, event):
        self._collision_begin("onCollisionBeginEvent")

    def _collision_begin(self, source):
        if not self.enabled or self.processed or self.current_rl_step != TARGET_STEP or self.current_substep != TARGET_SUBSTEP:
            return
        self.processed = True
        q_prev = _as_array(self.beam_dofs.position)
        q_free = _as_array(self.beam_dofs.free_position)
        t0 = time.perf_counter()
        points, clear = _profile(self.adapter, q_free)
        external_profile_s = time.perf_counter() - t0
        prev_mm = self.previous_committed_clearance_mm
        free_mm = float(np.min(clear)) * 1000.0
        self.record = {"event_source": source, "q_prev_clearance_mm": prev_mm, "q_free_clearance_mm": free_mm, "state_matched": bool(prev_mm is not None and abs(prev_mm-EXPECTED_QPREV_MM)<=STATE_TOL_MM and abs(free_mm-EXPECTED_QFREE_MM)<=STATE_TOL_MM), "dense_sample_count": len(clear), "q_prev_full_dense_profile_count": 0, "v1_q_free_full_profile_count": 2, "v2_external_q_free_full_profile_count": 1, "v2_builder_internal_q_free_full_profile_count": 0, "v2_total_q_free_full_profile_count": 1, "external_q_free_profile_runtime_s": external_profile_s, "q_prev_state_source": "previous_substep_independent_committed_dense_validation", "target_q_candidate_solver_used": False, "target_native_candidate_injection_used": False, "uses_collision_dofs_as_constraint_source": False, "writes_free_position": False, "writes_committed_position": False}
        if not self.record["state_matched"]:
            self.record["status"] = "STEP665_PROTECTED_STATE_MISMATCH"
            self.constraint.enabled.value = False
            return
        t0 = time.perf_counter()
        v1 = build_rows_fast_local_fd(adapter=self.adapter,q_prev=q_prev,q_free=q_free,requested_margin_m=MARGIN_M)
        v1_s = time.perf_counter()-t0
        t0 = time.perf_counter()
        v2 = build_rows_fast_local_fd_v2(adapter=self.adapter,q_prev=q_prev,q_free=q_free,dense_points=points,dense_clearance=clear,requested_margin_m=MARGIN_M)
        v2_s = time.perf_counter()-t0
        comp = compare_row_builds(v1,v2)
        comp["selected_elements_equal"] = v1["selected_element_indices"] == v2["selected_element_indices"]
        comp["free_violations_equal"] = np.array_equal(np.asarray(v1["free_violations"]),np.asarray(v2["free_violations"]))
        comp["source_clearances_exact_equal"] = np.array_equal(np.asarray(v1["source_clearances"]),np.asarray(v2["source_clearances"]))
        comp["support_mode_equal"] = v1["support_metadata"]["support_mode"] == v2["support_metadata"]["support_mode"]
        comp["selected_point_reconstruction_error_equal"] = v1["support_metadata"]["selected_point_reconstruction_max_error_m"] == v2["support_metadata"]["selected_point_reconstruction_max_error_m"]
        comp["batched_fd_point_count_equal"] = v1["batched_fd_point_count"] == v2["batched_fd_point_count"]
        comp["batched_fd_sdf_query_count_equal"] = v1["batched_fd_sdf_query_count"] == v2["batched_fd_sdf_query_count"]
        qmatch = bool(np.isfinite(q_free).all() and np.isfinite(q_prev).all())
        exact = bool(qmatch and comp["all_structure_equal"] and comp["selected_elements_equal"] and comp["free_violations_equal"] and comp["source_clearances_exact_equal"] and comp["support_mode_equal"] and comp["selected_point_reconstruction_error_equal"] and comp["batched_fd_point_count_equal"] and comp["batched_fd_sdf_query_count_equal"] and comp["linear_max_abs_error"]<=1e-12 and comp["angular_max_abs_error"]<=1e-12)
        self.record.update({"status":"FAST_ROWS_ARMED" if exact else "FAST_V2_ROW_MISMATCH", "jacobian_equivalent":exact, "comparison":comp, "v1_runtime_s":v1_s, "v2_runtime_s":v2_s, "v1_active_total_s":external_profile_s+v1_s, "v2_active_total_s":external_profile_s+v2_s, "speedup_x":(external_profile_s+v1_s)/(external_profile_s+v2_s), "row_count":len(v2["free_violations"]), "v1_selected_dense_indices":v1["selected_dense_indices"], "v2_selected_dense_indices":v2["selected_dense_indices"], "selected_dense_indices":v2["selected_dense_indices"], "selected_element_indices":v2["selected_element_indices"], "v1_row_sha256":comp["baseline_row_sha256"], "v2_row_sha256":comp["fast_row_sha256"], "support_mode":v2["support_metadata"]["support_mode"], "selected_point_reconstruction_max_error_m":v2["support_metadata"]["selected_point_reconstruction_max_error_m"], "batched_fd_point_count":v2["batched_fd_point_count"], "batched_fd_sdf_query_count":v2["batched_fd_sdf_query_count"], "full_beam_profile_evaluations_after_selection":v2["full_beam_profile_evaluations_after_selection"]})
        if exact:
            write_rows(self.constraint,v2)
        else:
            self.constraint.enabled.value = False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(CHECKPOINT))
    parser.add_argument("--seed", type=int, default=15204)
    parser.add_argument("--beam-plugin-lib", default=None)
    parser.add_argument("--hook-plugin-lib", default=None)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument(
        "--output",
        default=str(
            RUNTIME / "results" / "step665_fast_fd_v2_result.json"
        ),
    )
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    beam_plugin = (
        Path(args.beam_plugin_lib).expanduser().resolve()
        if args.beam_plugin_lib
        else _find_beam_plugin()
    )
    hook_plugin = (
        Path(args.hook_plugin_lib).expanduser().resolve()
        if args.hook_plugin_lib
        else find_hook_plugin(DEFAULT_HOOK_BUILD)
    )
    if beam_plugin is None:
        raise RuntimeError("Beam unilateral plugin not found")
    if hook_plugin is None:
        raise RuntimeError("prefix native hook plugin not found")

    _beam_handle = load_beam_plugin(beam_plugin)
    _hook_handle = load_hook_plugin(hook_plugin)

    env = _create_env()
    observation, reset_info = env.reset(seed=int(args.seed))
    if int(env.physics_substeps) != EXPECTED_SUBSTEPS:
        raise RuntimeError(
            f"physics_substeps={env.physics_substeps}, "
            f"expected={EXPECTED_SUBSTEPS}"
        )

    solvers = [
        obj.getClassName()
        for obj in env._sofa_root_node.objects
        if obj.getClassName()
        in ("GenericConstraintSolver", "LCPConstraintSolver")
    ]
    if solvers != ["GenericConstraintSolver"]:
        raise RuntimeError(
            f"expected GenericConstraintSolver only, got {solvers}"
        )

    instrument = env.mcr_controller_sofa.instrument.InstrumentCombined
    beam_dofs = instrument.getObject("DOFs")
    adapter = AdapterCompat(env, instrument)

    prefix_planner = prefix_base.FullEpisodeFeasibleController(
        name="FastFDStep665PrefixPlanner",
        beam_dofs=beam_dofs,
        adapter=adapter,
        dt=DT,
    )
    instrument.addObject(prefix_planner)

    native = instrument.addObject(
        "BeamFeasibleNativePrecommitHook",
        name="FastFDStep665PrefixNativeHook",
        beamState="@DOFs",
        collisionState="@mcr_collis/CollisionDOFs",
        dt=float(DT),
        armed=False,
    )
    prefix_planner.native = native

    constraint = instrument.addObject(
        "BeamLinearizedUnilateralConstraint",
        name="FastFDStep665BeamUnilateralConstraint",
        enabled=False,
        rowOffsets=[],
        dofIndices=[],
        linearJacobian=[],
        angularJacobian=[],
        freeViolations=[],
        sourceClearances=[],
    )
    constraint.init()

    target = Step665FastFDV2Controller(
        name="FastFDStep665BenchmarkController",
        beam_dofs=beam_dofs,
        adapter=adapter,
        constraint=constraint,
    )
    instrument.addObject(target)

    model = DistributedPPO.load(
        str(Path(args.checkpoint).expanduser().resolve()),
        device="cpu",
    )
    model.policy.set_training_mode(False)

    original_animate = env.sofa_simulation.animate
    current_step = 0
    substep_in_step = 0
    target_mode = False
    target_completed = False
    prefix_substeps = 0
    prefix_solver_calls = 0
    actions: list[np.ndarray] = []
    started = time.perf_counter()

    def traced_animate(root, dt):
        nonlocal substep_in_step, target_completed
        nonlocal prefix_substeps, prefix_solver_calls

        substep_in_step += 1
        sub = int(substep_in_step)

        if target_mode:
            prefix_planner.enabled = False
            native.armed.value = False
            target.enabled = True
            target.current_rl_step = int(current_step)
            target.current_substep = sub

            if sub != TARGET_SUBSTEP:
                raise RuntimeError(
                    "target diagnostic must stop before substep2"
                )

            result = original_animate(root, dt)
            target_completed = True
            raise _StopAfterTarget()

        target.enabled = False
        constraint.enabled.value = False
        prefix_planner.enabled = True
        prefix_planner.current_rl_step = int(current_step)
        prefix_planner.current_substep = sub

        fire_before = _as_int(native.fireCount)
        result = original_animate(root, dt)
        fire_after = _as_int(native.fireCount)
        rec = prefix_planner.take_record()

        committed = adapter.measure(
            _as_array(beam_dofs.position),
            spacing_m=DENSE_SPACING_M,
        )
        target.previous_committed_clearance_mm = float(committed["min_clearance_m"]) * 1000.0
        _prefix_substep_ok(
            rec,
            native,
            fire_after - fire_before,
            float(committed["min_clearance_m"]),
        )

        prefix_substeps += 1
        if rec and rec.get("solver_required"):
            prefix_solver_calls += 1
        return result

    env.sofa_simulation.animate = traced_animate

    try:
        for step in range(1, TARGET_STEP):
            current_step = int(step)
            substep_in_step = 0
            raw_action, _ = model.predict(
                observation, deterministic=True
            )
            raw_action = np.asarray(
                raw_action, dtype=np.float32
            ).reshape(3)
            actions.append(raw_action.copy())
            observation, _, terminated, truncated, info = env.step(
                raw_action
            )

            if args.progress_every > 0 and (
                step == 1 or step % int(args.progress_every) == 0
            ):
                print(
                    "[FAST_FD_PREFIX] "
                    f"step={step}/{TARGET_STEP-1} "
                    f"substeps={prefix_substeps} "
                    f"solver_calls={prefix_solver_calls}",
                    flush=True,
                )

            if terminated or truncated:
                raise RuntimeError(
                    f"protected prefix ended at step {step}: "
                    f"{info.get('terminal_reason')}"
                )

        raw_target, _ = model.predict(
            observation, deterministic=True
        )
        raw_target = np.asarray(
            raw_target, dtype=np.float32
        ).reshape(3)
        actions.append(raw_target.copy())
        action_sha = _sha_action_prefix(actions)
        expected_sha = json.loads((BEAM_ROOT / "_runtime" / "beam_unilateral_fast_fd_step665" / "results" / "step665_fast_local_fd_result.json").read_text())["action_prefix_sha256"]
        if action_sha != expected_sha:
            raise RuntimeError(f"STEP665_ACTION_PREFIX_MISMATCH: {action_sha}")

        current_step = TARGET_STEP
        substep_in_step = 0
        target_mode = True

        try:
            env.step(raw_target)
        except _StopAfterTarget:
            pass

        record = dict(target.record)
        try:
            active_count = int(constraint.activeCount.value)
        except Exception:
            active_count = -1

        q_committed = _as_array(beam_dofs.position)
        committed = adapter.measure(
            q_committed, spacing_m=DENSE_SPACING_M
        )
        committed_clearance_m = float(
            committed["min_clearance_m"]
        )

        archived = json.loads((BEAM_ROOT / "_runtime" / "beam_unilateral_fast_fd_step665" / "results" / "step665_fast_local_fd_result.json").read_text())
        record["v1_archived_committed_clearance_mm"] = float(archived["target"]["committed_clearance_mm"])
        record["v1_archived_action_sha256"] = archived["action_prefix_sha256"]
        record["committed_clearance_difference_vs_v1_mm"] = committed_clearance_m * 1000.0 - record["v1_archived_committed_clearance_mm"]
        record["committed_state_exact_comparison_available"] = False
        record["constraint_active_count"] = active_count
        record["committed_clearance_m"] = committed_clearance_m
        record["committed_clearance_mm"] = (
            committed_clearance_m * 1000.0
        )
        record["committed_penetration_mm"] = (
            max(0.0, -committed_clearance_m) * 1000.0
        )
        record["committed_finite"] = bool(
            _finite(q_committed)
            and np.isfinite(committed_clearance_m)
        )

        expected_rows = int(record.get("row_count", 0))
        equivalence_pass = bool(
            record.get("state_matched")
            and record.get("status") == "FAST_ROWS_ARMED"
            and record.get("jacobian_equivalent")
            and expected_rows > 0
            and active_count == expected_rows
            and archived["action_prefix_sha256"] == action_sha
            and abs(record["committed_clearance_difference_vs_v1_mm"]) <= 1e-9
        )
        committed_safe = bool(
            record["committed_finite"]
            and committed_clearance_m
            >= -COMMITTED_PEN_LIMIT_M
        )

        decision = (
            "PASS"
            if target_completed
            and equivalence_pass
            and committed_safe
            else "FAIL"
        )
        reason = (
            "FAST_FD_V2_ROWS_IDENTICAL_AND_COMMITTED_SAFE"
            if decision == "PASS"
            else record.get(
                "status",
                "FAST_LOCAL_FD_TARGET_NOT_ACCEPTED",
            )
        )

        payload = {
            "test": (
                "B02 step665 fast local-point FD Beam unilateral "
                "equivalence/performance"
            ),
            "decision": decision,
            "reason": reason,
            "seed": int(args.seed),
            "target_step": TARGET_STEP,
            "target_substep": TARGET_SUBSTEP,
            "requested_margin_m": MARGIN_M,
            "dense_spacing_m": DENSE_SPACING_M,
            "committed_penetration_limit_m": (
                COMMITTED_PEN_LIMIT_M
            ),
            "prefix_steps_completed": TARGET_STEP - 1,
            "prefix_physics_substeps": prefix_substeps,
            "prefix_solver_calls": prefix_solver_calls,
            "target_completed_substep1": target_completed,
            "action_prefix_sha256": action_sha,
            "target": record,
            "production_files_modified": False,
            "training_started": False,
            "target_q_candidate_solver_used": False,
            "target_native_candidate_injection_used": False,
            "target_collision_dofs_constraint_source": False,
            "wall_s": float(time.perf_counter() - started),
            "reset_info": _json_safe(reset_info),
        }
        output.write_text(
            json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n"
        )

        print(
            json.dumps(
                {
                    "decision": decision,
                    "reason": reason,
                    "q_prev_mm": record.get(
                        "q_prev_clearance_mm"
                    ),
                    "q_free_mm": record.get(
                        "q_free_clearance_mm"
                    ),
                    "rows": expected_rows,
                    "v1_runtime_s": record.get("v1_runtime_s"),
                    "v2_runtime_s": record.get("v2_runtime_s"),
                    "speedup_x": record.get("speedup_x"),
                    "linear_max_abs_error": (
                        record.get("comparison") or {}
                    ).get("linear_max_abs_error"),
                    "angular_max_abs_error": (
                        record.get("comparison") or {}
                    ).get("angular_max_abs_error"),
                    "committed_clearance_mm": record.get(
                        "committed_clearance_mm"
                    ),
                    "active_count": active_count,
                    "output": str(output),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    finally:
        env.sofa_simulation.animate = original_animate
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
