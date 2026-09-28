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
RUNTIME = BEAM_ROOT / "_runtime" / "beam_unilateral_fast_fd_step665"
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
from beam_unilateral_fast_fd import (
    build_rows_fast_local_fd,
    build_rows_local_support,
    compare_row_builds,
)
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


class Step665FastFDBenchmarkController(Sofa.Core.Controller):
    def __init__(
        self,
        *,
        beam_dofs,
        adapter,
        constraint,
        **kwargs,
    ):
        kwargs["listening"] = True
        Sofa.Core.Controller.__init__(self, **kwargs)
        self.beam_dofs = beam_dofs
        self.adapter = adapter
        self.constraint = constraint
        self.current_rl_step = 0
        self.current_substep = 0
        self.enabled = False
        self.processed = False
        self.record: dict[str, Any] = {}

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

    def _collision_begin(self, source: str) -> None:
        if (
            not self.enabled
            or self.processed
            or self.current_rl_step != TARGET_STEP
            or self.current_substep != TARGET_SUBSTEP
        ):
            return

        self.processed = True
        q_prev = _as_array(self.beam_dofs.position)
        q_free = _as_array(self.beam_dofs.free_position)

        prev = self.adapter.measure(q_prev, spacing_m=DENSE_SPACING_M)
        free = self.adapter.measure(q_free, spacing_m=DENSE_SPACING_M)
        prev_mm = float(prev["min_clearance_m"]) * 1000.0
        free_mm = float(free["min_clearance_m"]) * 1000.0

        self.record = {
            "event_source": source,
            "q_prev_clearance_mm": prev_mm,
            "q_free_clearance_mm": free_mm,
            "state_matched": bool(
                abs(prev_mm - EXPECTED_QPREV_MM) <= STATE_TOL_MM
                and abs(free_mm - EXPECTED_QFREE_MM) <= STATE_TOL_MM
            ),
            "target_q_candidate_solver_used": False,
            "target_native_candidate_injection_used": False,
            "uses_collision_dofs_as_constraint_source": False,
            "writes_free_position": False,
            "writes_committed_position": False,
        }

        if not self.record["state_matched"]:
            self.record["status"] = "STEP665_PROTECTED_STATE_MISMATCH"
            self.constraint.enabled.value = False
            return

        t0 = time.perf_counter()
        baseline = build_rows_local_support(
            adapter=self.adapter,
            q_prev=q_prev,
            q_free=q_free,
            requested_margin_m=MARGIN_M,
        )
        baseline_s = float(time.perf_counter() - t0)

        t1 = time.perf_counter()
        fast = build_rows_fast_local_fd(
            adapter=self.adapter,
            q_prev=q_prev,
            q_free=q_free,
            requested_margin_m=MARGIN_M,
        )
        fast_s = float(time.perf_counter() - t1)

        comparison = compare_row_builds(baseline, fast)
        jacobian_equivalent = _jacobian_equivalent(baseline, fast)
        speedup = baseline_s / fast_s if fast_s > 0.0 else float("inf")

        self.record.update(
            {
                "status": "FAST_ROWS_ARMED",
                "baseline_runtime_s": baseline_s,
                "fast_runtime_s": fast_s,
                "speedup_x": speedup,
                "jacobian_equivalent": jacobian_equivalent,
                "comparison": comparison,
                "row_count": len(fast["free_violations"]),
                "dense_sample_count": int(fast["dense_sample_count"]),
                "selected_dense_indices": list(
                    fast["selected_dense_indices"]
                ),
                "selected_element_indices": list(
                    fast["selected_element_indices"]
                ),
                "row_support_nodes": list(fast["row_support_nodes"]),
                "selected_clearances_mm": [
                    float(v) * 1000.0
                    for v in fast["selected_clearances_m"]
                ],
                "minimum_free_violation_mm": (
                    float(min(fast["free_violations"])) * 1000.0
                ),
                "batched_fd_point_count": int(
                    fast["batched_fd_point_count"]
                ),
                "batched_fd_sdf_query_count": int(
                    fast["batched_fd_sdf_query_count"]
                ),
                "selected_point_reconstruction_max_error_m": float(
                    fast["support_metadata"][
                        "selected_point_reconstruction_max_error_m"
                    ]
                ),
                "full_beam_profile_evaluations_after_selection": int(
                    fast["full_beam_profile_evaluations_after_selection"]
                ),
            }
        )

        if not comparison["all_structure_equal"]:
            self.record["status"] = "FAST_STRUCTURE_MISMATCH"
            self.constraint.enabled.value = False
            return
        if not comparison["source_clearances_equal"]:
            self.record["status"] = "FAST_SOURCE_CLEARANCE_MISMATCH"
            self.constraint.enabled.value = False
            return
        if not jacobian_equivalent:
            self.record["status"] = "FAST_JACOBIAN_MISMATCH"
            self.constraint.enabled.value = False
            return

        # Only the fast rows participate in the target Generic solve.
        write_rows(self.constraint, fast)


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
            RUNTIME / "results" / "step665_fast_local_fd_result.json"
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

    target = Step665FastFDBenchmarkController(
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
            "FAST_LOCAL_FD_EQUIVALENT_AND_COMMITTED_SAFE"
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
            "reset_info": reset_info,
        }
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
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
                    "baseline_runtime_s": record.get(
                        "baseline_runtime_s"
                    ),
                    "fast_runtime_s": record.get(
                        "fast_runtime_s"
                    ),
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
