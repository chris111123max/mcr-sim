#!/usr/bin/env python3
"""Step665 same-state Beam-level SDF unilateral constraint solver A/B.

Prefix:
  current dense-aware strict 0.100 mm feasible solver through RL step 664.

Fork:
  same in-memory SOFA state and same step665 policy action.

Branches:
  G: GenericConstraintSolver + BeamLinearizedUnilateralConstraint
  L: LCPConstraintSolver     + BeamLinearizedUnilateralConstraint

At step665/substep1 the SLSQP q_candidate solver is DISABLED.  A CollisionBegin
controller linearizes the real BeamAdapter 10-um SDF clearance directly with
respect to the Rigid3 Beam DOFs and supplies unilateral g>=0 rows to SOFA.

This is test-only.  It does not write Beam free/committed positions and does
not use CollisionDOFs as the constraint source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

THIS = Path(__file__).resolve()
TEST_DIR = THIS.parent
BEAM_ROOT = THIS.parents[1]
PYTHON_ROOT = THIS.parents[3]
UNILATERAL_DIR = BEAM_ROOT / "beam_unilateral_lcp"
RUNTIME = BEAM_ROOT / "_runtime" / "beam_unilateral_lcp_step665"
DEFAULT_BEAM_BUILD = RUNTIME / "native_build"
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
    Step665BeamUnilateralController,
    find_plugin as find_beam_plugin,
    load_plugin as load_beam_plugin,
)
from mcr_sim.distributed import DistributedPPO
from mcr_sim.training_config import (
    CONSTRAINT_MAX_ITERATIONS,
    CONSTRAINT_TOLERANCE,
    FRICTION_COEFFICIENT,
)


TARGET_STEP = 665
TARGET_SUBSTEP = 1
EXPECTED_QPREV_MM = 0.468945
EXPECTED_QFREE_MM = -0.032075
STATE_TOL_MM = 0.002
COMMITTED_PEN_LIMIT_M = 1.0e-6
EXPECTED_SUBSTEPS = 2


class _StopAfterTarget(Exception):
    pass


def _sha_action_prefix(actions: list[np.ndarray]) -> str:
    h = hashlib.sha256()
    for action in actions:
        h.update(np.asarray(action, dtype="<f4").reshape(3).tobytes(order="C"))
    return h.hexdigest()


def _state_fingerprint(env) -> str:
    controller = env.mcr_controller_sofa
    instrument = controller.instrument.InstrumentCombined
    beam = instrument.getObject("DOFs")
    collision = instrument.getChild("mcr_collis").getObject("CollisionDOFs")
    chunks = [
        _as_array(beam.position).tobytes(),
        _as_array(beam.free_position).tobytes(),
        _as_array(beam.velocity).tobytes(),
        _as_array(collision.position).tobytes(),
        _as_array(collision.free_position).tobytes(),
        np.asarray(
            [
                float(controller._getXTipValue()),
                float(controller.pending_insert_delta),
                float(env._sofa_root_node.getTime()),
            ],
            dtype=np.float64,
        ).tobytes(),
        np.asarray(env._last_smoothed_action, dtype=np.float32).tobytes(),
    ]
    return hashlib.sha256(b"".join(chunks)).hexdigest()


def _root_object_by_class(env, names):
    wanted = set(names)
    for obj in env._sofa_root_node.objects:
        if obj.getClassName() in wanted:
            return obj
    return None


def _solver_summary(env) -> dict:
    solver = _root_object_by_class(
        env, ("GenericConstraintSolver", "LCPConstraintSolver")
    )
    if solver is None:
        return {"class": None}
    return {"class": solver.getClassName()}


def _switch_to_lcp(env, expected_fingerprint: str) -> dict:
    before = _state_fingerprint(env)
    if before != expected_fingerprint:
        raise RuntimeError("fork state changed before LCP switch")

    root = env._sofa_root_node
    old = _root_object_by_class(
        env, ("GenericConstraintSolver", "LCPConstraintSolver")
    )
    loop = _root_object_by_class(env, ("FreeMotionAnimationLoop",))
    if old is None or loop is None:
        raise RuntimeError("constraint solver or FreeMotionAnimationLoop missing")

    old_class = old.getClassName()
    if old_class == "LCPConstraintSolver":
        return {
            "old": old_class,
            "new": old_class,
            "state_unchanged": True,
        }

    cleanup = getattr(old, "cleanup", None)
    if callable(cleanup):
        cleanup()
    root.removeObject(old)

    lcp = root.addObject(
        "LCPConstraintSolver",
        mu=str(FRICTION_COEFFICIENT),
        tolerance=str(CONSTRAINT_TOLERANCE),
        maxIt=str(CONSTRAINT_MAX_ITERATIONS),
        build_lcp="false",
    )
    lcp.init()
    loop.init()

    after = _state_fingerprint(env)
    if after != expected_fingerprint:
        raise RuntimeError("switching Generic -> LCP changed fork state")

    return {
        "old": old_class,
        "new": lcp.getClassName(),
        "state_unchanged": True,
        "mu": float(FRICTION_COEFFICIENT),
        "tolerance": float(CONSTRAINT_TOLERANCE),
        "maxIt": int(CONSTRAINT_MAX_ITERATIONS),
    }


def _prefix_substep_ok(rec, native, fire_delta, committed_clearance_m) -> None:
    if rec is None:
        raise RuntimeError("prefix CollisionBeginEvent planner record missing")

    planning = str(rec.get("planning_status", ""))
    if planning in ("FAIL", "INCONCLUSIVE"):
        raise RuntimeError(
            f"prefix feasible planner failed: {planning}: {rec.get('reason')}"
        )

    if bool(rec.get("solver_required")):
        accepted = float(rec.get("accepted_clearance_m", np.nan))
        if not np.isfinite(accepted) or accepted + 1e-12 < MARGIN_M:
            raise RuntimeError(
                f"prefix candidate below strict margin: {accepted}"
            )
        if int(fire_delta) != 1:
            raise RuntimeError(
                f"prefix native hook fire delta {fire_delta}, expected 1"
            )
        if _as_str(native.status) != "PASS_NATIVE_WRITE_AND_PROPAGATE":
            raise RuntimeError(
                f"prefix native status failed: {_as_str(native.status)}"
            )
    elif int(fire_delta) != 0:
        raise RuntimeError("unexpected prefix native injection on safe free state")

    if (
        not np.isfinite(committed_clearance_m)
        or committed_clearance_m < -COMMITTED_PEN_LIMIT_M
    ):
        raise RuntimeError(
            f"prefix committed clearance violation: "
            f"{committed_clearance_m * 1000.0} mm"
        )


def _branch_result(
    *,
    env,
    adapter,
    unilateral_controller,
    constraint,
    branch: str,
    solver_switch,
    target_completed: bool,
    action_sha: str,
    fork_fingerprint: str,
    target_wall_s: float,
) -> dict:
    q_committed = _as_array(
        env.mcr_controller_sofa.instrument.InstrumentCombined
        .getObject("DOFs").position
    )
    committed = adapter.measure(q_committed, spacing_m=DENSE_SPACING_M)
    record = dict(unilateral_controller.record)
    active_count = int(constraint.activeCount.value)

    state_matched = bool(record.get("state_matched", False))
    rows_armed = record.get("status") == "ROWS_ARMED"
    committed_clearance = float(committed["min_clearance_m"])
    safe = (
        target_completed
        and state_matched
        and rows_armed
        and active_count > 0
        and np.isfinite(committed_clearance)
        and committed_clearance >= -COMMITTED_PEN_LIMIT_M
    )

    return {
        "branch": branch,
        "solver": _solver_summary(env),
        "solver_switch": solver_switch,
        "fork_fingerprint_sha256": fork_fingerprint,
        "action_prefix_sha256": action_sha,
        "target_completed_substep1": bool(target_completed),
        "constraint_component_class": constraint.getClassName(),
        "constraint_active_count": active_count,
        "target_record": record,
        "committed_dense": committed,
        "committed_penetration_m": max(0.0, -committed_clearance),
        "finite": bool(_finite(q_committed) and np.isfinite(committed_clearance)),
        "collision_dofs_used_as_constraint_source": False,
        "q_candidate_solver_used_at_target": False,
        "free_position_written_at_target": False,
        "committed_position_written_at_target": False,
        "target_wall_s": float(target_wall_s),
        "decision": "PASS" if safe else "FAIL",
        "reason": (
            "BEAM_LEVEL_UNILATERAL_COMMITTED_SAFE"
            if safe
            else "BEAM_LEVEL_UNILATERAL_TARGET_NOT_ACCEPTED"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(CHECKPOINT))
    parser.add_argument("--seed", type=int, default=15204)
    parser.add_argument("--beam-build-dir", default=str(DEFAULT_BEAM_BUILD))
    parser.add_argument("--beam-plugin-lib", default=None)
    parser.add_argument("--hook-build-dir", default=str(DEFAULT_HOOK_BUILD))
    parser.add_argument("--hook-plugin-lib", default=None)
    parser.add_argument(
        "--output",
        default=str(RUNTIME / "results" / "step665_beam_unilateral_ab.json"),
    )
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    beam_plugin = (
        Path(args.beam_plugin_lib).expanduser().resolve()
        if args.beam_plugin_lib
        else find_beam_plugin(Path(args.beam_build_dir).expanduser().resolve())
    )
    hook_plugin = (
        Path(args.hook_plugin_lib).expanduser().resolve()
        if args.hook_plugin_lib
        else find_hook_plugin(Path(args.hook_build_dir).expanduser().resolve())
    )
    if beam_plugin is None:
        raise RuntimeError("MCRBeamLinearizedUnilateral plugin not built")
    if hook_plugin is None:
        raise RuntimeError("MCRBeamFeasibleHook prefix plugin not built")

    _beam_handle = load_beam_plugin(beam_plugin)
    _hook_handle = load_hook_plugin(hook_plugin)

    env = _create_env()
    observation, reset_info = env.reset(seed=int(args.seed))
    if int(env.physics_substeps) != EXPECTED_SUBSTEPS:
        raise RuntimeError(
            f"physics_substeps={env.physics_substeps}, expected {EXPECTED_SUBSTEPS}"
        )

    instrument = env.mcr_controller_sofa.instrument.InstrumentCombined
    beam_dofs = instrument.getObject("DOFs")
    collision_dofs = instrument.getChild("mcr_collis").getObject("CollisionDOFs")
    adapter = AdapterCompat(env, instrument)

    prefix_planner = prefix_base.FullEpisodeFeasibleController(
        name="Step665PrefixStrictPlanner",
        beam_dofs=beam_dofs,
        adapter=adapter,
        dt=DT,
    )
    instrument.addObject(prefix_planner)

    native = instrument.addObject(
        "BeamFeasibleNativePrecommitHook",
        name="Step665PrefixNativeHook",
        beamState="@DOFs",
        collisionState="@mcr_collis/CollisionDOFs",
        dt=float(DT),
        armed=False,
    )
    prefix_planner.native = native

    constraint = instrument.addObject(
        "BeamLinearizedUnilateralConstraint",
        name="Step665BeamLinearizedUnilateralConstraint",
        enabled=False,
        rowOffsets=[],
        dofIndices=[],
        linearJacobian=[],
        angularJacobian=[],
        freeViolations=[],
        sourceClearances=[],
    )
    constraint.init()

    unilateral_controller = Step665BeamUnilateralController(
        name="Step665BeamUnilateralRows",
        beam_dofs=beam_dofs,
        adapter=adapter,
        constraint=constraint,
        expected_qprev_mm=EXPECTED_QPREV_MM,
        expected_qfree_mm=EXPECTED_QFREE_MM,
        state_tol_mm=STATE_TOL_MM,
    )
    instrument.addObject(unilateral_controller)

    model = DistributedPPO.load(
        str(Path(args.checkpoint).expanduser().resolve()), device="cpu"
    )
    model.policy.set_training_mode(False)

    original_animate = env.sofa_simulation.animate
    current_step = 0
    substep_in_step = 0
    target_mode = False
    target_completed = False
    prefix_substeps = 0
    prefix_solver_calls = 0

    def traced_animate(root, dt):
        nonlocal substep_in_step, target_completed
        nonlocal prefix_substeps, prefix_solver_calls

        substep_in_step += 1
        sub = int(substep_in_step)

        if target_mode:
            prefix_planner.enabled = False
            native.armed.value = False
            unilateral_controller.enabled = True
            unilateral_controller.current_rl_step = int(current_step)
            unilateral_controller.current_substep = sub
            if sub != TARGET_SUBSTEP:
                raise RuntimeError("target diagnostic must stop before substep2")

            result = original_animate(root, dt)
            target_completed = True
            raise _StopAfterTarget()

        prefix_planner.enabled = True
        unilateral_controller.enabled = False
        constraint.enabled.value = False
        prefix_planner.current_rl_step = int(current_step)
        prefix_planner.current_substep = sub

        fire_before = _as_int(native.fireCount)
        result = original_animate(root, dt)
        fire_after = _as_int(native.fireCount)
        rec = prefix_planner.take_record()
        committed = adapter.measure(
            _as_array(beam_dofs.position), spacing_m=DENSE_SPACING_M
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

    actions: list[np.ndarray] = []
    started = time.perf_counter()

    try:
        for step in range(1, TARGET_STEP):
            current_step = int(step)
            substep_in_step = 0
            raw_action, _ = model.predict(observation, deterministic=True)
            raw_action = np.asarray(raw_action, dtype=np.float32).reshape(3)
            actions.append(raw_action.copy())
            observation, _, terminated, truncated, info = env.step(raw_action)

            if args.progress_every > 0 and (
                step == 1 or step % int(args.progress_every) == 0
            ):
                print(
                    f"[BEAM_UNILATERAL_PREFIX] step={step}/{TARGET_STEP-1} "
                    f"substeps={prefix_substeps} solver_calls={prefix_solver_calls}",
                    flush=True,
                )
            if terminated or truncated:
                raise RuntimeError(
                    f"protected prefix ended at step {step}: "
                    f"{info.get('terminal_reason')}"
                )

        fork_fingerprint = _state_fingerprint(env)
        raw_target, _ = model.predict(observation, deterministic=True)
        raw_target = np.asarray(raw_target, dtype=np.float32).reshape(3)
        target_actions = actions + [raw_target.copy()]
        action_sha = _sha_action_prefix(target_actions)

        results_dir = output.parent
        generic_file = results_dir / "step665_generic_beam_unilateral.json"
        lcp_file = results_dir / "step665_lcp_beam_unilateral.json"
        try:
            lcp_file.unlink()
        except FileNotFoundError:
            pass

        child = os.fork()
        if child == 0:
            try:
                switch = _switch_to_lcp(env, fork_fingerprint)
                current_step = TARGET_STEP
                substep_in_step = 0
                target_mode = True
                target_completed = False
                t0 = time.perf_counter()
                try:
                    env.step(raw_target)
                except _StopAfterTarget:
                    pass
                result = _branch_result(
                    env=env,
                    adapter=adapter,
                    unilateral_controller=unilateral_controller,
                    constraint=constraint,
                    branch="L_LCP_BEAM_UNILATERAL",
                    solver_switch=switch,
                    target_completed=target_completed,
                    action_sha=action_sha,
                    fork_fingerprint=fork_fingerprint,
                    target_wall_s=time.perf_counter() - t0,
                )
                lcp_file.write_text(
                    json.dumps(result, indent=2, sort_keys=True) + "\n"
                )
                os._exit(0)
            except BaseException:
                lcp_file.write_text(
                    json.dumps(
                        {"decision": "ERROR", "error": traceback.format_exc()},
                        indent=2,
                    )
                    + "\n"
                )
                os._exit(1)

        current_step = TARGET_STEP
        substep_in_step = 0
        target_mode = True
        target_completed = False
        t0 = time.perf_counter()
        try:
            env.step(raw_target)
        except _StopAfterTarget:
            pass
        generic_result = _branch_result(
            env=env,
            adapter=adapter,
            unilateral_controller=unilateral_controller,
            constraint=constraint,
            branch="G_GENERIC_BEAM_UNILATERAL",
            solver_switch={
                "old": _solver_summary(env)["class"],
                "new": _solver_summary(env)["class"],
                "state_unchanged": True,
            },
            target_completed=target_completed,
            action_sha=action_sha,
            fork_fingerprint=fork_fingerprint,
            target_wall_s=time.perf_counter() - t0,
        )
        generic_file.write_text(
            json.dumps(generic_result, indent=2, sort_keys=True) + "\n"
        )

        _, child_status = os.waitpid(child, 0)
        if not lcp_file.is_file():
            lcp_result = {
                "decision": "ERROR",
                "error": f"LCP child exited status={child_status} without JSON",
            }
        else:
            lcp_result = json.loads(lcp_file.read_text())

        comparison = {
            "same_fork_state": bool(
                generic_result.get("fork_fingerprint_sha256")
                == lcp_result.get("fork_fingerprint_sha256")
                == fork_fingerprint
            ),
            "same_action_prefix_sha256": bool(
                generic_result.get("action_prefix_sha256")
                == lcp_result.get("action_prefix_sha256")
                == action_sha
            ),
            "generic_decision": generic_result.get("decision"),
            "lcp_decision": lcp_result.get("decision"),
            "generic_committed_clearance_mm": (
                generic_result.get("committed_dense") or {}
            ).get("min_clearance_mm"),
            "lcp_committed_clearance_mm": (
                lcp_result.get("committed_dense") or {}
            ).get("min_clearance_mm"),
            "generic_active_rows": generic_result.get(
                "constraint_active_count"
            ),
            "lcp_active_rows": lcp_result.get("constraint_active_count"),
        }

        payload = {
            "test": "B02 step665 Beam-level SDF unilateral same-state Generic/LCP A/B",
            "seed": int(args.seed),
            "target_step": TARGET_STEP,
            "target_substep": TARGET_SUBSTEP,
            "requested_margin_m": MARGIN_M,
            "dense_spacing_m": DENSE_SPACING_M,
            "committed_penetration_limit_m": COMMITTED_PEN_LIMIT_M,
            "prefix_solver": "strict dense-aware 0.100 mm feasible-domain",
            "prefix_steps_completed": TARGET_STEP - 1,
            "prefix_physics_substeps": prefix_substeps,
            "prefix_solver_calls": prefix_solver_calls,
            "fork_fingerprint_sha256": fork_fingerprint,
            "action_prefix_sha256": action_sha,
            "beam_plugin": str(beam_plugin),
            "hook_plugin": str(hook_plugin),
            "generic": generic_result,
            "lcp": lcp_result,
            "comparison": comparison,
            "production_files_modified": False,
            "training_started": False,
            "target_q_candidate_solver_used": False,
            "collision_dofs_constraint_source": False,
            "wall_s": float(time.perf_counter() - started),
        }
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

        print(
            json.dumps(
                {
                    "comparison": comparison,
                    "output": str(output),
                    "generic_file": str(generic_file),
                    "lcp_file": str(lcp_file),
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
