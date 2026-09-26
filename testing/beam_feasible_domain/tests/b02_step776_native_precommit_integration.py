#!/usr/bin/env python3
"""B02 step776/substep1 native pre-commit mapping validation.

This is the gate after the Python CollisionBeginEvent hook proved that the
event timing is correct but could not refresh MultiAdaptiveBeamMapping through
its Python binding.

At exactly B02/target04/seed15204/RL-step776/physics-substep1, a test-only
native SOFA component:
  1. captures the live unsafe Rigid3 Beam free_position;
  2. replaces it with the already-PASSed offline feasible candidate;
  3. updates free_velocity coherently in Rigid3 tangent space;
  4. propagates freePosition/freeVelocity through SOFA's native mechanical
     mapping visitor;
  5. lets the ordinary collision/constraint phase of the SAME substep finish.

The runner intentionally aborts env.step before its second animate call, so no
physics substep2 is executed. CollisionDOFs are diagnostics only.
"""
from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

THIS = Path(__file__).resolve()
TEST_DIR = THIS.parent
BEAM_ROOT = THIS.parents[1]
RESULTS_DIR = BEAM_ROOT / "_runtime" / "results"
DEFAULT_BUILD_DIR = BEAM_ROOT / "_runtime" / "native_build"

if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from b02_step776_precommit_integration import (
    AdapterCompat,
    CHECKPOINT,
    DT,
    EXPECTED_ACTION_SHA256,
    NUM_TOL_M,
    TARGET_STEP,
    _as_array,
    _create_env,
    _finite,
    _load_candidate_artifact,
    _sha256_actions,
    _state_delta,
)
from mcr_sim.distributed import DistributedPPO


TARGET_SUBSTEP = 1


class _StopBeforeSecondTargetAnimate(RuntimeError):
    """Internal control-flow exception; raised before substep2 starts."""


def _scalar(data: Any) -> Any:
    try:
        return data.value
    except Exception:
        return data


def _as_bool(data: Any) -> bool:
    value = _scalar(data)
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value).reshape(-1)
        return bool(arr[0]) if arr.size else False
    return bool(value)


def _as_float(data: Any) -> float:
    value = _scalar(data)
    return float(np.asarray(value, dtype=np.float64).reshape(-1)[0])


def _as_str(data: Any) -> str:
    value = _scalar(data)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _find_plugin(build_dir: Path) -> Path | None:
    for pattern in (
        "libMCRBeamFeasibleHook.so",
        "MCRBeamFeasibleHook.so",
        "libMCRBeamFeasibleHook.dylib",
        "MCRBeamFeasibleHook.dll",
    ):
        hits = sorted(build_dir.rglob(pattern))
        if hits:
            return hits[0]
    return None


def _load_plugin(plugin: Path) -> Any:
    if not plugin.is_file():
        raise FileNotFoundError(plugin)
    mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    return ctypes.CDLL(str(plugin), mode=mode)


def _native_payload(obj: Any) -> dict[str, Any]:
    return {
        "status": _as_str(obj.status),
        "armed": _as_bool(obj.armed),
        "fired": _as_bool(obj.fired),
        "propagated": _as_bool(obj.propagated),
        "velocity_corrected": _as_bool(obj.velocityCorrected),
        "mapped_child_max_change_mm": _as_float(obj.mappedChildMaxChangeMm),
        "parent_write_max_error_mm": _as_float(obj.parentWriteMaxErrorMm),
    }


def _write_report(payload: dict[str, Any], path: Path) -> None:
    native = payload.get("native_hook", {})
    final = payload.get("final_committed", {})
    dense = payload.get("final_independent_dense", {})
    free = payload.get("captured_free", {})
    lines = [
        "B02 STEP776 NATIVE PRE-COMMIT MAPPING HOOK",
        "==========================================",
        "",
        f"Action prefix: {payload.get('action_prefix_status')}",
        f"SHA256: {payload.get('action_sha256')}",
        f"Completed physics substeps at target: {payload.get('target_completed_substeps')}",
        f"Second target substep executed: {payload.get('target_substep2_executed')}",
        "",
        f"Native hook fired: {native.get('fired')}",
        f"Native propagation ran: {native.get('propagated')}",
        f"Velocity corrected coherently: {native.get('velocity_corrected')}",
        f"Native status: {native.get('status')}",
        f"Parent candidate write error: {native.get('parent_write_max_error_mm')} mm",
        f"Mapped CollisionDOF free-state change: {native.get('mapped_child_max_change_mm')} mm",
        "NOTE: CollisionDOFs are diagnostic only, never a feasibility source.",
        "",
        f"Captured unsafe free clearance: {free.get('min_clearance_mm')} mm",
        f"Final committed clearance: {final.get('min_clearance_mm')} mm",
        f"Independent dense clearance: {dense.get('min_clearance_mm')} mm",
        f"NaN/Inf: {payload.get('non_finite')}",
        "",
        f"FINAL DECISION: {payload.get('decision')}",
        f"Reason: {payload.get('reason')}",
        "",
        "This validates at most one physical substep. It does not establish multi-step invariance.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


def _write_and_exit(
    payload: dict[str, Any],
    output: Path,
    report: Path,
    env: Any | None = None,
) -> None:
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_report(payload, report)
    print(json.dumps(payload, sort_keys=True), flush=True)
    if env is not None:
        try:
            env.close()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(CHECKPOINT))
    parser.add_argument("--seed", type=int, default=15204)
    parser.add_argument("--build-dir", default=str(DEFAULT_BUILD_DIR))
    parser.add_argument("--plugin-lib", default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--output",
        default=str(RESULTS_DIR / "b02_step776_native_precommit_integration.json"),
    )
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = output.with_name("b02_step776_native_precommit_integration_report.md")

    build_dir = Path(args.build_dir).expanduser().resolve()
    plugin = (
        Path(args.plugin_lib).expanduser().resolve()
        if args.plugin_lib
        else _find_plugin(build_dir)
    )
    if plugin is None:
        _write_and_exit(
            {
                "decision": "INCONCLUSIVE",
                "reason": "NATIVE_PLUGIN_NOT_BUILT",
                "build_dir": str(build_dir),
                "training_started": False,
                "production_files_modified": False,
            },
            output,
            report,
        )
        return

    try:
        _plugin_handle = _load_plugin(plugin)
    except Exception as exc:
        _write_and_exit(
            {
                "decision": "INCONCLUSIVE",
                "reason": f"NATIVE_PLUGIN_LOAD_FAILED: {type(exc).__name__}: {exc}",
                "plugin": str(plugin),
                "training_started": False,
                "production_files_modified": False,
            },
            output,
            report,
        )
        return

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    env = _create_env()
    observation, _ = env.reset(seed=int(args.seed))
    controller = env.mcr_controller_sofa
    instrument = controller.instrument.InstrumentCombined
    beam_dofs = instrument.getObject("DOFs")
    collision_dofs = instrument.getChild("mcr_collis").getObject("CollisionDOFs")

    adapter = AdapterCompat(env, instrument)
    candidate = _load_candidate_artifact(len(_as_array(beam_dofs.position)))
    if not candidate["ok"]:
        _write_and_exit(
            {
                "decision": "INCONCLUSIVE",
                "reason": candidate["reason"],
                "plugin": str(plugin),
                "training_started": False,
                "production_files_modified": False,
            },
            output,
            report,
            env,
        )
        return

    q_acc = np.asarray(candidate["accepted"], dtype=np.float64)
    accepted_dense = adapter.measure(q_acc, spacing_m=0.00001)
    if accepted_dense["min_clearance_m"] < -NUM_TOL_M:
        _write_and_exit(
            {
                "decision": "FAIL",
                "reason": "ACCEPTED_ARTIFACT_NOT_FEASIBLE_UNDER_CURRENT_ADAPTER",
                "accepted_dense": accepted_dense,
                "training_started": False,
                "production_files_modified": False,
            },
            output,
            report,
            env,
        )
        return

    native = instrument.addObject(
        "BeamFeasibleNativePrecommitHook",
        name="BeamFeasibleNativePrecommitHook",
        beamState="@DOFs",
        collisionState="@mcr_collis/CollisionDOFs",
        candidateFreePosition=q_acc.tolist(),
        dt=float(DT),
        armed=False,
    )

    model = DistributedPPO.load(str(checkpoint), device="cpu")
    model.policy.set_training_mode(False)

    original_animate = env.sofa_simulation.animate
    current_rl_step = 0
    target_animate_attempts = 0
    target_completed_substeps = 0
    target_substep2_executed = False

    def guarded_animate(root, dt):
        nonlocal target_animate_attempts
        nonlocal target_completed_substeps
        nonlocal target_substep2_executed

        if current_rl_step != TARGET_STEP:
            return original_animate(root, dt)

        target_animate_attempts += 1
        if target_animate_attempts == 1:
            result = original_animate(root, dt)
            target_completed_substeps = 1
            return result

        # Critical gate: raise BEFORE SOFA animate is called for substep2.
        target_substep2_executed = False
        raise _StopBeforeSecondTargetAnimate(
            "target substep1 completed; intentionally stop before substep2"
        )

    env.sofa_simulation.animate = guarded_animate

    actions: list[np.ndarray] = []
    terminal_reason = None
    target_substep1_finished = False
    started = time.perf_counter()

    try:
        for step in range(1, TARGET_STEP + 1):
            current_rl_step = int(step)
            raw_action, _ = model.predict(observation, deterministic=True)
            raw_action = np.asarray(raw_action, dtype=np.float32).reshape(3)
            actions.append(raw_action.copy())

            if step == TARGET_STEP:
                prefix_sha = _sha256_actions(actions)
                if prefix_sha != EXPECTED_ACTION_SHA256:
                    _write_and_exit(
                        {
                            "decision": "FAIL",
                            "reason": "ACTION_PREFIX_MISMATCH",
                            "action_sha256": prefix_sha,
                            "expected_action_sha256": EXPECTED_ACTION_SHA256,
                            "training_started": False,
                            "production_files_modified": False,
                        },
                        output,
                        report,
                        env,
                    )
                    return
                native.armed.value = True

            try:
                observation, _, terminated, truncated, info = env.step(raw_action)
            except _StopBeforeSecondTargetAnimate:
                if step != TARGET_STEP or target_completed_substeps != 1:
                    raise
                target_substep1_finished = True
                break

            if args.progress_every > 0 and (
                step == 1
                or step % int(args.progress_every) == 0
            ):
                print(f"[NATIVE_PRECOMMIT_REPLAY] step={step}/{TARGET_STEP}", flush=True)

            if terminated or truncated:
                terminal_reason = str(
                    info.get("terminal_reason")
                    or ("terminated" if terminated else "truncated")
                )
                break
    finally:
        env.sofa_simulation.animate = original_animate

    action_sha = _sha256_actions(actions)
    native_result = _native_payload(native)

    payload: dict[str, Any] = {
        "test": "B02 step776/substep1 native pre-commit mapping hook",
        "training_started": False,
        "production_files_modified": False,
        "seed": int(args.seed),
        "target_rl_step": TARGET_STEP,
        "target_substep": TARGET_SUBSTEP,
        "physics_dt_s": DT,
        "configured_physics_substeps": int(env.physics_substeps),
        "target_animate_attempts": int(target_animate_attempts),
        "target_completed_substeps": int(target_completed_substeps),
        "target_substep2_executed": bool(target_substep2_executed),
        "plugin": str(plugin),
        "action_sha256": action_sha,
        "expected_action_sha256": EXPECTED_ACTION_SHA256,
        "action_prefix_status": (
            "PASS" if action_sha == EXPECTED_ACTION_SHA256 else "FAIL"
        ),
        "candidate_artifact": {
            "accepted_source": candidate.get("accepted_source"),
            "free_source": candidate.get("free_source"),
        },
        "accepted_independent_dense_before_replay": accepted_dense,
        "native_hook": native_result,
        "terminal_reason": terminal_reason,
        "target_substep1_finished": bool(target_substep1_finished),
        "wall_s": float(time.perf_counter() - started),
    }

    decision = "INCONCLUSIVE"
    reason = "TARGET_SUBSTEP1_NOT_COMPLETED"
    non_finite = False

    if target_substep1_finished and native_result["fired"]:
        captured_free = _as_array(native.capturedFreePosition)
        payload["captured_free"] = adapter.measure(
            captured_free, spacing_m=0.00001
        )

        q_final = _as_array(beam_dofs.position)
        q_final_free = _as_array(beam_dofs.free_position)
        payload["final_committed"] = adapter.measure(
            q_final, spacing_m=0.00025
        )
        payload["final_independent_dense"] = adapter.measure(
            q_final, spacing_m=0.00001
        )
        payload["final_vs_offline_accepted"] = _state_delta(q_acc, q_final)
        payload["final_free_vs_committed"] = _state_delta(q_final_free, q_final)
        payload["collision_free_state_finite_diagnostic"] = _finite(
            _as_array(collision_dofs.free_position)
        )

        non_finite = not (
            _finite(captured_free)
            and _finite(q_final)
            and _finite(q_final_free)
        )

        free_c = payload["captured_free"]["min_clearance_m"]
        c_main = payload["final_committed"]["min_clearance_m"]
        c_dense = payload["final_independent_dense"]["min_clearance_m"]

        if target_completed_substeps != 1 or target_substep2_executed:
            decision, reason = "FAIL", "TARGET_SUBSTEP_ISOLATION_FAILED"
        elif native_result["status"] != "PASS_NATIVE_WRITE_AND_PROPAGATE":
            decision, reason = (
                "FAIL",
                f"NATIVE_HOOK_STATUS_{native_result['status']}",
            )
        elif not native_result["propagated"]:
            decision, reason = "FAIL", "NATIVE_MAPPING_PROPAGATION_DID_NOT_RUN"
        elif not native_result["velocity_corrected"]:
            decision, reason = "FAIL", "FREE_VELOCITY_CORRECTION_DID_NOT_RUN"
        elif native_result["parent_write_max_error_mm"] > 1e-6:
            decision, reason = "FAIL", "PARENT_CANDIDATE_WRITE_MISMATCH"
        elif native_result["mapped_child_max_change_mm"] <= 1e-7:
            decision, reason = "FAIL", "MAPPED_COLLISION_FREE_STATE_DID_NOT_CHANGE"
        elif free_c >= 0.0:
            decision, reason = "FAIL", "TARGET_FREE_STATE_NOT_UNSAFE"
        elif non_finite:
            decision, reason = "FAIL", "NON_FINITE_STATE"
        elif c_main < -0.00005 or c_dense < -0.00005:
            decision, reason = "FAIL", "MEANINGFUL_POST_COMMIT_PENETRATION"
        elif c_main >= -NUM_TOL_M and c_dense >= -NUM_TOL_M:
            decision = "PASS"
            reason = "NATIVE_FREE_STATE_WRITE_PROPAGATED_AND_SURVIVED_SAME_SUBSTEP"
        else:
            decision = "PARTIAL"
            reason = "SMALL_RESIDUAL_OR_NUMERICAL_VIOLATION"

    elif target_substep1_finished and not native_result["fired"]:
        reason = "NATIVE_COLLISION_BEGIN_HOOK_DID_NOT_FIRE"

    payload["non_finite"] = bool(non_finite)
    payload["decision"] = decision
    payload["reason"] = reason

    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_report(payload, report)

    print(
        json.dumps(
            {
                "decision": decision,
                "reason": reason,
                "action_sha256": action_sha,
                "target_completed_substeps": target_completed_substeps,
                "target_substep2_executed": target_substep2_executed,
                "native_status": native_result.get("status"),
                "fired": native_result.get("fired"),
                "propagated": native_result.get("propagated"),
                "mapped_child_max_change_mm": native_result.get(
                    "mapped_child_max_change_mm"
                ),
                "captured_free_clearance_mm": (
                    payload.get("captured_free") or {}
                ).get("min_clearance_mm"),
                "final_clearance_mm": (
                    payload.get("final_committed") or {}
                ).get("min_clearance_mm"),
                "dense_clearance_mm": (
                    payload.get("final_independent_dense") or {}
                ).get("min_clearance_mm"),
                "output": str(output),
                "report": str(report),
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
