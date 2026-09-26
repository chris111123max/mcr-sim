#!/usr/bin/env python3
"""B02 step776 one-RL-step / two-substep feasibility-invariance test.

Target:
  B02 / target_04 / seed 15204 / RL step 776
  exactly the configured two 5 ms physics substeps.

For each target CollisionBeginEvent:
  - read the real parent Beam committed state q_prev and live free state q_free;
  - if q_free is already feasible, do not intervene;
  - if q_free is unsafe, synchronously call the already-validated real-Beam
    feasible solver through b02_online_feasible_solver_bridge;
  - verify the candidate independently with the production BeamAdapter/SDF;
  - arm the native precommit hook so the candidate is written and mapped in
    the same CollisionBeginEvent before native collision/constraint completion;
  - after animate returns, verify the committed real-Beam state is feasible.

CollisionDOFs are mapping diagnostics only.  This file does not implement a new
solver and does not modify production code.
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
import Sofa

THIS = Path(__file__).resolve()
TEST_DIR = THIS.parent
BEAM_ROOT = THIS.parents[1]
RESULTS_DIR = BEAM_ROOT / "_runtime" / "results"
DEFAULT_BUILD_DIR = BEAM_ROOT / "_runtime" / "native_build"

if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from b02_online_feasible_solver_bridge import (
    ValidatedSolverBridgeUnavailable,
    solve_validated_feasible_state,
)
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
    _sha256_actions,
    _state_delta,
)
from mcr_sim.distributed import DistributedPPO


EXPECTED_SUBSTEPS = 2
MEANINGFUL_FAIL_M = 0.00005


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


def _as_int(data: Any) -> int:
    return int(np.asarray(_scalar(data)).reshape(-1)[0])


def _as_float(data: Any) -> float:
    return float(np.asarray(_scalar(data), dtype=np.float64).reshape(-1)[0])


def _as_str(data: Any) -> str:
    value = _scalar(data)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


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


def _measure(adapter: AdapterCompat, q: np.ndarray, spacing_m: float) -> dict[str, Any]:
    return _json_safe(adapter.measure(np.asarray(q, dtype=np.float64), spacing_m=spacing_m))


def _native_snapshot(native: Any) -> dict[str, Any]:
    return {
        "status": _as_str(native.status),
        "armed": _as_bool(native.armed),
        "fired": _as_bool(native.fired),
        "fire_count": _as_int(native.fireCount),
        "propagated": _as_bool(native.propagated),
        "velocity_corrected": _as_bool(native.velocityCorrected),
        "mapped_child_max_change_mm": _as_float(native.mappedChildMaxChangeMm),
        "parent_write_max_error_mm": _as_float(native.parentWriteMaxErrorMm),
    }


class OnlineFeasibleController(Sofa.Core.Controller):
    """Plan candidate before the native hook receives the same collision event."""

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
        self.enabled = False
        self.records: dict[int, dict[str, Any]] = {}
        self._processed: set[tuple[int, int]] = set()

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
        if not self.enabled:
            return
        if self.current_rl_step != TARGET_STEP:
            return
        if self.current_substep not in (1, 2):
            return

        key = (self.current_rl_step, self.current_substep)
        if key in self._processed:
            return
        self._processed.add(key)

        sub = int(self.current_substep)
        rec: dict[str, Any] = {
            "substep": sub,
            "event_source": source,
            "solver_required": False,
            "solver_called": False,
            "native_arm_requested": False,
            "uses_collision_dofs_as_constraint_source": False,
            "used_native_post_contact_state_as_solver_input": False,
            "rollback_used": False,
            "projection_used": False,
        }
        self.records[sub] = rec

        q_prev = _as_array(self.beam_dofs.position)
        q_free = _as_array(self.beam_dofs.free_position)
        rec["q_prev_finite"] = _finite(q_prev)
        rec["q_free_finite"] = _finite(q_free)
        rec["previous_committed"] = _measure(self.adapter, q_prev, 0.00001)
        rec["free_dense"] = _measure(self.adapter, q_free, 0.00001)
        rec["free_vs_previous"] = _json_safe(_state_delta(q_prev, q_free))

        if not (_finite(q_prev) and _finite(q_free)):
            rec["planning_status"] = "FAIL"
            rec["reason"] = "NON_FINITE_PRECOMMIT_STATE"
            return

        free_clearance_m = float(rec["free_dense"]["min_clearance_m"])
        if free_clearance_m >= -NUM_TOL_M:
            rec["planning_status"] = "SAFE_FREE_NO_INTERVENTION"
            rec["accepted"] = rec["free_dense"]
            rec["candidate_vs_free"] = _json_safe(_state_delta(q_free, q_free))
            return

        rec["solver_required"] = True
        rec["solver_called"] = True
        started = time.perf_counter()

        context = {
            "step": TARGET_STEP,
            "substep": sub,
            "dt": self.dt,
            "beam_dofs": self.beam_dofs,
            "adapter": self.adapter,
        }
        try:
            solved = solve_validated_feasible_state(
                q_prev=q_prev,
                q_free=q_free,
                adapter=self.adapter,
                context=context,
            )
        except ValidatedSolverBridgeUnavailable as exc:
            rec["planning_status"] = "INCONCLUSIVE"
            rec["reason"] = "VALIDATED_SOLVER_BRIDGE_UNAVAILABLE"
            rec["bridge_error"] = str(exc)
            return
        except Exception as exc:
            rec["planning_status"] = "FAIL"
            rec["reason"] = f"ONLINE_FEASIBLE_SOLVE_FAILED: {type(exc).__name__}: {exc}"
            return

        rec["solver_runtime_s"] = float(time.perf_counter() - started)
        q_acc = np.asarray(solved.accepted, dtype=np.float64)
        rec["solver_source"] = solved.source
        rec["solver_metadata"] = _json_safe(solved.metadata)
        rec["candidate_shape"] = list(q_acc.shape)
        rec["candidate_finite"] = _finite(q_acc)

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

        accepted_dense = _measure(self.adapter, q_acc, 0.00001)
        rec["accepted"] = accepted_dense
        rec["candidate_vs_free"] = _json_safe(_state_delta(q_free, q_acc))

        if float(accepted_dense["min_clearance_m"]) < -NUM_TOL_M:
            rec["planning_status"] = "FAIL"
            rec["reason"] = "ONLINE_ACCEPTED_CANDIDATE_NOT_FEASIBLE"
            return

        if self.native is None:
            rec["planning_status"] = "INCONCLUSIVE"
            rec["reason"] = "NATIVE_HOOK_NOT_ATTACHED"
            return

        self.native.candidateFreePosition.value = q_acc.tolist()
        self.native.armed.value = True
        rec["native_arm_requested"] = True
        rec["planning_status"] = "CANDIDATE_ARMED_FOR_NATIVE_HOOK"


def _write_report(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "B02 STEP776 TWO-SUBSTEP FEASIBILITY INVARIANCE",
        "=============================================",
        "",
        f"Action prefix: {payload.get('action_prefix_status')}",
        f"SHA256: {payload.get('action_sha256')}",
        f"Target animate count: {payload.get('target_animate_count')}",
        f"Configured physics substeps: {payload.get('configured_physics_substeps')}",
        "",
    ]
    for rec in payload.get("substeps", []):
        lines.extend(
            [
                f"Substep {rec.get('substep')}",
                f"  free clearance: {(rec.get('free_dense') or {}).get('min_clearance_mm')} mm",
                f"  solver required: {rec.get('solver_required')}",
                f"  planning status: {rec.get('planning_status')}",
                f"  native fire delta: {rec.get('native_fire_delta')}",
                f"  mapped child change: {rec.get('mapped_child_max_change_mm')} mm",
                f"  committed clearance: {(rec.get('committed_dense') or {}).get('min_clearance_mm')} mm",
                f"  status: {rec.get('substep_status')}",
                "",
            ]
        )
    lines.extend(
        [
            f"NaN/Inf: {payload.get('non_finite')}",
            f"FINAL DECISION: {payload.get('decision')}",
            f"Reason: {payload.get('reason')}",
            "",
            "This is exactly one RL step / two physics substeps.",
            "It does not establish long-horizon invariance or training readiness.",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def _finish(
    payload: dict[str, Any],
    output: Path,
    report: Path,
    env: Any | None = None,
) -> None:
    output.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n")
    _write_report(_json_safe(payload), report)
    print(json.dumps(_json_safe(payload), sort_keys=True), flush=True)
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
        default=str(RESULTS_DIR / "b02_step776_two_substep_invariance.json"),
    )
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = output.with_name("b02_step776_two_substep_invariance_report.md")

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    build_dir = Path(args.build_dir).expanduser().resolve()
    plugin = (
        Path(args.plugin_lib).expanduser().resolve()
        if args.plugin_lib
        else _find_plugin(build_dir)
    )
    if plugin is None:
        _finish(
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
        _finish(
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

    env = _create_env()
    observation, _ = env.reset(seed=int(args.seed))
    if int(env.physics_substeps) != EXPECTED_SUBSTEPS:
        _finish(
            {
                "decision": "FAIL",
                "reason": "PHYSICS_SUBSTEP_CONFIGURATION_MISMATCH",
                "configured_physics_substeps": int(env.physics_substeps),
                "expected_physics_substeps": EXPECTED_SUBSTEPS,
            },
            output,
            report,
            env,
        )
        return

    controller = env.mcr_controller_sofa
    instrument = controller.instrument.InstrumentCombined
    beam_dofs = instrument.getObject("DOFs")
    collision_dofs = instrument.getChild("mcr_collis").getObject("CollisionDOFs")
    adapter = AdapterCompat(env, instrument)

    # Add Python planner FIRST.  During event propagation it must compute/arm
    # the candidate before the native hook object receives the same event.
    planner = OnlineFeasibleController(
        name="B02OnlineFeasiblePlanner",
        beam_dofs=beam_dofs,
        adapter=adapter,
        dt=DT,
    )
    instrument.addObject(planner)

    native = instrument.addObject(
        "BeamFeasibleNativePrecommitHook",
        name="BeamFeasibleNativePrecommitHookTwoSubstep",
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
    target_animate_count = 0
    substep_records: dict[int, dict[str, Any]] = {}

    def traced_animate(root, dt):
        nonlocal target_animate_count

        if current_step != TARGET_STEP:
            return original_animate(root, dt)

        target_animate_count += 1
        sub = int(target_animate_count)
        planner.current_rl_step = TARGET_STEP
        planner.current_substep = sub

        fire_before = _as_int(native.fireCount)
        result = original_animate(root, dt)
        fire_after = _as_int(native.fireCount)

        rec = dict(planner.records.get(sub, {}))
        rec.setdefault("substep", sub)
        rec["native_fire_before"] = fire_before
        rec["native_fire_after"] = fire_after
        rec["native_fire_delta"] = int(fire_after - fire_before)

        q_committed = _as_array(beam_dofs.position)
        rec["committed_dense"] = _measure(adapter, q_committed, 0.00001)
        rec["committed_finite"] = _finite(q_committed)
        rec["collision_free_finite_diagnostic"] = _finite(
            _as_array(collision_dofs.free_position)
        )

        if rec.get("native_fire_delta", 0) > 0:
            snap = _native_snapshot(native)
            rec["native_status"] = snap["status"]
            rec["native_propagated"] = snap["propagated"]
            rec["native_velocity_corrected"] = snap["velocity_corrected"]
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

        substep_records[sub] = rec
        return result

    env.sofa_simulation.animate = traced_animate

    actions: list[np.ndarray] = []
    terminal_reason = None
    target_step_completed = False
    started = time.perf_counter()

    try:
        for step in range(1, TARGET_STEP + 1):
            current_step = int(step)
            raw_action, _ = model.predict(observation, deterministic=True)
            raw_action = np.asarray(raw_action, dtype=np.float32).reshape(3)
            actions.append(raw_action.copy())

            if step == TARGET_STEP:
                prefix_sha = _sha256_actions(actions)
                if prefix_sha != EXPECTED_ACTION_SHA256:
                    _finish(
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
                planner.enabled = True

            observation, _, terminated, truncated, info = env.step(raw_action)

            if step == TARGET_STEP:
                planner.enabled = False
                target_step_completed = True
                break

            if args.progress_every > 0 and (
                step == 1 or step % int(args.progress_every) == 0
            ):
                print(f"[TWO_SUBSTEP_INVARIANCE] step={step}/{TARGET_STEP}", flush=True)

            if terminated or truncated:
                terminal_reason = str(
                    info.get("terminal_reason")
                    or ("terminated" if terminated else "truncated")
                )
                break
    finally:
        env.sofa_simulation.animate = original_animate

    action_sha = _sha256_actions(actions)
    ordered = [substep_records[k] for k in sorted(substep_records)]
    payload: dict[str, Any] = {
        "test": "B02 step776 one-RL-step two-substep feasibility invariance",
        "training_started": False,
        "production_files_modified": False,
        "seed": int(args.seed),
        "target_rl_step": TARGET_STEP,
        "physics_dt_s": DT,
        "configured_physics_substeps": int(env.physics_substeps),
        "target_animate_count": int(target_animate_count),
        "target_step_completed": bool(target_step_completed),
        "plugin": str(plugin),
        "action_sha256": action_sha,
        "expected_action_sha256": EXPECTED_ACTION_SHA256,
        "action_prefix_status": (
            "PASS" if action_sha == EXPECTED_ACTION_SHA256 else "FAIL"
        ),
        "terminal_reason_before_target": terminal_reason,
        "substeps": ordered,
        "wall_s": float(time.perf_counter() - started),
    }

    decision = "PASS"
    reason = "TWO_SUBSTEP_FEASIBILITY_INVARIANT_HELD"
    non_finite = False

    if action_sha != EXPECTED_ACTION_SHA256:
        decision, reason = "FAIL", "ACTION_PREFIX_MISMATCH"
    elif not target_step_completed:
        decision, reason = "INCONCLUSIVE", "TARGET_RL_STEP_NOT_COMPLETED"
    elif target_animate_count != EXPECTED_SUBSTEPS or len(ordered) != EXPECTED_SUBSTEPS:
        decision, reason = "FAIL", "TARGET_SUBSTEP_COUNT_MISMATCH"
    else:
        for rec in ordered:
            sub = int(rec["substep"])
            planning = str(rec.get("planning_status", "MISSING"))
            free = rec.get("free_dense") or {}
            committed = rec.get("committed_dense") or {}
            free_c = float(free.get("min_clearance_m", np.nan))
            committed_c = float(committed.get("min_clearance_m", np.nan))
            solver_required = bool(rec.get("solver_required", False))
            fire_delta = int(rec.get("native_fire_delta", 0))

            sub_status = "PASS"
            sub_reason = "SAFE_FREE_PASSED_NATIVELY"

            if not np.isfinite(free_c) or not np.isfinite(committed_c):
                sub_status, sub_reason = "FAIL", "NON_FINITE_CLEARANCE"
            elif solver_required:
                if planning == "INCONCLUSIVE":
                    sub_status, sub_reason = "INCONCLUSIVE", str(
                        rec.get("reason", "SOLVER_BRIDGE_INCONCLUSIVE")
                    )
                elif planning != "CANDIDATE_ARMED_FOR_NATIVE_HOOK":
                    sub_status, sub_reason = "FAIL", str(
                        rec.get("reason", "ONLINE_SOLVER_DID_NOT_ARM_CANDIDATE")
                    )
                elif fire_delta != 1:
                    sub_status, sub_reason = "FAIL", "NATIVE_HOOK_DID_NOT_FIRE_EXACTLY_ONCE"
                elif rec.get("native_status") != "PASS_NATIVE_WRITE_AND_PROPAGATE":
                    sub_status, sub_reason = "FAIL", "NATIVE_HOOK_STATUS_FAILED"
                elif not rec.get("native_propagated", False):
                    sub_status, sub_reason = "FAIL", "NATIVE_MAPPING_PROPAGATION_NOT_RUN"
                elif not rec.get("native_velocity_corrected", False):
                    sub_status, sub_reason = "FAIL", "FREE_VELOCITY_CORRECTION_NOT_RUN"
                elif float(rec.get("parent_write_max_error_mm", np.inf)) > 1e-6:
                    sub_status, sub_reason = "FAIL", "PARENT_CANDIDATE_WRITE_MISMATCH"
                elif float(rec.get("mapped_child_max_change_mm", 0.0)) <= 1e-7:
                    sub_status, sub_reason = "FAIL", "MAPPED_COLLISION_FREE_STATE_DID_NOT_CHANGE"
                else:
                    accepted_c = float(
                        (rec.get("accepted") or {}).get("min_clearance_m", np.nan)
                    )
                    if not np.isfinite(accepted_c) or accepted_c < -NUM_TOL_M:
                        sub_status, sub_reason = "FAIL", "ACCEPTED_CANDIDATE_NOT_SAFE"
                    else:
                        sub_reason = "UNSAFE_FREE_CORRECTED_AND_COMMITTED_SAFE"
            else:
                if planning != "SAFE_FREE_NO_INTERVENTION":
                    sub_status, sub_reason = "FAIL", "SAFE_FREE_CLASSIFICATION_INCONSISTENT"
                elif fire_delta != 0:
                    sub_status, sub_reason = "FAIL", "UNEXPECTED_NATIVE_INTERVENTION"

            if sub_status == "PASS":
                if committed_c < -MEANINGFUL_FAIL_M:
                    sub_status, sub_reason = "FAIL", "MEANINGFUL_POST_COMMIT_PENETRATION"
                elif committed_c < -NUM_TOL_M:
                    sub_status, sub_reason = "PARTIAL", "SMALL_POST_COMMIT_RESIDUAL"

            rec["substep_status"] = sub_status
            rec["substep_reason"] = sub_reason
            non_finite = non_finite or not bool(rec.get("committed_finite", False))

            if sub_status == "FAIL":
                decision, reason = "FAIL", f"SUBSTEP_{sub}_{sub_reason}"
                break
            if sub_status == "INCONCLUSIVE" and decision == "PASS":
                decision, reason = "INCONCLUSIVE", f"SUBSTEP_{sub}_{sub_reason}"
            elif sub_status == "PARTIAL" and decision == "PASS":
                decision, reason = "PARTIAL", f"SUBSTEP_{sub}_{sub_reason}"

    if non_finite and decision == "PASS":
        decision, reason = "FAIL", "NON_FINITE_COMMITTED_STATE"

    payload["substeps"] = ordered
    payload["non_finite"] = bool(non_finite)
    payload["decision"] = decision
    payload["reason"] = reason

    output.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n")
    _write_report(_json_safe(payload), report)

    print(
        json.dumps(
            {
                "decision": decision,
                "reason": reason,
                "action_sha256": action_sha,
                "target_animate_count": target_animate_count,
                "substeps": [
                    {
                        "substep": r.get("substep"),
                        "free_clearance_mm": (r.get("free_dense") or {}).get(
                            "min_clearance_mm"
                        ),
                        "solver_required": r.get("solver_required"),
                        "planning_status": r.get("planning_status"),
                        "native_fire_delta": r.get("native_fire_delta"),
                        "mapped_child_max_change_mm": r.get(
                            "mapped_child_max_change_mm"
                        ),
                        "committed_clearance_mm": (
                            r.get("committed_dense") or {}
                        ).get("min_clearance_mm"),
                        "substep_status": r.get("substep_status"),
                    }
                    for r in ordered
                ],
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
