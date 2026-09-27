#!/usr/bin/env python3
"""One protected-prefix, one-substep robust-margin diagnosis at B02 step 1293."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

THIS = Path(__file__).resolve()
BEAM_ROOT = THIS.parents[1]
TEST_DIR = BEAM_ROOT / "tests"
for entry in (THIS.parents[3], TEST_DIR):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import b02_full_episode_safety_acceptance as full
import b02_online_feasible_solver_bridge as bridge
import b02_step776_feasible_solve as validated_solver
from b02_step776_precommit_integration import _state_delta
from mcr_sim.distributed import DistributedPPO

TARGET_STEP = 1293
TARGET_SUBSTEP = 1
CONTROL_RESULTS = BEAM_ROOT / "_runtime" / "results"
RUNTIME = BEAM_ROOT / "_runtime" / "postconstraint_diagnosis"
ACCEPT_LIMIT_MM = 0.001


class TargetCaptured(Exception):
    pass


class PrefixMismatch(Exception):
    pass


def finite(*values):
    return all(np.isfinite(np.asarray(value)).all() for value in values)


def measure(adapter, q):
    return full._measure(adapter, q, full.DENSE_SPACING_M)


class CapturePlanner(full.FullEpisodeFeasibleController):
    def __init__(self, *, collision_dofs, **kwargs):
        super().__init__(**kwargs)
        self.collision_dofs = collision_dofs
        self.target_states = {}

    def _collision_begin(self, source):
        key = (int(self.current_rl_step), int(self.current_substep))
        target = key == (TARGET_STEP, TARGET_SUBSTEP)
        fresh = target and self._last_processed_key != key
        if fresh:
            self.target_states["q_prev"] = full._as_array(self.beam_dofs.position)
            self.target_states["q_free"] = full._as_array(self.beam_dofs.free_position)
            self.target_states["collision_free_before"] = full._as_array(
                self.collision_dofs.free_position
            )
        super()._collision_begin(source)
        if fresh and self._record:
            self.target_states["solver_geometry"] = full._json_safe(
                self.adapter.adapter.solver_geometry()
            )
        if fresh and self._record and self._record.get("native_arm_requested"):
            self.target_states["q_accepted"] = full._as_array(
                self.native.candidateFreePosition
            )


def run(margin_mm, output, scientific_iteration=1):
    output.parent.mkdir(parents=True, exist_ok=True)
    control = json.loads(
        (CONTROL_RESULTS / "b02_full_episode_safety_acceptance.json").read_text()
    )
    baseline = None
    with (CONTROL_RESULTS / "b02_full_episode_safety_acceptance_trace.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if (row.get("rl_step"), row.get("substep")) == (TARGET_STEP, TARGET_SUBSTEP):
                baseline = row
                break
    if control.get("decision") != "FAIL" or baseline is None:
        raise RuntimeError("validated full-episode failure control is unavailable")

    checkpoint = Path(full.CHECKPOINT)
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if checkpoint_sha != control.get("checkpoint_sha256"):
        raise RuntimeError("checkpoint SHA differs from protected control")
    plugin = full._find_plugin(full.DEFAULT_BUILD_DIR)
    if plugin is None:
        raise RuntimeError("native plugin missing")
    plugin_handle = full._load_plugin(plugin)

    env = full._create_env()
    observation, _ = env.reset(seed=15204)
    if int(env.physics_substeps) != 2:
        raise RuntimeError("expected exactly two 5 ms physics substeps")
    instrument = env.mcr_controller_sofa.instrument.InstrumentCombined
    beam_dofs = instrument.getObject("DOFs")
    collision_dofs = instrument.getChild("mcr_collis").getObject("CollisionDOFs")
    adapter = full.AdapterCompat(env, instrument)
    initial = measure(adapter, full._as_array(beam_dofs.position))
    if initial["min_clearance_mm"] < -ACCEPT_LIMIT_MM:
        raise RuntimeError("reset committed state is unsafe")

    planner = CapturePlanner(
        name="B02Iter1MarginPlanner",
        beam_dofs=beam_dofs,
        collision_dofs=collision_dofs,
        adapter=adapter,
        dt=full.DT,
    )
    instrument.addObject(planner)
    native = instrument.addObject(
        "BeamFeasibleNativePrecommitHook",
        name="B02Iter1MarginNativeHook",
        beamState="@DOFs",
        collisionState="@mcr_collis/CollisionDOFs",
        dt=float(full.DT),
        armed=False,
    )
    planner.native = native

    model = DistributedPPO.load(str(checkpoint), device="cpu")
    model.policy.set_training_mode(False)

    original_solve = full.solve_validated_feasible_state
    original_animate = env.sofa_simulation.animate
    action_hasher = hashlib.sha256()
    current_step = 0
    substep_in_step = 0
    target_result = None
    prefix_substeps = 0
    prefix_unsafe = 0
    started = time.perf_counter()

    def solve_with_margin(*, q_prev, q_free, adapter, context,
                          require_explicit_state_inputs=False):
        if (int(context["step"]), int(context["substep"])) != (TARGET_STEP, TARGET_SUBSTEP):
            return original_solve(
                q_prev=q_prev, q_free=q_free, adapter=adapter, context=context,
                require_explicit_state_inputs=require_explicit_state_inputs,
            )
        if not require_explicit_state_inputs:
            raise RuntimeError("strict bridge disabled at target")
        original_margin = validated_solver.MARGIN_M
        validated_solver.MARGIN_M = float(margin_mm) / 1000.0
        try:
            raw = validated_solver.solve_feasible_state(
                q_prev=q_prev, q_free=q_free, adapter=adapter, context=context
            )
            solved = bridge._extract(
                raw, "b02_step776_feasible_solve:solve_feasible_state"
            )
            solved.metadata["requested_margin_mm"] = float(margin_mm)
            return solved
        finally:
            validated_solver.MARGIN_M = original_margin

    def traced_animate(root, dt):
        nonlocal substep_in_step, target_result, prefix_substeps, prefix_unsafe
        substep_in_step += 1
        sub = substep_in_step
        planner.current_rl_step = current_step
        planner.current_substep = sub
        fire_before = full._as_int(native.fireCount)
        result = original_animate(root, dt)
        fire_after = full._as_int(native.fireCount)
        rec = planner.take_record()
        if rec is None:
            raise PrefixMismatch(f"CollisionBeginEvent missing at {current_step}/{sub}")
        q_committed = full._as_array(beam_dofs.position)
        committed = measure(adapter, q_committed)
        native_data = full._native_snapshot(native) if fire_after > fire_before else None
        free_mm = float(rec.get("free_clearance_m", np.nan)) * 1000.0
        committed_mm = float(committed["min_clearance_mm"])

        if (current_step, sub) == (TARGET_STEP, TARGET_SUBSTEP):
            states = planner.target_states
            q_prev = states.get("q_prev")
            q_free = states.get("q_free")
            q_accepted = states.get("q_accepted")
            accepted = measure(adapter, q_accepted) if q_accepted is not None else None
            prev = measure(adapter, q_prev) if q_prev is not None else None
            save = output.with_suffix(".npz")
            arrays = {
                "q_prev": q_prev, "q_free": q_free,
                "q_committed": q_committed,
                "collision_free_before": states.get("collision_free_before"),
                "collision_free_after": full._as_array(collision_dofs.free_position),
            }
            if q_accepted is not None:
                arrays["q_accepted"] = q_accepted
            np.savez_compressed(save, **{k:v for k,v in arrays.items() if v is not None})
            target_result = {
                "rl_step": current_step,
                "substep": sub,
                "physics_dt_s": float(dt),
                "requested_margin_mm": float(margin_mm),
                "q_prev": prev,
                "q_free": measure(adapter, q_free) if q_free is not None else None,
                "accepted": accepted,
                "committed": committed,
                "candidate_to_committed_clearance_loss_mm":
                    (float(accepted["min_clearance_mm"]) - committed_mm)
                    if accepted is not None else None,
                "candidate_vs_free": _state_delta(q_free, q_accepted)
                    if q_accepted is not None else None,
                "candidate_vs_committed": _state_delta(q_accepted, q_committed)
                    if q_accepted is not None else None,
                "native_fire_delta": int(fire_after - fire_before),
                "native": native_data,
                "planner_status": rec.get("planning_status"),
                "planner_reason": rec.get("reason"),
                "solver_runtime_s": rec.get("solver_runtime_s"),
                "solver_source": rec.get("solver_source"),
                "q_arrays_finite": finite(*[v for v in arrays.values() if v is not None]),
                "capture_npz": str(save),
                "solver_geometry": states.get("solver_geometry"),
            }
            raise TargetCaptured()

        prefix_substeps += 1
        if rec.get("solver_required"):
            prefix_unsafe += 1
            if (
                rec.get("planning_status") != "CANDIDATE_ARMED_FOR_NATIVE_HOOK"
                or fire_after - fire_before != 1
                or native_data is None
                or native_data["status"] != "PASS_NATIVE_WRITE_AND_PROPAGATE"
                or not native_data["propagated"]
                or not native_data["velocity_corrected"]
            ):
                raise PrefixMismatch(
                    f"protected prefix correction mismatch at {current_step}/{sub}: "
                    f"{rec.get('planning_status')} / {native_data}"
                )
        elif rec.get("planning_status") != "SAFE_FREE_NO_INTERVENTION" or fire_after != fire_before:
            raise PrefixMismatch(f"safe-free prefix mismatch at {current_step}/{sub}")
        if not finite(q_committed) or committed_mm < -ACCEPT_LIMIT_MM:
            raise PrefixMismatch(
                f"prefix committed violation at {current_step}/{sub}: {committed_mm} mm"
            )
        return result

    full.solve_validated_feasible_state = solve_with_margin
    env.sofa_simulation.animate = traced_animate
    captured = False
    failure = None
    try:
        for step in range(1, TARGET_STEP + 1):
            current_step = step
            substep_in_step = 0
            action, _ = model.predict(observation, deterministic=True)
            action = np.asarray(action, dtype=np.float32).reshape(3)
            action_hasher.update(np.asarray(action, dtype="<f4").tobytes(order="C"))
            try:
                observation, _, terminated, truncated, info = env.step(action)
            except TargetCaptured:
                captured = True
                break
            if step % 100 == 0:
                print(
                    f"[ITER1] protected_step={step}/{TARGET_STEP} "
                    f"prefix_substeps={prefix_substeps} unsafe={prefix_unsafe}",
                    flush=True,
                )
            if terminated or truncated:
                failure = f"episode terminated before target at step {step}: {info}"
                break
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        planner.enabled = False
        env.sofa_simulation.animate = original_animate
        full.solve_validated_feasible_state = original_solve
        try:
            env.close()
        except Exception:
            pass

    result = {
        "test": "B02 protected step1293/substep1 robust-margin diagnostic",
        "scientific_iteration": int(scientific_iteration),
        "requested_margin_mm": float(margin_mm),
        "checkpoint_sha256": checkpoint_sha,
        "action_stream_sha256": action_hasher.hexdigest(),
        "control_action_stream_sha256": control.get("action_stream_sha256"),
        "prefix_substeps": prefix_substeps,
        "prefix_unsafe_free_count": prefix_unsafe,
        "target_substeps_executed": substep_in_step if current_step == TARGET_STEP else 0,
        "captured_target": captured,
        "failure": failure,
        "target": target_result,
        "wall_s": time.perf_counter() - started,
        "production_files_modified": False,
        "training_launched": False,
    }
    if captured and target_result is not None:
        actual_prev = target_result["q_prev"]["min_clearance_mm"]
        actual_free = target_result["q_free"]["min_clearance_mm"]
        control_prev = float(baseline["previous_committed_clearance_m"]) * 1000.0
        control_free = float(baseline["free_clearance_m"]) * 1000.0
        result["control"] = {
            "previous_committed_clearance_mm": control_prev,
            "free_clearance_mm": control_free,
            "accepted_clearance_mm": float(baseline["accepted_clearance_m"]) * 1000.0,
            "committed_clearance_mm": float(baseline["committed_clearance_m"]) * 1000.0,
            "native_fire_delta": baseline["native_fire_delta"],
            "native_status": baseline["native_status"],
        }
        result["same_prefix_action_sha"] = (
            result["action_stream_sha256"] == control["action_stream_sha256"]
        )
        result["q_prev_clearance_delta_vs_control_mm"] = actual_prev - control_prev
        result["q_free_clearance_delta_vs_control_mm"] = actual_free - control_free
        result["same_target_free_state"] = bool(
            result["same_prefix_action_sha"]
            and abs(actual_prev - control_prev) <= 0.00001
            and abs(actual_free - control_free) <= 0.00001
        )
        accepted = target_result["accepted"]
        if not result["same_target_free_state"]:
            result["decision"] = "INCONCLUSIVE_OFF_TRAJECTORY"
        elif accepted is None:
            result["decision"] = "SOLVER_DID_NOT_RETURN_CANDIDATE"
        elif accepted["min_clearance_mm"] < margin_mm - 0.001:
            result["decision"] = "REQUESTED_MARGIN_NOT_ACHIEVED"
        elif target_result["committed"]["min_clearance_mm"] >= -ACCEPT_LIMIT_MM:
            result["decision"] = "MARGIN_PREVENTED_VIOLATION"
        else:
            result["decision"] = "MARGIN_DID_NOT_PREVENT_VIOLATION"
    else:
        result["decision"] = "INCONCLUSIVE_TARGET_NOT_CAPTURED"

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(full._json_safe(result), indent=2, sort_keys=True) + "\n")
    report = output.with_suffix(".md")
    report.write_text(
        f"B02 ITERATION {scientific_iteration} ROBUST MARGIN\n\n"
        f"Decision: {result['decision']}\n"
        f"Requested margin: {margin_mm} mm\n"
        f"Same target free state: {result.get('same_target_free_state')}\n"
        f"Control: {result.get('control')}\n"
        f"Target: {target_result}\n"
        f"Failure: {failure}\n"
    )
    print(json.dumps({
        "decision": result["decision"],
        "captured_target": captured,
        "same_target_free_state": result.get("same_target_free_state"),
        "target": target_result,
        "output": str(output),
        "wall_s": result["wall_s"],
    }, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--margin-mm", type=float, default=0.05)
    parser.add_argument(
        "--output",
        default=str(RUNTIME / "results" / "iter1_margin_0p050.json"),
    )
    args = parser.parse_args()
    if not (0.0 < args.margin_mm <= 0.2):
        raise ValueError("diagnostic requested margin must be in (0,0.2] mm")
    run(args.margin_mm, Path(args.output).resolve())


if __name__ == "__main__":
    main()
