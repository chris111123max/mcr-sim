#!/usr/bin/env python3
"""B02 step776/substep1 test-only pre-commit SOFA integration probe.

This is the next gate after the real-beam offline feasible solve passed.

The test injects the previously accepted Rigid3d beam candidate at
CollisionBeginEvent: after FreeMotionAnimationLoop has produced free_position,
before native collision detection / constraint correction completes.

It never edits production code. CollisionDOFs are diagnostics only. If the
production MultiAdaptiveBeamMapping cannot be refreshed coherently from Python
at this stage, the result is INCONCLUSIVE rather than a false PASS.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation
import Sofa

PYTHON_ROOT = Path(__file__).resolve().parents[3]
TEST_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = TEST_ROOT / "_runtime" / "results"
TEST_DIR = Path(__file__).resolve().parent
for p in (PYTHON_ROOT, TEST_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from mcr_sim.distributed import DistributedPPO
from mcr_sim.mcr_rl_env import EnvType, MCREnv
from mcr_sim.rl_core.base import RenderMode

EXPECTED_ACTION_SHA256 = "4304fdb75b63859597716b6016a489e68cdb1202e078cca9d844954e40f26ad2"
CHECKPOINT = Path(
    "/data/home/3220251075/mcr_sim/mcr_project/mCR_simulator-master/"
    "training_runs/ppo_v15_2b_physics_wall_10Nm_20260922_215738/models/"
    "ppo_base_epoch_074_episodes_07400.zip"
)
TARGET_STEP = 776
TARGET_SUBSTEP = 1
DT = 0.005
NUM_TOL_M = 1e-6


def _as_array(data: Any, dtype=np.float64) -> np.ndarray:
    try:
        return np.asarray(data.array(), dtype=dtype).copy()
    except Exception:
        return np.asarray(data.value, dtype=dtype).copy()


def _sha256_actions(actions: list[np.ndarray]) -> str:
    arr = np.asarray(actions, dtype=np.float32).reshape((-1, 3))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _quat_angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).reshape((-1, 4))
    b = np.asarray(b, dtype=np.float64).reshape((-1, 4))
    rel = Rotation.from_quat(a).inv() * Rotation.from_quat(b)
    return np.degrees(np.linalg.norm(rel.as_rotvec(), axis=1))


def _state_delta(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    t = np.linalg.norm(b[:, :3] - a[:, :3], axis=1)
    r = _quat_angle_deg(a[:, 3:7], b[:, 3:7])
    return {
        "translation_max_mm": float(np.max(t) * 1000.0),
        "translation_rms_mm": float(np.sqrt(np.mean(t * t)) * 1000.0),
        "rotation_max_deg": float(np.max(r)),
        "rotation_rms_deg": float(np.sqrt(np.mean(r * r))),
    }


def _finite(x: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(np.asarray(x))))


def _find_state_arrays(obj: Any, prefix: str = "") -> list[tuple[str, np.ndarray]]:
    out: list[tuple[str, np.ndarray]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(_find_state_arrays(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        arr = None
        try:
            arr = np.asarray(obj, dtype=np.float64)
        except Exception:
            pass
        if arr is not None and arr.ndim == 2 and arr.shape[1] == 7:
            out.append((prefix, arr))
        else:
            for i, v in enumerate(obj):
                out.extend(_find_state_arrays(v, f"{prefix}[{i}]"))
    return out


def _score_state_key(key: str, kind: str) -> int:
    key = key.lower()
    terms = {
        "accepted": ("accepted", "feasible", "safe", "candidate", "q_accept"),
        "free": ("free_position", "free_state", "q_free", "beam_free"),
    }[kind]
    score = 0
    for i, term in enumerate(terms):
        if term in key:
            score += 100 - i
    if "collision" in key or "collis" in key:
        score -= 200
    return score


def _load_candidate_artifact(n_nodes: int) -> dict[str, Any]:
    """Load accepted/free states exported by the already-passed offline test."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    accepted: list[tuple[int, str, np.ndarray]] = []
    free: list[tuple[int, str, np.ndarray]] = []

    for path in sorted(RESULTS_DIR.glob("*.npz")):
        try:
            with np.load(path, allow_pickle=False) as data:
                for key in data.files:
                    arr = np.asarray(data[key])
                    if arr.shape != (n_nodes, 7):
                        continue
                    tag = f"{path.name}:{key}"
                    accepted.append((_score_state_key(tag, "accepted"), tag, arr.astype(np.float64)))
                    free.append((_score_state_key(tag, "free"), tag, arr.astype(np.float64)))
        except Exception:
            continue

    for path in sorted(RESULTS_DIR.glob("*.json")):
        try:
            obj = json.loads(path.read_text())
        except Exception:
            continue
        for key, arr in _find_state_arrays(obj):
            if arr.shape != (n_nodes, 7):
                continue
            tag = f"{path.name}:{key}"
            accepted.append((_score_state_key(tag, "accepted"), tag, arr))
            free.append((_score_state_key(tag, "free"), tag, arr))

    accepted = sorted((x for x in accepted if x[0] > 0), reverse=True)
    free = sorted((x for x in free if x[0] > 0), reverse=True)
    if not accepted:
        return {
            "ok": False,
            "reason": "ACCEPTED_STATE_ARTIFACT_MISSING",
            "accepted": None,
            "reference_free": None,
        }

    return {
        "ok": True,
        "accepted": accepted[0][2],
        "reference_free": free[0][2] if free else None,
        "accepted_source": accepted[0][1],
        "free_source": free[0][1] if free else None,
    }


class AdapterCompat:
    """Compatibility wrapper for the real-beam adapter created by the prior test."""

    def __init__(self, env, instrument):
        try:
            module = importlib.import_module("b02_beam_adapter")
        except Exception as exc:
            raise RuntimeError(
                "Missing testing/beam_feasible_domain/tests/b02_beam_adapter.py "
                "from the prior offline validation"
            ) from exc

        cls = getattr(module, "BeamFeasibleAdapter", None)
        if cls is None:
            raise RuntimeError("b02_beam_adapter.BeamFeasibleAdapter not found")

        last = None
        for factory in (
            lambda: cls(env, instrument),
            lambda: cls(env=env, instrument=instrument),
            lambda: cls(env),
            lambda: cls(env=env),
        ):
            try:
                self.adapter = factory()
                break
            except Exception as exc:
                last = exc
        else:
            raise RuntimeError(f"Could not construct BeamFeasibleAdapter: {last}")

        self.irc = instrument.getObject("m_ircontroller")

    def _inserted_length(self) -> float:
        try:
            return float(np.asarray(self.irc.xtip.value, dtype=np.float64).reshape(-1)[0])
        except Exception:
            return float("nan")

    def sample(self, state: np.ndarray, spacing_m: float | None = None) -> np.ndarray:
        fn = getattr(self.adapter, "sample_state", None)
        if fn is None:
            raise RuntimeError("BeamFeasibleAdapter.sample_state missing")

        try:
            names = set(inspect.signature(fn).parameters)
        except Exception:
            names = set()
        kwargs = {}
        inserted = self._inserted_length()
        if "inserted_length" in names and np.isfinite(inserted):
            kwargs["inserted_length"] = inserted
        if spacing_m is not None:
            for name in ("spacing_m", "sample_spacing_m", "step_m", "max_step_m"):
                if name in names:
                    kwargs[name] = float(spacing_m)
                    break

        points = fn(np.asarray(state, dtype=np.float64), **kwargs)
        if isinstance(points, dict):
            for key in ("points", "sample_points", "positions", "xyz"):
                if key in points:
                    points = points[key]
                    break
        points = np.asarray(points, dtype=np.float64).reshape((-1, 3))
        if len(points) < 2 or not _finite(points):
            raise RuntimeError("Invalid real-beam samples")
        return points

    def query(self, points: np.ndarray) -> np.ndarray:
        fn = getattr(self.adapter, "query_production_sdf", None)
        if fn is None:
            raise RuntimeError("BeamFeasibleAdapter.query_production_sdf missing")
        result = fn(np.asarray(points, dtype=np.float64))
        if isinstance(result, tuple):
            clearance = result[0]
        elif isinstance(result, dict):
            clearance = None
            for key in (
                "clearance", "clearances",
                "radius_adjusted_clearance", "radius_adjusted_clearances",
            ):
                if key in result:
                    clearance = result[key]
                    break
            if clearance is None:
                raise RuntimeError("No clearance field in adapter query result")
        else:
            clearance = result
        return np.asarray(clearance, dtype=np.float64).reshape(-1)

    def measure(self, state: np.ndarray, spacing_m: float | None = None) -> dict[str, Any]:
        points = self.sample(state, spacing_m=spacing_m)
        clearance = self.query(points)
        if len(clearance) != len(points):
            raise RuntimeError("Adapter point/clearance count mismatch")
        valid = np.isfinite(clearance)
        if not np.any(valid):
            raise RuntimeError("No valid SDF clearances")
        ids = np.where(valid)[0]
        idx = int(ids[int(np.argmin(clearance[valid]))])
        c = float(clearance[idx])
        return {
            "sample_count": int(len(points)),
            "valid_count": int(np.sum(valid)),
            "min_clearance_m": c,
            "min_clearance_mm": c * 1000.0,
            "worst_point_m": points[idx].tolist(),
            "penetration_gt_0p01mm": int(np.sum(clearance < -0.00001)),
            "penetration_gt_0p05mm": int(np.sum(clearance < -0.00005)),
            "penetration_gt_0p1mm": int(np.sum(clearance < -0.0001)),
        }


def _create_env() -> MCREnv:
    # Preserve the same historical B02 dynamics used to reach the captured state.
    # The soft wall is only part of the replay baseline, not the new safety layer.
    os.environ["MCR_CONSTRAINT_SOLVER"] = "generic"
    os.environ["MCR_SOFA_DT"] = str(DT)
    return MCREnv(
        create_scene_kwargs={
            "force_model": "B02",
            "centerline_file": "target_04_centerline.vtk",
            "verbose_scene": False,
            "training_curriculum_enabled": False,
            "vessel_scale_min": 1.0,
            "vessel_scale_max": 1.0,
            "start_window_distance_m": 0.0,
            "target_window_distance_m": 0.0,
            "initial_orientation_max_angle_deg": 0.0,
            "sdf_physics_wall_enabled": True,
            "sdf_wall_stiffness_n_per_m": 10.0,
            "diagnostic_intersection_method": "local_min_distance",
            "use_vessel_line_point_collision": False,
            "use_vessel_point_collision": False,
            "use_vessel_line_collision": False,
            "sdf_hard_constraint_construct": False,
            "sdf_hard_constraint_enabled": False,
            "sdf_unilateral_constraint_construct": False,
            "sdf_unilateral_constraint_enabled": False,
        },
        env_type=EnvType.AORTIC,
        render_mode=RenderMode.NONE,
        max_episode_steps=2048,
        time_step=DT,
        frame_skip=1,
        physics_substeps=2,
    )


def _coherent_free_velocity(
    q_original_free: np.ndarray,
    q_accepted: np.ndarray,
    original_free_velocity: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Apply the accepted-vs-free correction to free velocity coherently."""
    out = np.asarray(original_free_velocity, dtype=np.float64).copy()
    out[:, :3] += (q_accepted[:, :3] - q_original_free[:, :3]) / float(dt)

    r_free = Rotation.from_quat(q_original_free[:, 3:7])
    r_acc = Rotation.from_quat(q_accepted[:, 3:7])
    correction = (r_acc * r_free.inv()).as_rotvec() / float(dt)
    if out.shape[1] >= 6:
        out[:, 3:6] += correction
    return out


class PrecommitHook(Sofa.Core.Controller):
    """Inject accepted free state only at the real pre-collision stage."""

    def __init__(
        self,
        *,
        beam_dofs,
        collision_dofs,
        collis_map,
        adapter: AdapterCompat,
        candidate: dict[str, Any],
        **kwargs,
    ):
        Sofa.Core.Controller.__init__(self, **kwargs)
        self.beam_dofs = beam_dofs
        self.collision_dofs = collision_dofs
        self.collis_map = collis_map
        self.adapter = adapter
        self.candidate = candidate
        self.current_rl_step = 0
        self.current_substep = 0
        self.enabled = False
        self.injected = False
        self.stage_seen = False
        self.event_names: list[str] = []
        self.result: dict[str, Any] = {
            "status": "NOT_RUN",
            "injection_stage": None,
            "mapping_refresh_method": None,
            "uses_collision_dofs_as_constraint_source": False,
            "uses_native_post_contact_state_as_solver_input": False,
            "position_written_at_precommit": False,
            "free_position_written_at_precommit": False,
            "free_velocity_written_at_precommit": False,
        }

    def _record_event(self, name: str) -> None:
        if name not in self.event_names and len(self.event_names) < 32:
            self.event_names.append(name)

    def onEvent(self, event):
        name = None
        if isinstance(event, dict):
            for key in ("type", "Type", "name", "event", "className"):
                if key in event:
                    name = str(event[key])
                    break
        if name is None:
            name = type(event).__name__
        self._record_event(name)
        if "CollisionBegin" in name:
            self._collision_begin("onEvent:" + name)

    def onCollisionBeginEvent(self, event):
        self._record_event("CollisionBeginEvent")
        self._collision_begin("onCollisionBeginEvent")

    def _try_refresh_mapping(self) -> tuple[bool, str, str | None]:
        before = _as_array(self.collision_dofs.free_position)
        errors = []
        for name in ("apply", "update"):
            fn = getattr(self.collis_map, name, None)
            if not callable(fn):
                continue
            try:
                fn()
                after = _as_array(self.collision_dofs.free_position)
                moved = float(np.max(np.linalg.norm(after[:, :3] - before[:, :3], axis=1)))
                if moved > 1e-10:
                    return True, name, None
                errors.append(f"{name}: call succeeded but mapped free_position unchanged")
            except Exception as exc:
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
        msg = "; ".join(errors) if errors else "No Python-bound mapping apply/update method"
        return False, "none", msg

    def _collision_begin(self, source: str) -> None:
        if self.injected or not self.enabled:
            return
        if self.current_rl_step != TARGET_STEP or self.current_substep != TARGET_SUBSTEP:
            return

        self.stage_seen = True
        q_free = _as_array(self.beam_dofs.free_position)
        v_free = _as_array(self.beam_dofs.free_velocity)
        coll_free_before = _as_array(self.collision_dofs.free_position)
        q_acc = np.asarray(self.candidate["accepted"], dtype=np.float64)

        self.result["injection_stage"] = source
        self.result["free_before"] = self.adapter.measure(q_free, spacing_m=0.00025)
        self.result["accepted_offline"] = self.adapter.measure(q_acc, spacing_m=0.00025)
        self.result["accepted_independent_dense"] = self.adapter.measure(q_acc, spacing_m=0.00001)
        self.result["candidate_vs_live_free"] = _state_delta(q_free, q_acc)

        ref_free = self.candidate.get("reference_free")
        if ref_free is not None:
            self.result["live_free_vs_offline_reference_free"] = _state_delta(
                np.asarray(ref_free, dtype=np.float64), q_free
            )

        # Pre-commit write: do not touch committed position.
        self.beam_dofs.free_position.value = q_acc.tolist()
        self.result["free_position_written_at_precommit"] = True

        try:
            v_acc = _coherent_free_velocity(q_free, q_acc, v_free, DT)
            self.beam_dofs.free_velocity.value = v_acc.tolist()
            self.result["free_velocity_written_at_precommit"] = True
            self.result["velocity_update_mode"] = (
                "accepted-free translational correction/dt; "
                "SO(3) world-frame correction rotvec/dt"
            )
        except Exception as exc:
            self.beam_dofs.free_position.value = q_free.tolist()
            self.result["status"] = "INCONCLUSIVE"
            self.result["reason"] = (
                f"FREE_VELOCITY_UPDATE_FAILED: {type(exc).__name__}: {exc}"
            )
            self.enabled = False
            return

        ok, method, error = self._try_refresh_mapping()
        self.result["mapping_refresh_method"] = method
        self.result["mapping_refresh_error"] = error
        coll_free_after = _as_array(self.collision_dofs.free_position)
        self.result["mapped_collision_free_change_mm"] = float(
            np.max(
                np.linalg.norm(
                    coll_free_after[:, :3] - coll_free_before[:, :3], axis=1
                )
            )
            * 1000.0
        )

        if not ok:
            # Restore original free state so an incoherent mapping cannot look like PASS.
            self.beam_dofs.free_position.value = q_free.tolist()
            self.beam_dofs.free_velocity.value = v_free.tolist()
            self.result["status"] = "INCONCLUSIVE"
            self.result["reason"] = (
                "PRODUCTION_MAPPING_REFRESH_NOT_AVAILABLE_AT_PRECOMMIT_STAGE"
            )
            self.enabled = False
            return

        self.result["accepted_write_error"] = _state_delta(
            q_acc, _as_array(self.beam_dofs.free_position)
        )
        self.result["status"] = "INJECTED"
        self.injected = True

    def finalize(self) -> dict[str, Any]:
        self.result["stage_seen"] = bool(self.stage_seen)
        self.result["injected"] = bool(self.injected)
        self.result["event_names"] = list(self.event_names)
        return self.result


def _write_report(payload: dict[str, Any], output: Path) -> None:
    hook = payload.get("hook", {})
    final = payload.get("final_committed", {})
    dense = payload.get("final_independent_dense", {})
    lines = [
        "B02 STEP776 PRE-COMMIT SOFA INTEGRATION",
        "=======================================",
        "",
        f"Action prefix: {payload.get('action_prefix_status')}",
        f"SHA256: {payload.get('action_sha256')}",
        "",
        f"CollisionBegin pre-commit stage seen: {hook.get('stage_seen')}",
        f"Safe candidate injected: {hook.get('injected')}",
        f"Mapping refresh method: {hook.get('mapping_refresh_method')}",
        (
            "CollisionDOFs used as feasible constraint source: "
            f"{hook.get('uses_collision_dofs_as_constraint_source')}"
        ),
        (
            "Native post-contact state used as solver input: "
            f"{hook.get('uses_native_post_contact_state_as_solver_input')}"
        ),
        "",
        f"Final committed clearance: {final.get('min_clearance_mm')} mm",
        f"Independent dense clearance: {dense.get('min_clearance_mm')} mm",
        f"NaN/Inf: {payload.get('non_finite')}",
        "",
        f"FINAL DECISION: {payload.get('decision')}",
        f"Reason: {payload.get('reason')}",
        "",
        "Interpretation:",
        "This validates only step776/substep1 pre-commit writeback.",
        "It does not establish multi-step invariance or training readiness.",
        "",
    ]
    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(CHECKPOINT))
    parser.add_argument("--seed", type=int, default=15204)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--output",
        default=str(RESULTS_DIR / "b02_step776_precommit_integration.json"),
    )
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path = output.with_name("b02_step776_precommit_integration_report.md")

    env = _create_env()
    observation, _ = env.reset(seed=int(args.seed))
    controller = env.mcr_controller_sofa
    instrument = controller.instrument.InstrumentCombined
    beam_dofs = instrument.getObject("DOFs")
    collision_node = instrument.getChild("mcr_collis")
    collision_dofs = collision_node.getObject("CollisionDOFs")
    collis_map = collision_node.getObject("collisMap")

    adapter = AdapterCompat(env, instrument)
    candidate = _load_candidate_artifact(len(_as_array(beam_dofs.position)))
    if not candidate["ok"]:
        payload = {
            "decision": "INCONCLUSIVE",
            "reason": candidate["reason"],
            "training_started": False,
            "production_files_modified": False,
        }
        output.write_text(json.dumps(payload, indent=2) + "\n")
        _write_report(payload, report_path)
        print(json.dumps(payload, sort_keys=True))
        env.close()
        return

    hook = PrecommitHook(
        name="BeamFeasiblePrecommitProbe",
        beam_dofs=beam_dofs,
        collision_dofs=collision_dofs,
        collis_map=collis_map,
        adapter=adapter,
        candidate=candidate,
    )
    env._sofa_root_node.addObject(hook)

    model = DistributedPPO.load(str(checkpoint), device="cpu")
    model.policy.set_training_mode(False)

    original_animate = env.sofa_simulation.animate
    current_step = 0
    substep_counts: dict[int, int] = {}

    def traced_animate(root, dt):
        sub = int(substep_counts.get(current_step, 0) + 1)
        substep_counts[current_step] = sub
        hook.current_rl_step = int(current_step)
        hook.current_substep = int(sub)
        return original_animate(root, dt)

    env.sofa_simulation.animate = traced_animate
    actions: list[np.ndarray] = []
    terminal_reason = None
    target_finished = False
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
                    hook.enabled = False
                    payload = {
                        "decision": "FAIL",
                        "reason": "ACTION_PREFIX_MISMATCH",
                        "action_sha256": prefix_sha,
                        "expected_action_sha256": EXPECTED_ACTION_SHA256,
                        "training_started": False,
                        "production_files_modified": False,
                    }
                    output.write_text(json.dumps(payload, indent=2) + "\n")
                    _write_report(payload, report_path)
                    print(json.dumps(payload, sort_keys=True))
                    return
                hook.enabled = True

            observation, _, terminated, truncated, info = env.step(raw_action)

            if args.progress_every > 0 and (
                step == 1
                or step % int(args.progress_every) == 0
                or step == TARGET_STEP
            ):
                print(f"[PRECOMMIT_REPLAY] step={step}/{TARGET_STEP}", flush=True)

            if step == TARGET_STEP:
                target_finished = True
                break

            if terminated or truncated:
                terminal_reason = str(
                    info.get("terminal_reason")
                    or ("terminated" if terminated else "truncated")
                )
                break
    finally:
        env.sofa_simulation.animate = original_animate

    action_sha = _sha256_actions(actions)
    hook_result = hook.finalize()
    payload: dict[str, Any] = {
        "test": "B02 step776/substep1 test-only pre-commit SOFA integration",
        "training_started": False,
        "production_files_modified": False,
        "seed": int(args.seed),
        "target_rl_step": TARGET_STEP,
        "target_substep": TARGET_SUBSTEP,
        "physics_dt_s": DT,
        "action_sha256": action_sha,
        "expected_action_sha256": EXPECTED_ACTION_SHA256,
        "action_prefix_status": (
            "PASS" if action_sha == EXPECTED_ACTION_SHA256 else "FAIL"
        ),
        "candidate_artifact": {
            "accepted_source": candidate.get("accepted_source"),
            "free_source": candidate.get("free_source"),
        },
        "hook": hook_result,
        "terminal_reason": terminal_reason,
        "target_finished": bool(target_finished),
        "wall_s": float(time.perf_counter() - started),
    }

    decision = "INCONCLUSIVE"
    reason = "TARGET_NOT_REACHED"
    non_finite = False

    if target_finished and hook_result.get("injected"):
        q_final = _as_array(beam_dofs.position)
        q_final_free = _as_array(beam_dofs.free_position)
        payload["final_committed"] = adapter.measure(q_final, spacing_m=0.00025)
        payload["final_independent_dense"] = adapter.measure(
            q_final, spacing_m=0.00001
        )
        payload["final_vs_offline_accepted"] = _state_delta(
            np.asarray(candidate["accepted"], dtype=np.float64), q_final
        )
        payload["final_free_vs_committed"] = _state_delta(q_final_free, q_final)
        non_finite = not (_finite(q_final) and _finite(q_final_free))

        c_main = payload["final_committed"]["min_clearance_m"]
        c_dense = payload["final_independent_dense"]["min_clearance_m"]
        if non_finite:
            decision, reason = "FAIL", "NON_FINITE_FINAL_STATE"
        elif c_main < -0.00005 or c_dense < -0.00005:
            decision, reason = "FAIL", "MEANINGFUL_POST_COMMIT_PENETRATION"
        elif c_main >= -NUM_TOL_M and c_dense >= -NUM_TOL_M:
            decision = "PASS"
            reason = "SAFE_CANDIDATE_SURVIVED_REAL_PRECOMMIT_AND_NATIVE_SOLVE"
        else:
            decision = "PARTIAL"
            reason = "SMALL_RESIDUAL_OR_NUMERICAL_VIOLATION"
    else:
        if hook_result.get("status") == "INCONCLUSIVE":
            reason = hook_result.get("reason", "PRECOMMIT_HOOK_INCONCLUSIVE")
        elif not hook_result.get("stage_seen"):
            reason = "COLLISION_BEGIN_EVENT_NOT_EXPOSED_TO_SOFAPYTHON"
        elif not hook_result.get("injected"):
            decision, reason = "FAIL", "SAFE_CANDIDATE_NOT_INJECTED"

    payload["non_finite"] = bool(non_finite)
    payload["decision"] = decision
    payload["reason"] = reason
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_report(payload, report_path)

    print(
        json.dumps(
            {
                "decision": decision,
                "reason": reason,
                "action_sha256": action_sha,
                "hook_status": hook_result.get("status"),
                "stage_seen": hook_result.get("stage_seen"),
                "injected": hook_result.get("injected"),
                "mapping_refresh_method": hook_result.get(
                    "mapping_refresh_method"
                ),
                "final_clearance_mm": (
                    payload.get("final_committed") or {}
                ).get("min_clearance_mm"),
                "independent_dense_clearance_mm": (
                    payload.get("final_independent_dense") or {}
                ).get("min_clearance_mm"),
                "output": str(output),
                "report": str(report_path),
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
