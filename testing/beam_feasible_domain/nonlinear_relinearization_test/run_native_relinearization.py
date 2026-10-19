#!/usr/bin/env python3
"""One saved B02 state; 1/2/3 genuine native same-substep constraint solves."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import sys
import traceback
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
TEST_DIR = Path(__file__).resolve().parent
RESULTS = TEST_DIR / "results"
FAIL_DIR = ROOT / "testing/beam_feasible_domain/formal_training_failure"
THREE_DIR = ROOT / "testing/beam_feasible_domain/three_cause_offline/results"
BASELINE_NPZ = THREE_DIR / "b02_worker23_episode3_step2009_normalized.npz"
BASELINE_META = FAIL_DIR / "final_stress_metadata.json"
SNAPSHOT = FAIL_DIR / "final_stress_snapshot.npz"
RUN = ROOT.parent / "training_runs/ppo_beam_unilateral_audit_formal_100ep_20260928_224143"
PLUGIN = RESULTS / "native_build/lib/libMCRNonlinearRelinearizationTest.so"
PROD_PLUGIN = ROOT / "mcr_sim/native/beam_unilateral/_build/lib/libMCRBeamLinearizedUnilateral.so"
MARGIN_M = 0.0001
STATE_TOL_M = 1.0e-10
QUAT_TOL = 1.0e-10

sys.path[:0] = [str(ROOT), str(ROOT / "training/py"), str(FAIL_DIR)]
import capture_failure as capture  # noqa: E402
from mcr_sim import beam_sdf_unilateral_fast as fast  # noqa: E402
from mcr_sim.beam_safety_env import BeamSafetyMCREnv  # noqa: E402
import train_ppo_beam_unilateral as factory  # noqa: E402
import Sofa  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

with np.load(SNAPSHOT, allow_pickle=False) as _source:
    SNAP = {key: np.array(_source[key]) for key in _source.files}
META = json.loads(BASELINE_META.read_text(encoding="utf-8"))
CASE = META["case"]
TARGET = int(META["rl_step"])
WORKER = int(CASE["seed"]) - 42
RAW_ACTIONS = np.asarray(SNAP["raw_actions"], dtype=np.float32)
ACTION_SHA = hashlib.sha256(RAW_ACTIONS.tobytes()).hexdigest()
RUN_ARM = None
RUN_TOLERANCE = 1e-6
RUN_PASSES = 3
RUN_ACTION_SHA = ACTION_SHA
LAST_ROW_BUILD_WALL_MS = None
ORIGINAL_ROW_BUILDER = None
REGISTRY = {}
ORIGINAL_SIM_INIT = Sofa.Simulation.init


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def _state_sha(q):
    return hashlib.sha256(np.asarray(q, dtype="<f8").tobytes(order="C")).hexdigest()


def _timed_row_builder(*args, **kwargs):
    global LAST_ROW_BUILD_WALL_MS
    if ORIGINAL_ROW_BUILDER is None:
        raise RuntimeError("ROW_BUILDER_TIMER_NOT_INITIALIZED")
    started = time.perf_counter()
    rows = ORIGINAL_ROW_BUILDER(*args, **kwargs)
    LAST_ROW_BUILD_WALL_MS = (time.perf_counter() - started) * 1000.0
    return rows


def _read_rows(component, selected):
    def data(name, dtype=float):
        return np.asarray(getattr(component, name).value, dtype=dtype).copy()
    return {
        "row_offsets": data("rowOffsets", np.int64).reshape(-1),
        "dof_indices": data("dofIndices", np.int64).reshape(-1),
        "linear_jacobian": data("linearJacobian", float).reshape((-1, 3)),
        "angular_jacobian": data("angularJacobian", float).reshape((-1, 3)),
        "free_violations": data("freeViolations", float).reshape(-1),
        "source_clearances": data("sourceClearances", float).reshape(-1),
        "selected_dense_indices": np.asarray(selected, dtype=np.int64).reshape(-1),
    }


def _linear_g(q_lin, q_eval, rows, convert_angular):
    ids, jt, ja = rows["dof_indices"], rows["linear_jacobian"], rows["angular_jacobian"]
    offsets = rows["row_offsets"]
    dt = q_eval[:, :3] - q_lin[:, :3]
    r0, r1 = Rotation.from_quat(q_lin[:, 3:7]), Rotation.from_quat(q_eval[:, 3:7])
    dr_world = (r1 * r0.inv()).as_rotvec()
    angular = r0[ids].apply(ja) if convert_angular else ja
    blocks = np.einsum("ij,ij->i", jt, dt[ids])
    blocks += np.einsum("ij,ij->i", angular, dr_world[ids])
    gaps = rows["free_violations"].copy()
    for row in range(len(gaps)):
        gaps[row] += float(np.sum(blocks[offsets[row]:offsets[row + 1]]))
    return gaps


def _stage_measurement(q_lin, q_stage, dense_clearance, rows):
    selected = rows["selected_dense_indices"]
    raw = _linear_g(q_lin, q_stage, rows, False)
    world = _linear_g(q_lin, q_stage, rows, True)
    actual_gap = dense_clearance[selected] - MARGIN_M
    return {
        "committed_clearance_mm": float(np.min(dense_clearance) * 1000.0),
        "dense_worst_index": int(np.argmin(dense_clearance)),
        "selected_count": int(len(selected)),
        "row_count": int(len(rows["free_violations"])),
        "min_linear_row_gap_mm": float(np.min(raw) * 1000.0),
        "min_linear_row_gap_world_angular_mm": float(np.min(world) * 1000.0),
        "max_abs_nonlinear_sdf_error_mm": float(np.max(np.abs(actual_gap - raw)) * 1000.0),
        "max_abs_nonlinear_sdf_error_world_angular_mm": float(np.max(np.abs(actual_gap - world)) * 1000.0),
        "min_selected_true_sdf_gap_mm": float(np.min(actual_gap) * 1000.0),
        "worst_dense_is_selected": bool(int(np.argmin(dense_clearance)) in set(selected.tolist())),
    }
class RelinearizationController(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        kwargs["listening"] = True
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.env = self.solver = self.constraint = self.dofs = self.adapter = None
        self.active = False
        self.branch = None
        self.q_prev = self.q_free = self.free_profile = None
        self.linearization_state = None
        self.rows = None
        self.records = {}
        self.initial_row_build_wall_ms = None
        self.pending_row_build_wall_ms = None

    def bind(self, env, solver):
        self.env = env
        self.solver = solver
        self.constraint = env._beam_constraint
        self.dofs = env._beam_dofs
        self.adapter = env._beam_adapter

    def arm(self, q_prev, q_free, free_profile, branch,
            initial_row_build_wall_ms=None):
        self.active = True
        self.branch = branch
        self.q_prev = np.asarray(q_prev, dtype=np.float64).copy()
        self.q_free = np.asarray(q_free, dtype=np.float64).copy()
        self.free_profile = np.asarray(free_profile, dtype=np.float64).copy()
        self.linearization_state = self.q_free.copy()
        self.rows = _read_rows(self.constraint, fast._select_rows(self.free_profile))
        self.records = {}
        self.initial_row_build_wall_ms = initial_row_build_wall_ms
        self.pending_row_build_wall_ms = None

    def onScriptEvent(self, event):
        if not self.active or not isinstance(event, dict):
            return
        event_name = str(event.get("event_name", ""))
        prefix = "MCR_NONLINEAR_RELINEARIZATION_STAGE_"
        if not event_name.startswith(prefix):
            return
        stage = int(event_name[len(prefix):])
        try:
            q_stage = fast._as_array(self.dofs.position).copy()
            points, profile = fast._profile(self.adapter, q_stage)
            rows = self.rows
            q_lin = self.linearization_state.copy()
            solver = self.solver
            record = {
                "stage": stage, "branch": self.branch,
                "q_linearization": q_lin, "q_stage_event": q_stage,
                "dense_clearance": profile.copy(),
                "rows": {k: v.copy() for k, v in rows.items()},
                "measurement": _stage_measurement(q_lin, q_stage, profile, rows),
                "solver_iterations": int(solver.currentIterations.value),
                "solver_error": float(solver.currentError.value),
                "solver_tolerance": float(solver.tolerance.value),
                "row_build_wall_ms": (self.initial_row_build_wall_ms if stage == 1
                                       else self.pending_row_build_wall_ms),
                "solver_constraint_count": int(solver.currentNumConstraints.value),
                "active_count": int(self.constraint.activeCount.value),
                "native_solve_calls_cumulative": int(solver.nativeSolveCalls.value),
                "completed_stages": int(solver.completedStages.value),
                "mapping_refresh_count": int(solver.mappingRefreshCount.value),
                "relinearization_status_at_callback": str(solver.relinearizationStatus.value),
                "event_stage": event_name,
            }
            self.records[stage] = record
            requested = int(solver.requestedPasses.value)
            if stage < requested:
                new_rows = fast.build_rows_fast_local_fd(
                    adapter=self.adapter, q_prev=q_stage, q_free=q_stage,
                    dense_points=points, dense_clearance=profile,
                    requested_margin_m=MARGIN_M,
                )
                self.pending_row_build_wall_ms = LAST_ROW_BUILD_WALL_MS
                fast.write_rows(self.constraint, new_rows)
                planner_record = getattr(self.env._beam_planner, "_record", None)
                if isinstance(planner_record, dict):
                    planner_record["row_count"] = int(len(new_rows["free_violations"]))
                    planner_record["rows_armed"] = bool(len(new_rows["free_violations"]) > 0)
                self.linearization_state = q_stage.copy()
                self.rows = {
                    "row_offsets": np.asarray(new_rows["row_offsets"], dtype=np.int64),
                    "dof_indices": np.asarray(new_rows["dof_indices"], dtype=np.int64),
                    "linear_jacobian": np.asarray(new_rows["linear_jacobian"], dtype=np.float64).reshape((-1, 3)),
                    "angular_jacobian": np.asarray(new_rows["angular_jacobian"], dtype=np.float64).reshape((-1, 3)),
                    "free_violations": np.asarray(new_rows["free_violations"], dtype=np.float64),
                    "source_clearances": np.asarray(new_rows["source_clearances"], dtype=np.float64),
                    "selected_dense_indices": np.asarray(new_rows["selected_dense_indices"], dtype=np.int64),
                }
                solver.stageEventStatus.value = f"ROWS_REBUILT_FOR_STAGE_{stage + 1}"
            else:
                solver.stageEventStatus.value = f"CAPTURED_STAGE_{stage}"
        except Exception as exc:
            self.solver.stageEventStatus.value = f"ERROR:{type(exc).__name__}:{exc}"
            self.records[-stage] = {"stage": stage, "error": traceback.format_exc()}


class TargetDone(BaseException):
    pass


class ForkEnv(BeamSafetyMCREnv):
    def __init__(self, *args, **kwargs):
        self._test_registry = self._test_solver = self._test_controller = None
        self._test_branch = "not_armed"
        self._test_child_process = False
        self._test_forked = False
        self._test_target_seen = False
        self._test_target_animate_calls = 0
        self._test_state_match = None
        self._test_branch_summary = None
        super().__init__(*args, **kwargs)

    def _validate_solver(self):
        solvers = [obj.getClassName() for obj in self._sofa_root_node.objects
                   if obj.getClassName() in ("GenericConstraintSolver", "LCPConstraintSolver",
                                             "TestOnlyRelinearizingGenericConstraintSolver")]
        if solvers != ["TestOnlyRelinearizingGenericConstraintSolver"]:
            raise RuntimeError(f"TEST_SOLVER_REPLACEMENT_FAILED:{solvers}")

    def _attach_beam_safety(self):
        super()._attach_beam_safety()
        self._test_registry = REGISTRY.get(id(self._sofa_root_node))
        if not self._test_registry:
            raise RuntimeError("TEST_NATIVE_SOLVER_REGISTRY_NOT_FOUND")
        self._test_solver = self._test_registry["solver"]
        self._test_controller = self._test_registry["controller"]
        self._test_controller.bind(self, self._test_solver)
        planner = self._beam_planner
        original = planner._collision_begin

        def target_hook(source):
            original(source)
            if (self._beam_rl_step != TARGET or self._beam_substep != 1
                    or self._test_target_seen):
                return
            self._test_target_seen = True
            q_prev = fast._as_array(self._beam_dofs.position).copy()
            q_free = fast._as_array(self._beam_dofs.free_position).copy()
            expected_prev = np.asarray(SNAP["q_prev"], dtype=np.float64)
            expected_free = np.asarray(SNAP["q_free"], dtype=np.float64)
            errors = {
                "q_prev_translation_max_abs_m": float(np.max(np.abs(q_prev[:, :3] - expected_prev[:, :3]))),
                "q_prev_quaternion_max_abs": float(np.max(np.abs(q_prev[:, 3:7] - expected_prev[:, 3:7]))),
                "q_free_translation_max_abs_m": float(np.max(np.abs(q_free[:, :3] - expected_free[:, :3]))),
                "q_free_quaternion_max_abs": float(np.max(np.abs(q_free[:, 3:7] - expected_free[:, 3:7]))),
            }
            self._test_state_match = {
                **errors,
                "pass": bool(errors["q_prev_translation_max_abs_m"] <= STATE_TOL_M
                              and errors["q_free_translation_max_abs_m"] <= STATE_TOL_M
                              and errors["q_prev_quaternion_max_abs"] <= QUAT_TOL
                              and errors["q_free_quaternion_max_abs"] <= QUAT_TOL),
            }
            if not self._test_state_match["pass"]:
                self._test_branch = "STATE_MISMATCH"
                return
            _, free_clearance = fast._profile(self._beam_adapter, q_free)
            self._test_solver.requestedPasses.value = 1
            self._test_solver.relinearizationEnabled.value = True
            if RUN_ARM is not None:
                self._test_branch = RUN_ARM
                # Change tolerance only after the saved q_prev/q_free state has
                # been proven identical, immediately before the target solve.
                self._test_solver.tolerance.value = RUN_TOLERANCE
                self._test_solver.requestedPasses.value = RUN_PASSES
                self._test_controller.arm(
                    q_prev, q_free, free_clearance, RUN_ARM,
                    initial_row_build_wall_ms=LAST_ROW_BUILD_WALL_MS,
                )
                return
            for passes in (2, 3):
                pid = os.fork()
                if pid == 0:
                    self._test_child_process = True
                    self._test_branch = f"{passes}_native_passes"
                    self._test_solver.requestedPasses.value = passes
                    self._test_controller.arm(q_prev, q_free, free_clearance, self._test_branch)
                    return
                _, status = os.waitpid(pid, 0)
                if not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 0:
                    print(f"CHILD_BRANCH_FAILED passes={passes} wait_status={status}", flush=True)

            self._test_branch = "1_native_pass"
            self._test_controller.arm(q_prev, q_free, free_clearance, self._test_branch)

        planner._collision_begin = target_hook

    def _save_stage(self, stage, record, final_q, branch_passes):
        q_commit = np.asarray(record["q_stage_event"], dtype=np.float64)
        profile = np.asarray(record["dense_clearance"], dtype=np.float64)
        event_to_final_translation = float(np.max(np.abs(q_commit[:, :3] - final_q[:, :3])))
        event_to_final_quaternion = float(np.max(np.abs(q_commit[:, 3:7] - final_q[:, 3:7])))
        if stage == branch_passes:
            q_commit = np.asarray(final_q, dtype=np.float64).copy()
            _, profile = fast._profile(self._beam_adapter, q_commit)
        rows = record["rows"]
        stem = f"native_{branch_passes}pass_stage{stage}"
        output = RESULTS / f"{stem}.npz"
        np.savez_compressed(
            output,
            q_prev=self._test_controller.q_prev,
            q_free=self._test_controller.q_free,
            q_committed=q_commit,
            q_linearization_state=record["q_linearization"],
            row_offsets=rows["row_offsets"],
            dof_indices=rows["dof_indices"],
            linear_jacobian=rows["linear_jacobian"],
            angular_jacobian=rows["angular_jacobian"],
            free_violations=rows["free_violations"],
            source_clearances=rows["source_clearances"],
            selected_dense_indices=rows["selected_dense_indices"],
            q_free_dense_clearance=self._test_controller.free_profile,
            q_committed_dense_clearance=profile,
        )
        meta = dict(META)
        meta["native_test_stage"] = {
            "branch": self._test_branch, "stage": stage,
            "requested_passes": branch_passes,
            "native_solve_calls_cumulative": record["native_solve_calls_cumulative"],
            "rows_rebuilt_from_native_state": bool(stage > 1),
        }
        (RESULTS / f"{stem}.json").write_text(
            json.dumps(_clean(meta), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return {
            "stage": stage, "npz": output.name,
            "committed_clearance_mm": float(np.min(profile) * 1000.0),
            "event_vs_final_translation_max_m": event_to_final_translation,
            "event_vs_final_quaternion_max": event_to_final_quaternion,
            **{k: v for k, v in record.items()
               if k not in ("q_linearization", "q_stage_event", "dense_clearance", "rows")},
        }

    def _finish_target_branch(self, error=None):
        q_final = fast._as_array(self._beam_dofs.position).copy()
        solver = self._test_solver
        passes = int(solver.requestedPasses.value)
        branch_summary = {
            "branch": self._test_branch,
            "requested_passes": passes,
            "state_match": self._test_state_match,
            "native_status": str(solver.relinearizationStatus.value),
            "native_solve_calls": int(solver.nativeSolveCalls.value),
            "completed_stages": int(solver.completedStages.value),
            "mapping_refresh_count": int(solver.mappingRefreshCount.value),
            "solver_iterations_last": int(solver.currentIterations.value),
            "solver_error_last": float(solver.currentError.value),
            "solver_tolerance": float(solver.tolerance.value),
            "target_substep2_executed": False,
            "target_animate_calls": int(self._test_target_animate_calls),
            "error": error,
            "stages": [],
        }
        if self._test_state_match and self._test_state_match["pass"]:
            for stage in sorted(k for k in self._test_controller.records if k > 0):
                branch_summary["stages"].append(
                    self._save_stage(stage, self._test_controller.records[stage], q_final, passes)
                )
        _, final_profile = fast._profile(self._beam_adapter, q_final)
        branch_summary["final_committed_clearance_mm"] = float(np.min(final_profile) * 1000.0)
        self._test_branch_summary = branch_summary
        (RESULTS / f"branch_{self._test_branch}.json").write_text(
            json.dumps(_clean(branch_summary), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    def _beam_traced_animate(self, root, dt):
        try:
            result = super()._beam_traced_animate(root, dt)
        except Exception as exc:
            if self._beam_rl_step == TARGET and self._beam_substep == 1:
                self._finish_target_branch(f"{type(exc).__name__}:{exc}")
                raise TargetDone()
            raise
        if self._beam_rl_step == TARGET:
            self._test_target_animate_calls += 1
        if self._beam_rl_step == TARGET and self._beam_substep == 1:
            self._finish_target_branch(None)
            raise TargetDone()
        return result


def _install_native_solver(root):
    existing = [obj for obj in root.objects if obj.getClassName() == "GenericConstraintSolver"]
    if len(existing) != 1:
        raise RuntimeError(f"EXPECTED_ONE_GENERIC_SOLVER_BEFORE_REPLACEMENT:{len(existing)}")
    old = existing[0]

    def value(name, default):
        try:
            return getattr(old, name).value
        except Exception:
            return default
    settings = {
        "name": "TestOnlyRelinearizingGenericConstraintSolver",
        # Preserve the saved trajectory tolerance during the prefix replay.
        # The requested arm tolerance is applied only at the matched target hook.
        "tolerance": value("tolerance", 1e-6),
        "maxIterations": value("maxIterations", 20000),
        "printLog": value("printLog", False),
        "relinearizationEnabled": False,
        "requestedPasses": 1,
    }
    root.removeObject(old)
    solver = root.addObject("TestOnlyRelinearizingGenericConstraintSolver", **settings)
    controller = RelinearizationController(name="TestOnlyNonlinearRelinearizationController")
    root.addObject(controller)
    REGISTRY[id(root)] = {"solver": solver, "controller": controller}


def _patched_sim_init(root, *args, **kwargs):
    _install_native_solver(root)
    return ORIGINAL_SIM_INIT(root, *args, **kwargs)


def main(argv=None):
    global RESULTS, RUN_ARM, RUN_TOLERANCE, RUN_PASSES, RUN_ACTION_SHA
    global ORIGINAL_ROW_BUILDER, LAST_ROW_BUILD_WALL_MS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("nominal", "tight"))
    parser.add_argument("--solver-tolerance", type=float)
    parser.add_argument("--native-passes", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--action-prefix-sha256")
    cli = parser.parse_args(argv)
    if cli.arm is None:
        if any(v is not None for v in (cli.solver_tolerance, cli.native_passes,
                                       cli.output_dir, cli.action_prefix_sha256)):
            parser.error("all native-arm CLI arguments require --arm")
        RUN_ARM = None
    else:
        expected_tolerance = 1e-6 if cli.arm == "nominal" else 1e-9
        if cli.solver_tolerance != expected_tolerance:
            parser.error("solver tolerance does not match requested arm")
        if cli.native_passes != 3:
            parser.error("only exactly three native passes are supported")
        if cli.output_dir is None or cli.action_prefix_sha256 is None:
            parser.error("--output-dir and --action-prefix-sha256 are required")
        if cli.action_prefix_sha256 != ACTION_SHA:
            parser.error("action prefix SHA does not match saved raw-action snapshot")
        RESULTS = cli.output_dir.expanduser().resolve()
        try:
            RESULTS.relative_to((ROOT / "testing").resolve())
        except ValueError:
            parser.error("test output must remain under repository testing/")
        RUN_ARM = cli.arm
        RUN_TOLERANCE = expected_tolerance
        RUN_PASSES = 3
        RUN_ACTION_SHA = cli.action_prefix_sha256
    ORIGINAL_ROW_BUILDER = fast.build_rows_fast_local_fd
    LAST_ROW_BUILD_WALL_MS = None
    fast.build_rows_fast_local_fd = _timed_row_builder
    RESULTS.mkdir(parents=True, exist_ok=True)
    if not BASELINE_NPZ.is_file() or not SNAPSHOT.is_file() or not PLUGIN.is_file():
        raise RuntimeError("BASELINE_OR_NATIVE_PLUGIN_MISSING")
    if CASE.get("vessel") != "B02" or CASE.get("target") != "target_04" or TARGET != 2009 or WORKER != 23:
        raise RuntimeError(f"UNEXPECTED_SNAPSHOT_CONTEXT:{CASE}:step={TARGET}:worker={WORKER}")
    if len(RAW_ACTIONS) != TARGET:
        raise RuntimeError(f"ACTION_PREFIX_LENGTH_MISMATCH:{len(RAW_ACTIONS)}:{TARGET}")
    source_action_sha = META.get("action_sha256")
    if source_action_sha and source_action_sha != ACTION_SHA:
        raise RuntimeError(f"ACTION_PREFIX_SHA_MISMATCH:{ACTION_SHA}:{source_action_sha}")

    ctypes.CDLL(str(PLUGIN), mode=getattr(ctypes, "RTLD_GLOBAL", 0))
    Sofa.Simulation.init = _patched_sim_init
    cfg = json.loads((RUN / "logs/run_config.json").read_text(encoding="utf-8"))
    args = SimpleNamespace(**cfg)
    args.local_n_envs = 1
    args.n_envs = 1
    args.distributed_rank = WORKER
    args.render = "headless"
    args.force_model = ""
    args.beam_safety_plugin = str(PROD_PLUGIN)
    factory.BeamSafetyMCREnv = ForkEnv

    result = {
        "snapshot": str(SNAPSHOT), "baseline_npz": str(BASELINE_NPZ),
        "target": {"vessel": "B02", "target": "target_04", "worker": WORKER,
                   "episode": 3, "rl_step": TARGET, "substep": 1},
        "action_count": int(len(RAW_ACTIONS)), "action_prefix_sha256": ACTION_SHA,
        "source_action_sha256": source_action_sha, "branches": {},
        "production_modified": False, "ppo_training_started": False,
        "extra_animate_calls": 0, "direct_position_write": False,
        "projection_used": False, "rollback_used": False,
        "action_shielding_used": False,
    }
    vec = None
    try:
        vec = factory.build_env(args)
        env = vec.envs[0].unwrapped
        env.set_training_model_sampling_weights({"B02": 1.0})
        env.set_training_route_sampling_weights({"B02": {4: 1.0}})
        vec.seed(int(CASE["seed"]))
        obs = vec.reset()
        if not isinstance(obs, np.ndarray):
            raise RuntimeError("RESET_OBSERVATION_INVALID")
        try:
            for index, action in enumerate(RAW_ACTIONS):
                obs, reward, done, info = vec.step(action[None, :])
                if bool(done[0]):
                    raise RuntimeError(f"PREFIX_TERMINATED_BEFORE_TARGET:{index + 1}")
                if (index + 1) % 500 == 0:
                    print(f"PREFIX {index + 1}/{TARGET}", flush=True)
            raise RuntimeError("TARGET_STOP_NOT_OBSERVED")
        except TargetDone:
            print(f"TARGET_DONE branch={env._test_branch}", flush=True)
        result["target_state_match"] = env._test_state_match
        result["parent_branch"] = env._test_branch_summary
        for passes in (2, 3):
            branch = RESULTS / f"branch_{passes}_native_passes.json"
            if branch.is_file():
                result["branches"][str(passes)] = json.loads(branch.read_text(encoding="utf-8"))
        result["branches"]["1"] = env._test_branch_summary
    except Exception as exc:
        result["status"] = "ERROR"
        result["error"] = f"{type(exc).__name__}:{exc}"
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        if vec is not None:
            vec.close()
        Sofa.Simulation.init = ORIGINAL_SIM_INIT
        if ORIGINAL_ROW_BUILDER is not None:
            fast.build_rows_fast_local_fd = ORIGINAL_ROW_BUILDER

    if env._test_child_process:
        os._exit(0)
    result["status"] = ("COMPLETED" if result.get("target_state_match", {}).get("pass")
                        else "STATE_MISMATCH_OR_TARGET_NOT_REACHED")
    (RESULTS / "native_relinearization_run.json").write_text(
        json.dumps(_clean(result), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if RUN_ARM is not None:
        branch = env._test_branch_summary
        if branch is None:
            raise RuntimeError("NATIVE_ARM_SUMMARY_MISSING")
        stages = []
        previous_committed = None
        for raw in branch.get("stages", []):
            stage_number = int(raw["stage"])
            npz_path = RESULTS / raw["npz"]
            with np.load(npz_path, allow_pickle=False) as snap:
                q_free = np.asarray(snap["q_free"], dtype=np.float64)
                q_linear = np.asarray(snap["q_linearization_state"], dtype=np.float64)
                q_committed = np.asarray(snap["q_committed"], dtype=np.float64)
                row_count = int(len(snap["free_violations"]))
            if stage_number == 1:
                if not np.array_equal(q_linear, q_free):
                    raise RuntimeError("PASS1_NOT_LINEARIZED_FROM_ORIGINAL_Q_FREE")
                parent_sha = None
            else:
                if previous_committed is None or not np.array_equal(q_linear, previous_committed):
                    raise RuntimeError(f"PASS{stage_number}_NOT_LINEARIZED_FROM_PREVIOUS_NATIVE_COMMIT")
                parent_sha = _state_sha(previous_committed)
            measurement = raw["measurement"]
            if int(measurement["row_count"]) != row_count:
                raise RuntimeError(f"PASS{stage_number}_ROW_COUNT_CAPTURE_MISMATCH")
            stage = {
                "pass_index": stage_number,
                "npz": raw["npz"],
                "solver_tolerance": float(raw["solver_tolerance"]),
                "native_solver": "GenericConstraintSolver",
                "native_solver_invocations_cumulative": int(raw["native_solve_calls_cumulative"]),
                "native_solver_iterations": int(raw["solver_iterations"]),
                "solver_error": float(raw["solver_error"]),
                "rebuilt_row_count": row_count,
                "active_count": int(raw["active_count"]),
                "native_linear_row_gap_min_mm": float(measurement["min_linear_row_gap_mm"]),
                "native_nonlinear_max_prediction_error_mm": float(measurement["max_abs_nonlinear_sdf_error_mm"]),
                "row_build_wall_ms": raw.get("row_build_wall_ms"),
                "direct_position_write": False,
                "projection_used": False,
                "rollback_used": False,
                "action_shielding_used": False,
                "modified_production": False,
            }
            if stage_number > 1:
                stage["rows_rebuilt_from_native_state"] = True
                stage["relinearized_about_committed_state_sha256"] = parent_sha
            stages.append(stage)
            previous_committed = q_committed.copy()
        arm_manifest = {
            "arm": RUN_ARM,
            "requested_tolerance": RUN_TOLERANCE,
            "physics_substeps_observed_in_fork": 1 if branch.get("target_animate_calls") == 1 else 0,
            "target_animate_calls": int(branch.get("target_animate_calls", 0)),
            "target_substep2_executed": bool(branch.get("target_substep2_executed", True)),
            "native_solver_invocations": int(branch.get("native_solve_calls", 0)),
            "mapping_refresh_count": int(branch.get("mapping_refresh_count", 0)),
            "solver_tolerance_applied_on_all_stages": bool(
                len(stages) == 3 and all(s["solver_tolerance"] == RUN_TOLERANCE for s in stages)),
            "native_solver": "GenericConstraintSolver",
            "direct_position_write": False,
            "projection_used": False,
            "rollback_used": False,
            "action_shielding_used": False,
            "modified_production": False,
            "action_prefix_sha256": ACTION_SHA,
            "state_match": branch.get("state_match"),
            "runner_error": branch.get("error"),
            "stages": stages,
        }
        (RESULTS / "arm_manifest.json").write_text(
            json.dumps(_clean(arm_manifest), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if branch.get("error") is not None:
            raise RuntimeError("NATIVE_ARM_ERROR:" + str(branch["error"]))
        if (arm_manifest["native_solver_invocations"] != 3 or
                arm_manifest["mapping_refresh_count"] != 2 or
                arm_manifest["target_animate_calls"] != 1 or
                arm_manifest["target_substep2_executed"] or
                not arm_manifest["solver_tolerance_applied_on_all_stages"]):
            raise RuntimeError("NATIVE_ARM_COUNTER_OR_TOLERANCE_MISMATCH")
    print(json.dumps(_clean(result), ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
