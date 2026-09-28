"""Test-only 32-env worker for Fast FD + Generic throughput benchmarking.

Each worker owns a real SOFA21 MCREnv.  The fast Beam-level SDF unilateral
constraint is attached after reset and runs on both 5 ms physics substeps.

The wrapper exposes compact per-step safety/performance telemetry through info
so the parent SubprocVecEnv benchmark can aggregate 32 workers without touching
production environment code.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import time
from typing import Any

import gymnasium as gym
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
    DT,
    _as_array,
    _create_env,
    _finite,
)
from beam_linearized_unilateral import (
    DENSE_SPACING_M,
    MARGIN_M,
    load_plugin,
)
from beam_unilateral_fast_full_episode_v2 import (
    FastFullEpisodeBeamUnilateralControllerV2,
)


EXPECTED_SUBSTEPS = 2
COMMITTED_PENETRATION_LIMIT_M = 1.0e-6
FLOAT_TOL_M = 1.0e-12
INFO_KEY = "_mcr_fast_fd_v2_component_profile"


class FastFDVectorBenchmarkEnv(gym.Wrapper):
    """One real SOFA worker with test-only Fast FD safety instrumentation."""

    def __init__(
        self,
        *,
        worker_rank: int,
        plugin_path: str,
        base_seed: int,
    ):
        # Set solver/dt inside each spawn worker before scene construction.
        os.environ["MCR_CONSTRAINT_SOLVER"] = "generic"
        os.environ["MCR_SOFA_DT"] = str(DT)

        self.worker_rank = int(worker_rank)
        self.base_seed = int(base_seed)
        self.plugin_path = str(plugin_path)
        self._plugin_handle = load_plugin(Path(self.plugin_path))

        env = _create_env()
        super().__init__(env)

        self._attached = False
        self._original_animate = None
        self._planner = None
        self._constraint = None
        self._adapter = None
        self._beam_dofs = None

        self._episode_rl_step = 0
        self._substep_in_step = 0
        self._substep_records: list[dict[str, Any]] = []

    def _attach_after_reset(self) -> None:
        if self._attached:
            return

        instrument = self.env.mcr_controller_sofa.instrument.InstrumentCombined
        self._beam_dofs = instrument.getObject("DOFs")
        self._adapter = AdapterCompat(self.env, instrument)

        self._constraint = instrument.addObject(
            "BeamLinearizedUnilateralConstraint",
            name=f"FastFDVectorBenchConstraint{self.worker_rank:02d}",
            enabled=False,
            rowOffsets=[],
            dofIndices=[],
            linearJacobian=[],
            angularJacobian=[],
            freeViolations=[],
            sourceClearances=[],
        )
        self._constraint.init()

        self._planner = FastFullEpisodeBeamUnilateralControllerV2(
            name=f"FastFDVectorBenchController{self.worker_rank:02d}",
            beam_dofs=self._beam_dofs,
            adapter=self._adapter,
            constraint=self._constraint,
            requested_margin_m=MARGIN_M,
        )
        instrument.addObject(self._planner)

        self._original_animate = self.env.sofa_simulation.animate
        self.env.sofa_simulation.animate = self._traced_animate
        self._attached = True

    def _traced_animate(self, root, dt):
        self._substep_in_step += 1
        sub = int(self._substep_in_step)

        self._planner.current_rl_step = int(self._episode_rl_step)
        self._planner.current_substep = sub

        substep_start_perf = time.perf_counter()
        substep_start_cpu = time.process_time()
        result = self._original_animate(root, dt)
        animate_end_perf = time.perf_counter()
        animate_end_cpu = time.process_time()

        rec = self._planner.take_record() or {
            "rl_step": int(self._episode_rl_step),
            "substep": sub,
            "planning_status": "FAIL",
            "reason": "COLLISION_BEGIN_EVENT_NOT_OBSERVED",
            "rows_required": False,
            "rows_armed": False,
            "row_count": 0,
            "row_build_runtime_s": 0.0,
        }

        try:
            active_count = int(self._constraint.activeCount.value)
        except Exception:
            active_count = -1
        rec["constraint_active_count"] = active_count

        q_committed = _as_array(self._beam_dofs.position)
        committed_finite = _finite(q_committed)

        committed_start_perf = time.perf_counter()
        committed_start_cpu = time.process_time()
        if committed_finite:
            committed = self._adapter.measure(
                q_committed,
                spacing_m=DENSE_SPACING_M,
            )
            committed_clearance = float(committed["min_clearance_m"])
        else:
            committed_clearance = float("nan")

        rec["committed_dense_validation_s"] = time.perf_counter() - committed_start_perf
        rec["committed_dense_validation_cpu_s"] = time.process_time() - committed_start_cpu
        rec["post_planner_sofa_remainder_s"] = max(0.0, animate_end_perf - float(rec.get("planner_end_perf", animate_end_perf)))
        rec["post_planner_sofa_remainder_cpu_s"] = max(0.0, animate_end_cpu - float(rec.get("planner_end_cpu", animate_end_cpu)))
        rec["committed_clearance_m"] = committed_clearance
        rec["committed_penetration_m"] = (
            max(0.0, -committed_clearance)
            if np.isfinite(committed_clearance)
            else float("nan")
        )

        rows_required = bool(rec.get("rows_required", False))
        intended_rows = int(rec.get("row_count", 0))
        if rows_required:
            active_mismatch = bool(
                intended_rows <= 0
                or active_count != intended_rows
                or not bool(rec.get("rows_armed", False))
            )
        else:
            active_mismatch = active_count not in (0, -1)
        rec["constraint_active_mismatch"] = active_mismatch

        forbidden = bool(
            rec.get("uses_collision_dofs_as_constraint_source", False)
            or rec.get("q_candidate_solver_used", False)
            or rec.get("native_candidate_injection_used", False)
            or rec.get("writes_free_position", False)
            or rec.get("writes_committed_position", False)
            or rec.get("projection_used", False)
            or rec.get("rollback_used", False)
            or rec.get("action_shielding_used", False)
        )
        rec["forbidden_fallback_or_write"] = forbidden

        q_free_clearance = float(
            rec.get("q_free_clearance_m", np.nan)
        )
        rec["non_finite"] = bool(
            not committed_finite
            or not np.isfinite(committed_clearance)
            or not np.isfinite(q_free_clearance)
        )

        invariant_failure = False
        if rows_required:
            invariant_failure = bool(
                rec.get("support_mode")
                != "exact_selected_point_local_fd"
                or float(
                    rec.get(
                        "selected_point_reconstruction_max_error_m",
                        np.inf,
                    )
                )
                > 1.0e-10
                or int(rec.get("batched_fd_sdf_query_count", 0)) != 1
                or int(
                    rec.get(
                        "full_beam_profile_evaluations_after_selection",
                        -1,
                    )
                )
                != 0
            )
        rec["fast_builder_invariant_failure"] = invariant_failure

        failure_reason = None
        if str(rec.get("planning_status")) == "FAIL":
            failure_reason = str(
                rec.get("reason", "FAST_PLANNER_FAILED")
            )
        elif forbidden:
            failure_reason = "FORBIDDEN_FALLBACK_OR_POSITION_WRITE"
        elif active_mismatch:
            failure_reason = "CONSTRAINT_ACTIVE_COUNT_MISMATCH"
        elif invariant_failure:
            failure_reason = "FAST_BUILDER_INVARIANT_FAILURE"
        elif rec["non_finite"]:
            failure_reason = "NON_FINITE_POST_SUBSTEP_STATE"
        elif committed_clearance < -COMMITTED_PENETRATION_LIMIT_M:
            failure_reason = "COMMITTED_DENSE_PENETRATION_VIOLATION"

        rec["failure_reason"] = failure_reason
        rec["total_physics_substep_s"] = time.perf_counter() - substep_start_perf
        rec["total_physics_substep_cpu_s"] = time.process_time() - substep_start_cpu
        components = ("q_prev_measure", "q_free_measure", "gate", "fast_builder_base_profile", "selected_dense_indices", "support_mapping", "fd_point_generation", "fd_batched_sdf", "jacobian_assembly", "constraint_write", "post_planner_sofa_remainder", "committed_dense_validation")
        rec["other_remainder_s"] = max(0.0, rec["total_physics_substep_s"] - sum(float(rec.get(k + "_s", 0.0)) for k in components))
        rec["other_remainder_cpu_s"] = max(0.0, rec["total_physics_substep_cpu_s"] - sum(float(rec.get(k + "_cpu_s", 0.0)) for k in components))
        rec["full_q_free_dense_profile_count"] = int(rec.get("q_free_external_dense_count", 0)) + int(rec.get("q_free_internal_dense_count", 0))
        if int(rec.get("q_prev_full_dense_profile_count", -1)) != 0 or int(rec.get("q_free_external_dense_count", -1)) != 1 or int(rec.get("q_free_internal_dense_count", -1)) != 0 or rec["full_q_free_dense_profile_count"] != 1:
            rec["failure_reason"] = rec.get("failure_reason") or "FAST_FD_V2_DENSE_PROFILE_COUNT_MISMATCH"
        self._substep_records.append(rec)
        return result

    @staticmethod
    def _min_finite(values, default=None):
        values = [float(v) for v in values if np.isfinite(float(v))]
        return min(values) if values else default

    @staticmethod
    def _max_finite(values, default=None):
        values = [float(v) for v in values if np.isfinite(float(v))]
        return max(values) if values else default

    def _summarize_step(self, wall_s: float, cpu_s: float) -> dict[str, Any]:
        records = list(self._substep_records)
        row_records = [r for r in records if bool(r.get("rows_required"))]
        free_values = [
            float(r.get("q_free_clearance_m", np.nan))
            for r in records
        ]
        committed_values = [
            float(r.get("committed_clearance_m", np.nan))
            for r in records
        ]
        failures = [
            str(r["failure_reason"])
            for r in records
            if r.get("failure_reason")
        ]

        return {
            "worker_rank": self.worker_rank,
            "episode_rl_step": int(self._episode_rl_step),
            "physics_substeps": len(records),
            "step_wall_s": float(wall_s),
            "step_cpu_s": float(cpu_s),
            "row_build_substeps": len(row_records),
            "row_build_runtime_s": float(
                sum(
                    float(r.get("row_build_runtime_s", 0.0))
                    for r in row_records
                )
            ),
            "row_build_runtimes_s": [
                float(r.get("row_build_runtime_s", 0.0))
                for r in row_records
            ],
            "row_counts": [
                int(r.get("row_count", 0))
                for r in row_records
            ],
            "row_count_total": int(
                sum(int(r.get("row_count", 0)) for r in row_records)
            ),
            "negative_free_substeps": int(
                sum(
                    1
                    for v in free_values
                    if np.isfinite(v) and v < 0.0
                )
            ),
            "below_minus_0p001mm_free_substeps": int(
                sum(
                    1
                    for v in free_values
                    if np.isfinite(v)
                    and v < -COMMITTED_PENETRATION_LIMIT_M
                )
            ),
            "min_free_clearance_m": self._min_finite(free_values),
            "min_committed_clearance_m": self._min_finite(
                committed_values
            ),
            "max_committed_penetration_m": self._max_finite(
                [
                    max(0.0, -v)
                    for v in committed_values
                    if np.isfinite(v)
                ],
                default=0.0,
            ),
            "committed_violation_count": int(
                sum(
                    1
                    for v in committed_values
                    if np.isfinite(v)
                    and v < -COMMITTED_PENETRATION_LIMIT_M
                )
            ),
            "active_count_mismatch_count": int(
                sum(
                    bool(r.get("constraint_active_mismatch", False))
                    for r in records
                )
            ),
            "fast_builder_invariant_failure_count": int(
                sum(
                    bool(r.get("fast_builder_invariant_failure", False))
                    for r in records
                )
            ),
            "non_finite_count": int(
                sum(bool(r.get("non_finite", False)) for r in records)
            ),
            "planning_failure_count": int(
                sum(
                    str(r.get("planning_status")) == "FAIL"
                    for r in records
                )
            ),
            "failure_reasons": failures,
            "batched_fd_point_count": int(
                sum(
                    int(r.get("batched_fd_point_count", 0))
                    for r in row_records
                )
            ),
            "batched_fd_sdf_query_count": int(
                sum(
                    int(r.get("batched_fd_sdf_query_count", 0))
                    for r in row_records
                )
            ),
            "post_selection_full_beam_profile_count": int(
                sum(
                    int(
                        r.get(
                            "full_beam_profile_evaluations_after_selection",
                            0,
                        )
                    )
                    for r in row_records
                )
            ),
            "profile_substeps": [dict(r) for r in records],
            "env_postphysics_remainder_s": max(0.0, float(wall_s) - sum(float(r.get("total_physics_substep_s", 0.0)) for r in records)),
            "env_postphysics_remainder_cpu_s": max(0.0, float(cpu_s) - sum(float(r.get("total_physics_substep_cpu_s", 0.0)) for r in records)),
            "support_modes": [
                str(r.get("support_mode", "MISSING"))
                for r in row_records
            ],
        }

    def reset(self, *, seed=None, options=None):
        if seed is None:
            seed = self.base_seed + self.worker_rank

        # MCREnv intentionally unloads/rebuilds its SOFA scene on reset.
        # Restore the module animate function before that reload, then attach
        # fresh constraint/controller objects to the newly created scene.
        if self._attached and self._original_animate is not None:
            try:
                self.env.sofa_simulation.animate = self._original_animate
            except Exception:
                pass
        self._attached = False
        self._original_animate = None
        self._planner = None
        self._constraint = None
        self._adapter = None
        self._beam_dofs = None

        observation, info = self.env.reset(
            seed=int(seed),
            options=options,
        )
        self._attach_after_reset()

        self._episode_rl_step = 0
        self._substep_in_step = 0
        self._substep_records = []
        self._planner.reset()
        return observation, info

    def step(self, action):
        self._episode_rl_step += 1
        self._substep_in_step = 0
        self._substep_records = []

        wall0 = time.perf_counter()
        cpu0 = time.process_time()
        observation, reward, terminated, truncated, info = self.env.step(action)
        cpu_s = float(time.process_time() - cpu0)
        wall_s = float(time.perf_counter() - wall0)

        bench = self._summarize_step(wall_s, cpu_s)
        if self._substep_in_step != EXPECTED_SUBSTEPS:
            bench["failure_reasons"].append(
                "PHYSICS_SUBSTEP_COUNT_MISMATCH"
            )
            bench["physics_substep_count_mismatch"] = True
        else:
            bench["physics_substep_count_mismatch"] = False

        if not isinstance(info, dict):
            info = {}
        info = dict(info)
        info[INFO_KEY] = bench
        return observation, reward, terminated, truncated, info

    def close(self):
        if self._attached and self._original_animate is not None:
            try:
                self.env.sofa_simulation.animate = self._original_animate
            except Exception:
                pass
        try:
            return self.env.close()
        finally:
            self._attached = False


def make_fast_fd_v2_profile_env(
    rank: int,
    plugin_path: str,
    base_seed: int,
):
    """Cloudpickle-safe SubprocVecEnv factory."""

    def _init():
        return FastFDVectorBenchmarkEnv(
            worker_rank=int(rank),
            plugin_path=str(plugin_path),
            base_seed=int(base_seed),
        )

    return _init
