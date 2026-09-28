"""Fast full-episode Beam-level SDF unilateral controller.

Test-only safety route:
    real Beam Rigid3 free state
    -> one 10 um dense Beam/SDF profile
    -> select worst/near-worst dense samples
    -> reconstruct only selected Beam points under local Rigid3 FD perturbations
    -> one batched production-SDF query for all FD points
    -> scalar Beam unilateral rows
    -> GenericConstraintSolver

This keeps the validated finite-difference convention and safety formulation
unchanged.  It only removes repeated full-Beam profile evaluations during
Jacobian construction.

No q_candidate solve, native candidate injection, direct Beam position write,
projection, rollback, action shielding, or CollisionDOFs safety source is used.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import Sofa

from beam_linearized_unilateral import (
    DENSE_SPACING_M,
    MARGIN_M,
    write_rows,
)
from beam_linearized_unilateral import _profile
from beam_unilateral_fast_fd_v2 import build_rows_fast_local_fd_v2


FLOAT_TOL_M = 1.0e-12


class FastFullEpisodeBeamUnilateralControllerV2(Sofa.Core.Controller):
    def __init__(
        self,
        *,
        beam_dofs,
        adapter,
        constraint,
        requested_margin_m: float = MARGIN_M,
        **kwargs,
    ):
        kwargs["listening"] = True
        Sofa.Core.Controller.__init__(self, **kwargs)
        self.beam_dofs = beam_dofs
        self.adapter = adapter
        self.constraint = constraint
        self.requested_margin_m = float(requested_margin_m)

        self.current_rl_step = 0
        self.current_substep = 0
        self.enabled = True
        self._last_key: tuple[int, int] | None = None
        self._record: dict[str, Any] | None = None

    def reset(self) -> None:
        self._last_key = None
        self._record = None
        self.constraint.enabled.value = False

    def take_record(self) -> dict[str, Any] | None:
        record = self._record
        self._record = None
        return record

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
        if self.current_rl_step <= 0 or self.current_substep not in (1, 2):
            return

        key = (int(self.current_rl_step), int(self.current_substep))
        if key == self._last_key:
            return
        self._last_key = key
        planner_start_perf = time.perf_counter()
        planner_start_cpu = time.process_time()

        q_prev = np.asarray(
            self.beam_dofs.position.array(), dtype=np.float64
        ).copy()
        q_free = np.asarray(
            self.beam_dofs.free_position.array(), dtype=np.float64
        ).copy()

        record: dict[str, Any] = {
            "rl_step": key[0],
            "substep": key[1],
            "event_source": source,
            "requested_margin_m": self.requested_margin_m,
            "dense_spacing_m": DENSE_SPACING_M,
            "rows_required": False,
            "rows_armed": False,
            "row_count": 0,
            "row_build_runtime_s": 0.0,
            "builder": "fast_local_point_fd_v2",
            "q_prev_full_dense_profile_count": 0,
            "q_free_external_dense_count": 0,
            "q_free_internal_dense_count": 0,
            "planner_start_perf": planner_start_perf,
            "planner_start_cpu": planner_start_cpu,
            "uses_collision_dofs_as_constraint_source": False,
            "q_candidate_solver_used": False,
            "native_candidate_injection_used": False,
            "writes_free_position": False,
            "writes_committed_position": False,
            "projection_used": False,
            "rollback_used": False,
            "action_shielding_used": False,
        }

        if (
            q_prev.ndim != 2
            or q_prev.shape[1] != 7
            or q_free.shape != q_prev.shape
            or not np.isfinite(q_prev).all()
            or not np.isfinite(q_free).all()
        ):
            self.constraint.enabled.value = False
            record["planning_status"] = "FAIL"
            record["reason"] = "NON_FINITE_OR_INVALID_BEAM_STATE"
            record["planner_end_perf"] = time.perf_counter()
            record["planner_end_cpu"] = time.process_time()
            record["planner_total_s"] = record["planner_end_perf"] - planner_start_perf
            record["planner_total_cpu_s"] = record["planner_end_cpu"] - planner_start_cpu
            self._record = record
            return

        try:
            stamp = (time.perf_counter(), time.process_time())
            dense_points, dense_clearance = _profile(self.adapter, q_free)
            record["q_free_measure_s"] = time.perf_counter() - stamp[0]
            record["q_free_measure_cpu_s"] = time.process_time() - stamp[1]
            record["q_free_external_dense_count"] = 1
            record["q_prev_measure_s"] = 0.0
            record["q_prev_measure_cpu_s"] = 0.0
        except Exception as exc:
            self.constraint.enabled.value = False
            record["planning_status"] = "FAIL"
            record["reason"] = (
                f"DENSE_BEAM_MEASUREMENT_FAILED: "
                f"{type(exc).__name__}: {exc}"
            )
            record["planner_end_perf"] = time.perf_counter()
            record["planner_end_cpu"] = time.process_time()
            record["planner_total_s"] = record["planner_end_perf"] - planner_start_perf
            record["planner_total_cpu_s"] = record["planner_end_cpu"] - planner_start_cpu
            self._record = record
            return

        worst_index = int(np.argmin(dense_clearance))
        free_clearance = float(dense_clearance[worst_index])
        record.update(
            {
                "q_free_clearance_m": free_clearance,
                "q_free_clearance_mm": free_clearance * 1000.0,
                "q_free_penetration_m": max(0.0, -free_clearance),
                "q_free_worst_point_m": dense_points[worst_index].tolist(),
                "dense_sample_count": int(len(dense_clearance)),
            }
        )

        if not (
            np.isfinite(free_clearance)
        ):
            self.constraint.enabled.value = False
            record["planning_status"] = "FAIL"
            record["reason"] = "NON_FINITE_DENSE_CLEARANCE"
            record["planner_end_perf"] = time.perf_counter()
            record["planner_end_cpu"] = time.process_time()
            record["planner_total_s"] = record["planner_end_perf"] - planner_start_perf
            record["planner_total_cpu_s"] = record["planner_end_cpu"] - planner_start_cpu
            self._record = record
            return

        stamp = (time.perf_counter(), time.process_time())
        safe_free = free_clearance + FLOAT_TOL_M >= self.requested_margin_m
        record["gate_s"] = time.perf_counter() - stamp[0]
        record["gate_cpu_s"] = time.process_time() - stamp[1]
        if safe_free:
            record["total_q_free_full_profile_count"] = record["q_free_external_dense_count"]
            self.constraint.enabled.value = False
            record["planning_status"] = "SAFE_FREE_NO_ROWS"
            record["reason"] = "FREE_BEAM_ALREADY_AT_OR_ABOVE_MARGIN"
            record["planner_end_perf"] = time.perf_counter()
            record["planner_end_cpu"] = time.process_time()
            record["planner_total_s"] = record["planner_end_perf"] - planner_start_perf
            record["planner_total_cpu_s"] = record["planner_end_cpu"] - planner_start_cpu
            self._record = record
            return

        record["rows_required"] = True
        started = time.perf_counter()
        try:
            rows = build_rows_fast_local_fd_v2(
                adapter=self.adapter,
                q_prev=q_prev,
                q_free=q_free,
                dense_points=dense_points,
                dense_clearance=dense_clearance,
                requested_margin_m=self.requested_margin_m,
            )
            stamp = (time.perf_counter(), time.process_time())
            write_rows(self.constraint, rows)
            record["constraint_write_s"] = time.perf_counter() - stamp[0]
            record["constraint_write_cpu_s"] = time.process_time() - stamp[1]
            record.update(rows["profile_timings"])
            record["q_free_internal_dense_count"] = rows["q_free_internal_dense_count"]
            record["total_q_free_full_profile_count"] = record["q_free_external_dense_count"] + record["q_free_internal_dense_count"]
        except Exception as exc:
            self.constraint.enabled.value = False
            record["row_build_runtime_s"] = float(
                time.perf_counter() - started
            )
            record["planning_status"] = "FAIL"
            record["reason"] = (
                f"FAST_BEAM_UNILATERAL_ROW_BUILD_FAILED: "
                f"{type(exc).__name__}: {exc}"
            )
            record["planner_end_perf"] = time.perf_counter()
            record["planner_end_cpu"] = time.process_time()
            record["planner_total_s"] = record["planner_end_perf"] - planner_start_perf
            record["planner_total_cpu_s"] = record["planner_end_cpu"] - planner_start_cpu
            self._record = record
            return

        runtime = float(time.perf_counter() - started)
        row_count = len(rows["free_violations"])
        support_meta = rows["support_metadata"]

        record.update(
            {
                "row_build_runtime_s": runtime,
                "row_count": int(row_count),
                "rows_armed": bool(row_count > 0),
                "dense_sample_count": int(rows["dense_sample_count"]),
                "free_dense_worst_index": int(
                    rows["free_dense_worst_index"]
                ),
                "selected_dense_indices": list(
                    rows["selected_dense_indices"]
                ),
                "selected_element_indices": list(
                    rows["selected_element_indices"]
                ),
                "row_support_nodes": list(rows["row_support_nodes"]),
                "row_block_counts": list(rows["row_block_counts"]),
                "selected_clearances_m": list(
                    rows["selected_clearances_m"]
                ),
                "selected_clearances_mm": [
                    float(v) * 1000.0
                    for v in rows["selected_clearances_m"]
                ],
                "minimum_free_violation_m": float(
                    min(rows["free_violations"])
                ),
                "minimum_free_violation_mm": float(
                    min(rows["free_violations"])
                )
                * 1000.0,
                "support_mode": str(support_meta["support_mode"]),
                "selected_point_reconstruction_max_error_m": float(
                    support_meta[
                        "selected_point_reconstruction_max_error_m"
                    ]
                ),
                "batched_fd_sdf_query_count": int(
                    rows["batched_fd_sdf_query_count"]
                ),
                "batched_fd_point_count": int(
                    rows["batched_fd_point_count"]
                ),
                "full_beam_profile_evaluations_after_selection": int(
                    rows[
                        "full_beam_profile_evaluations_after_selection"
                    ]
                ),
                "translation_fd_m": float(rows["translation_fd_m"]),
                "rotation_fd_rad": float(rows["rotation_fd_rad"]),
            }
        )

        if row_count <= 0:
            self.constraint.enabled.value = False
            record["planning_status"] = "FAIL"
            record["reason"] = "NO_FAST_BEAM_UNILATERAL_ROWS_BUILT"
        elif (
            record["support_mode"] != "exact_selected_point_local_fd"
            or record["selected_point_reconstruction_max_error_m"] > 1.0e-10
            or record["batched_fd_sdf_query_count"] != 1
            or record["full_beam_profile_evaluations_after_selection"] != 0
        ):
            self.constraint.enabled.value = False
            record["planning_status"] = "FAIL"
            record["reason"] = "FAST_BUILDER_INVARIANT_VIOLATION"
        else:
            record["planning_status"] = "ROWS_ARMED"
            record["reason"] = (
                "FAST_LOCAL_POINT_FD_ROWS_READY_FOR_GENERIC_SOLVE"
            )

        record["planner_end_perf"] = time.perf_counter()
        record["planner_end_cpu"] = time.process_time()
        record["planner_total_s"] = record["planner_end_perf"] - planner_start_perf
        record["planner_total_cpu_s"] = record["planner_end_cpu"] - planner_start_cpu
        self._record = record
