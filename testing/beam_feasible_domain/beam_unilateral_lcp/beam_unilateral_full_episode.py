"""Full-episode controller for the test-only Beam-level SDF unilateral route.

Unlike the original step665 diagnostic row builder, this full-episode version
uses the exact local support of each dense Beam sample: each cubic Beam sample
depends on the two Rigid3 endpoint DOFs of its active element. Finite
differences are therefore computed only for the union of those endpoint DOFs.

The controller runs at CollisionBeginEvent on every physics substep. It reads
only the real Rigid3 Beam committed/free states plus production BeamAdapter/SDF
geometry. When the live free Beam violates the +0.100 mm safety margin it
builds linearized unilateral rows for GenericConstraintSolver.

It never writes Beam free_position or committed position.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import Sofa

from beam_linearized_unilateral import (
    DENSE_SPACING_M,
    GRADIENT_BLOCK_EPS,
    MARGIN_M,
    ROTATION_FD_RAD,
    TRANSLATION_FD_M,
    _perturb,
    _profile,
    _select_rows,
    write_rows,
)


FLOAT_TOL_M = 1.0e-12


def _dense_index_element_support(
    *,
    adapter,
    selected_indices: list[int],
    dense_sample_count: int,
) -> tuple[list[list[int]], dict[str, Any]]:
    """Map each dense sample index to its exact active Beam element endpoints."""
    impl = getattr(adapter, "adapter", adapter)
    geometry_fn = getattr(impl, "solver_geometry", None)
    if not callable(geometry_fn):
        raise RuntimeError("live BeamAdapter solver_geometry unavailable")

    geometry = geometry_fn()
    specs = list(geometry["active_elements"])
    if not specs:
        raise RuntimeError("no active Beam elements")

    ranges = []
    cursor = 0
    for i, sp in enumerate(specs):
        length = float(sp["rest_length_m"])
        raw_count = max(
            2, int(np.ceil(length / DENSE_SPACING_M)) + 1
        )
        # beam_curve drops the first point of every connected element after
        # the first to avoid duplicate element endpoints.
        count = raw_count if i == 0 else raw_count - 1
        start = cursor
        end = cursor + count
        ranges.append(
            {
                "element_index": int(i),
                "start": int(start),
                "end": int(end),
                "nodes": [int(v) for v in sp["nodes"]],
                "rest_length_m": length,
                "sample_count": int(count),
            }
        )
        cursor = end

    exact_count_match = cursor == int(dense_sample_count)
    if not exact_count_match:
        # This should not happen for the production connected active Beam.
        # Fall back conservatively to all active nodes rather than silently
        # assigning a wrong local support.
        all_nodes = sorted(
            {
                int(n)
                for sp in specs
                for n in sp["nodes"]
            }
        )
        return (
            [list(all_nodes) for _ in selected_indices],
            {
                "support_mode": "all_active_nodes_fallback",
                "derived_dense_sample_count": int(cursor),
                "actual_dense_sample_count": int(dense_sample_count),
                "ranges": ranges,
            },
        )

    supports = []
    selected_elements = []
    for idx in selected_indices:
        idx = int(idx)
        hit = None
        for entry in ranges:
            if entry["start"] <= idx < entry["end"]:
                hit = entry
                break
        if hit is None:
            raise RuntimeError(
                f"dense sample index {idx} has no active Beam element support"
            )
        supports.append(list(hit["nodes"]))
        selected_elements.append(int(hit["element_index"]))

    return (
        supports,
        {
            "support_mode": "exact_two_endpoint_beam_element",
            "derived_dense_sample_count": int(cursor),
            "actual_dense_sample_count": int(dense_sample_count),
            "selected_element_indices": selected_elements,
            "ranges": ranges,
        },
    )


def build_rows_local_support(
    *,
    adapter,
    q_prev: np.ndarray,
    q_free: np.ndarray,
    requested_margin_m: float = MARGIN_M,
) -> dict[str, Any]:
    """Build dense SDF unilateral rows using exact two-node Beam support."""
    q_prev = np.asarray(q_prev, dtype=np.float64)
    q_free = np.asarray(q_free, dtype=np.float64)

    if q_prev.shape != q_free.shape or q_free.ndim != 2 or q_free.shape[1] != 7:
        raise ValueError("expected q_prev/q_free shape (N,7)")
    if not np.isfinite(q_prev).all() or not np.isfinite(q_free).all():
        raise ValueError("non-finite Beam state")

    points, base = _profile(adapter, q_free)
    selected = [int(v) for v in _select_rows(base)]
    if not selected:
        raise RuntimeError("no dense unilateral samples selected")

    row_supports, support_meta = _dense_index_element_support(
        adapter=adapter,
        selected_indices=selected,
        dense_sample_count=len(base),
    )
    support_nodes = sorted(
        {int(n) for row in row_supports for n in row}
    )
    if not support_nodes:
        raise RuntimeError("no Beam DOFs support selected unilateral rows")

    row_count = len(selected)
    node_to_col = {node: i for i, node in enumerate(support_nodes)}
    linear = np.zeros(
        (row_count, len(support_nodes), 3), dtype=np.float64
    )
    angular = np.zeros(
        (row_count, len(support_nodes), 3), dtype=np.float64
    )

    rows_by_node: dict[int, list[int]] = {n: [] for n in support_nodes}
    for r, nodes in enumerate(row_supports):
        for node in nodes:
            rows_by_node[int(node)].append(int(r))

    for node in support_nodes:
        affected_rows = sorted(set(rows_by_node[node]))
        col = node_to_col[node]

        for axis in range(3):
            qp = _perturb(
                q_free,
                node,
                axis,
                TRANSLATION_FD_M,
                rotational=False,
            )
            qm = _perturb(
                q_free,
                node,
                axis,
                -TRANSLATION_FD_M,
                rotational=False,
            )
            _, cp = _profile(adapter, qp)
            _, cm = _profile(adapter, qm)
            if len(cp) != len(base) or len(cm) != len(base):
                raise RuntimeError(
                    "dense profile count changed under translation FD"
                )
            for row in affected_rows:
                idx = selected[row]
                linear[row, col, axis] = (
                    cp[idx] - cm[idx]
                ) / (2.0 * TRANSLATION_FD_M)

        for axis in range(3):
            qp = _perturb(
                q_free,
                node,
                axis,
                ROTATION_FD_RAD,
                rotational=True,
            )
            qm = _perturb(
                q_free,
                node,
                axis,
                -ROTATION_FD_RAD,
                rotational=True,
            )
            _, cp = _profile(adapter, qp)
            _, cm = _profile(adapter, qm)
            if len(cp) != len(base) or len(cm) != len(base):
                raise RuntimeError(
                    "dense profile count changed under rotation FD"
                )
            for row in affected_rows:
                idx = selected[row]
                angular[row, col, axis] = (
                    cp[idx] - cm[idx]
                ) / (2.0 * ROTATION_FD_RAD)

    row_offsets = [0]
    dof_indices: list[int] = []
    linear_blocks: list[list[float]] = []
    angular_blocks: list[list[float]] = []
    row_block_counts: list[int] = []

    for row, support in enumerate(row_supports):
        before = len(dof_indices)
        for node in support:
            col = node_to_col[int(node)]
            jl = linear[row, col]
            ja = angular[row, col]
            if (
                float(np.linalg.norm(jl)) < GRADIENT_BLOCK_EPS
                and float(np.linalg.norm(ja)) < GRADIENT_BLOCK_EPS
            ):
                continue
            dof_indices.append(int(node))
            linear_blocks.append(jl.tolist())
            angular_blocks.append(ja.tolist())

        count = len(dof_indices) - before
        row_block_counts.append(int(count))
        row_offsets.append(len(dof_indices))

    if any(v <= 0 for v in row_block_counts):
        raise RuntimeError(
            "at least one selected dense row has zero Beam Jacobian blocks: "
            f"{row_block_counts}"
        )

    clearances = np.asarray(base[selected], dtype=np.float64)
    violations = clearances - float(requested_margin_m)

    return {
        "row_offsets": row_offsets,
        "dof_indices": dof_indices,
        "linear_jacobian": linear_blocks,
        "angular_jacobian": angular_blocks,
        "free_violations": violations.tolist(),
        "source_clearances": clearances.tolist(),
        "selected_dense_indices": selected,
        "selected_points_m": points[selected].tolist(),
        "selected_clearances_m": clearances.tolist(),
        "support_nodes": support_nodes,
        "row_support_nodes": row_supports,
        "row_block_counts": row_block_counts,
        "support_metadata": support_meta,
        "dense_sample_count": int(len(base)),
        "free_dense_min_clearance_m": float(np.min(base)),
        "free_dense_worst_index": int(np.argmin(base)),
        "requested_margin_m": float(requested_margin_m),
        "translation_fd_m": TRANSLATION_FD_M,
        "rotation_fd_rad": ROTATION_FD_RAD,
        "uses_collision_dofs_as_constraint_source": False,
        "writes_committed_position": False,
        "writes_free_position": False,
        "projection_used": False,
        "rollback_used": False,
    }


class FullEpisodeBeamUnilateralController(Sofa.Core.Controller):
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
            "uses_collision_dofs_as_constraint_source": False,
            "q_candidate_solver_used": False,
            "writes_free_position": False,
            "writes_committed_position": False,
            "projection_used": False,
            "rollback_used": False,
            "rows_required": False,
            "rows_armed": False,
            "row_count": 0,
            "row_build_runtime_s": 0.0,
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
            self._record = record
            return

        try:
            prev = self.adapter.measure(
                q_prev, spacing_m=DENSE_SPACING_M
            )
            free = self.adapter.measure(
                q_free, spacing_m=DENSE_SPACING_M
            )
        except Exception as exc:
            self.constraint.enabled.value = False
            record["planning_status"] = "FAIL"
            record["reason"] = (
                f"DENSE_BEAM_MEASUREMENT_FAILED: "
                f"{type(exc).__name__}: {exc}"
            )
            self._record = record
            return

        prev_clearance = float(prev["min_clearance_m"])
        free_clearance = float(free["min_clearance_m"])
        record.update(
            {
                "q_prev_clearance_m": prev_clearance,
                "q_prev_clearance_mm": prev_clearance * 1000.0,
                "q_free_clearance_m": free_clearance,
                "q_free_clearance_mm": free_clearance * 1000.0,
                "q_free_penetration_m": max(0.0, -free_clearance),
                "q_free_worst_point_m": free.get("worst_point_m"),
            }
        )

        if not (
            np.isfinite(prev_clearance) and np.isfinite(free_clearance)
        ):
            self.constraint.enabled.value = False
            record["planning_status"] = "FAIL"
            record["reason"] = "NON_FINITE_DENSE_CLEARANCE"
            self._record = record
            return

        if free_clearance + FLOAT_TOL_M >= self.requested_margin_m:
            self.constraint.enabled.value = False
            record["planning_status"] = "SAFE_FREE_NO_ROWS"
            record["reason"] = "FREE_BEAM_ALREADY_AT_OR_ABOVE_MARGIN"
            self._record = record
            return

        record["rows_required"] = True
        started = time.perf_counter()
        try:
            rows = build_rows_local_support(
                adapter=self.adapter,
                q_prev=q_prev,
                q_free=q_free,
                requested_margin_m=self.requested_margin_m,
            )
            write_rows(self.constraint, rows)
        except Exception as exc:
            self.constraint.enabled.value = False
            record["row_build_runtime_s"] = float(
                time.perf_counter() - started
            )
            record["planning_status"] = "FAIL"
            record["reason"] = (
                f"BEAM_UNILATERAL_ROW_BUILD_FAILED: "
                f"{type(exc).__name__}: {exc}"
            )
            self._record = record
            return

        runtime = float(time.perf_counter() - started)
        row_count = len(rows["free_violations"])
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
                "support_nodes": list(rows["support_nodes"]),
                "row_support_nodes": list(rows["row_support_nodes"]),
                "row_block_counts": list(rows["row_block_counts"]),
                "support_mode": rows["support_metadata"]["support_mode"],
                "selected_element_indices": rows[
                    "support_metadata"
                ].get("selected_element_indices"),
                "translation_fd_m": float(rows["translation_fd_m"]),
                "rotation_fd_rad": float(rows["rotation_fd_rad"]),
            }
        )

        if row_count <= 0:
            self.constraint.enabled.value = False
            record["planning_status"] = "FAIL"
            record["reason"] = "NO_BEAM_UNILATERAL_ROWS_BUILT"
        else:
            record["planning_status"] = "ROWS_ARMED"
            record["reason"] = "BEAM_UNILATERAL_ROWS_READY_FOR_GENERIC_SOLVE"

        self._record = record
