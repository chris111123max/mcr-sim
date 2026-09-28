"""Fast test-only Beam unilateral Jacobian builder.

Scientific equivalence target:
- same 10 um dense profile used to select safety rows;
- same selected dense indices;
- same exact two-endpoint Beam support;
- same translation/rotation finite-difference step sizes;
- same production SDF clearance function.

Optimization:
The baseline full-episode builder recomputes the ENTIRE ~5300-point Beam
profile for every +/- finite-difference perturbation.  This builder instead
reconstructs only the selected dense point on its Beam element for each
perturbation, batches all perturbed points, and queries the production SDF once.

No analytic-gradient assumption is introduced yet.  This intentionally keeps
the numerical differentiation convention identical to the validated baseline.
"""
from __future__ import annotations

import hashlib
import time
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from beam_linearized_unilateral import (
    DENSE_SPACING_M,
    GRADIENT_BLOCK_EPS,
    MARGIN_M,
    ROTATION_FD_RAD,
    TRANSLATION_FD_M,
    _profile,
    _select_rows,
)
from beam_unilateral_full_episode import build_rows_local_support


def _element_point(
    q: np.ndarray,
    *,
    node0: int,
    node1: int,
    length_m: float,
    t: float,
) -> np.ndarray:
    """Exact point formula used by the production BeamAdapter audit."""
    a = np.asarray(q[int(node0)], dtype=np.float64)
    b = np.asarray(q[int(node1)], dtype=np.float64)

    p0 = a[:3]
    p3 = b[:3]
    r0 = Rotation.from_quat(a[3:7])
    r1 = Rotation.from_quat(b[3:7])

    length = float(length_m)
    p1 = p0 + r0.apply([length / 3.0, 0.0, 0.0])
    p2 = p3 + r1.apply([-length / 3.0, 0.0, 0.0])

    # Keep the same compressed-element fallback as beam_curve().
    linear = (
        float(np.dot(p1 - p0, p2 - p1)) < 0.0
        and float(np.linalg.norm(p3 - p0)) < 0.4 * length
    )

    tt = float(t)
    if linear:
        return p0 * (1.0 - tt) + p3 * tt

    u = 1.0 - tt
    return (
        (u ** 3) * p0
        + (3.0 * u * u * tt) * p1
        + (3.0 * u * tt * tt) * p2
        + (tt ** 3) * p3
    )


def _selected_dense_records(
    *,
    adapter,
    q_free: np.ndarray,
    dense_points: np.ndarray,
    selected_indices: list[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Recover exact Beam element/t for each selected dense sample."""
    impl = getattr(adapter, "adapter", adapter)
    geometry_fn = getattr(impl, "solver_geometry", None)
    if not callable(geometry_fn):
        raise RuntimeError("live BeamAdapter solver_geometry unavailable")

    geometry = geometry_fn()
    specs = list(geometry["active_elements"])
    if not specs:
        raise RuntimeError("no active Beam elements")

    ranges: list[dict[str, Any]] = []
    cursor = 0
    for element_index, spec in enumerate(specs):
        length = float(spec["rest_length_m"])
        raw_count = max(
            2, int(np.ceil(length / DENSE_SPACING_M)) + 1
        )
        t_values = np.linspace(0.0, 1.0, raw_count)
        if element_index > 0:
            t_values = t_values[1:]

        start = int(cursor)
        end = int(cursor + len(t_values))
        ranges.append(
            {
                "element_index": int(element_index),
                "start": start,
                "end": end,
                "nodes": [int(v) for v in spec["nodes"]],
                "rest_length_m": length,
                "t_values": t_values,
            }
        )
        cursor = end

    if cursor != len(dense_points):
        raise RuntimeError(
            "FAST_DENSE_INDEX_MAPPING_COUNT_MISMATCH: "
            f"derived={cursor} actual={len(dense_points)}"
        )

    records: list[dict[str, Any]] = []
    max_reconstruction_error = 0.0

    for dense_index in selected_indices:
        dense_index = int(dense_index)
        hit = None
        for entry in ranges:
            if entry["start"] <= dense_index < entry["end"]:
                hit = entry
                break
        if hit is None:
            raise RuntimeError(
                f"dense index {dense_index} has no Beam element"
            )

        local_index = dense_index - int(hit["start"])
        t = float(hit["t_values"][local_index])
        node0, node1 = hit["nodes"]
        point = _element_point(
            q_free,
            node0=node0,
            node1=node1,
            length_m=float(hit["rest_length_m"]),
            t=t,
        )
        error = float(
            np.linalg.norm(point - dense_points[dense_index])
        )
        max_reconstruction_error = max(
            max_reconstruction_error, error
        )

        records.append(
            {
                "dense_index": dense_index,
                "element_index": int(hit["element_index"]),
                "nodes": [int(node0), int(node1)],
                "rest_length_m": float(hit["rest_length_m"]),
                "t": t,
                "baseline_point_m": dense_points[dense_index].tolist(),
                "reconstructed_point_m": point.tolist(),
                "reconstruction_error_m": error,
            }
        )

    if max_reconstruction_error > 1.0e-10:
        raise RuntimeError(
            "FAST_SELECTED_POINT_RECONSTRUCTION_MISMATCH: "
            f"max_error_m={max_reconstruction_error}"
        )

    return records, {
        "support_mode": "exact_selected_point_local_fd",
        "dense_sample_count": int(len(dense_points)),
        "active_element_count": int(len(specs)),
        "selected_point_reconstruction_max_error_m": (
            max_reconstruction_error
        ),
    }


def _perturbed_point(
    q_free: np.ndarray,
    record: dict[str, Any],
    *,
    node: int,
    axis: int,
    signed_amount: float,
    rotational: bool,
) -> np.ndarray:
    """Reconstruct only one selected Beam point under one Rigid3 perturbation."""
    node0, node1 = [int(v) for v in record["nodes"]]
    local = np.asarray(q_free[[node0, node1]], dtype=np.float64).copy()

    target = 0 if int(node) == node0 else 1
    if int(node) not in (node0, node1):
        raise RuntimeError("perturbed node is outside row Beam support")

    if rotational:
        delta = np.zeros(3, dtype=np.float64)
        delta[int(axis)] = float(signed_amount)
        r = Rotation.from_quat(local[target, 3:7])
        local[target, 3:7] = (
            r * Rotation.from_rotvec(delta)
        ).as_quat()
    else:
        local[target, int(axis)] += float(signed_amount)

    pair_q = np.zeros((2, 7), dtype=np.float64)
    pair_q[0] = local[0]
    pair_q[1] = local[1]
    return _element_point(
        pair_q,
        node0=0,
        node1=1,
        length_m=float(record["rest_length_m"]),
        t=float(record["t"]),
    )


def build_rows_fast_local_fd(
    *,
    adapter,
    q_prev: np.ndarray,
    q_free: np.ndarray,
    requested_margin_m: float = MARGIN_M,
) -> dict[str, Any]:
    """Build equivalent FD rows without repeated full-Beam reconstruction."""
    q_prev = np.asarray(q_prev, dtype=np.float64)
    q_free = np.asarray(q_free, dtype=np.float64)

    if q_prev.shape != q_free.shape or q_free.ndim != 2 or q_free.shape[1] != 7:
        raise ValueError("expected q_prev/q_free shape (N,7)")
    if not np.isfinite(q_prev).all() or not np.isfinite(q_free).all():
        raise ValueError("non-finite Beam state")

    profile_times = {}
    def mark(name, start):
        profile_times[name + "_s"] = time.perf_counter() - start[0]
        profile_times[name + "_cpu_s"] = time.process_time() - start[1]
    def tick():
        return time.perf_counter(), time.process_time()

    stamp = tick()
    dense_points, base = _profile(adapter, q_free)
    mark("fast_builder_base_profile", stamp)
    stamp = tick()
    selected = [int(v) for v in _select_rows(base)]
    mark("selected_dense_indices", stamp)
    if not selected:
        raise RuntimeError("no dense unilateral samples selected")

    stamp = tick()
    records, support_meta = _selected_dense_records(
        adapter=adapter,
        q_free=q_free,
        dense_points=dense_points,
        selected_indices=selected,
    )

    mark("support_mapping", stamp)

    # Batch every single-point finite-difference SDF query.  Metadata maps each
    # query result back to one row/node/axis/sign/translation-or-rotation slot.
    query_points: list[np.ndarray] = []
    query_meta: list[tuple[int, int, int, int, bool]] = []

    stamp = tick()
    for row, record in enumerate(records):
        for node in record["nodes"]:
            for rotational, h in (
                (False, TRANSLATION_FD_M),
                (True, ROTATION_FD_RAD),
            ):
                for axis in range(3):
                    for sign in (-1, 1):
                        point = _perturbed_point(
                            q_free,
                            record,
                            node=int(node),
                            axis=axis,
                            signed_amount=float(sign) * float(h),
                            rotational=rotational,
                        )
                        query_points.append(point)
                        query_meta.append(
                            (
                                int(row),
                                int(node),
                                int(axis),
                                int(sign),
                                bool(rotational),
                            )
                        )

    query_points_array = np.asarray(
        query_points, dtype=np.float64
    ).reshape((-1, 3))
    mark("fd_point_generation", stamp)
    stamp = tick()
    query_clearance = np.asarray(
        adapter.query(query_points_array), dtype=np.float64
    ).reshape(-1)
    mark("fd_batched_sdf", stamp)

    if len(query_clearance) != len(query_meta):
        raise RuntimeError("batched SDF query length mismatch")
    if not np.isfinite(query_clearance).all():
        raise RuntimeError("non-finite batched SDF clearance")

    stamp = tick()
    row_offsets = [0]
    dof_indices: list[int] = []
    linear_blocks: list[list[float]] = []
    angular_blocks: list[list[float]] = []
    row_block_counts: list[int] = []

    # Collect +/- values before converting them into Jacobian blocks.
    samples: dict[tuple[int, int, bool, int, int], float] = {}
    for meta, clearance in zip(query_meta, query_clearance):
        row, node, axis, sign, rotational = meta
        samples[(row, node, rotational, axis, sign)] = float(
            clearance
        )

    for row, record in enumerate(records):
        before = len(dof_indices)
        for node in record["nodes"]:
            jl = np.zeros(3, dtype=np.float64)
            ja = np.zeros(3, dtype=np.float64)
            for axis in range(3):
                jl[axis] = (
                    samples[(row, int(node), False, axis, 1)]
                    - samples[(row, int(node), False, axis, -1)]
                ) / (2.0 * TRANSLATION_FD_M)
                ja[axis] = (
                    samples[(row, int(node), True, axis, 1)]
                    - samples[(row, int(node), True, axis, -1)]
                ) / (2.0 * ROTATION_FD_RAD)

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
            "fast local FD produced a zero-Jacobian row: "
            f"{row_block_counts}"
        )

    clearances = np.asarray(base[selected], dtype=np.float64)
    violations = clearances - float(requested_margin_m)

    mark("jacobian_assembly", stamp)
    return {
        "profile_timings": profile_times,
        "q_free_internal_dense_count": 1,
        "q_prev_used_for_row_math": False,
        "row_offsets": row_offsets,
        "dof_indices": dof_indices,
        "linear_jacobian": linear_blocks,
        "angular_jacobian": angular_blocks,
        "free_violations": violations.tolist(),
        "source_clearances": clearances.tolist(),
        "selected_dense_indices": selected,
        "selected_points_m": dense_points[selected].tolist(),
        "selected_clearances_m": clearances.tolist(),
        "row_support_nodes": [list(r["nodes"]) for r in records],
        "selected_element_indices": [
            int(r["element_index"]) for r in records
        ],
        "selected_t": [float(r["t"]) for r in records],
        "row_block_counts": row_block_counts,
        "dense_sample_count": int(len(base)),
        "free_dense_min_clearance_m": float(np.min(base)),
        "free_dense_worst_index": int(np.argmin(base)),
        "requested_margin_m": float(requested_margin_m),
        "translation_fd_m": TRANSLATION_FD_M,
        "rotation_fd_rad": ROTATION_FD_RAD,
        "support_metadata": support_meta,
        "batched_fd_sdf_query_count": 1,
        "batched_fd_point_count": int(len(query_points_array)),
        "full_beam_profile_evaluations_after_selection": 0,
        "uses_collision_dofs_as_constraint_source": False,
        "writes_committed_position": False,
        "writes_free_position": False,
        "projection_used": False,
        "rollback_used": False,
    }


def row_numeric_arrays(rows: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(rows["linear_jacobian"], dtype=np.float64),
        np.asarray(rows["angular_jacobian"], dtype=np.float64),
    )


def compare_row_builds(
    baseline: dict[str, Any],
    fast: dict[str, Any],
) -> dict[str, Any]:
    structural = {
        "selected_dense_indices_equal": (
            baseline["selected_dense_indices"]
            == fast["selected_dense_indices"]
        ),
        "row_offsets_equal": (
            baseline["row_offsets"] == fast["row_offsets"]
        ),
        "dof_indices_equal": (
            baseline["dof_indices"] == fast["dof_indices"]
        ),
        "row_block_counts_equal": (
            baseline["row_block_counts"]
            == fast["row_block_counts"]
        ),
    }

    a_lin, a_ang = row_numeric_arrays(baseline)
    b_lin, b_ang = row_numeric_arrays(fast)
    same_shapes = (
        a_lin.shape == b_lin.shape and a_ang.shape == b_ang.shape
    )

    if same_shapes and a_lin.size:
        lin_abs = np.abs(a_lin - b_lin)
        ang_abs = np.abs(a_ang - b_ang)
        lin_max = float(np.max(lin_abs))
        ang_max = float(np.max(ang_abs))
        lin_scale = max(float(np.max(np.abs(a_lin))), 1.0e-15)
        ang_scale = max(float(np.max(np.abs(a_ang))), 1.0e-15)
        lin_rel = lin_max / lin_scale
        ang_rel = ang_max / ang_scale
    else:
        lin_max = ang_max = lin_rel = ang_rel = float("inf")

    exact_source = bool(
        np.allclose(
            np.asarray(baseline["source_clearances"], dtype=np.float64),
            np.asarray(fast["source_clearances"], dtype=np.float64),
            rtol=0.0,
            atol=1.0e-14,
        )
    )

    return {
        **structural,
        "same_jacobian_shapes": bool(same_shapes),
        "source_clearances_equal": exact_source,
        "linear_max_abs_error": lin_max,
        "angular_max_abs_error": ang_max,
        "linear_max_relative_error": lin_rel,
        "angular_max_relative_error": ang_rel,
        "all_structure_equal": bool(all(structural.values())),
        "baseline_row_sha256": row_sha256(baseline),
        "fast_row_sha256": row_sha256(fast),
    }


def row_sha256(rows: dict[str, Any]) -> str:
    h = hashlib.sha256()
    for key in (
        "row_offsets",
        "dof_indices",
        "linear_jacobian",
        "angular_jacobian",
        "free_violations",
    ):
        arr = np.asarray(rows[key], dtype=np.float64)
        h.update(arr.astype("<f8", copy=False).tobytes(order="C"))
    return h.hexdigest()


__all__ = [
    "build_rows_fast_local_fd",
    "build_rows_local_support",
    "compare_row_builds",
]
