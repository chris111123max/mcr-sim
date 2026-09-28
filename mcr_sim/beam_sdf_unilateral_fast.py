"""Production Fast-FD Beam/SDF unilateral safety helpers.

This module is the production form of the validated test-only Fast FD V2
formulation.  It reconstructs live BeamAdapter geometry directly from the
SOFA WireBeamInterpolation and RegularGrid topology; it does not depend on
captured B02 artifacts or files under testing/.

The physical safety formulation is unchanged:
    live Rigid3 Beam free state
    -> one 10 um dense Beam/SDF profile
    -> local selected-point finite differences
    -> scalar g = clearance - 0.100 mm >= 0 rows
    -> GenericConstraintSolver

No projection, rollback, action shielding, CollisionDOF safety source, or
committed/free-position overwrite is used.
"""
from __future__ import annotations

import ctypes
from pathlib import Path
import time
from typing import Any

import numpy as np
import Sofa
from scipy.spatial.transform import Rotation

from .sdf_hard_constraint import sample_sdf_clearance_and_outward


MARGIN_M = 1.0e-4
DENSE_SPACING_M = 1.0e-5
TRANSLATION_FD_M = 1.0e-6
ROTATION_FD_RAD = 1.0e-4
MAX_ROWS = 12
ACTIVE_BAND_M = 2.0e-4
NEIGHBOR_OFFSETS = (-4, -2, 0, 2, 4)
GRADIENT_BLOCK_EPS = 1.0e-10
FLOAT_TOL_M = 1.0e-12

MODULE_DIR = Path(__file__).resolve().parent
NATIVE_DIR = MODULE_DIR / "native" / "beam_unilateral"
DEFAULT_PLUGIN_BUILD_DIR = NATIVE_DIR / "_build"


def _as_array(data, dtype=np.float64) -> np.ndarray:
    try:
        return np.asarray(data.array(), dtype=dtype).copy()
    except Exception:
        return np.asarray(data.value, dtype=dtype).copy()


def find_plugin(build_dir: Path = DEFAULT_PLUGIN_BUILD_DIR) -> Path | None:
    build_dir = Path(build_dir)
    for pattern in (
        "libMCRBeamLinearizedUnilateral.so",
        "MCRBeamLinearizedUnilateral.so",
        "libMCRBeamLinearizedUnilateral.dylib",
        "MCRBeamLinearizedUnilateral.dll",
    ):
        hits = sorted(build_dir.rglob(pattern)) if build_dir.exists() else []
        if hits:
            return hits[0]
    return None


def resolve_plugin_path(explicit: str | Path | None = None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
    else:
        path = find_plugin()
        if path is None:
            raise FileNotFoundError(
                "Beam unilateral native plugin is not built. Run "
                "mcr_sim/native/beam_unilateral/build.sh first, or pass "
                "--beam-safety-plugin."
            )
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_plugin(path: str | Path):
    path = resolve_plugin_path(path)
    mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    return ctypes.CDLL(str(path), mode=mode)


def _data_array(component, name: str, dtype=None) -> np.ndarray:
    data = getattr(component, name, None)
    if data is None:
        try:
            data = component.findData(name)
        except Exception as exc:
            raise RuntimeError(f"SOFA data {name!r} unavailable") from exc
    last = None
    for accessor in ("array", "value"):
        try:
            value = getattr(data, accessor)
            value = value() if callable(value) else value
            return np.asarray(value, dtype=dtype)
        except Exception as exc:
            last = exc
    raise RuntimeError(f"Unable to read SOFA data {name!r}: {last}")


def _bezier(p0, p1, p2, p3, t: np.ndarray) -> np.ndarray:
    u = 1.0 - t
    return (
        (u ** 3)[:, None] * p0
        + (3.0 * u * u * t)[:, None] * p1
        + (3.0 * u * t * t)[:, None] * p2
        + (t ** 3)[:, None] * p3
    )


class LiveBeamSDFAdapter:
    """Live production Beam interpolation + production vessel SDF adapter."""

    def __init__(self, env, instrument):
        self.env = env
        self.instrument = instrument
        self.interpolation = instrument.getObject("InterpolGuide")
        self.topology = instrument.getObject("meshLinesCombined")
        self.irc = instrument.getObject("m_ircontroller")
        if self.interpolation is None or self.topology is None or self.irc is None:
            raise RuntimeError("Required live BeamAdapter SOFA objects are missing")

    def _inserted_length(self) -> float:
        value = _data_array(self.irc, "xtip", dtype=np.float64).reshape(-1)
        if value.size == 0 or not np.isfinite(value[0]):
            raise RuntimeError("Invalid m_ircontroller.xtip")
        return float(value[0])

    def _geometry(self) -> dict[str, Any]:
        edge_ids = _data_array(
            self.interpolation, "edgeList", dtype=np.int64
        ).reshape(-1)
        lengths = _data_array(
            self.interpolation, "lengthList", dtype=np.float64
        ).reshape(-1)
        edges = _data_array(
            self.topology, "edges", dtype=np.int64
        ).reshape((-1, 2))

        if (
            edge_ids.size == 0
            or edge_ids.size != lengths.size
            or not np.isfinite(lengths).all()
            or np.any(lengths <= 0.0)
        ):
            raise RuntimeError("Invalid live WireBeamInterpolation edgeList/lengthList")
        if np.any(edge_ids < 0) or np.any(edge_ids >= len(edges)):
            raise RuntimeError("WireBeamInterpolation edgeList exceeds live topology")

        inserted = self._inserted_length()
        length_sum = float(np.sum(lengths))
        if abs(length_sum - inserted) > 1.0e-6:
            raise RuntimeError(
                "Live Beam active-length mismatch: "
                f"sum(lengthList)={length_sum:.12g}, xtip={inserted:.12g}"
            )

        specs = []
        for i, (edge_id, length) in enumerate(zip(edge_ids, lengths)):
            n0, n1 = edges[int(edge_id)]
            specs.append(
                {
                    "edge_list_index": int(i),
                    "topology_edge_id": int(edge_id),
                    "nodes": [int(n0), int(n1)],
                    "rest_length_m": float(length),
                }
            )
        return {
            "inserted_length_m": inserted,
            "active_elements": specs,
        }

    @staticmethod
    def _element_points(
        q: np.ndarray,
        *,
        node0: int,
        node1: int,
        length_m: float,
        spacing_m: float,
    ) -> np.ndarray:
        a = np.asarray(q[int(node0)], dtype=np.float64)
        b = np.asarray(q[int(node1)], dtype=np.float64)
        p0 = a[:3]
        p3 = b[:3]
        r0 = Rotation.from_quat(a[3:7])
        r1 = Rotation.from_quat(b[3:7])
        length = float(length_m)
        p1 = p0 + r0.apply([length / 3.0, 0.0, 0.0])
        p2 = p3 + r1.apply([-length / 3.0, 0.0, 0.0])

        linear = (
            float(np.dot(p1 - p0, p2 - p1)) < 0.0
            and float(np.linalg.norm(p3 - p0)) < 0.4 * length
        )
        t = np.linspace(
            0.0,
            1.0,
            max(2, int(np.ceil(length / float(spacing_m))) + 1),
        )
        if linear:
            return p0[None, :] * (1.0 - t[:, None]) + p3[None, :] * t[:, None]
        return _bezier(p0, p1, p2, p3, t)

    def sample(self, state: np.ndarray, spacing_m: float = DENSE_SPACING_M) -> np.ndarray:
        q = np.asarray(state, dtype=np.float64)
        if q.ndim != 2 or q.shape[1] != 7 or not np.isfinite(q).all():
            raise RuntimeError("Invalid live Beam Rigid3 state")
        spacing = float(spacing_m)
        if not np.isfinite(spacing) or spacing <= 0.0:
            raise ValueError("spacing_m must be finite and positive")

        chunks = []
        for i, spec in enumerate(self._geometry()["active_elements"]):
            pts = self._element_points(
                q,
                node0=int(spec["nodes"][0]),
                node1=int(spec["nodes"][1]),
                length_m=float(spec["rest_length_m"]),
                spacing_m=spacing,
            )
            if i > 0:
                pts = pts[1:]
            chunks.append(pts)

        if not chunks:
            raise RuntimeError("No active Beam elements")
        points = np.concatenate(chunks, axis=0)
        if len(points) < 2 or not np.isfinite(points).all():
            raise RuntimeError("Invalid dense live Beam samples")
        return points

    def query(self, points: np.ndarray) -> np.ndarray:
        clearance, _outward, _valid = sample_sdf_clearance_and_outward(
            np.asarray(points, dtype=np.float64),
            self.env.sdf_grid,
            self.env.asset_T_env_sim,
            self.env.asset_offset_sim,
            self.env.asset_source_to_sim_scale,
            self.env.catheter_radius,
        )
        return np.asarray(clearance, dtype=np.float64).reshape(-1)

    def measure(self, state: np.ndarray, spacing_m: float = DENSE_SPACING_M) -> dict[str, Any]:
        points = self.sample(state, spacing_m=spacing_m)
        clearance = self.query(points)
        if len(clearance) != len(points) or not np.isfinite(clearance).all():
            raise RuntimeError("Invalid live Beam/SDF measurement")
        index = int(np.argmin(clearance))
        value = float(clearance[index])
        return {
            "sample_count": int(len(points)),
            "min_clearance_m": value,
            "min_clearance_mm": value * 1000.0,
            "worst_index": index,
            "worst_point_m": points[index].tolist(),
        }

    def solver_geometry(self) -> dict[str, Any]:
        return self._geometry()


def _profile(adapter: LiveBeamSDFAdapter, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = adapter.sample(q, spacing_m=DENSE_SPACING_M)
    clearance = adapter.query(points)
    if (
        len(points) != len(clearance)
        or len(clearance) < 2
        or not np.isfinite(points).all()
        or not np.isfinite(clearance).all()
    ):
        raise RuntimeError("Invalid dense BeamAdapter/SDF profile")
    return points, clearance


def _local_minima(clearance: np.ndarray) -> list[int]:
    out = []
    for i in range(len(clearance)):
        left = i == 0 or clearance[i] <= clearance[i - 1]
        right = i == len(clearance) - 1 or clearance[i] <= clearance[i + 1]
        if left and right:
            out.append(i)
    return out


def _select_rows(clearance: np.ndarray) -> list[int]:
    worst = int(np.argmin(clearance))
    candidates = set()
    for offset in NEIGHBOR_OFFSETS:
        index = worst + int(offset)
        if 0 <= index < len(clearance):
            candidates.add(index)
    for index in _local_minima(clearance):
        if float(clearance[index]) <= MARGIN_M + ACTIVE_BAND_M:
            candidates.add(int(index))

    ordered = sorted(
        candidates,
        key=lambda index: (float(clearance[index]), abs(index - worst)),
    )
    selected: list[int] = []
    for index in ordered:
        if index == worst or all(abs(index - other) >= 2 for other in selected):
            selected.append(int(index))
        if len(selected) >= MAX_ROWS:
            break
    if worst not in selected:
        selected = [worst] + selected[: MAX_ROWS - 1]
    return selected


def _element_point(
    q: np.ndarray,
    *,
    node0: int,
    node1: int,
    length_m: float,
    t: float,
) -> np.ndarray:
    a = np.asarray(q[int(node0)], dtype=np.float64)
    b = np.asarray(q[int(node1)], dtype=np.float64)
    p0 = a[:3]
    p3 = b[:3]
    r0 = Rotation.from_quat(a[3:7])
    r1 = Rotation.from_quat(b[3:7])
    length = float(length_m)
    p1 = p0 + r0.apply([length / 3.0, 0.0, 0.0])
    p2 = p3 + r1.apply([-length / 3.0, 0.0, 0.0])

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
    adapter: LiveBeamSDFAdapter,
    q_free: np.ndarray,
    dense_points: np.ndarray,
    selected_indices: list[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    specs = list(adapter.solver_geometry()["active_elements"])
    if not specs:
        raise RuntimeError("No active Beam elements")

    ranges = []
    cursor = 0
    for element_index, spec in enumerate(specs):
        length = float(spec["rest_length_m"])
        raw_count = max(2, int(np.ceil(length / DENSE_SPACING_M)) + 1)
        t_values = np.linspace(0.0, 1.0, raw_count)
        if element_index > 0:
            t_values = t_values[1:]
        start = cursor
        end = cursor + len(t_values)
        ranges.append(
            {
                "element_index": int(element_index),
                "start": int(start),
                "end": int(end),
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

    records = []
    max_error = 0.0
    for dense_index in selected_indices:
        hit = None
        for entry in ranges:
            if entry["start"] <= int(dense_index) < entry["end"]:
                hit = entry
                break
        if hit is None:
            raise RuntimeError(f"Dense index {dense_index} has no Beam element")
        local_index = int(dense_index) - int(hit["start"])
        t = float(hit["t_values"][local_index])
        node0, node1 = hit["nodes"]
        point = _element_point(
            q_free,
            node0=node0,
            node1=node1,
            length_m=float(hit["rest_length_m"]),
            t=t,
        )
        error = float(np.linalg.norm(point - dense_points[int(dense_index)]))
        max_error = max(max_error, error)
        records.append(
            {
                "dense_index": int(dense_index),
                "element_index": int(hit["element_index"]),
                "nodes": [int(node0), int(node1)],
                "rest_length_m": float(hit["rest_length_m"]),
                "t": t,
            }
        )

    if max_error > 1.0e-10:
        raise RuntimeError(
            "FAST_SELECTED_POINT_RECONSTRUCTION_MISMATCH: "
            f"max_error_m={max_error}"
        )
    return records, {
        "support_mode": "exact_selected_point_local_fd",
        "dense_sample_count": int(len(dense_points)),
        "active_element_count": int(len(specs)),
        "selected_point_reconstruction_max_error_m": max_error,
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
    node0, node1 = [int(v) for v in record["nodes"]]
    local = np.asarray(q_free[[node0, node1]], dtype=np.float64).copy()
    if int(node) not in (node0, node1):
        raise RuntimeError("Perturbed node is outside row Beam support")
    target = 0 if int(node) == node0 else 1

    if rotational:
        delta = np.zeros(3, dtype=np.float64)
        delta[int(axis)] = float(signed_amount)
        rotation = Rotation.from_quat(local[target, 3:7])
        local[target, 3:7] = (
            rotation * Rotation.from_rotvec(delta)
        ).as_quat()
    else:
        local[target, int(axis)] += float(signed_amount)

    pair = np.zeros((2, 7), dtype=np.float64)
    pair[0] = local[0]
    pair[1] = local[1]
    return _element_point(
        pair,
        node0=0,
        node1=1,
        length_m=float(record["rest_length_m"]),
        t=float(record["t"]),
    )


def build_rows_fast_local_fd(
    *,
    adapter: LiveBeamSDFAdapter,
    q_prev: np.ndarray,
    q_free: np.ndarray,
    dense_points: np.ndarray,
    dense_clearance: np.ndarray,
    requested_margin_m: float = MARGIN_M,
) -> dict[str, Any]:
    """Validated Fast FD V2 row builder using a shared q_free profile."""
    q_prev = np.asarray(q_prev, dtype=np.float64)
    q_free = np.asarray(q_free, dtype=np.float64)
    if q_prev.shape != q_free.shape or q_free.ndim != 2 or q_free.shape[1] != 7:
        raise ValueError("Expected q_prev/q_free shape (N,7)")
    if not np.isfinite(q_prev).all() or not np.isfinite(q_free).all():
        raise ValueError("Non-finite Beam state")

    dense_points = np.asarray(dense_points, dtype=np.float64).reshape((-1, 3))
    base = np.asarray(dense_clearance, dtype=np.float64).reshape(-1)
    if (
        len(base) < 2
        or len(base) != len(dense_points)
        or not np.isfinite(dense_points).all()
        or not np.isfinite(base).all()
    ):
        raise RuntimeError("Invalid shared q_free dense profile")

    selected = [int(v) for v in _select_rows(base)]
    if not selected:
        raise RuntimeError("No dense unilateral samples selected")
    records, support_meta = _selected_dense_records(
        adapter=adapter,
        q_free=q_free,
        dense_points=dense_points,
        selected_indices=selected,
    )

    query_points = []
    query_meta = []
    for row, record in enumerate(records):
        for node in record["nodes"]:
            for rotational, h in (
                (False, TRANSLATION_FD_M),
                (True, ROTATION_FD_RAD),
            ):
                for axis in range(3):
                    for sign in (-1, 1):
                        query_points.append(
                            _perturbed_point(
                                q_free,
                                record,
                                node=int(node),
                                axis=axis,
                                signed_amount=float(sign) * float(h),
                                rotational=rotational,
                            )
                        )
                        query_meta.append(
                            (
                                int(row),
                                int(node),
                                int(axis),
                                int(sign),
                                bool(rotational),
                            )
                        )

    query_points_array = np.asarray(query_points, dtype=np.float64).reshape((-1, 3))
    query_clearance = adapter.query(query_points_array)
    if len(query_clearance) != len(query_meta) or not np.isfinite(query_clearance).all():
        raise RuntimeError("Invalid batched local FD SDF query")

    samples: dict[tuple[int, int, bool, int, int], float] = {}
    for meta, value in zip(query_meta, query_clearance):
        row, node, axis, sign, rotational = meta
        samples[(row, node, rotational, axis, sign)] = float(value)

    row_offsets = [0]
    dof_indices: list[int] = []
    linear_blocks: list[list[float]] = []
    angular_blocks: list[list[float]] = []
    row_block_counts = []

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

    if any(value <= 0 for value in row_block_counts):
        raise RuntimeError(
            f"Fast local FD produced a zero-Jacobian row: {row_block_counts}"
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
        "selected_element_indices": [
            int(record["element_index"]) for record in records
        ],
        "row_support_nodes": [list(record["nodes"]) for record in records],
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
        "q_free_internal_dense_count": 0,
        "q_prev_used_for_row_math": False,
        "uses_collision_dofs_as_constraint_source": False,
        "writes_committed_position": False,
        "writes_free_position": False,
        "projection_used": False,
        "rollback_used": False,
    }


def write_rows(component, rows: dict[str, Any]) -> None:
    component.enabled.value = False
    component.rowOffsets.value = [int(v) for v in rows["row_offsets"]]
    component.dofIndices.value = [int(v) for v in rows["dof_indices"]]
    component.linearJacobian.value = rows["linear_jacobian"]
    component.angularJacobian.value = rows["angular_jacobian"]
    component.freeViolations.value = [float(v) for v in rows["free_violations"]]
    component.sourceClearances.value = [
        float(v) for v in rows["source_clearances"]
    ]
    component.enabled.value = True


class FastBeamUnilateralPlanner(Sofa.Core.Controller):
    """Build Beam-level unilateral rows at CollisionBeginEvent."""

    def __init__(
        self,
        *,
        beam_dofs,
        adapter: LiveBeamSDFAdapter,
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

    def _finish(self, record: dict[str, Any], started: float) -> None:
        record["planner_runtime_s"] = float(time.perf_counter() - started)
        self._record = record

    def _collision_begin(self, source: str) -> None:
        if not self.enabled:
            return
        if self.current_rl_step <= 0 or self.current_substep <= 0:
            return

        key = (int(self.current_rl_step), int(self.current_substep))
        if key == self._last_key:
            return
        self._last_key = key
        started = time.perf_counter()

        q_prev = _as_array(self.beam_dofs.position)
        q_free = _as_array(self.beam_dofs.free_position)
        record: dict[str, Any] = {
            "rl_step": key[0],
            "substep": key[1],
            "event_source": source,
            "planning_status": "FAIL",
            "rows_required": False,
            "rows_armed": False,
            "row_count": 0,
            "q_prev_full_dense_profile_count": 0,
            "q_free_external_dense_count": 0,
            "q_free_internal_dense_count": 0,
            "uses_collision_dofs_as_constraint_source": False,
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
            record["reason"] = "NON_FINITE_OR_INVALID_BEAM_STATE"
            self._finish(record, started)
            return

        try:
            dense_points, dense_clearance = _profile(self.adapter, q_free)
        except Exception as exc:
            self.constraint.enabled.value = False
            record["reason"] = (
                f"DENSE_BEAM_MEASUREMENT_FAILED:{type(exc).__name__}:{exc}"
            )
            self._finish(record, started)
            return

        record["q_free_external_dense_count"] = 1
        worst = int(np.argmin(dense_clearance))
        free_clearance = float(dense_clearance[worst])
        record.update(
            {
                "q_free_clearance_m": free_clearance,
                "q_free_worst_index": worst,
                "dense_sample_count": int(len(dense_clearance)),
            }
        )

        if not np.isfinite(free_clearance):
            self.constraint.enabled.value = False
            record["reason"] = "NON_FINITE_DENSE_CLEARANCE"
            self._finish(record, started)
            return

        if free_clearance + FLOAT_TOL_M >= self.requested_margin_m:
            self.constraint.enabled.value = False
            record["planning_status"] = "SAFE_FREE_NO_ROWS"
            record["reason"] = "FREE_BEAM_ALREADY_AT_OR_ABOVE_MARGIN"
            record["total_q_free_full_profile_count"] = 1
            self._finish(record, started)
            return

        record["rows_required"] = True
        try:
            rows = build_rows_fast_local_fd(
                adapter=self.adapter,
                q_prev=q_prev,
                q_free=q_free,
                dense_points=dense_points,
                dense_clearance=dense_clearance,
                requested_margin_m=self.requested_margin_m,
            )
            write_rows(self.constraint, rows)
        except Exception as exc:
            self.constraint.enabled.value = False
            record["reason"] = (
                "FAST_BEAM_UNILATERAL_ROW_BUILD_FAILED:"
                f"{type(exc).__name__}:{exc}"
            )
            self._finish(record, started)
            return

        support = rows["support_metadata"]
        row_count = int(len(rows["free_violations"]))
        record.update(
            {
                "row_count": row_count,
                "rows_armed": bool(row_count > 0),
                "q_free_internal_dense_count": int(
                    rows["q_free_internal_dense_count"]
                ),
                "total_q_free_full_profile_count": 1
                + int(rows["q_free_internal_dense_count"]),
                "selected_dense_indices": list(rows["selected_dense_indices"]),
                "selected_element_indices": list(
                    rows["selected_element_indices"]
                ),
                "row_support_nodes": list(rows["row_support_nodes"]),
                "support_mode": str(support["support_mode"]),
                "selected_point_reconstruction_max_error_m": float(
                    support["selected_point_reconstruction_max_error_m"]
                ),
                "batched_fd_sdf_query_count": int(
                    rows["batched_fd_sdf_query_count"]
                ),
                "full_beam_profile_evaluations_after_selection": int(
                    rows["full_beam_profile_evaluations_after_selection"]
                ),
            }
        )

        if row_count <= 0:
            self.constraint.enabled.value = False
            record["reason"] = "NO_FAST_BEAM_UNILATERAL_ROWS_BUILT"
        elif (
            record["support_mode"] != "exact_selected_point_local_fd"
            or record["selected_point_reconstruction_max_error_m"] > 1.0e-10
            or record["batched_fd_sdf_query_count"] != 1
            or record["full_beam_profile_evaluations_after_selection"] != 0
            or record["q_free_internal_dense_count"] != 0
        ):
            self.constraint.enabled.value = False
            record["reason"] = "FAST_BUILDER_INVARIANT_VIOLATION"
        else:
            record["planning_status"] = "ROWS_ARMED"
            record["reason"] = "FAST_LOCAL_POINT_FD_ROWS_READY_FOR_GENERIC_SOLVE"
        self._finish(record, started)


__all__ = [
    "ACTIVE_BAND_M",
    "DENSE_SPACING_M",
    "FLOAT_TOL_M",
    "FastBeamUnilateralPlanner",
    "LiveBeamSDFAdapter",
    "MARGIN_M",
    "build_rows_fast_local_fd",
    "find_plugin",
    "load_plugin",
    "resolve_plugin_path",
]
