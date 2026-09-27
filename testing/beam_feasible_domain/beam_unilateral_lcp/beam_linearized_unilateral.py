"""Test-only BeamAdapter/SDF -> Rigid3 unilateral row builder for step665.

No position projection and no q_candidate solve occurs here.  The live Beam
free state is sampled densely, SDF clearance is linearized with respect to the
actual Rigid3 Beam DOFs, and those Jacobian rows are passed to a native SOFA
Constraint<Rigid3Types> component.
"""
from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

import numpy as np
import Sofa
from scipy.spatial.transform import Rotation


MARGIN_M = 1.0e-4
DENSE_SPACING_M = 1.0e-5
TRANSLATION_FD_M = 1.0e-6
ROTATION_FD_RAD = 1.0e-4
MAX_ROWS = 12
ACTIVE_BAND_M = 2.0e-4
NEIGHBOR_OFFSETS = (-4, -2, 0, 2, 4)
GRADIENT_BLOCK_EPS = 1.0e-10


def find_plugin(build_dir: Path) -> Path | None:
    build_dir = Path(build_dir)
    for pattern in (
        "libMCRBeamLinearizedUnilateral.so",
        "MCRBeamLinearizedUnilateral.so",
        "libMCRBeamLinearizedUnilateral.dylib",
        "MCRBeamLinearizedUnilateral.dll",
    ):
        hits = sorted(build_dir.rglob(pattern))
        if hits:
            return hits[0]
    return None


def load_plugin(path: Path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    return ctypes.CDLL(str(path), mode=mode)


def _profile(adapter, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(
        adapter.sample(np.asarray(q, dtype=np.float64), spacing_m=DENSE_SPACING_M),
        dtype=np.float64,
    ).reshape((-1, 3))
    clear = np.asarray(adapter.query(pts), dtype=np.float64).reshape(-1)
    if len(pts) != len(clear) or len(clear) < 2:
        raise RuntimeError("invalid dense BeamAdapter/SDF profile")
    if not np.isfinite(pts).all() or not np.isfinite(clear).all():
        raise RuntimeError("non-finite dense BeamAdapter/SDF profile")
    return pts, clear


def _local_minima(clear: np.ndarray) -> list[int]:
    out = []
    for i in range(len(clear)):
        left = i == 0 or clear[i] <= clear[i - 1]
        right = i == len(clear) - 1 or clear[i] <= clear[i + 1]
        if left and right:
            out.append(i)
    return out


def _select_rows(clear: np.ndarray) -> list[int]:
    worst = int(np.argmin(clear))
    candidates = set()
    for off in NEIGHBOR_OFFSETS:
        idx = worst + int(off)
        if 0 <= idx < len(clear):
            candidates.add(idx)

    for i in _local_minima(clear):
        if float(clear[i]) <= MARGIN_M + ACTIVE_BAND_M:
            candidates.add(int(i))

    ordered = sorted(candidates, key=lambda i: (float(clear[i]), abs(i - worst)))
    selected: list[int] = []
    for idx in ordered:
        if idx == worst or all(abs(idx - j) >= 2 for j in selected):
            selected.append(int(idx))
        if len(selected) >= MAX_ROWS:
            break
    if worst not in selected:
        selected = [worst] + selected[: MAX_ROWS - 1]
    return selected


def _moved_nodes(q_prev: np.ndarray, q_free: np.ndarray, adapter) -> list[int]:
    impl = getattr(adapter, "adapter", adapter)
    geometry_fn = getattr(impl, "solver_geometry", None)
    if not callable(geometry_fn):
        raise RuntimeError("live BeamAdapter solver_geometry unavailable")
    geometry = geometry_fn()
    nodes = sorted(
        {
            int(n)
            for sp in geometry["active_elements"]
            for n in sp["nodes"]
        }
    )

    moved = []
    for n in nodes:
        t = float(np.linalg.norm(q_free[n, :3] - q_prev[n, :3]))
        r_prev = Rotation.from_quat(q_prev[n, 3:7])
        r_free = Rotation.from_quat(q_free[n, 3:7])
        a = float(np.linalg.norm((r_prev.inv() * r_free).as_rotvec()))
        if t > 1.0e-11 or a > 1.0e-10:
            moved.append(n)

    if not moved:
        raise RuntimeError("no moving active Beam Rigid3 DOFs")
    return moved


def _perturb(
    q: np.ndarray,
    node: int,
    axis: int,
    amount: float,
    rotational: bool,
) -> np.ndarray:
    out = np.asarray(q, dtype=np.float64).copy()
    if not rotational:
        out[node, axis] += float(amount)
        return out

    delta = np.zeros(3, dtype=np.float64)
    delta[axis] = float(amount)
    r = Rotation.from_quat(out[node, 3:7])
    out[node, 3:7] = (r * Rotation.from_rotvec(delta)).as_quat()
    return out


def build_rows(
    *,
    adapter,
    q_prev: np.ndarray,
    q_free: np.ndarray,
    requested_margin_m: float = MARGIN_M,
) -> dict[str, Any]:
    q_prev = np.asarray(q_prev, dtype=np.float64)
    q_free = np.asarray(q_free, dtype=np.float64)
    if q_prev.shape != q_free.shape or q_free.ndim != 2 or q_free.shape[1] != 7:
        raise ValueError("expected q_prev/q_free shape (N,7)")
    if not np.isfinite(q_prev).all() or not np.isfinite(q_free).all():
        raise ValueError("non-finite Beam state")

    points, base = _profile(adapter, q_free)
    selected = _select_rows(base)
    moved = _moved_nodes(q_prev, q_free, adapter)

    row_count = len(selected)
    node_count = len(moved)
    linear = np.zeros((row_count, node_count, 3), dtype=np.float64)
    angular = np.zeros((row_count, node_count, 3), dtype=np.float64)

    for ni, node in enumerate(moved):
        for axis in range(3):
            qp = _perturb(
                q_free, node, axis, TRANSLATION_FD_M, rotational=False
            )
            qm = _perturb(
                q_free, node, axis, -TRANSLATION_FD_M, rotational=False
            )
            _, cp = _profile(adapter, qp)
            _, cm = _profile(adapter, qm)
            if len(cp) != len(base) or len(cm) != len(base):
                raise RuntimeError("dense profile count changed under translation FD")
            linear[:, ni, axis] = (
                cp[selected] - cm[selected]
            ) / (2.0 * TRANSLATION_FD_M)

        for axis in range(3):
            qp = _perturb(
                q_free, node, axis, ROTATION_FD_RAD, rotational=True
            )
            qm = _perturb(
                q_free, node, axis, -ROTATION_FD_RAD, rotational=True
            )
            _, cp = _profile(adapter, qp)
            _, cm = _profile(adapter, qm)
            if len(cp) != len(base) or len(cm) != len(base):
                raise RuntimeError("dense profile count changed under rotation FD")
            angular[:, ni, axis] = (
                cp[selected] - cm[selected]
            ) / (2.0 * ROTATION_FD_RAD)

    row_offsets = [0]
    dof_indices: list[int] = []
    linear_blocks: list[list[float]] = []
    angular_blocks: list[list[float]] = []
    row_block_counts = []

    for r in range(row_count):
        before = len(dof_indices)
        for ni, node in enumerate(moved):
            jl = linear[r, ni]
            ja = angular[r, ni]
            if (
                float(np.linalg.norm(jl)) < GRADIENT_BLOCK_EPS
                and float(np.linalg.norm(ja)) < GRADIENT_BLOCK_EPS
            ):
                continue
            dof_indices.append(int(node))
            linear_blocks.append(jl.tolist())
            angular_blocks.append(ja.tolist())
        row_block_counts.append(len(dof_indices) - before)
        row_offsets.append(len(dof_indices))

    if any(v <= 0 for v in row_block_counts):
        raise RuntimeError(
            f"at least one selected dense row has zero Beam Jacobian blocks: "
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
        "selected_dense_indices": [int(i) for i in selected],
        "selected_points_m": points[selected].tolist(),
        "selected_clearances_m": clearances.tolist(),
        "moved_nodes": [int(n) for n in moved],
        "row_block_counts": row_block_counts,
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


def write_rows(component, rows: dict[str, Any]) -> None:
    component.enabled.value = False
    component.rowOffsets.value = [int(v) for v in rows["row_offsets"]]
    component.dofIndices.value = [int(v) for v in rows["dof_indices"]]
    component.linearJacobian.value = rows["linear_jacobian"]
    component.angularJacobian.value = rows["angular_jacobian"]
    component.freeViolations.value = [
        float(v) for v in rows["free_violations"]
    ]
    component.sourceClearances.value = [
        float(v) for v in rows["source_clearances"]
    ]
    component.enabled.value = True


class Step665BeamUnilateralController(Sofa.Core.Controller):
    def __init__(
        self,
        *,
        beam_dofs,
        adapter,
        constraint,
        expected_qprev_mm: float,
        expected_qfree_mm: float,
        state_tol_mm: float = 0.002,
        **kwargs,
    ):
        Sofa.Core.Controller.__init__(self, **kwargs)
        self.beam_dofs = beam_dofs
        self.adapter = adapter
        self.constraint = constraint
        self.expected_qprev_mm = float(expected_qprev_mm)
        self.expected_qfree_mm = float(expected_qfree_mm)
        self.state_tol_mm = float(state_tol_mm)
        self.current_rl_step = 0
        self.current_substep = 0
        self.enabled = False
        self.processed = False
        self.record: dict[str, Any] = {}

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
        if (
            not self.enabled
            or self.processed
            or self.current_rl_step != 665
            or self.current_substep != 1
        ):
            return

        self.processed = True
        q_prev = np.asarray(self.beam_dofs.position.array(), dtype=np.float64)
        q_free = np.asarray(self.beam_dofs.free_position.array(), dtype=np.float64)
        prev = self.adapter.measure(q_prev, spacing_m=DENSE_SPACING_M)
        free = self.adapter.measure(q_free, spacing_m=DENSE_SPACING_M)

        self.record = {
            "event_source": source,
            "q_prev_clearance_m": float(prev["min_clearance_m"]),
            "q_free_clearance_m": float(free["min_clearance_m"]),
            "q_prev_clearance_mm": float(prev["min_clearance_m"]) * 1000.0,
            "q_free_clearance_mm": float(free["min_clearance_m"]) * 1000.0,
            "state_matched": False,
        }

        if (
            abs(self.record["q_prev_clearance_mm"] - self.expected_qprev_mm)
            > self.state_tol_mm
            or abs(self.record["q_free_clearance_mm"] - self.expected_qfree_mm)
            > self.state_tol_mm
        ):
            self.record["status"] = "STEP665_PROTECTED_STATE_MISMATCH"
            self.constraint.enabled.value = False
            return

        rows = build_rows(
            adapter=self.adapter,
            q_prev=q_prev,
            q_free=q_free,
            requested_margin_m=MARGIN_M,
        )
        write_rows(self.constraint, rows)
        self.record["state_matched"] = True
        self.record["status"] = "ROWS_ARMED"
        self.record["rows"] = rows
