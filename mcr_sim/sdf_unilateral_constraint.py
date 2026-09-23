"""Python bridge for the SOFA SDFUnilateralConstraint plugin.

Python performs only geometry sampling and updates the plugin's Data fields.
The C++ component contributes the actual unilateral Lagrange rows to SOFA's
constraint solve.  No catheter position is projected or overwritten here.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import Sofa
import SofaRuntime

from .sdf_hard_constraint import sample_sdf_clearance_and_outward
from .training_config import CATHETER_RADIUS_M, SDF_WALL_ACTIVATION_CLEARANCE_M
from .vessel_assets import load_signed_distance_grid


def _plugin_search_paths():
    python_root = Path(__file__).resolve().parents[1]
    build_root = python_root / "cpp" / "SDFUnilateralConstraint" / "build"

    configured = os.environ.get("MCR_SDF_UNILATERAL_PLUGIN_DIR")
    candidates = []
    if configured:
        candidates.append(Path(configured).expanduser().resolve())

    candidates.extend(
        [
            build_root.resolve(),
            (build_root / "lib").resolve(),
            (build_root / "bin").resolve(),
        ]
    )

    seen = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            yield path


def load_sdf_unilateral_plugin() -> str:
    """Load the external SOFA plugin and return the path used if known."""

    existing_dirs = []
    for candidate in _plugin_search_paths():
        if candidate.is_dir():
            existing_dirs.append(str(candidate))
            SofaRuntime.PluginRepository.addFirstPath(str(candidate))

    try:
        SofaRuntime.importPlugin("SDFUnilateralConstraint")
    except Exception as exc:
        searched = "\n  - ".join(existing_dirs or [str(p) for p in _plugin_search_paths()])
        raise RuntimeError(
            "Could not load SOFA plugin 'SDFUnilateralConstraint'. Build it first. "
            "Searched plugin directories:\n  - " + searched
        ) from exc

    return existing_dirs[0] if existing_dirs else ""


class SDFUnilateralConstraintController(Sofa.Core.Controller):
    """Update SDF unilateral rows on catheter CollisionDOFs before each solve."""

    def __init__(
        self,
        instrument,
        sdf_vti,
        asset_T_env_sim,
        asset_offset_sim,
        asset_source_to_sim_scale,
        catheter_radius_m: float = CATHETER_RADIUS_M,
        activation_clearance_m: float = SDF_WALL_ACTIVATION_CLEARANCE_M,
        max_constraints: int = 24,
        min_sample_separation_m: Optional[float] = None,
        enabled: bool = False,
        verbose: bool = False,
        *args,
        **kwargs,
    ):
        kwargs["listening"] = True
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.instrument = instrument
        self.sdf_grid = load_signed_distance_grid(sdf_vti)
        self.asset_T_env_sim = np.asarray(asset_T_env_sim, dtype=np.float64).reshape(7)
        self.asset_offset_sim = np.asarray(asset_offset_sim, dtype=np.float64).reshape(3)
        self.asset_source_to_sim_scale = float(asset_source_to_sim_scale)
        self.catheter_radius_m = float(catheter_radius_m)
        self.activation_clearance_m = float(activation_clearance_m)
        self.max_constraints = int(max_constraints)
        self.min_sample_separation_m = float(
            self.catheter_radius_m
            if min_sample_separation_m is None
            else min_sample_separation_m
        )
        self.enabled = bool(enabled)
        self.verbose = bool(verbose)

        if self.asset_source_to_sim_scale <= 0.0:
            raise ValueError("asset_source_to_sim_scale must be positive")
        if self.catheter_radius_m <= 0.0:
            raise ValueError("catheter_radius_m must be positive")
        if self.activation_clearance_m <= 0.0:
            raise ValueError("activation_clearance_m must be positive")
        if self.max_constraints < 1:
            raise ValueError("max_constraints must be positive")
        if self.min_sample_separation_m <= 0.0:
            raise ValueError("min_sample_separation_m must be positive")

        collision_node = self.instrument.InstrumentCombined.getChild("mcr_collis")
        if collision_node is None:
            raise RuntimeError("Catheter collision node mcr_collis not found")

        self.collision_node = collision_node
        self.collision_dofs = collision_node.getObject("CollisionDOFs")
        self.collision_topology = collision_node.getObject("collisEdgeSet")
        if self.collision_dofs is None or self.collision_topology is None:
            raise RuntimeError("CollisionDOFs/collisEdgeSet not found")

        load_sdf_unilateral_plugin()
        self.constraint = collision_node.addObject(
            "SDFUnilateralConstraint",
            name="SDFUnilateralConstraint",
            enabled=False,
            indices0=[],
            indices1=[],
            weights0=[],
            weights1=[],
            normals=[],
            anchors=[],
            sourceClearances=[],
        )

        self.reset_diagnostics()
        self.set_enabled(self.enabled)

        if self.verbose:
            print(
                "[SDF_UNILATERAL_CONSTRAINT]",
                "enabled=", self.enabled,
                "activation_clearance_m=", self.activation_clearance_m,
                "max_constraints=", self.max_constraints,
                "min_sample_separation_m=", self.min_sample_separation_m,
            )

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self.constraint.enabled.value = bool(self.enabled)
        if not self.enabled:
            self._write_rows([], [], [], [], [], [], [])
            self.active_constraints = 0

    def reset_diagnostics(self) -> None:
        self.update_count = 0
        self.sample_count = 0
        self.valid_samples = 0
        self.candidate_constraints = 0
        self.active_constraints = 0
        self.dropped_constraints = 0
        self.min_clearance_m = np.nan
        self.min_clearance_episode_m = np.inf
        self.max_penetration_episode_m = 0.0
        self.max_active_constraints_episode = 0
        self.selected_sample_kinds = []
        self.selected_clearances_m = []

    def reset(self) -> None:
        self.reset_diagnostics()
        self.set_enabled(self.enabled)

    def _edges(self, n_positions: int) -> np.ndarray:
        try:
            edges = np.asarray(
                self.collision_topology.edges.array(), dtype=np.int64
            ).reshape((-1, 2))
        except Exception:
            try:
                edges = np.asarray(
                    self.collision_topology.edges.value, dtype=np.int64
                ).reshape((-1, 2))
            except Exception:
                edges = np.zeros((0, 2), dtype=np.int64)

        valid = (
            (edges[:, 0] >= 0)
            & (edges[:, 1] >= 0)
            & (edges[:, 0] < n_positions)
            & (edges[:, 1] < n_positions)
        ) if len(edges) else np.zeros(0, dtype=bool)
        return edges[valid]

    def _build_samples(self, positions: np.ndarray):
        n = len(positions)

        # Node samples.
        sample_points = [positions.copy()]
        i0 = [np.arange(n, dtype=np.int64)]
        i1 = [np.arange(n, dtype=np.int64)]
        w0 = [np.ones(n, dtype=np.float64)]
        w1 = [np.zeros(n, dtype=np.float64)]
        kinds = [np.asarray(["node"] * n, dtype=object)]

        # Mid-edge samples catch inter-node penetration without adding physical
        # collision primitives.  The resulting Jacobian row is distributed to
        # the two mapped CollisionDOFs with 0.5/0.5 weights.
        edges = self._edges(n)
        if len(edges):
            mid = 0.5 * (positions[edges[:, 0]] + positions[edges[:, 1]])
            sample_points.append(mid)
            i0.append(edges[:, 0])
            i1.append(edges[:, 1])
            w0.append(np.full(len(edges), 0.5, dtype=np.float64))
            w1.append(np.full(len(edges), 0.5, dtype=np.float64))
            kinds.append(np.asarray(["mid"] * len(edges), dtype=object))

        return (
            np.concatenate(sample_points, axis=0),
            np.concatenate(i0),
            np.concatenate(i1),
            np.concatenate(w0),
            np.concatenate(w1),
            np.concatenate(kinds),
        )

    def _write_rows(self, indices0, indices1, weights0, weights1, normals, anchors, clearances):
        self.constraint.indices0.value = [int(v) for v in indices0]
        self.constraint.indices1.value = [int(v) for v in indices1]
        self.constraint.weights0.value = [float(v) for v in weights0]
        self.constraint.weights1.value = [float(v) for v in weights1]
        self.constraint.normals.value = np.asarray(normals, dtype=np.float64).reshape((-1, 3)).tolist()
        self.constraint.anchors.value = np.asarray(anchors, dtype=np.float64).reshape((-1, 3)).tolist()
        self.constraint.sourceClearances.value = [float(v) for v in clearances]

    def onAnimateBeginEvent(self, event) -> None:
        self.update_count += 1

        if not self.enabled:
            self.constraint.enabled.value = False
            self._write_rows([], [], [], [], [], [], [])
            self.sample_count = 0
            self.valid_samples = 0
            self.candidate_constraints = 0
            self.active_constraints = 0
            self.dropped_constraints = 0
            self.min_clearance_m = np.nan
            self.selected_sample_kinds = []
            self.selected_clearances_m = []
            return

        self.constraint.enabled.value = True
        positions = np.asarray(
            self.collision_dofs.position.array(), dtype=np.float64
        )[:, :3]
        (
            samples,
            indices0,
            indices1,
            weights0,
            weights1,
            kinds,
        ) = self._build_samples(positions)

        clearance, outward, valid = sample_sdf_clearance_and_outward(
            samples,
            sdf_grid=self.sdf_grid,
            asset_T_env_sim=self.asset_T_env_sim,
            asset_offset_sim=self.asset_offset_sim,
            asset_source_to_sim_scale=self.asset_source_to_sim_scale,
            catheter_radius_m=self.catheter_radius_m,
        )
        inward = -outward

        active_mask = (
            valid
            & np.isfinite(clearance)
            & (clearance <= self.activation_clearance_m)
        )
        candidates = np.flatnonzero(active_mask)
        self.candidate_constraints = int(len(candidates))

        # Deepest first, suppress almost coincident samples (e.g. collapsed
        # proximal mapped points), and cap rows to keep the solve small.
        ordered = sorted(
            (int(i) for i in candidates),
            key=lambda i: float(clearance[i]),
        )
        selected = []
        selected_points = []
        for idx in ordered:
            point = samples[idx]
            if selected_points:
                nearest = min(
                    float(np.linalg.norm(point - p))
                    for p in selected_points
                )
                if nearest < self.min_sample_separation_m:
                    continue
            selected.append(idx)
            selected_points.append(point)
            if len(selected) >= self.max_constraints:
                break

        if selected:
            sel = np.asarray(selected, dtype=np.int64)
            normals = inward[sel]
            # Exact local center-boundary anchor: g(current)=clearance.
            anchors = samples[sel] - normals * clearance[sel, None]
            self._write_rows(
                indices0[sel],
                indices1[sel],
                weights0[sel],
                weights1[sel],
                normals,
                anchors,
                clearance[sel],
            )
            self.selected_sample_kinds = [str(v) for v in kinds[sel]]
            self.selected_clearances_m = [float(v) for v in clearance[sel]]
        else:
            self._write_rows([], [], [], [], [], [], [])
            self.selected_sample_kinds = []
            self.selected_clearances_m = []

        finite_clearance = clearance[np.isfinite(clearance)]
        self.sample_count = int(len(samples))
        self.valid_samples = int(np.count_nonzero(valid))
        self.active_constraints = int(len(selected))
        self.dropped_constraints = int(
            self.candidate_constraints - self.active_constraints
        )
        self.min_clearance_m = (
            float(np.min(finite_clearance)) if len(finite_clearance) else np.nan
        )

        if np.isfinite(self.min_clearance_m):
            self.min_clearance_episode_m = min(
                self.min_clearance_episode_m, self.min_clearance_m
            )
            self.max_penetration_episode_m = max(
                self.max_penetration_episode_m,
                max(0.0, -self.min_clearance_m),
            )
        self.max_active_constraints_episode = max(
            self.max_active_constraints_episode,
            self.active_constraints,
        )

    def get_diagnostics(self) -> dict:
        try:
            cpp_active = int(self.constraint.activeCount.value)
        except Exception:
            cpp_active = -1
        return {
            "enabled": bool(self.enabled),
            "update_count": int(self.update_count),
            "sample_count": int(self.sample_count),
            "valid_samples": int(self.valid_samples),
            "candidate_constraints": int(self.candidate_constraints),
            "active_constraints": int(self.active_constraints),
            "cpp_active_count": int(cpp_active),
            "dropped_constraints": int(self.dropped_constraints),
            "selected_sample_kinds": list(self.selected_sample_kinds),
            "selected_clearances_m": list(self.selected_clearances_m),
            "min_clearance_m": float(self.min_clearance_m),
            "min_clearance_episode_m": (
                float(self.min_clearance_episode_m)
                if np.isfinite(self.min_clearance_episode_m)
                else np.nan
            ),
            "max_penetration_episode_m": float(self.max_penetration_episode_m),
            "max_active_constraints_episode": int(
                self.max_active_constraints_episode
            ),
            "activation_clearance_m": float(self.activation_clearance_m),
            "max_constraints": int(self.max_constraints),
            "min_sample_separation_m": float(self.min_sample_separation_m),
        }
