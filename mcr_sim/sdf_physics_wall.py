"""SOFA-native SDF virtual wall for inserted catheter beam nodes.

The controller only writes translational entries of a ConstantForceField.  It
never changes positions, insertion state, policy actions, observations, or
rewards.  Native SOFA collision/contact remains enabled as the hard constraint.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import Sofa

from .training_config import (
    CATHETER_RADIUS_M,
    SDF_PHYSICS_WALL_ENABLED,
    SDF_WALL_ACTIVATION_CLEARANCE_M,
    SDF_WALL_MAX_FORCE_N,
    SDF_WALL_STIFFNESS_N_PER_M,
)
from .vessel_assets import (
    asset_vectors_to_sim,
    load_signed_distance_grid,
    sim_points_to_asset_source,
)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    normalized = str(value).strip().lower()
    if normalized in ("1", "true", "yes", "y", "on"):
        return True
    if normalized in ("0", "false", "no", "n", "off"):
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def wall_force_magnitude(
    clearance_m,
    activation_clearance_m: float = SDF_WALL_ACTIVATION_CLEARANCE_M,
    stiffness_n_per_m: float = SDF_WALL_STIFFNESS_N_PER_M,
    max_force_n: float = SDF_WALL_MAX_FORCE_N,
) -> np.ndarray:
    """Return the bounded wall-force magnitude for catheter-surface clearance."""

    clearance = np.asarray(clearance_m, dtype=np.float64)
    penetration_into_buffer = np.maximum(
        0.0, float(activation_clearance_m) - clearance
    )
    return np.minimum(
        float(max_force_n), float(stiffness_n_per_m) * penetration_into_buffer
    )


def compute_sdf_wall_forces(
    points_sim,
    sdf_grid,
    asset_T_env_sim,
    asset_offset_sim,
    asset_source_to_sim_scale: float,
    catheter_radius_m: float = CATHETER_RADIUS_M,
    activation_clearance_m: float = SDF_WALL_ACTIVATION_CLEARANCE_M,
    stiffness_n_per_m: float = SDF_WALL_STIFFNESS_N_PER_M,
    max_force_n: float = SDF_WALL_MAX_FORCE_N,
):
    """Compute force vectors and diagnostics for SOFA positions.

    The generated VTI convention is negative inside and positive outside.
    Therefore its normalized gradient points out of the lumen and the physical
    restoring direction is the negative normalized gradient.
    """

    points = np.asarray(points_sim, dtype=np.float64).reshape((-1, 3))
    if len(points) == 0:
        empty = np.zeros((0, 3), dtype=np.float64)
        return empty, np.zeros(0), empty, np.zeros(0), np.zeros(0, dtype=bool)

    source_points = sim_points_to_asset_source(
        points,
        asset_T_env_sim=asset_T_env_sim,
        asset_offset_sim=asset_offset_sim,
        asset_source_to_sim_scale=asset_source_to_sim_scale,
    )
    signed_source = np.asarray(
        sdf_grid.sample(source_points), dtype=np.float64
    ).reshape(-1)
    signed_sim = signed_source * float(asset_source_to_sim_scale)
    clearance = -signed_sim - float(catheter_radius_m)

    gradients_source = np.asarray(
        sdf_grid.gradient(source_points), dtype=np.float64
    ).reshape((-1, 3))
    gradients_sim = asset_vectors_to_sim(
        gradients_source, asset_T_env_sim=asset_T_env_sim
    )
    gradient_norm = np.linalg.norm(gradients_sim, axis=1)
    valid = (
        np.isfinite(signed_sim)
        & np.all(np.isfinite(gradients_sim), axis=1)
        & (gradient_norm > 1e-9)
    )

    inward = np.zeros_like(gradients_sim)
    inward[valid] = -gradients_sim[valid] / gradient_norm[valid, None]
    magnitudes = wall_force_magnitude(
        clearance,
        activation_clearance_m=activation_clearance_m,
        stiffness_n_per_m=stiffness_n_per_m,
        max_force_n=max_force_n,
    )
    magnitudes = np.where(valid, magnitudes, 0.0)
    forces = magnitudes[:, None] * inward
    return forces, clearance, inward, magnitudes, valid


class SDFPhysicsWallController(Sofa.Core.Controller):
    """Update a SOFA ConstantForceField on currently inserted beam nodes."""

    def __init__(
        self,
        instrument,
        sdf_vti,
        asset_T_env_sim,
        asset_offset_sim,
        asset_source_to_sim_scale,
        enabled: Optional[bool] = None,
        activation_clearance_m: Optional[float] = None,
        stiffness_n_per_m: Optional[float] = None,
        max_force_n: Optional[float] = None,
        catheter_radius_m: float = CATHETER_RADIUS_M,
        verbose: bool = False,
        *args,
        **kwargs,
    ):
        kwargs["listening"] = True
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.instrument = instrument
        self.sdf_grid = load_signed_distance_grid(sdf_vti)
        self.asset_T_env_sim = np.asarray(
            asset_T_env_sim, dtype=np.float64
        ).reshape(7)
        self.asset_offset_sim = np.asarray(
            asset_offset_sim, dtype=np.float64
        ).reshape(3)
        self.asset_source_to_sim_scale = float(asset_source_to_sim_scale)
        self.catheter_radius_m = float(catheter_radius_m)
        self.enabled = (
            _env_bool("MCR_SDF_PHYSICS_WALL", SDF_PHYSICS_WALL_ENABLED)
            if enabled is None
            else bool(enabled)
        )
        self.activation_clearance_m = float(
            os.environ.get(
                "MCR_SDF_WALL_ACTIVATION_CLEARANCE",
                SDF_WALL_ACTIVATION_CLEARANCE_M
                if activation_clearance_m is None
                else activation_clearance_m,
            )
        )
        self.stiffness_n_per_m = float(
            os.environ.get(
                "MCR_SDF_WALL_STIFFNESS",
                SDF_WALL_STIFFNESS_N_PER_M
                if stiffness_n_per_m is None
                else stiffness_n_per_m,
            )
        )
        self.max_force_n = float(
            os.environ.get(
                "MCR_SDF_WALL_MAX_FORCE",
                SDF_WALL_MAX_FORCE_N if max_force_n is None else max_force_n,
            )
        )
        if not (
            self.asset_source_to_sim_scale > 0.0
            and self.catheter_radius_m > 0.0
            and self.activation_clearance_m > 0.0
            and self.stiffness_n_per_m > 0.0
            and self.max_force_n > 0.0
        ):
            raise ValueError("Invalid SDF physics-wall configuration.")

        self.force_field = self.instrument.SDFWallCFF
        self.num_nodes = int(len(self.instrument.MO.position))
        self.verbose = bool(verbose)
        self.reset_diagnostics()
        self._write_force_rows(np.zeros((self.num_nodes, 6), dtype=np.float64))

        if self.verbose:
            print(
                "[SDF_PHYSICS_WALL]",
                "enabled=", self.enabled,
                "activation_clearance_m=", self.activation_clearance_m,
                "stiffness_n_per_m=", self.stiffness_n_per_m,
                "max_force_n=", self.max_force_n,
                "mechanical_nodes=", self.num_nodes,
            )

    def reset_diagnostics(self) -> None:
        self.active_nodes = 0
        self.sampled_nodes = 0
        self.invalid_nodes = 0
        self.max_force_N = 0.0
        self.total_force_N = 0.0
        self.min_clearance_m = np.nan
        self.max_force_episode_N = 0.0
        self.max_total_force_episode_N = 0.0
        self.min_clearance_episode_m = np.inf
        self.active_steps_episode = 0
        self.update_count = 0

    def reset(self) -> None:
        self.reset_diagnostics()
        self._write_force_rows(np.zeros((self.num_nodes, 6), dtype=np.float64))

    def _first_inserted_node(self) -> int:
        raw = getattr(self.instrument.IRC.indexFirstNode, "value", 0)
        values = np.asarray(raw).reshape(-1)
        first = int(values[0]) if values.size else 0
        return int(np.clip(first, 0, max(self.num_nodes - 1, 0)))

    def _write_force_rows(self, force_rows: np.ndarray) -> None:
        rows = np.asarray(force_rows, dtype=np.float64).reshape((self.num_nodes, 6))
        self.force_field.forces.value = rows.tolist()

    def onAnimateBeginEvent(self, event) -> None:
        force_rows = np.zeros((self.num_nodes, 6), dtype=np.float64)
        self.update_count += 1
        if not self.enabled or self.num_nodes == 0:
            self.active_nodes = 0
            self.sampled_nodes = 0
            self.invalid_nodes = 0
            self.max_force_N = 0.0
            self.total_force_N = 0.0
            self.min_clearance_m = np.nan
            self._write_force_rows(force_rows)
            return

        positions = np.asarray(
            self.instrument.MO.position.array(), dtype=np.float64
        )[:, :3]
        first = self._first_inserted_node()
        node_indices = np.arange(first, self.num_nodes, dtype=np.int64)
        node_positions = positions[node_indices]
        forces, clearance, _, magnitudes, valid = compute_sdf_wall_forces(
            node_positions,
            sdf_grid=self.sdf_grid,
            asset_T_env_sim=self.asset_T_env_sim,
            asset_offset_sim=self.asset_offset_sim,
            asset_source_to_sim_scale=self.asset_source_to_sim_scale,
            catheter_radius_m=self.catheter_radius_m,
            activation_clearance_m=self.activation_clearance_m,
            stiffness_n_per_m=self.stiffness_n_per_m,
            max_force_n=self.max_force_n,
        )
        force_rows[node_indices, :3] = forces
        self._write_force_rows(force_rows)

        finite_clearance = clearance[np.isfinite(clearance)]
        self.sampled_nodes = int(len(node_indices))
        self.invalid_nodes = int(np.count_nonzero(~valid))
        self.active_nodes = int(np.count_nonzero(magnitudes > 0.0))
        self.max_force_N = float(np.max(magnitudes)) if len(magnitudes) else 0.0
        self.total_force_N = float(np.sum(magnitudes))
        self.min_clearance_m = (
            float(np.min(finite_clearance)) if len(finite_clearance) else np.nan
        )
        self.max_force_episode_N = max(
            self.max_force_episode_N, self.max_force_N
        )
        self.max_total_force_episode_N = max(
            self.max_total_force_episode_N, self.total_force_N
        )
        if np.isfinite(self.min_clearance_m):
            self.min_clearance_episode_m = min(
                self.min_clearance_episode_m, self.min_clearance_m
            )
        if self.active_nodes > 0:
            self.active_steps_episode += 1

    def get_diagnostics(self) -> dict:
        min_episode = (
            float(self.min_clearance_episode_m)
            if np.isfinite(self.min_clearance_episode_m)
            else np.nan
        )
        return {
            "enabled": bool(self.enabled),
            "active_nodes": int(self.active_nodes),
            "sampled_nodes": int(self.sampled_nodes),
            "invalid_nodes": int(self.invalid_nodes),
            "max_force_N": float(self.max_force_N),
            "total_force_N": float(self.total_force_N),
            "min_clearance_m": float(self.min_clearance_m),
            "max_force_episode_N": float(self.max_force_episode_N),
            "max_total_force_episode_N": float(self.max_total_force_episode_N),
            "min_clearance_episode_m": min_episode,
            "active_steps_episode": int(self.active_steps_episode),
            "update_count": int(self.update_count),
            "stiffness_n_per_m": float(self.stiffness_n_per_m),
            "activation_clearance_m": float(self.activation_clearance_m),
            "max_force_limit_N": float(self.max_force_n),
        }
