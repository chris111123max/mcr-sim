"""SDF-derived native hard-contact geometry for diagnostic anti-penetration tests.

This module does NOT project or overwrite catheter positions. Instead, it
samples the vessel SDF at the dense catheter CollisionDOFs and creates small
local tangent triangles on the actual vessel surface. The ordinary SOFA
collision pipeline then turns catheter Line/Point versus these triangles into
FrictionContactConstraint constraints solved by the existing LCP solver.

The production vessel mesh remains Triangle-only. Vessel Point/Line models are
not required by this module.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import Sofa

from .training_config import (
    CATHETER_COLLISION_BODY_EDGES,
    CATHETER_COLLISION_TIP_EDGES,
    CATHETER_RADIUS_M,
    SDF_WALL_ACTIVATION_CLEARANCE_M,
)
from .vessel_assets import (
    asset_vectors_to_sim,
    load_signed_distance_grid,
    sim_points_to_asset_source,
)


def sample_sdf_clearance_and_outward(
    points_sim,
    sdf_grid,
    asset_T_env_sim,
    asset_offset_sim,
    asset_source_to_sim_scale: float,
    catheter_radius_m: float = CATHETER_RADIUS_M,
):
    """Return centerline clearance and outward SDF normals in simulation frame."""

    points = np.asarray(points_sim, dtype=np.float64).reshape((-1, 3))
    if len(points) == 0:
        empty = np.zeros((0, 3), dtype=np.float64)
        return np.zeros(0), empty, np.zeros(0, dtype=bool)

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

    outward = np.zeros_like(gradients_sim)
    outward[valid] = gradients_sim[valid] / gradient_norm[valid, None]
    clearance = -signed_sim - float(catheter_radius_m)
    return clearance, outward, valid


def _tangent_triangle(center, outward, circumradius):
    """Create one equilateral tangent triangle centered at center."""

    n = np.asarray(outward, dtype=np.float64).reshape(3)
    n_norm = float(np.linalg.norm(n))
    if n_norm <= 1e-12:
        raise ValueError("Invalid SDF normal")
    n = n / n_norm

    ref = (
        np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(n[2])) < 0.9
        else np.array([0.0, 1.0, 0.0], dtype=np.float64)
    )
    t1 = np.cross(n, ref)
    t1 /= float(np.linalg.norm(t1)) + 1e-15
    t2 = np.cross(n, t1)
    t2 /= float(np.linalg.norm(t2)) + 1e-15

    r = float(circumradius)
    s60 = np.sqrt(3.0) / 2.0
    center = np.asarray(center, dtype=np.float64).reshape(3)
    return np.stack(
        [
            center + r * t1,
            center + r * (-0.5 * t1 + s60 * t2),
            center + r * (-0.5 * t1 - s60 * t2),
        ],
        axis=0,
    )


def _parking_vertices(num_patches: int, circumradius: float) -> np.ndarray:
    """Return non-degenerate triangles parked far from the MCR scene."""

    vertices = np.zeros((3 * int(num_patches), 3), dtype=np.float64)
    for i in range(int(num_patches)):
        center = np.array([10.0 + 0.01 * i, 10.0, 10.0], dtype=np.float64)
        vertices[3 * i : 3 * i + 3] = _tangent_triangle(
            center, np.array([0.0, 0.0, 1.0]), circumradius
        )
    return vertices


class SDFHardConstraintController(Sofa.Core.Controller):
    """Create SDF-local tangent collision triangles before each physics solve."""

    def __init__(
        self,
        root_node,
        instrument,
        sdf_vti,
        asset_T_env_sim,
        asset_offset_sim,
        asset_source_to_sim_scale,
        catheter_radius_m: float = CATHETER_RADIUS_M,
        activation_clearance_m: float = SDF_WALL_ACTIVATION_CLEARANCE_M,
        collision_proximity_m: float = 0.0,
        collision_exclusion_group: int = 2,
        enabled: bool = False,
        verbose: bool = False,
        *args,
        **kwargs,
    ):
        kwargs["listening"] = True
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.root_node = root_node
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
        self.activation_clearance_m = float(activation_clearance_m)
        self.collision_proximity_m = float(collision_proximity_m)
        self.collision_exclusion_group = int(collision_exclusion_group)
        self.enabled = bool(enabled)
        self.verbose = bool(verbose)

        if self.asset_source_to_sim_scale <= 0.0:
            raise ValueError("asset_source_to_sim_scale must be positive")
        if self.catheter_radius_m <= 0.0:
            raise ValueError("catheter_radius_m must be positive")
        if self.activation_clearance_m <= 0.0:
            raise ValueError("activation_clearance_m must be positive")

        collision_node = self.instrument.InstrumentCombined.getChild("mcr_collis")
        self.catheter_collision_dofs = collision_node.getObject("CollisionDOFs")
        if self.catheter_collision_dofs is None:
            raise RuntimeError("Catheter CollisionDOFs not found")

        self.max_samples = int(
            CATHETER_COLLISION_BODY_EDGES + CATHETER_COLLISION_TIP_EDGES + 1
        )

        # Derived from catheter size, not a solver tuning knob. An equilateral
        # triangle with circumradius 4r has inradius 2r around its source sample.
        self.patch_circumradius_m = 4.0 * self.catheter_radius_m

        triangles = np.arange(3 * self.max_samples, dtype=np.int64).reshape(
            (self.max_samples, 3)
        )
        self._parked_vertices = _parking_vertices(
            self.max_samples, self.patch_circumradius_m
        )

        self.collision_node = self.root_node.addChild("SDFHardConstraintCollision")
        self.topology = self.collision_node.addObject(
            "MeshTopology",
            name="SDFHardTopology",
            position=self._parked_vertices.tolist(),
            triangles=triangles.tolist(),
        )
        self.mechanical_object = self.collision_node.addObject(
            "MechanicalObject",
            name="SDFHardWallDOFs",
            template="Vec3d",
            position=self._parked_vertices.tolist(),
        )
        self.collision_model = self.collision_node.addObject(
            "TriangleCollisionModel",
            name="SDFHardWallTriangles",
            moving=True,
            simulated=False,
            bothSide=True,
            proximity=self.collision_proximity_m,
            group=self.collision_exclusion_group,
        )

        self.reset_diagnostics()
        self._write_vertices(self._parked_vertices)

        if self.verbose:
            print(
                "[SDF_HARD_CONSTRAINT]",
                "enabled=", self.enabled,
                "max_samples=", self.max_samples,
                "activation_clearance_m=", self.activation_clearance_m,
                "patch_circumradius_m=", self.patch_circumradius_m,
                "collision_proximity_m=", self.collision_proximity_m,
                "exclusion_group=", self.collision_exclusion_group,
            )

    def _write_vertices(self, vertices) -> None:
        arr = np.asarray(vertices, dtype=np.float64).reshape(
            (3 * self.max_samples, 3)
        )
        self.mechanical_object.position.value = arr.tolist()

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        if not self.enabled:
            self._write_vertices(self._parked_vertices)
            self.active_patches = 0

    def reset_diagnostics(self) -> None:
        self.update_count = 0
        self.sample_count = 0
        self.valid_samples = 0
        self.active_patches = 0
        self.invalid_samples = 0
        self.min_clearance_m = np.nan
        self.min_clearance_episode_m = np.inf
        self.max_penetration_episode_m = 0.0
        self.max_active_patches_episode = 0

    def reset(self) -> None:
        self.reset_diagnostics()
        self._write_vertices(self._parked_vertices)

    def onAnimateBeginEvent(self, event) -> None:
        self.update_count += 1

        if not self.enabled:
            self.sample_count = 0
            self.valid_samples = 0
            self.active_patches = 0
            self.invalid_samples = 0
            self.min_clearance_m = np.nan
            self._write_vertices(self._parked_vertices)
            return

        positions = np.asarray(
            self.catheter_collision_dofs.position.array(), dtype=np.float64
        )[:, :3]
        if len(positions) > self.max_samples:
            raise RuntimeError(
                "Catheter CollisionDOFs exceed preallocated SDF hard-wall patches: "
                f"{len(positions)} > {self.max_samples}"
            )

        clearance, outward, valid = sample_sdf_clearance_and_outward(
            positions,
            sdf_grid=self.sdf_grid,
            asset_T_env_sim=self.asset_T_env_sim,
            asset_offset_sim=self.asset_offset_sim,
            asset_source_to_sim_scale=self.asset_source_to_sim_scale,
            catheter_radius_m=self.catheter_radius_m,
        )

        vertices = self._parked_vertices.copy()
        active = valid & np.isfinite(clearance) & (
            clearance <= self.activation_clearance_m
        )

        active_indices = np.flatnonzero(active)
        for idx in active_indices:
            # Place the virtual triangle on the actual vessel surface. Catheter
            # collision proximity remains the representation of catheter radius,
            # just as in the original Triangle-vessel collision configuration.
            surface_distance = float(clearance[idx] + self.catheter_radius_m)
            surface_center = positions[idx] + outward[idx] * surface_distance
            vertices[3 * idx : 3 * idx + 3] = _tangent_triangle(
                surface_center,
                outward[idx],
                self.patch_circumradius_m,
            )

        self._write_vertices(vertices)

        finite_clearance = clearance[np.isfinite(clearance)]
        self.sample_count = int(len(positions))
        self.valid_samples = int(np.count_nonzero(valid))
        self.invalid_samples = int(np.count_nonzero(~valid))
        self.active_patches = int(len(active_indices))
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
        self.max_active_patches_episode = max(
            self.max_active_patches_episode, self.active_patches
        )

    def _number_of_contacts(self) -> Optional[int]:
        try:
            value = self.collision_model.numberOfContacts.value
            arr = np.asarray(value).reshape(-1)
            return int(arr[0]) if arr.size else 0
        except Exception:
            return None

    def get_diagnostics(self) -> dict:
        return {
            "enabled": bool(self.enabled),
            "update_count": int(self.update_count),
            "sample_count": int(self.sample_count),
            "valid_samples": int(self.valid_samples),
            "invalid_samples": int(self.invalid_samples),
            "active_patches": int(self.active_patches),
            "max_active_patches_episode": int(self.max_active_patches_episode),
            "min_clearance_m": float(self.min_clearance_m),
            "min_clearance_episode_m": (
                float(self.min_clearance_episode_m)
                if np.isfinite(self.min_clearance_episode_m)
                else np.nan
            ),
            "max_penetration_episode_m": float(
                self.max_penetration_episode_m
            ),
            "hard_wall_contact_count": self._number_of_contacts(),
            "activation_clearance_m": float(self.activation_clearance_m),
            "patch_circumradius_m": float(self.patch_circumradius_m),
            "collision_proximity_m": float(self.collision_proximity_m),
            "collision_exclusion_group": int(self.collision_exclusion_group),
        }
