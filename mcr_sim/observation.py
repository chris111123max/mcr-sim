"""Fixed-scale, tip-local 41D observation; recurrent memory lives in the LSTM."""
from dataclasses import dataclass
import numpy as np
from gymnasium import spaces

from .config import ActionConfig, InstrumentConfig, ObservationConfig
from .pathfinder import PathResult
from .vessel_bundle import VesselBundle


def sample_polyline_m(points_m, distances_m) -> np.ndarray:
    """Interpolate physical arc length; short paths clamp to their real endpoint."""
    points = np.asarray(points_m, dtype=float).reshape(-1, 3)
    distances = np.asarray(distances_m, dtype=float).reshape(-1)
    if len(points) == 0 or not np.all(np.isfinite(points)):
        raise ValueError("Polyline must contain finite physical coordinates")
    if not np.all(np.isfinite(distances)) or np.any(distances < 0):
        raise ValueError("Sampling distances must be finite and nonnegative")
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    keep = np.r_[True, np.diff(arc) > 1e-12]
    points, arc = points[keep], arc[keep]
    return np.column_stack([np.interp(distances, arc, points[:, axis]) for axis in range(3)])


def points_to_tip_local(points_world_m, tip_world_m, tip_frame_world) -> np.ndarray:
    points = np.asarray(points_world_m, dtype=float).reshape(-1, 3)
    tip = np.asarray(tip_world_m, dtype=float).reshape(3)
    frame = np.asarray(tip_frame_world, dtype=float).reshape(3, 3)
    return (frame.T @ (points - tip).T).T


@dataclass(frozen=True)
class ObservationParts:
    magnetic_field_local: np.ndarray
    reference_points_local: np.ndarray
    remaining_path: float
    shape_points_local: np.ndarray
    remaining_time: float
    inserted_length: float
    local_radius: float
    target_local: np.ndarray
    last_action: np.ndarray
    tip_displacement_local: np.ndarray
    last_insertion: float

    def flatten(self) -> np.ndarray:
        return np.concatenate([
            self.magnetic_field_local.reshape(-1),             # 0:3
            self.reference_points_local.reshape(-1),           # 3:18
            [self.remaining_path],                             # 18
            self.shape_points_local.reshape(-1),               # 19:28
            [self.remaining_time, self.inserted_length, self.local_radius],  # 28:31
            self.target_local.reshape(-1),                     # 31:34
            self.last_action.reshape(-1),                      # 34:37
            self.tip_displacement_local.reshape(-1),            # 37:40
            [self.last_insertion],                             # 40
        ]).astype(np.float32)


class MagneticObservationBuilder:
    def __init__(self, bundle: VesselBundle, config: ObservationConfig = ObservationConfig(), *,
                 insertion_length_scale_m=InstrumentConfig().total_length_m,
                 insertion_step_scale_m=ActionConfig().max_insertion_per_rl_step_m):
        self.bundle, self.config = bundle, config
        self.insertion_length_scale_m = float(insertion_length_scale_m)
        self.insertion_step_scale_m = float(insertion_step_scale_m)
        scales = (config.spatial_scale_m, config.remaining_path_scale_m, config.target_scale_m,
                  config.radius_scale_m, config.tip_displacement_scale_m,
                  self.insertion_length_scale_m, self.insertion_step_scale_m)
        if not all(np.isfinite(x) and x > 0 for x in scales):
            raise ValueError("Observation normalization scales must be finite and positive")
        if len(config.lookahead_distances_m) != 5 or len(config.shape_offsets_m) != 3:
            raise ValueError("41D observation requires five path points and three shape points")
        self.space = spaces.Box(-1.0, 1.0, shape=(config.dimension,), dtype=np.float32)

    @staticmethod
    def _normalized(value, scale, nonnegative=False):
        value = np.asarray(value, dtype=float)
        if not np.all(np.isfinite(value)):
            raise ValueError("Observation input contains NaN or Inf")
        return np.clip(value / scale, 0.0 if nonnegative else -1.0, 1.0)

    def build(self, catheter_tracking_m, target_m, last_action, field_world_t, tip_frame_world, *,
              path: PathResult, remaining_time_ratio, inserted_length_m,
              tip_displacement_world_m, last_insertion_m) -> ObservationParts:
        tracking = np.asarray(catheter_tracking_m, dtype=float).reshape(-1, 3)
        frame = np.asarray(tip_frame_world, dtype=float).reshape(3, 3)
        field = np.asarray(field_world_t, dtype=float).reshape(3)
        if not np.all(np.isfinite(frame)) or not np.allclose(frame.T @ frame, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(frame), 1.0, atol=1e-5):
            raise ValueError("Tip frame must be a finite proper rotation matrix")
        if not np.all(np.isfinite(field)):
            raise ValueError("Magnetic field must be finite")
        # Real SOFA node positions, tip first; never substitute vessel geometry.
        shape_world = sample_polyline_m(tracking, self.config.shape_offsets_m)
        tip = tracking[0]
        reference_world = sample_polyline_m(path.points_mm * 0.001, self.config.lookahead_distances_m)
        projection = path.start_projection
        branch = self.bundle.branches[projection.branch]
        i, fraction = projection.segment_index, projection.fraction
        radius_m = ((1.0 - fraction) * branch.radii_mm[i] + fraction * branch.radii_mm[i + 1]) * 0.001
        if not np.isfinite(radius_m) or radius_m <= 0 or not np.isfinite(path.length_mm) or path.length_mm < 0:
            raise ValueError("Centerline path/radius must be physically valid")
        norm = float(np.linalg.norm(field))
        field_local = frame.T @ (field / norm) if norm > 1e-15 else np.zeros(3)
        normed = self._normalized
        parts = ObservationParts(
            magnetic_field_local=normed(field_local, 1.0),
            reference_points_local=normed(points_to_tip_local(reference_world, tip, frame), self.config.spatial_scale_m),
            remaining_path=float(normed(path.length_mm * 0.001, self.config.remaining_path_scale_m, True)),
            shape_points_local=normed(points_to_tip_local(shape_world, tip, frame), self.config.spatial_scale_m),
            remaining_time=float(normed(remaining_time_ratio, 1.0, True)),
            inserted_length=float(normed(inserted_length_m, self.insertion_length_scale_m, True)),
            local_radius=float(normed(radius_m, self.config.radius_scale_m, True)),
            target_local=normed(points_to_tip_local(target_m, tip, frame).reshape(3), self.config.target_scale_m),
            last_action=normed(np.asarray(last_action).reshape(3), 1.0),
            tip_displacement_local=normed(frame.T @ np.asarray(tip_displacement_world_m, dtype=float).reshape(3), self.config.tip_displacement_scale_m),
            last_insertion=float(normed(last_insertion_m, self.insertion_step_scale_m)),
        )
        flat = parts.flatten()
        if flat.shape != (41,) or not np.all(np.isfinite(flat)) or not self.space.contains(flat):
            raise ValueError("Observation violates the finite 41D Gym space")
        return parts
