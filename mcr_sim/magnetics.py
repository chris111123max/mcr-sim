"""Nelson dipole physics and overlap-weighted distributed magnetic torque."""
from dataclasses import dataclass
from typing import Sequence
import numpy as np

from .config import InstrumentConfig
from .instrument_layout import MagnetOverlap, NelsonInstrumentLayout

MU_0 = 4.0 * np.pi * 1e-7


def annular_magnet_volume_m3(length_m: float, outer_diameter_m: float, inner_diameter_m: float) -> float:
    return float(length_m * np.pi * ((outer_diameter_m / 2.0) ** 2 - (inner_diameter_m / 2.0) ** 2))


def nelson_dipole_moment_a_m2(config: InstrumentConfig = InstrumentConfig()) -> float:
    volume = annular_magnet_volume_m3(config.magnet_length_m, config.magnet_outer_diameter_m, config.magnet_inner_diameter_m)
    return float(config.magnet_remanence_t * volume / MU_0)


def quaternion_xyzw_to_matrix(quaternion) -> np.ndarray:
    q = np.asarray(quaternion, dtype=float).reshape(4)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm < 1e-15:
        raise ValueError("Invalid orientation quaternion")
    x, y, z, w = q / norm
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ], dtype=float)


@dataclass(frozen=True)
class DistributedTorque:
    node_index: int
    weight: float
    moment_world_a_m2: np.ndarray
    torque_world_n_m: np.ndarray


def distributed_magnetic_torques(
    field_world_t,
    segment_orientations_xyzw: Sequence[Sequence[float]],
    layout: NelsonInstrumentLayout,
) -> tuple:
    field = np.asarray(field_world_t, dtype=float).reshape(3)
    if len(segment_orientations_xyzw) != len(layout.magnet_overlaps):
        raise ValueError("One orientation is required for every magnet-overlap segment")
    total_moment = nelson_dipole_moment_a_m2(layout.config)
    result = []
    for overlap, quaternion in zip(layout.magnet_overlaps, segment_orientations_xyzw):
        direction = quaternion_xyzw_to_matrix(quaternion) @ np.array([1.0, 0.0, 0.0])
        moment = total_moment * overlap.weight * direction
        result.append(DistributedTorque(overlap.node_index, overlap.weight, moment, np.cross(moment, field)))
    return tuple(result)


def total_torque(torques: Sequence[DistributedTorque]) -> np.ndarray:
    return np.sum([item.torque_world_n_m for item in torques], axis=0) if torques else np.zeros(3)
