"""Continuous local magnetic-field and insertion action planning."""
from dataclasses import dataclass
import math
import numpy as np

from .config import ActionConfig, InstrumentConfig, SofaConfig


def quantize_half_away_from_zero(value: float, resolution: float) -> float:
    """Deterministic symmetric rounding; exact halves move away from zero."""
    value = float(np.clip(value, -1.0, 1.0))
    scaled = abs(value) / float(resolution)
    quantized = math.floor(scaled + 0.5) * float(resolution)
    return float(np.clip(math.copysign(quantized, value), -1.0, 1.0))


def _normalize(vector, fallback=None):
    vector = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-15:
        if fallback is None:
            raise ValueError("Cannot normalize a zero vector")
        return _normalize(fallback)
    return vector / norm


def rotation_matrix_from_rotvec(rotvec) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=float).reshape(3)
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-15:
        return np.eye(3)
    axis = rotvec / angle
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def slerp_directions(start, target, fraction: float) -> np.ndarray:
    start = _normalize(start); target = _normalize(target)
    dot = float(np.clip(np.dot(start, target), -1.0, 1.0))
    angle = math.acos(dot)
    if angle < 1e-12:
        return _normalize((1.0 - fraction) * start + fraction * target, start)
    return (math.sin((1.0 - fraction) * angle) * start + math.sin(fraction * angle) * target) / math.sin(angle)


@dataclass(frozen=True)
class ActionPlan:
    raw_action: np.ndarray
    quantized_action: np.ndarray
    executed_action: np.ndarray
    start_field_world_t: np.ndarray
    target_field_world_t: np.ndarray
    field_substeps_world_t: np.ndarray
    requested_insertion_m: float
    insertion_substeps_m: np.ndarray
    magnetic_rotation_command_deg: float
    magnetic_rotation_executed_deg: float


class MagneticActionController:
    def __init__(self, action: ActionConfig = ActionConfig(), sofa: SofaConfig = SofaConfig(), instrument: InstrumentConfig = InstrumentConfig()):
        self.action = action
        self.sofa = sofa
        self.instrument = instrument

    def plan(self, raw_action, current_field_world_t, tip_frame_world) -> ActionPlan:
        raw = np.asarray(raw_action, dtype=float).reshape(3)
        if not np.all(np.isfinite(raw)) or np.any(raw < -1.0) or np.any(raw > 1.0):
            raise ValueError("Action must be finite and inside [-1, 1]^3")
        quantized = raw.copy()
        quantized[0] = quantize_half_away_from_zero(raw[0], self.action.magnetic_resolution)
        quantized[1] = quantize_half_away_from_zero(raw[1], self.action.magnetic_resolution)
        magnetic = quantized[:2].copy()
        magnetic_norm = float(np.linalg.norm(magnetic))
        if magnetic_norm > 1.0:
            magnetic /= magnetic_norm
        executed = np.array([magnetic[0], magnetic[1], raw[2]], dtype=float)

        frame = np.asarray(tip_frame_world, dtype=float).reshape(3, 3)
        if not np.allclose(frame.T @ frame, np.eye(3), atol=1e-6):
            raise ValueError("Tip frame must be orthonormal")
        normal_world = frame[:, 1]
        binormal_world = frame[:, 2]
        max_angle = math.radians(self.action.max_field_rotation_deg)
        rotvec = max_angle * (magnetic[0] * normal_world + magnetic[1] * binormal_world)

        start_field = np.asarray(current_field_world_t, dtype=float).reshape(3)
        if float(np.linalg.norm(start_field)) < 1e-15:
            start_direction = _normalize(self.instrument.initial_field_t)
        else:
            start_direction = _normalize(start_field)
        target_direction = rotation_matrix_from_rotvec(rotvec) @ start_direction
        magnitude = self.instrument.initial_field_magnitude_t
        start_field_fixed = magnitude * start_direction
        target_field = magnitude * _normalize(target_direction)
        fields = np.asarray([
            magnitude * slerp_directions(start_direction, target_direction, (index + 1) / self.sofa.physics_substeps)
            for index in range(self.sofa.physics_substeps)
        ])
        requested = self.action.max_insertion_per_rl_step_m * raw[2]
        insertion = np.full(self.sofa.physics_substeps, requested / self.sofa.physics_substeps, dtype=float)
        executed_angle = math.degrees(math.acos(float(np.clip(np.dot(start_direction, target_direction), -1.0, 1.0))))
        return ActionPlan(
            raw, quantized, executed, start_field_fixed, target_field, fields,
            float(requested), insertion,
            math.degrees(float(np.linalg.norm(rotvec))), executed_angle,
        )
