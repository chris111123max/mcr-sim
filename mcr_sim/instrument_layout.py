"""Nelson catheter geometry and discretization independent of SOFA imports."""
from dataclasses import dataclass
from typing import Tuple
import numpy as np

from .config import BEAM_PRESETS, BeamDiscretization, InstrumentConfig


@dataclass(frozen=True)
class BeamSegment:
    beam_index: int
    distal_node_index: int
    region: str
    start_m: float
    end_m: float

    @property
    def length_m(self) -> float:
        return self.end_m - self.start_m


@dataclass(frozen=True)
class MagnetOverlap:
    beam_index: int
    node_index: int
    overlap_m: float
    weight: float
    segment_start_m: float
    segment_end_m: float


class NelsonInstrumentLayout:
    def __init__(self, config: InstrumentConfig = InstrumentConfig()):
        self.config = config
        self.segments = self._segments()
        self.magnet_overlaps = self._magnet_overlaps()
        self.validate()

    def _segments(self) -> Tuple[BeamSegment, ...]:
        d = self.config.discretization
        body_step = self.config.body_length_m / d.body_beams
        tip_step = self.config.tip_length_m / d.tip_beams
        segments = []
        for index in range(d.body_beams):
            segments.append(BeamSegment(index, index + 1, "body", index * body_step, (index + 1) * body_step))
        for local in range(d.tip_beams):
            index = d.body_beams + local
            start = self.config.body_length_m + local * tip_step
            segments.append(BeamSegment(index, index + 1, "tip", start, start + tip_step))
        return tuple(segments)

    def _magnet_overlaps(self) -> Tuple[MagnetOverlap, ...]:
        magnet_start = self.config.total_length_m - self.config.magnet_length_m
        raw = []
        for segment in self.segments:
            overlap = max(0.0, min(segment.end_m, self.config.total_length_m) - max(segment.start_m, magnet_start))
            if overlap > 1e-15:
                raw.append((segment, overlap))
        total = sum(overlap for _, overlap in raw)
        if not np.isclose(total, self.config.magnet_length_m, rtol=0.0, atol=1e-12):
            raise ValueError(f"Magnet coverage {total} != {self.config.magnet_length_m}")
        return tuple(
            MagnetOverlap(
                segment.beam_index, segment.distal_node_index, overlap,
                overlap / self.config.magnet_length_m,
                segment.start_m, segment.end_m,
            )
            for segment, overlap in raw
        )

    @property
    def magnetic_node_indices(self) -> Tuple[int, ...]:
        return tuple(item.node_index for item in self.magnet_overlaps)

    def validate(self):
        d = self.config.discretization
        if len(self.segments) != d.total_beams:
            raise ValueError("Physical beam count mismatch")
        if d.mechanical_dofs != d.total_beams + 1:
            raise ValueError("Rigid DOF count must equal beam count + 1")
        if not np.isclose(sum(item.weight for item in self.magnet_overlaps), 1.0, atol=1e-12):
            raise ValueError("Magnet torque weights do not sum to one")
        if max(self.magnetic_node_indices) >= d.mechanical_dofs:
            raise ValueError("Magnetic node index outside mechanical topology")
        if d.collision_edges_body == d.body_beams and d.collision_edges_tip == d.tip_beams:
            # Equality is allowed, but no code derives one from the other.
            pass
        if not np.isclose(self.segments[-1].end_m, self.config.total_length_m, atol=1e-12):
            raise ValueError("Beam layout does not span physical catheter length")


def instrument_with_preset(name: str) -> NelsonInstrumentLayout:
    if name not in BEAM_PRESETS:
        raise KeyError(name)
    return NelsonInstrumentLayout(InstrumentConfig(discretization=BEAM_PRESETS[name]))
