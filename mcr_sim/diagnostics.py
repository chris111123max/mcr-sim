"""Non-reward safety and solver diagnostics.

Missing SOFA metrics remain explicitly unavailable; they are never silently
replaced by zeros because that would make later safety analysis misleading.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class StepDiagnostics:
    contact_count: Optional[int] = None
    max_penetration_m: Optional[float] = None
    lcp_iterations: Optional[int] = None
    minimum_clearance_m: Optional[float] = None
    simulation_error: bool = False
    unavailable_reasons: Dict[str, str] = field(default_factory=dict)

    def as_info(self) -> Dict[str, object]:
        return {
            "contact_count": self.contact_count,
            "max_penetration_m": self.max_penetration_m,
            "lcp_iterations": self.lcp_iterations,
            "minimum_clearance_m": self.minimum_clearance_m,
            "simulation_error": self.simulation_error,
            "diagnostic_unavailable_reasons": dict(self.unavailable_reasons),
        }


class DiagnosticAccumulator:
    def __init__(self):
        self._samples: List[StepDiagnostics] = []

    def add(self, sample: StepDiagnostics) -> None:
        self._samples.append(sample)

    @staticmethod
    def _maximum(values):
        available = [value for value in values if value is not None]
        return max(available) if available else None

    @staticmethod
    def _minimum(values):
        available = [value for value in values if value is not None]
        return min(available) if available else None

    def finish(self) -> StepDiagnostics:
        if not self._samples:
            return StepDiagnostics(
                unavailable_reasons={"all": "backend returned no substep diagnostics"}
            )
        reasons: Dict[str, str] = {}
        for sample in self._samples:
            reasons.update(sample.unavailable_reasons)
        return StepDiagnostics(
            contact_count=self._maximum(s.contact_count for s in self._samples),
            max_penetration_m=self._maximum(
                s.max_penetration_m for s in self._samples
            ),
            lcp_iterations=self._maximum(s.lcp_iterations for s in self._samples),
            minimum_clearance_m=self._minimum(
                s.minimum_clearance_m for s in self._samples
            ),
            simulation_error=any(s.simulation_error for s in self._samples),
            unavailable_reasons=reasons,
        )
