"""Test-only dense adaptive SDF sampling for unilateral constraints.

This module intentionally lives under testing/py/diagnostics.  It does not
change production/training behavior.

The production controller currently samples CollisionDOF nodes + edge
midpoints and spatially suppresses selected rows using a separation close to
the catheter radius.  For the diagnostic prototype we instead:

1. keep every CollisionDOF node once;
2. subdivide every non-degenerate CollisionDOF edge so the sample spacing is
   no larger than the same voxel-derived step used by body SDF safety checks;
3. reduce active-row spatial suppression to a small fraction of that sampling
   step so dense candidates are not immediately thinned back to ~radius scale;
4. raise the diagnostic-only active-row cap.

All dense samples still use barycentric weights on the existing mapped
CollisionDOFs.  No MechanicalObject nodes are added and no positions are
projected or overwritten.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from types import MethodType

import numpy as np


@dataclass(frozen=True)
class DenseAdaptiveSamplingConfig:
    sample_step_fraction: float
    sdf_cell_sim_m: float
    max_sample_step_m: float
    min_active_separation_m: float
    max_constraints: int

    def to_dict(self):
        return asdict(self)


def _build_dense_samples(self, positions: np.ndarray):
    positions = np.asarray(positions, dtype=np.float64).reshape((-1, 3))
    n = len(positions)

    sample_points = [positions.copy()]
    i0 = [np.arange(n, dtype=np.int64)]
    i1 = [np.arange(n, dtype=np.int64)]
    w0 = [np.ones(n, dtype=np.float64)]
    w1 = [np.zeros(n, dtype=np.float64)]
    kinds = [np.asarray(["node"] * n, dtype=object)]

    max_step = float(self._diagnostic_dense_max_sample_step_m)
    edges = self._edges(n)

    for edge_index, edge in enumerate(edges):
        a = int(edge[0])
        b = int(edge[1])
        start = positions[a]
        end = positions[b]
        length = float(np.linalg.norm(end - start))
        if not np.isfinite(length) or length <= 1e-12:
            continue

        subdivisions = max(1, int(np.ceil(length / max_step)))
        if subdivisions <= 1:
            continue

        # Endpoints are already represented by the global node samples.
        # Only add strict interior barycentric points to avoid duplicate rows.
        ks = np.arange(1, subdivisions, dtype=np.float64)
        t = ks / float(subdivisions)
        pts = start[None, :] * (1.0 - t[:, None]) + end[None, :] * t[:, None]

        sample_points.append(pts)
        i0.append(np.full(len(t), a, dtype=np.int64))
        i1.append(np.full(len(t), b, dtype=np.int64))
        w0.append(1.0 - t)
        w1.append(t)
        kinds.append(
            np.asarray(
                [f"dense_edge_{edge_index}" for _ in range(len(t))],
                dtype=object,
            )
        )

    return (
        np.concatenate(sample_points, axis=0),
        np.concatenate(i0),
        np.concatenate(i1),
        np.concatenate(w0),
        np.concatenate(w1),
        np.concatenate(kinds),
    )


def install_dense_adaptive_sampling(
    controller,
    *,
    sample_step_fraction: float,
    max_constraints: int = 128,
    min_separation_fraction_of_step: float = 0.25,
) -> DenseAdaptiveSamplingConfig:
    """Install the diagnostic-only dense sampler on one controller instance."""

    fraction = float(sample_step_fraction)
    if not (0.0 < fraction <= 1.0):
        raise ValueError("sample_step_fraction must be in (0, 1].")
    max_constraints = int(max_constraints)
    if max_constraints < 1:
        raise ValueError("max_constraints must be >= 1.")
    separation_fraction = float(min_separation_fraction_of_step)
    if not (0.0 < separation_fraction <= 1.0):
        raise ValueError(
            "min_separation_fraction_of_step must be in (0, 1]."
        )

    sdf_cell_sim_m = float(
        np.min(np.asarray(controller.sdf_grid.spacing, dtype=np.float64))
        * float(controller.asset_source_to_sim_scale)
    )
    if not np.isfinite(sdf_cell_sim_m) or sdf_cell_sim_m <= 0.0:
        raise RuntimeError(
            f"Invalid SDF cell size in sim coordinates: {sdf_cell_sim_m}"
        )

    max_sample_step_m = max(1e-6, fraction * sdf_cell_sim_m)
    min_active_separation_m = max(
        1e-9,
        separation_fraction * max_sample_step_m,
    )

    # Test-only instance attributes consumed by the bound builder.
    controller._diagnostic_dense_max_sample_step_m = max_sample_step_m
    controller._diagnostic_original_build_samples = controller._build_samples
    controller._build_samples = MethodType(_build_dense_samples, controller)

    # These affect only the diagnostic controller instance.
    controller.min_sample_separation_m = min_active_separation_m
    controller.max_constraints = max_constraints

    config = DenseAdaptiveSamplingConfig(
        sample_step_fraction=fraction,
        sdf_cell_sim_m=sdf_cell_sim_m,
        max_sample_step_m=max_sample_step_m,
        min_active_separation_m=min_active_separation_m,
        max_constraints=max_constraints,
    )
    controller._diagnostic_dense_sampling_config = config
    return config


def dense_sampler_snapshot(controller, positions=None) -> dict:
    config = getattr(
        controller, "_diagnostic_dense_sampling_config", None
    )
    result = {
        "installed": config is not None,
        "config": config.to_dict() if config is not None else None,
    }
    if positions is not None and config is not None:
        samples, i0, i1, w0, w1, kinds = controller._build_samples(positions)
        result.update(
            {
                "sample_count": int(len(samples)),
                "node_sample_count": int(
                    np.count_nonzero(np.asarray(kinds, dtype=object) == "node")
                ),
                "dense_edge_sample_count": int(
                    len(samples)
                    - np.count_nonzero(
                        np.asarray(kinds, dtype=object) == "node"
                    )
                ),
                "indices_count": int(
                    min(len(i0), len(i1), len(w0), len(w1))
                ),
            }
        )
    return result
