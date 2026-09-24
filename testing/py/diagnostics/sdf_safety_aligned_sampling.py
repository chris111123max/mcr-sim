"""Test-only safety-aligned sampler for the SDF unilateral constraint.

This module lives only under testing/py/diagnostics and does not alter
production/training behavior.

Goal
----
Use the *same longitudinal sample set* as MCREnv._get_sdf_sample_points():
the exact distal-to-proximal arclength locations used by the production body
SDF safety diagnostic.

The current C++ SDFUnilateralConstraint is attached to mapped Vec3
CollisionDOFs, while the safety diagnostic is built from the Rigid3d beam
centers.  Therefore a test-only bridge is required:

1. ask the environment for its exact production safety sample points;
2. compute their cumulative arclength from the catheter tip;
3. order the CollisionDOF topology as a tip-to-proximal chain;
4. place one unilateral sample at the same cumulative arclength on that chain;
5. express it with two CollisionDOF indices and barycentric weights.

The helper records the geometric representation error between each original
safety point and the CollisionDOF-represented point.  If that error is not
small, the experiment demonstrates that a CollisionDOF-attached constraint
cannot exactly share the production safety geometry and the next design should
move the constraint closer to the beam state rather than add more sampling
patches.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import MethodType

import numpy as np


@dataclass(frozen=True)
class SafetyAlignedSamplingConfig:
    max_constraints: int
    min_active_separation_m: float

    def to_dict(self):
        return asdict(self)


def _ordered_collision_chain(controller, positions, tip):
    positions = np.asarray(positions, dtype=np.float64).reshape((-1, 3))
    edges = np.asarray(controller._edges(len(positions)), dtype=np.int64).reshape((-1, 2))
    if len(edges) < 1:
        raise RuntimeError("Collision topology has no valid edges")

    adjacency = {i: [] for i in range(len(positions))}
    for edge_id, (a, b) in enumerate(edges):
        a = int(a)
        b = int(b)
        adjacency[a].append((b, edge_id))
        adjacency[b].append((a, edge_id))

    active_vertices = [v for v, nbrs in adjacency.items() if nbrs]
    degrees = {v: len(adjacency[v]) for v in active_vertices}
    bad = {v: d for v, d in degrees.items() if d > 2}
    if bad:
        raise RuntimeError(
            "Collision topology is not a simple chain; branching degrees="
            + repr(bad)
        )

    endpoints = [v for v, d in degrees.items() if d == 1]
    if len(endpoints) != 2:
        raise RuntimeError(
            f"Expected exactly two collision-chain endpoints, got {endpoints}"
        )

    tip = np.asarray(tip, dtype=np.float64).reshape(3)
    start = min(
        endpoints,
        key=lambda v: float(np.linalg.norm(positions[v] - tip)),
    )

    ordered = [int(start)]
    used_edges = set()
    previous = None
    current = int(start)

    while True:
        options = [
            (neighbor, edge_id)
            for neighbor, edge_id in adjacency[current]
            if edge_id not in used_edges and neighbor != previous
        ]
        if not options:
            break
        if len(options) != 1:
            raise RuntimeError(
                f"Ambiguous collision-chain traversal at vertex {current}: {options}"
            )
        neighbor, edge_id = options[0]
        used_edges.add(int(edge_id))
        previous, current = current, int(neighbor)
        ordered.append(current)

    if len(used_edges) != len(edges):
        raise RuntimeError(
            f"Collision chain traversal used {len(used_edges)}/{len(edges)} edges"
        )

    chain_points = positions[np.asarray(ordered, dtype=np.int64)]
    segment_lengths = np.linalg.norm(np.diff(chain_points, axis=0), axis=1)
    cumulative = np.concatenate(
        ([0.0], np.cumsum(segment_lengths, dtype=np.float64))
    )
    return np.asarray(ordered, dtype=np.int64), chain_points, cumulative


def _arc_lengths(points):
    points = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    if len(points) == 0:
        return np.zeros(0, dtype=np.float64)
    if len(points) == 1:
        return np.zeros(1, dtype=np.float64)
    return np.concatenate(
        (
            [0.0],
            np.cumsum(
                np.linalg.norm(np.diff(points, axis=0), axis=1),
                dtype=np.float64,
            ),
        )
    )


def _sample_chain_at_arclengths(
    ordered_vertices,
    chain_points,
    chain_cumulative,
    targets,
):
    ordered_vertices = np.asarray(ordered_vertices, dtype=np.int64).reshape(-1)
    chain_points = np.asarray(chain_points, dtype=np.float64).reshape((-1, 3))
    chain_cumulative = np.asarray(chain_cumulative, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)

    if len(chain_points) < 2:
        raise RuntimeError("Collision chain needs at least two points")

    total = float(chain_cumulative[-1])
    clamped = np.clip(targets, 0.0, total)

    samples = []
    i0 = []
    i1 = []
    w0 = []
    w1 = []

    for target in clamped:
        seg = int(np.searchsorted(chain_cumulative, target, side="right") - 1)
        seg = int(np.clip(seg, 0, len(chain_points) - 2))
        start_s = float(chain_cumulative[seg])
        end_s = float(chain_cumulative[seg + 1])
        denom = max(end_s - start_s, 1e-12)
        t = float(np.clip((float(target) - start_s) / denom, 0.0, 1.0))

        a = int(ordered_vertices[seg])
        b = int(ordered_vertices[seg + 1])
        point = (1.0 - t) * chain_points[seg] + t * chain_points[seg + 1]

        samples.append(point)
        i0.append(a)
        i1.append(b)
        w0.append(1.0 - t)
        w1.append(t)

    return (
        np.asarray(samples, dtype=np.float64),
        np.asarray(i0, dtype=np.int64),
        np.asarray(i1, dtype=np.int64),
        np.asarray(w0, dtype=np.float64),
        np.asarray(w1, dtype=np.float64),
        clamped,
    )


def _build_safety_aligned_samples(self, collision_positions):
    collision_positions = np.asarray(
        collision_positions, dtype=np.float64
    ).reshape((-1, 3))

    env = self._diagnostic_safety_aligned_env
    controller = env.mcr_controller_sofa
    tip = np.asarray(
        controller.get_pos_quat_catheter_tip()[:3],
        dtype=np.float64,
    ).reshape(3)

    # Exact production safety sample set, in distal-to-proximal order.
    safety_points = np.asarray(
        env._get_sdf_sample_points(tip),
        dtype=np.float64,
    ).reshape((-1, 3))
    safety_arc = _arc_lengths(safety_points)

    ordered, chain_points, chain_arc = _ordered_collision_chain(
        self,
        collision_positions,
        tip,
    )

    samples, i0, i1, w0, w1, used_arc = _sample_chain_at_arclengths(
        ordered,
        chain_points,
        chain_arc,
        safety_arc,
    )

    representation_error = np.linalg.norm(samples - safety_points, axis=1)
    kinds = np.asarray(["safety_aligned"] * len(samples), dtype=object)

    self._diagnostic_safety_aligned_snapshot = {
        "safety_sample_count": int(len(safety_points)),
        "collision_chain_vertex_count": int(len(ordered)),
        "safety_total_arclength_m": (
            float(safety_arc[-1]) if len(safety_arc) else 0.0
        ),
        "collision_chain_total_arclength_m": (
            float(chain_arc[-1]) if len(chain_arc) else 0.0
        ),
        "clamped_sample_count": int(
            np.count_nonzero(
                np.abs(used_arc - safety_arc) > 1e-12
            )
        ),
        "representation_error_mean_mm": (
            float(np.mean(representation_error) * 1000.0)
            if len(representation_error)
            else 0.0
        ),
        "representation_error_p95_mm": (
            float(np.percentile(representation_error, 95) * 1000.0)
            if len(representation_error)
            else 0.0
        ),
        "representation_error_max_mm": (
            float(np.max(representation_error) * 1000.0)
            if len(representation_error)
            else 0.0
        ),
        "safety_points_m": safety_points.tolist(),
        "represented_points_m": samples.tolist(),
        "sample_arclength_m": safety_arc.tolist(),
        "represented_arclength_m": used_arc.tolist(),
        "representation_error_mm": (
            representation_error * 1000.0
        ).tolist(),
    }

    return samples, i0, i1, w0, w1, kinds


def install_safety_aligned_sampling(
    controller,
    env,
    *,
    max_constraints: int = 256,
    min_active_separation_m: float = 1e-6,
) -> SafetyAlignedSamplingConfig:
    """Bind production safety arclength sampling into one diagnostic controller."""

    max_constraints = int(max_constraints)
    min_active_separation_m = float(min_active_separation_m)
    if max_constraints < 1:
        raise ValueError("max_constraints must be >= 1")
    if min_active_separation_m <= 0.0:
        raise ValueError("min_active_separation_m must be positive")

    controller._diagnostic_safety_aligned_env = env
    controller._diagnostic_original_build_samples = controller._build_samples
    controller._build_samples = MethodType(
        _build_safety_aligned_samples,
        controller,
    )
    controller.max_constraints = max_constraints
    controller.min_sample_separation_m = min_active_separation_m
    controller._diagnostic_safety_aligned_config = SafetyAlignedSamplingConfig(
        max_constraints=max_constraints,
        min_active_separation_m=min_active_separation_m,
    )
    controller._diagnostic_safety_aligned_snapshot = None
    return controller._diagnostic_safety_aligned_config


def safety_aligned_snapshot(controller) -> dict:
    config = getattr(
        controller,
        "_diagnostic_safety_aligned_config",
        None,
    )
    snapshot = getattr(
        controller,
        "_diagnostic_safety_aligned_snapshot",
        None,
    )
    return {
        "installed": config is not None,
        "config": config.to_dict() if config is not None else None,
        "latest": snapshot,
    }
