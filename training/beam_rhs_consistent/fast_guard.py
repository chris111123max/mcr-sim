"""Conservative coarse-to-fine Beam/SDF free-motion gate (candidate only).

The 100-um spatial step is bounded using each Bezier segment's maximum
derivative, not its rest length alone.  A coarse scan can ONLY disable the
expensive 10-um free-state scan when a conservative distance lower bound
is comfortably above the candidate margin.  Otherwise the unchanged
production fine planner builds the exact 10-um Beam rows.

A trilinearly interpolated grid is not a mathematically exact signed-distance
field.  The 2x Lipschitz allowance and 50-um extra cushion are deliberately
conservative training heuristics, NOT a certified nonpenetration proof.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

# Upper allowance for potential interpolated-SDF variation between samples.
SDF_SLOPE_ALLOWANCE = 2.0
EXTRA_ERROR_CUSHION_M = 5.0e-5


def coarse_beam_clearance_guard(adapter, q_free, *, spacing_m=1.0e-4,
                                switch_margin_m=2.5e-4):
    """Return clearance-based fast-skip decision and auditable uncertainty bound.

    Only uses the same active topology, cubic Bezier control points, linear
    fallback criterion and signed clearance convention as LiveBeamSDFAdapter.
    No Jacobians or SDF gradients are required.
    """
    q = np.asarray(q_free, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 7 or not np.isfinite(q).all():
        raise RuntimeError("COARSE_GATE_INVALID_BEAM_STATE")
    spacing = float(spacing_m)
    switch_margin = float(switch_margin_m)
    if not np.isfinite(spacing) or spacing <= 0 or not np.isfinite(switch_margin) or switch_margin <= 0:
        raise ValueError("Invalid coarse clearance gate spacing or margin")
    elements = adapter._geometry()["active_elements"]
    if not elements:
        raise RuntimeError("COARSE_GATE_EMPTY_ACTIVE_GEOMETRY")

    chunks = []
    max_gap_bound = 0.0
    for idx, spec in enumerate(elements):
        a, b = q[np.asarray(spec["nodes"], dtype=int)]
        length = float(spec["rest_length_m"])
        p0, p3 = a[:3], b[:3]
        p1 = p0 + Rotation.from_quat(a[3:7]).apply([length / 3.0, 0.0, 0.0])
        p2 = p3 + Rotation.from_quat(b[3:7]).apply([-length / 3.0, 0.0, 0.0])
        chord = float(np.linalg.norm(p3 - p0))
        linear = bool(np.dot(p1 - p0, p2 - p1) < 0 and chord < 0.4 * length)
        if linear:
            derivative_bound = chord
        else:
            derivative_bound = 3.0 * max(
                float(np.linalg.norm(p1 - p0)),
                float(np.linalg.norm(p2 - p1)),
                float(np.linalg.norm(p3 - p2)),
            )
        # Bezier derivative is a convex combination of the three control
        # polygon differences.  Hence each arc between adjacent t samples is
        # <= derivative_bound / n and no more than the configured spacing.
        n = max(1, int(np.ceil(max(length, derivative_bound) / spacing)))
        max_gap_bound = max(max_gap_bound, derivative_bound / n)
        t = np.linspace(0.0, 1.0, n + 1)
        u = 1.0 - t
        if linear:
            points = u[:, None] * p0 + t[:, None] * p3
        else:
            points = (
                (u ** 3)[:, None] * p0
                + (3 * u * u * t)[:, None] * p1
                + (3 * u * t * t)[:, None] * p2
                + (t ** 3)[:, None] * p3
            )
        if idx:
            points = points[1:]
        chunks.append(points)

    points = np.concatenate(chunks, axis=0)
    clearance = np.asarray(adapter.query(points), dtype=np.float64).reshape(-1)
    if len(clearance) != len(points):
        raise RuntimeError("COARSE_GATE_SDF_SIZE_MISMATCH")
    valid = np.isfinite(clearance)
    valid_count = int(valid.sum())
    if valid_count:
        checked = np.where(valid, clearance, np.inf)
        worst_index = int(np.argmin(checked))
        minimum = float(checked[worst_index])
    else:
        worst_index, minimum = -1, None
    invalid = not bool(valid.all())
    # Every continuous Bezier point is at most half one inter-sample arc away
    # from one sample.  The additional conservative cushion compensates for
    # SDF discretization and interpolation not being an exact distance field.
    unobserved_gap_m = SDF_SLOPE_ALLOWANCE * max_gap_bound / 2.0 + EXTRA_ERROR_CUSHION_M
    lower_bound = None if invalid else float(minimum - unobserved_gap_m)
    return {
        "safe": bool(lower_bound is not None and lower_bound >= switch_margin),
        "invalid_sdf": bool(invalid),
        "minimum_clearance_m": minimum,
        "lower_bound_m": lower_bound,
        "worst_index": worst_index,
        "sample_count": int(len(points)),
        "max_curve_gap_bound_m": float(max_gap_bound),
        "interpolation_allowance_m": float(unobserved_gap_m),
        "coarse_spacing_m": spacing,
        "switch_margin_m": switch_margin,
    }
