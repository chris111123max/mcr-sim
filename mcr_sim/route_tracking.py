"""Continuous selected-route tracking without discrete waypoint gates.

The tracker deliberately separates initial localization from recurrent
tracking.  Initial localization may inspect the complete selected route.  Once
an episode has started, projection is restricted by previous arc length and by
the maximum physically plausible progress change.  This prevents spatially
nearby arms of a tight U-turn (or nearby branches) from creating artificial
forward progress.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class RouteProjection:
    progress: float
    segment_index: int
    distance: float
    point: np.ndarray
    tangent: np.ndarray
    jump_rejected: bool = False


def _validate_route(points: np.ndarray, cumulative: np.ndarray) -> tuple:
    points = np.asarray(points, dtype=np.float64)
    cumulative = np.asarray(cumulative, dtype=np.float64).reshape(-1)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or len(points) < 2
        or len(cumulative) != len(points)
        or not np.all(np.isfinite(points))
        or not np.all(np.isfinite(cumulative))
        or np.any(np.diff(cumulative) <= 0.0)
    ):
        raise ValueError("route points/cumulative arc length are invalid")
    return points, cumulative


def project_to_route(
    points: np.ndarray,
    cumulative: np.ndarray,
    point: np.ndarray,
    *,
    previous_progress: Optional[float] = None,
    backward_window: float = 0.020,
    forward_window: float = 0.040,
    ambiguity_tolerance: float = 0.00075,
    max_progress_step: float = 0.002,
) -> RouteProjection:
    """Project ``point`` onto a continuous selected route.

    When ``previous_progress`` is supplied, only a local arc-length window is
    considered.  Near-equal Euclidean candidates prefer continuity.  A final
    physical step gate rejects a remote candidate rather than clipping toward
    it; clipping would allow a wrong projection to creep across a U-turn over
    several environment steps.
    """

    points, cumulative = _validate_route(points, cumulative)
    query = np.asarray(point, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(query)):
        raise ValueError("projection point is non-finite")

    seg_start = points[:-1]
    seg_vec = points[1:] - seg_start
    seg_len = np.linalg.norm(seg_vec, axis=1)
    valid_segments = seg_len > 1e-12
    if not np.any(valid_segments):
        raise ValueError("route has no non-degenerate segment")

    candidate_mask = valid_segments.copy()
    previous = None
    if previous_progress is not None and np.isfinite(float(previous_progress)):
        previous = float(
            np.clip(float(previous_progress), cumulative[0], cumulative[-1])
        )
        low = previous - max(float(backward_window), 0.0)
        high = previous + max(float(forward_window), 0.0)
        candidate_mask &= (cumulative[:-1] <= high) & (cumulative[1:] >= low)
        if not np.any(candidate_mask):
            candidate_mask = valid_segments.copy()

    candidate_indices = np.flatnonzero(candidate_mask)
    starts = seg_start[candidate_indices]
    vectors = seg_vec[candidate_indices]
    lengths = seg_len[candidate_indices]
    length_sq = lengths * lengths
    fractions = np.clip(
        np.sum((query[None, :] - starts) * vectors, axis=1) / length_sq,
        0.0,
        1.0,
    )
    projections = starts + fractions[:, None] * vectors
    distances = np.linalg.norm(projections - query[None, :], axis=1)
    progresses = cumulative[candidate_indices] + fractions * lengths

    min_distance = float(np.min(distances))
    ambiguous = distances <= min_distance + max(float(ambiguity_tolerance), 0.0)
    ambiguous_indices = np.flatnonzero(ambiguous)
    if previous is None:
        local_choice = int(np.argmin(distances))
    else:
        gaps = np.abs(progresses[ambiguous_indices] - previous)
        order = np.lexsort((distances[ambiguous_indices], gaps))
        local_choice = int(ambiguous_indices[int(order[0])])

    jump_rejected = False
    if previous is not None:
        maximum_step = max(float(max_progress_step), 1e-9)
        physically_reachable = np.abs(progresses - previous) <= maximum_step + 1e-12
        if not bool(physically_reachable[local_choice]):
            jump_rejected = True
            reachable_indices = np.flatnonzero(physically_reachable)
            if reachable_indices.size > 0:
                local_choice = int(
                    reachable_indices[
                        int(np.argmin(distances[reachable_indices]))
                    ]
                )
            else:
                # Keep the previous continuous state if no projected segment is
                # physically reachable.  Interpolating the route is preferable
                # to accepting a false arm/branch jump.
                progress = previous
                idx = int(
                    np.clip(
                        np.searchsorted(cumulative, progress, side="right") - 1,
                        0,
                        len(points) - 2,
                    )
                )
                local_len = float(cumulative[idx + 1] - cumulative[idx])
                fraction = (progress - float(cumulative[idx])) / max(local_len, 1e-12)
                projection = points[idx] + fraction * (points[idx + 1] - points[idx])
                tangent = points[idx + 1] - points[idx]
                tangent /= max(float(np.linalg.norm(tangent)), 1e-12)
                return RouteProjection(
                    progress=float(progress),
                    segment_index=idx,
                    distance=float(np.linalg.norm(projection - query)),
                    point=projection.astype(np.float32),
                    tangent=tangent.astype(np.float32),
                    jump_rejected=True,
                )

    idx = int(candidate_indices[local_choice])
    tangent = vectors[local_choice] / max(float(lengths[local_choice]), 1e-12)
    return RouteProjection(
        progress=float(progresses[local_choice]),
        segment_index=idx,
        distance=float(distances[local_choice]),
        point=projections[local_choice].astype(np.float32),
        tangent=tangent.astype(np.float32),
        jump_rejected=jump_rejected,
    )


def normalized_route_progress(
    progress: float,
    start_progress: float,
    target_progress: float,
) -> float:
    """Return bounded continuous completion on the selected route."""

    start = float(start_progress)
    target = float(target_progress)
    current = float(progress)
    if not all(np.isfinite(value) for value in (start, target, current)):
        return 0.0
    length = target - start
    if length <= 1e-9:
        return 0.0
    return float(np.clip((current - start) / length, 0.0, 1.0))
