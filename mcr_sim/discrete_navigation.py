"""Episode-local ordered navigation references. All distances are metres."""
import numpy as np


def compile_points(route, branch_points=(), start=0.0, end=None):
    route = np.asarray(route, dtype=float).reshape(-1, 3)
    if len(route) < 2 or not np.isfinite(route).all():
        raise ValueError("Discrete navigation requires a finite matched route")
    keep = np.r_[True, np.linalg.norm(np.diff(route, axis=0), axis=1) > 1e-10]
    route = route[keep]
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))]
    end = arc[-1] if end is None else min(float(end), arc[-1])
    start = max(float(start), 0.)
    if end <= start:
        raise ValueError("Target must follow start on its matched route")
    def point(s):
        return np.array([np.interp(s, arc, route[:, j]) for j in range(3)])
    # Offline bend classification: direction change over a 4 mm window >=10deg.
    before = np.array([point(max(0., s-.002)) for s in arc])
    after = np.array([point(min(arc[-1], s+.002)) for s in arc])
    a, b = route-before, after-route
    denom = np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1)
    angles = np.arccos(np.clip(np.sum(a*b, axis=1)/np.maximum(denom, 1e-15), -1, 1))
    bends = arc[(denom > 1e-14) & (angles >= np.deg2rad(10.))]
    branches = np.asarray(branch_points, dtype=float).reshape(-1, 3)
    if len(branches):
        close = np.min(np.linalg.norm(route[:, None]-branches[None], axis=2), axis=1) <= .002
        bends = np.r_[bends, arc[close]]
    points, lengths, radii, kinds = [], [], [], []
    s = start
    while s < end-1e-10:
        # Look ahead across the proposed interval so 4mm steps cannot skip a bend.
        tight = bool(np.any((bends >= s-.004) & (bends <= s+.008)))
        s = min(s + (.002 if tight else .004), end)
        points.append(point(s)); lengths.append(s)
        radii.append(.0008 if tight else .0012); kinds.append("bend_branch" if tight else "gentle")
    radii[-1] = .003
    kinds[-1] = "final"
    return np.asarray(points), np.asarray(lengths), np.asarray(radii), kinds


class PointTracker:
    def __init__(self, points, arc, radii, kinds, tip, initial_index=0):
        self.points, self.arc, self.radii, self.kinds = points, arc, radii, kinds
        # The first few route samples are optional initialization samples.  They
        # can lie behind the physical tip after reset and should not force the
        # policy to turn back before it starts navigating.
        self.initial_index = int(np.clip(initial_index, 0, max(len(points) - 1, 0)))
        self.index = self.initial_index
        self.previous_distance = float(np.linalg.norm(tip-points[self.index]))
        self.last_delta = 0.
        self.switches = 0

    def update(self, tip, inside):
        distance = float(np.linalg.norm(tip-self.points[self.index]))
        self.last_delta = self.previous_distance-distance
        self.previous_distance = distance
        self.switches = 0
        # Never choose a nearest point; never advance through an unsafe transition.
        while inside and self.index < len(self.points)-1 and distance <= self.radii[self.index]:
            self.index += 1; self.switches += 1
            distance = float(np.linalg.norm(tip-self.points[self.index]))
        self.previous_distance = distance
        return self.last_delta

    def window(self):
        return self.points[np.minimum(self.index+np.arange(5), len(self.points)-1)]

    def preview(self, distances):
        """Interpolate route references at fixed arc-length lookaheads."""

        query = np.minimum(
            float(self.arc[self.index]) + np.asarray(distances, dtype=float),
            float(self.arc[-1]),
        )
        return np.stack(
            [np.interp(query, self.arc, self.points[:, axis]) for axis in range(3)],
            axis=1,
        )
