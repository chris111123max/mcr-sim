"""Saved-centerline adapter with stEVE BruteForceBFS path semantics."""
from dataclasses import dataclass
from heapq import heappop, heappush
import random
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

from .vessel_bundle import BranchData, VesselBundle


@dataclass(frozen=True)
class Projection:
    branch: str
    segment_index: int
    fraction: float
    point_mm: np.ndarray
    distance_mm: float


@dataclass(frozen=True)
class PathResult:
    length_mm: float
    points_mm: np.ndarray
    start_projection: Projection
    target_projection: Projection


@dataclass(frozen=True)
class TargetPoint:
    branch: str
    coordinates_mm: np.ndarray

    @property
    def coordinates_m(self):
        return self.coordinates_mm * 0.001


class StEVEBruteForceBFSAdapter:
    """Weighted centerline graph equivalent of official branch-level BFS.

    Official stEVE enumerates branch-junction paths and compares their physical
    lengths.  Here the saved VTK polylines are merged at exact attachment points
    and Dijkstra is applied to the same positive centerline arc lengths.
    """

    def __init__(self, bundle: VesselBundle, merge_tolerance_mm: float = 1e-5):
        self.bundle = bundle
        self.merge_tolerance_mm = float(merge_tolerance_mm)
        self.node_points_mm: List[np.ndarray] = []
        self.branch_nodes: Dict[str, np.ndarray] = {}
        self.adjacency: Dict[int, Dict[int, float]] = {}
        self._build_graph()

    def _find_or_add_node(self, point: np.ndarray) -> int:
        point = np.asarray(point, dtype=float)
        for index, existing in enumerate(self.node_points_mm):
            if np.linalg.norm(existing - point) <= self.merge_tolerance_mm:
                return index
        index = len(self.node_points_mm)
        self.node_points_mm.append(point.copy())
        self.adjacency[index] = {}
        return index

    def _add_edge(self, first: int, second: int, length: float):
        previous = self.adjacency[first].get(second, float("inf"))
        self.adjacency[first][second] = min(previous, float(length))
        previous = self.adjacency[second].get(first, float("inf"))
        self.adjacency[second][first] = min(previous, float(length))

    def _build_graph(self):
        for name, branch in self.bundle.branches.items():
            ids = np.asarray([self._find_or_add_node(point) for point in branch.coordinates_mm])
            self.branch_nodes[name] = ids
            for first, second, p0, p1 in zip(ids[:-1], ids[1:], branch.coordinates_mm[:-1], branch.coordinates_mm[1:]):
                self._add_edge(int(first), int(second), np.linalg.norm(p1 - p0))
        for child, parent in self.bundle.parents.items():
            child_id = int(self.branch_nodes[child][0])
            parent_points = self.bundle.branches[parent].coordinates_mm
            nearest = int(np.argmin(np.linalg.norm(parent_points - self.node_points_mm[child_id], axis=1)))
            parent_id = int(self.branch_nodes[parent][nearest])
            gap = float(np.linalg.norm(self.node_points_mm[child_id] - self.node_points_mm[parent_id]))
            if gap > self.merge_tolerance_mm:
                raise ValueError(f"Disconnected branch {child}->{parent}: {gap} mm")
            if child_id != parent_id:
                self._add_edge(child_id, parent_id, gap)

    @staticmethod
    def _project_to_branch(point_mm: np.ndarray, branch: BranchData) -> Projection:
        point = np.asarray(point_mm, dtype=float).reshape(3)
        best = None
        for index, (p0, p1) in enumerate(zip(branch.coordinates_mm[:-1], branch.coordinates_mm[1:])):
            delta = p1 - p0
            denom = float(np.dot(delta, delta))
            fraction = 0.0 if denom <= 1e-20 else float(np.clip(np.dot(point - p0, delta) / denom, 0.0, 1.0))
            projected = p0 + fraction * delta
            distance = float(np.linalg.norm(point - projected))
            if best is None or distance < best.distance_mm:
                best = Projection(branch.name, index, fraction, projected, distance)
        return best

    def project(self, point_mm: np.ndarray) -> Projection:
        projections = [self._project_to_branch(point_mm, branch) for branch in self.bundle.branches.values()]
        return min(projections, key=lambda item: item.distance_mm)

    def _with_virtual(self, projection: Projection, virtual_id: int, adjacency):
        branch = self.bundle.branches[projection.branch]
        ids = self.branch_nodes[projection.branch]
        index = projection.segment_index
        segment_length = float(np.linalg.norm(branch.coordinates_mm[index + 1] - branch.coordinates_mm[index]))
        adjacency[virtual_id] = {
            int(ids[index]): projection.fraction * segment_length,
            int(ids[index + 1]): (1.0 - projection.fraction) * segment_length,
        }
        for node, weight in adjacency[virtual_id].items():
            adjacency[node][virtual_id] = weight

    @staticmethod
    def _dijkstra(adjacency, start, target):
        queue = [(0.0, start)]
        distance = {start: 0.0}
        parent = {start: None}
        while queue:
            current_distance, node = heappop(queue)
            if current_distance != distance[node]:
                continue
            if node == target:
                break
            for neighbor, weight in adjacency[node].items():
                candidate = current_distance + weight
                if candidate < distance.get(neighbor, float("inf")):
                    distance[neighbor] = candidate
                    parent[neighbor] = node
                    heappush(queue, (candidate, neighbor))
        if target not in distance:
            raise RuntimeError("No connected centerline path")
        nodes = []
        node = target
        while node is not None:
            nodes.append(node); node = parent[node]
        return distance[target], nodes[::-1]

    def shortest_path_mm(self, start_mm: np.ndarray, target_mm: np.ndarray) -> PathResult:
        start_projection = self.project(start_mm)
        target_projection = self.project(target_mm)
        adjacency = {node: dict(neighbors) for node, neighbors in self.adjacency.items()}
        start_id, target_id = -1, -2
        self._with_virtual(start_projection, start_id, adjacency)
        self._with_virtual(target_projection, target_id, adjacency)
        if (
            start_projection.branch == target_projection.branch
            and start_projection.segment_index == target_projection.segment_index
        ):
            branch = self.bundle.branches[start_projection.branch]
            index = start_projection.segment_index
            segment_length = float(np.linalg.norm(branch.coordinates_mm[index + 1] - branch.coordinates_mm[index]))
            direct = abs(start_projection.fraction - target_projection.fraction) * segment_length
            adjacency[start_id][target_id] = direct
            adjacency[target_id][start_id] = direct
        length, nodes = self._dijkstra(adjacency, start_id, target_id)
        points = []
        for node in nodes:
            if node == start_id:
                points.append(start_projection.point_mm)
            elif node == target_id:
                points.append(target_projection.point_mm)
            else:
                points.append(self.node_points_mm[node])
        return PathResult(
            float(length), np.asarray(points, dtype=np.float64),
            start_projection, target_projection,
        )

    def path_length_mm(self, start_mm: np.ndarray, target_mm: np.ndarray) -> float:
        return self.shortest_path_mm(start_mm, target_mm).length_mm

    def is_at_vessel_end(self, point_mm: np.ndarray, threshold_mm: float = 1.0) -> bool:
        child_branches = set(self.bundle.parents.values())
        terminal_names = [name for name in self.bundle.branches if name not in child_branches and name != "aorta"]
        terminals = np.vstack([self.bundle.branches[name].coordinates_mm[-1] for name in terminal_names])
        return bool(np.min(np.linalg.norm(terminals - np.asarray(point_mm), axis=1)) < threshold_mm)


def _point_inside_branch(point_mm: np.ndarray, branch: BranchData) -> bool:
    projection = StEVEBruteForceBFSAdapter._project_to_branch(point_mm, branch)
    index = projection.segment_index
    radius = (1.0 - projection.fraction) * branch.radii_mm[index] + projection.fraction * branch.radii_mm[index + 1]
    return bool(projection.distance_mm <= float(radius))


class CenterlineRandomAdapter:
    """Seeded target sampler matching stEVE CenterlineRandom point-cloud semantics."""

    def __init__(self, bundle: VesselBundle, branches: Optional[Sequence[str]] = None):
        self.bundle = bundle
        self.branches = tuple(branches) if branches is not None else tuple(bundle.branches)
        unknown = set(self.branches) - set(bundle.branches)
        if unknown:
            raise ValueError(f"Target branches absent from vessel: {sorted(unknown)}")
        excluded = tuple(name for name in bundle.branches if name not in self.branches)
        candidates: List[TargetPoint] = []
        for name in sorted(self.branches):
            for point in bundle.branches[name].coordinates_mm:
                if any(_point_inside_branch(point, bundle.branches[other]) for other in excluded):
                    continue
                candidates.append(TargetPoint(name, np.asarray(point, dtype=np.float64).copy()))
        if not candidates:
            raise ValueError("No potential centerline targets after branch filtering")
        self.potential_targets = tuple(candidates)
        self._rng = random.Random()

    def reset(self, seed: Optional[int] = None) -> TargetPoint:
        if seed is not None:
            self._rng = random.Random(int(seed))
        return self._rng.choice(self.potential_targets)
