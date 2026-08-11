#!/usr/bin/env python3
"""Generate deterministic artificial vessels for mCR SOFA training.

The source assets use millimetres.  SOFA scenes must load these models with a
0.001 scale so geometry and centerline radii are converted to metres together.

Each model directory contains:
  collision_inner.stl      collision-only luminal surface
  Segmentation.stl         compatibility alias used by the current scene
  visual_wall.stl          smoother/larger visual-only surface
  Centerline model.vtk     default single root-to-target centerline
  centerline.vtk           compatibility alias of the default centerline
  centerline_graph.vtk     complete graph (branching models only)
  target_XX_centerline.vtk one root-to-outlet path (branching models only)
  vessel_sdf.vti           signed distance field in millimetres
  metadata.json            generation parameters and validation statistics
  preview.png              deterministic projected preview

Only NumPy and the Python standard library are required.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import shutil
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


MM_PER_M = 1000.0
SOURCE_UNITS = "mm"
SOFA_SCALE = 0.001
CATHETER_OUTER_DIAMETER_MM = 1.33
CATHETER_RADIUS_MM = CATHETER_OUTER_DIAMETER_MM / 2.0


@dataclass(frozen=True)
class CurvedSpec:
    model_id: str
    seed: int
    length_mm: float
    turns: float
    amplitude_mm: float
    torsion_mm: float
    radius_min_mm: float
    radius_max_mm: float
    phase: float
    difficulty: str


@dataclass(frozen=True)
class BranchSpec:
    model_id: str
    seed: int
    x_scale: float
    z_scale_mm: float
    curve_mm: float
    trunk_radius_mm: float
    outlet_radius_mm: float
    angle_range_deg: Tuple[float, float]
    difficulty: str


CURVED_SPECS: Tuple[CurvedSpec, ...] = (
    CurvedSpec("C01", 101, 260.0, 1.65, 23.0, 7.0, 4.5, 4.5, 0.20, "low"),
    CurvedSpec("C02", 102, 285.0, 2.05, 29.0, 12.0, 4.0, 4.0, 0.65, "medium-low"),
    CurvedSpec("C03", 103, 310.0, 2.45, 34.0, 19.0, 4.0, 4.0, 1.05, "medium"),
    CurvedSpec("C04", 104, 300.0, 2.15, 38.0, 25.0, 3.5, 4.5, 1.45, "medium-high"),
    CurvedSpec("C05", 105, 330.0, 2.70, 41.0, 31.0, 3.2, 4.5, 2.00, "high"),
)


BRANCH_SPECS: Tuple[BranchSpec, ...] = (
    BranchSpec("B01", 201, 0.80, 5.0, 2.0, 5.0, 3.4, (45.0, 65.0), "low"),
    BranchSpec("B02", 202, 1.00, 11.0, 4.0, 5.0, 3.2, (35.0, 80.0), "medium-low"),
    BranchSpec("B03", 203, 0.90, 17.0, 6.0, 5.1, 3.0, (25.0, 55.0), "medium"),
    BranchSpec("B04", 204, 1.28, 22.0, 7.0, 5.2, 2.9, (65.0, 110.0), "medium-high"),
    BranchSpec("B05", 205, 1.12, 31.0, 10.0, 5.2, 2.8, (30.0, 105.0), "high"),
)


def _normalize(v: np.ndarray, fallback: Sequence[float] = (1.0, 0.0, 0.0)) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.asarray(fallback, dtype=np.float64)
    return v / n


def _rodrigues(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    axis = _normalize(axis)
    c = math.cos(float(angle))
    s = math.sin(float(angle))
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1.0 - c)


def parallel_transport_frames(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    tangents = np.empty_like(points)
    tangents[0] = _normalize(points[1] - points[0], (0.0, 1.0, 0.0))
    tangents[-1] = _normalize(points[-1] - points[-2], (0.0, 1.0, 0.0))
    tangents[1:-1] = points[2:] - points[:-2]
    tangents = np.asarray([_normalize(t, (0.0, 1.0, 0.0)) for t in tangents])

    normals = np.empty_like(points)
    binormals = np.empty_like(points)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(ref, tangents[0]))) > 0.90:
        ref = np.array([1.0, 0.0, 0.0])
    normals[0] = _normalize(np.cross(ref, tangents[0]), (1.0, 0.0, 0.0))
    binormals[0] = _normalize(np.cross(tangents[0], normals[0]), (0.0, 0.0, 1.0))

    for i in range(1, len(points)):
        prev_t = tangents[i - 1]
        cur_t = tangents[i]
        axis = np.cross(prev_t, cur_t)
        axis_norm = float(np.linalg.norm(axis))
        normal = normals[i - 1]
        if axis_norm > 1e-10:
            axis /= axis_norm
            angle = math.atan2(axis_norm, float(np.clip(np.dot(prev_t, cur_t), -1.0, 1.0)))
            normal = _rodrigues(normal, axis, angle)
        normal = normal - cur_t * float(np.dot(normal, cur_t))
        normals[i] = _normalize(normal, normals[i - 1])
        binormals[i] = _normalize(np.cross(cur_t, normals[i]), binormals[i - 1])
    return tangents, normals, binormals


def sweep_tube(points: np.ndarray, radii: np.ndarray, sides: int = 28) -> np.ndarray:
    """Return outward-oriented open tube triangles."""
    points = np.asarray(points, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    _, normals, binormals = parallel_transport_frames(points)
    angles = np.linspace(0.0, 2.0 * math.pi, int(sides), endpoint=False)
    rings = []
    for p, r, n, b in zip(points, radii, normals, binormals):
        ring = p + r * (np.cos(angles)[:, None] * n + np.sin(angles)[:, None] * b)
        rings.append(ring)
    rings = np.asarray(rings)

    triangles: List[np.ndarray] = []
    for i in range(len(points) - 1):
        for j in range(sides):
            jn = (j + 1) % sides
            quad = (rings[i, j], rings[i + 1, j], rings[i + 1, jn], rings[i, jn])
            for tri in ((quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3])):
                tri_arr = np.asarray(tri, dtype=np.float64)
                centroid = tri_arr.mean(axis=0)
                radial = centroid - 0.5 * (points[i] + points[i + 1])
                normal = np.cross(tri_arr[1] - tri_arr[0], tri_arr[2] - tri_arr[0])
                if float(np.dot(normal, radial)) < 0.0:
                    tri_arr[[1, 2]] = tri_arr[[2, 1]]
                triangles.append(tri_arr)
    return np.asarray(triangles, dtype=np.float32)


def curved_centerline(spec: CurvedSpec, samples: int = 181) -> Tuple[np.ndarray, np.ndarray]:
    u = np.linspace(0.0, 1.0, int(samples), dtype=np.float64)
    collar_fraction = 0.11
    core_u = np.clip((u - collar_fraction) / (1.0 - 2.0 * collar_fraction), 0.0, 1.0)
    active = (u >= collar_fraction) & (u <= 1.0 - collar_fraction)
    envelope = np.sin(math.pi * core_u) ** 2

    x = np.zeros_like(u)
    z = np.zeros_like(u)
    x[active] = (
        spec.amplitude_mm
        * np.sin(2.0 * math.pi * spec.turns * core_u[active] + spec.phase)
        * envelope[active]
    )
    z[active] = (
        spec.torsion_mm
        * np.sin(2.0 * math.pi * (spec.turns * 0.73 + 0.37) * core_u[active] + 0.5 * spec.phase)
        * envelope[active]
    )
    y = spec.length_mm * u
    points = np.column_stack((x, y, z))

    if abs(spec.radius_max_mm - spec.radius_min_mm) < 1e-9:
        radii = np.full_like(u, spec.radius_min_mm)
    else:
        wave = 0.5 + 0.5 * np.sin(2.0 * math.pi * 1.35 * u + spec.phase)
        radii = spec.radius_min_mm + (spec.radius_max_mm - spec.radius_min_mm) * wave
        edge_blend = np.sin(math.pi * u) ** 2
        radii = radii * edge_blend + spec.radius_max_mm * (1.0 - edge_blend)
    return points, radii


def _branch_base_nodes(spec: BranchSpec) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    rng = np.random.default_rng(spec.seed)
    # One root, three branching levels, exactly six terminal outlets.
    raw = {
        "root": (0.0, 0.0, 0.0),
        "j0": (0.0, 46.0, 0.0),
        "l1": (-24.0, 91.0, -0.20),
        "r1": (24.0, 94.0, 0.20),
        "lo1": (-58.0, 150.0, -0.32),
        "l2": (-9.0, 143.0, 0.18),
        "lo2": (-30.0, 207.0, -0.12),
        "lo3": (18.0, 205.0, 0.34),
        "ro1": (59.0, 153.0, 0.29),
        "r2": (10.0, 146.0, -0.19),
        "ro2": (-12.0, 211.0, -0.38),
        "ro3": (39.0, 208.0, 0.10),
    }
    nodes: Dict[str, np.ndarray] = {}
    for name, (x, y, z_unit) in raw.items():
        jitter = np.zeros(3)
        if name not in ("root", "j0"):
            jitter = rng.normal(0.0, (1.6, 2.3, 0.7), size=3)
        nodes[name] = np.array(
            [x * spec.x_scale + jitter[0], y + jitter[1], z_unit * spec.z_scale_mm + jitter[2]],
            dtype=np.float64,
        )

    radii = {
        "root": spec.trunk_radius_mm,
        "j0": spec.trunk_radius_mm,
        "l1": 0.88 * spec.trunk_radius_mm,
        "r1": 0.88 * spec.trunk_radius_mm,
        "l2": 0.75 * spec.trunk_radius_mm,
        "r2": 0.75 * spec.trunk_radius_mm,
        "lo1": 1.08 * spec.outlet_radius_mm,
        "ro1": 1.08 * spec.outlet_radius_mm,
        "lo2": spec.outlet_radius_mm,
        "lo3": spec.outlet_radius_mm,
        "ro2": spec.outlet_radius_mm,
        "ro3": spec.outlet_radius_mm,
    }
    return nodes, radii


BRANCH_EDGES: Tuple[Tuple[str, str], ...] = (
    ("root", "j0"),
    ("j0", "l1"),
    ("j0", "r1"),
    ("l1", "lo1"),
    ("l1", "l2"),
    ("l2", "lo2"),
    ("l2", "lo3"),
    ("r1", "ro1"),
    ("r1", "r2"),
    ("r2", "ro2"),
    ("r2", "ro3"),
)

BRANCH_OUTLETS: Tuple[str, ...] = ("lo1", "lo2", "lo3", "ro1", "ro2", "ro3")


def _bezier_edge(
    p0: np.ndarray,
    p1: np.ndarray,
    radius0: float,
    radius1: float,
    edge_index: int,
    spec: BranchSpec,
    samples: int = 15,
) -> Tuple[np.ndarray, np.ndarray]:
    d = p1 - p0
    tangent = _normalize(d, (0.0, 1.0, 0.0))
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(ref, tangent))) > 0.92:
        ref = np.array([1.0, 0.0, 0.0])
    lateral = _normalize(np.cross(tangent, ref), (1.0, 0.0, 0.0))
    vertical = _normalize(np.cross(tangent, lateral), (0.0, 0.0, 1.0))
    sign = -1.0 if edge_index % 2 else 1.0
    offset = sign * spec.curve_mm * lateral + 0.35 * spec.curve_mm * vertical
    c1 = p0 + 0.33 * d + offset
    c2 = p0 + 0.68 * d - 0.35 * offset
    t = np.linspace(0.0, 1.0, int(samples), dtype=np.float64)
    omt = 1.0 - t
    points = (
        omt[:, None] ** 3 * p0
        + 3.0 * omt[:, None] ** 2 * t[:, None] * c1
        + 3.0 * omt[:, None] * t[:, None] ** 2 * c2
        + t[:, None] ** 3 * p1
    )
    smooth = t * t * (3.0 - 2.0 * t)
    radii = radius0 + (radius1 - radius0) * smooth
    return points, radii


def branch_polylines(spec: BranchSpec):
    nodes, node_radii = _branch_base_nodes(spec)
    polylines = []
    for edge_index, (a, b) in enumerate(BRANCH_EDGES):
        pts, radii = _bezier_edge(
            nodes[a], nodes[b], node_radii[a], node_radii[b], edge_index, spec
        )
        polylines.append({"start": a, "end": b, "points": pts, "radii": radii})
    return nodes, node_radii, polylines


def polylines_to_segments(polylines) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts, ends, rpair = [], [], []
    for poly in polylines:
        p = np.asarray(poly["points"], dtype=np.float64)
        r = np.asarray(poly["radii"], dtype=np.float64)
        starts.append(p[:-1])
        ends.append(p[1:])
        rpair.append(np.column_stack((r[:-1], r[1:])))
    return np.vstack(starts), np.vstack(ends), np.vstack(rpair)


def path_to_segments(points: np.ndarray, radii: np.ndarray):
    return points[:-1], points[1:], np.column_stack((radii[:-1], radii[1:]))


def compute_sdf_grid(
    seg_a: np.ndarray,
    seg_b: np.ndarray,
    seg_radii: np.ndarray,
    spacing_mm: float,
    margin_mm: float = 7.0,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int]]:
    all_points = np.vstack((seg_a, seg_b))
    max_radius = float(np.max(seg_radii))
    lo = np.floor((all_points.min(axis=0) - max_radius - margin_mm) / spacing_mm) * spacing_mm
    hi = np.ceil((all_points.max(axis=0) + max_radius + margin_mm) / spacing_mm) * spacing_mm
    dims = np.maximum(3, np.rint((hi - lo) / spacing_mm).astype(int) + 1)
    nx, ny, nz = [int(v) for v in dims]
    xs = lo[0] + spacing_mm * np.arange(nx, dtype=np.float32)
    ys = lo[1] + spacing_mm * np.arange(ny, dtype=np.float32)
    zs = lo[2] + spacing_mm * np.arange(nz, dtype=np.float32)

    a = np.asarray(seg_a, dtype=np.float32)
    ab = np.asarray(seg_b - seg_a, dtype=np.float32)
    len2 = np.maximum(np.sum(ab * ab, axis=1), 1e-9)
    r0 = np.asarray(seg_radii[:, 0], dtype=np.float32)
    r1 = np.asarray(seg_radii[:, 1], dtype=np.float32)
    field = np.empty((nz, ny, nx), dtype=np.float32)

    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    xy = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float32)
    segment_batch = 16
    for zi, z in enumerate(zs):
        points = np.empty((len(xy), 3), dtype=np.float32)
        points[:, :2] = xy
        points[:, 2] = z
        dmin = np.full(len(points), np.inf, dtype=np.float32)
        for s0 in range(0, len(a), segment_batch):
            s1 = min(len(a), s0 + segment_batch)
            rel = points[:, None, :] - a[None, s0:s1, :]
            t = np.sum(rel * ab[None, s0:s1, :], axis=2) / len2[None, s0:s1]
            t = np.clip(t, 0.0, 1.0)
            nearest = a[None, s0:s1, :] + t[:, :, None] * ab[None, s0:s1, :]
            distance = np.sqrt(np.sum((points[:, None, :] - nearest) ** 2, axis=2))
            local_radius = r0[None, s0:s1] + t * (r1 - r0)[None, s0:s1]
            dmin = np.minimum(dmin, np.min(distance - local_radius, axis=1))
        field[zi] = dmin.reshape((ny, nx))
    return field, lo.astype(np.float64), (nx, ny, nz)


_CUBE_OFFSETS = np.asarray(
    [
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
    ],
    dtype=np.float64,
)
_TETS = (
    (0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
    (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6),
)


def _interpolate_iso(p0, p1, v0, v1):
    denom = v0 - v1
    denom = np.where(np.abs(denom) < 1e-12, np.sign(denom) * 1e-12 + 1e-12, denom)
    t = np.clip(v0 / denom, 0.0, 1.0)
    return p0 + t[:, None] * (p1 - p0)


def _orient_triangles(triangles: np.ndarray, outward_hint: np.ndarray) -> np.ndarray:
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    flip = np.sum(normals * outward_hint, axis=1) < 0.0
    if np.any(flip):
        triangles[flip, 1], triangles[flip, 2] = (
            triangles[flip, 2].copy(), triangles[flip, 1].copy()
        )
    return triangles


def _march_tetra_batch(coords: np.ndarray, vals: np.ndarray) -> List[np.ndarray]:
    inside = vals < 0.0
    counts = np.sum(inside, axis=1)
    output: List[np.ndarray] = []

    for ii in range(4):
        mask = (counts == 1) & inside[:, ii]
        if np.any(mask):
            outs = [j for j in range(4) if j != ii]
            c, v = coords[mask], vals[mask]
            pts = [_interpolate_iso(c[:, ii], c[:, j], v[:, ii], v[:, j]) for j in outs]
            tri = np.stack(pts, axis=1)
            hint = c[:, outs].mean(axis=1) - c[:, ii]
            output.append(_orient_triangles(tri, hint))

    for oo in range(4):
        mask = (counts == 3) & (~inside[:, oo])
        if np.any(mask):
            ins = [j for j in range(4) if j != oo]
            c, v = coords[mask], vals[mask]
            pts = [_interpolate_iso(c[:, j], c[:, oo], v[:, j], v[:, oo]) for j in ins]
            tri = np.stack(pts, axis=1)
            hint = c[:, oo] - c[:, ins].mean(axis=1)
            output.append(_orient_triangles(tri, hint))

    pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    for i0, i1 in pairs:
        mask = (counts == 2) & inside[:, i0] & inside[:, i1]
        if np.any(mask):
            outs = [j for j in range(4) if j not in (i0, i1)]
            o0, o1 = outs
            c, v = coords[mask], vals[mask]
            a = _interpolate_iso(c[:, i0], c[:, o0], v[:, i0], v[:, o0])
            b = _interpolate_iso(c[:, i0], c[:, o1], v[:, i0], v[:, o1])
            cc = _interpolate_iso(c[:, i1], c[:, o0], v[:, i1], v[:, o0])
            d = _interpolate_iso(c[:, i1], c[:, o1], v[:, i1], v[:, o1])
            hint = c[:, outs].mean(axis=1) - c[:, [i0, i1]].mean(axis=1)
            tri1 = _orient_triangles(np.stack((a, b, cc), axis=1), hint)
            tri2 = _orient_triangles(np.stack((b, d, cc), axis=1), hint)
            output.extend((tri1, tri2))
    return output


def marching_tetrahedra(field: np.ndarray, origin: np.ndarray, spacing_mm: float) -> np.ndarray:
    nz, ny, nx = field.shape
    triangles: List[np.ndarray] = []
    x0, y0 = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="xy")
    base_xy = np.column_stack((x0.ravel(), y0.ravel())).astype(np.float64)

    for z in range(nz - 1):
        base = np.column_stack(
            (base_xy[:, 0], base_xy[:, 1], np.full(len(base_xy), z, dtype=np.float64))
        )
        cube_coords = origin[None, None, :] + spacing_mm * (
            base[:, None, :] + _CUBE_OFFSETS[None, :, :]
        )
        cube_vals = np.column_stack(
            (
                field[z, :-1, :-1].ravel(),
                field[z, :-1, 1:].ravel(),
                field[z, 1:, 1:].ravel(),
                field[z, 1:, :-1].ravel(),
                field[z + 1, :-1, :-1].ravel(),
                field[z + 1, :-1, 1:].ravel(),
                field[z + 1, 1:, 1:].ravel(),
                field[z + 1, 1:, :-1].ravel(),
            )
        )
        active = (np.min(cube_vals, axis=1) <= 0.0) & (np.max(cube_vals, axis=1) >= 0.0)
        if not np.any(active):
            continue
        cube_coords = cube_coords[active]
        cube_vals = cube_vals[active]
        for tet in _TETS:
            triangles.extend(_march_tetra_batch(cube_coords[:, tet, :], cube_vals[:, tet]))
    if not triangles:
        raise RuntimeError("Marching tetrahedra produced no surface triangles.")
    out = np.concatenate(triangles, axis=0)
    # Weld interpolation points that differ only by floating-point noise near
    # an isosurface/grid-vertex coincidence.  A 0.001 mm grid is three orders
    # below the collision spacing and prevents microscopic sliver triangles
    # from reaching SOFA without changing the vessel geometry materially.
    out = np.round(out, decimals=3)
    area2 = np.linalg.norm(np.cross(out[:, 1] - out[:, 0], out[:, 2] - out[:, 0]), axis=1)
    return out[area2 > 2e-8].astype(np.float32)


def remove_terminal_caps(
    triangles: np.ndarray,
    terminal_data: Iterable[Tuple[np.ndarray, np.ndarray, float]],
    spacing_mm: float,
) -> np.ndarray:
    """Open terminals by clipping against their exact cross-section planes.

    The previous implementation deleted whole triangles by centroid inside a
    broad cap region.  At a coarse collision-grid spacing that could remove
    unrelated side-wall triangles and leave dozens of catheter-sized holes.
    Half-space polygon clipping preserves every shared cut edge and therefore
    produces one boundary loop per inlet/outlet.
    """

    def clip_halfspace(
        mesh: np.ndarray,
        endpoint: np.ndarray,
        outward: np.ndarray,
        radius: float,
    ) -> np.ndarray:
        endpoint = np.asarray(endpoint, dtype=np.float64)
        outward = _normalize(outward).astype(np.float64)
        output: List[np.ndarray] = []
        epsilon = max(1e-8, 1e-7 * float(spacing_mm))

        for triangle in np.asarray(mesh, dtype=np.float64):
            centroid_rel = triangle.mean(axis=0) - endpoint
            centroid_axial = float(np.dot(centroid_rel, outward))
            centroid_radial = float(
                np.linalg.norm(centroid_rel - centroid_axial * outward)
            )
            # A terminal clipping plane is infinite, while a vessel tree has
            # several outlets in different directions. Restrict clipping to a
            # short cylinder around this terminal so the plane cannot slice
            # neighbouring branches elsewhere in the tree.
            local_terminal_region = (
                centroid_axial > -2.5 * float(spacing_mm)
                and centroid_axial < float(radius) + 3.0 * float(spacing_mm)
                and centroid_radial < float(radius) + 2.0 * float(spacing_mm)
            )
            if not local_terminal_region:
                output.append(triangle)
                continue

            polygon = [triangle[0], triangle[1], triangle[2]]
            clipped: List[np.ndarray] = []
            previous = polygon[-1]
            previous_distance = float(np.dot(previous - endpoint, outward))
            previous_inside = previous_distance <= epsilon

            for current in polygon:
                current_distance = float(np.dot(current - endpoint, outward))
                current_inside = current_distance <= epsilon
                if current_inside != previous_inside:
                    denominator = previous_distance - current_distance
                    if abs(denominator) > 1e-15:
                        t = previous_distance / denominator
                        intersection = previous + t * (current - previous)
                        # Snap to the plane so adjacent triangles create exactly
                        # matching boundary vertices after float32 serialization.
                        intersection = intersection - float(
                            np.dot(intersection - endpoint, outward)
                        ) * outward
                        clipped.append(intersection)
                if current_inside:
                    clipped.append(current)
                previous = current
                previous_distance = current_distance
                previous_inside = current_inside

            if len(clipped) >= 3:
                anchor = clipped[0]
                for index in range(1, len(clipped) - 1):
                    tri = np.asarray([anchor, clipped[index], clipped[index + 1]])
                    area2 = float(np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])))
                    if area2 > 1e-12:
                        output.append(tri)

        if not output:
            raise RuntimeError("Terminal plane clipping removed the complete vessel surface.")
        return np.asarray(output, dtype=np.float32)

    clipped = np.asarray(triangles, dtype=np.float32)
    for endpoint, outward, radius in terminal_data:
        clipped = clip_halfspace(clipped, endpoint, outward, radius)
    return clipped


def clean_triangles(
    triangles: np.ndarray,
    min_edge_mm: float = 0.01,
    min_area_mm2: float = 1e-4,
) -> np.ndarray:
    """Remove numerical slivers that are harmful to SOFA collision detection."""
    triangles = np.asarray(triangles, dtype=np.float32)
    e01 = np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1)
    e12 = np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1)
    e20 = np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1)
    area = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    keep = (
        (np.minimum(np.minimum(e01, e12), e20) >= min_edge_mm)
        & (area >= min_area_mm2)
        & np.isfinite(triangles).all(axis=(1, 2))
    )
    return triangles[keep]


def write_binary_stl(path: Path, triangles: np.ndarray, solid_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    triangles = np.asarray(triangles, dtype=np.float32)
    with path.open("wb") as f:
        header = (f"mCR artificial vessel {solid_name}".encode("ascii")[:80]).ljust(80, b"\0")
        f.write(header)
        f.write(struct.pack("<I", len(triangles)))
        for tri in triangles:
            normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            normal = _normalize(normal, (0.0, 0.0, 0.0)).astype(np.float32)
            f.write(struct.pack("<12fH", *(normal.tolist() + tri.reshape(-1).tolist()), 0))


def write_legacy_vtk_polydata(
    path: Path,
    points: np.ndarray,
    cells: Sequence[Sequence[int]],
    radii: np.ndarray,
) -> None:
    points = np.asarray(points, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    total_line_values = sum(1 + len(cell) for cell in cells)
    lines = [
        "# vtk DataFile Version 3.0",
        "mCR artificial vessel centerline (millimetres)",
        "ASCII",
        "DATASET POLYDATA",
        f"POINTS {len(points)} float",
    ]
    lines.extend(f"{p[0]:.9g} {p[1]:.9g} {p[2]:.9g}" for p in points)
    lines.append(f"LINES {len(cells)} {total_line_values}")
    lines.extend(f"{len(cell)} " + " ".join(str(int(i)) for i in cell) for cell in cells)
    lines.extend(
        [
            f"POINT_DATA {len(points)}",
            "SCALARS Radius float 1",
            "LOOKUP_TABLE default",
        ]
    )
    lines.extend(f"{r:.9g}" for r in radii)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_vti(path: Path, field: np.ndarray, origin: np.ndarray, spacing_mm: float) -> None:
    field = np.asarray(field, dtype="<f4")
    nz, ny, nx = field.shape
    raw = field.tobytes(order="C")
    compressed = zlib.compress(raw, level=6)
    header = struct.pack("<4I", 1, len(raw), len(raw), len(compressed))
    payload = base64.b64encode(header + compressed).decode("ascii")
    extent = f"0 {nx - 1} 0 {ny - 1} 0 {nz - 1}"
    origin_text = " ".join(f"{float(v):.9g}" for v in origin)
    spacing_text = f"{spacing_mm:.9g} {spacing_mm:.9g} {spacing_mm:.9g}"
    xml = f'''<?xml version="1.0"?>
<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian" header_type="UInt32" compressor="vtkZLibDataCompressor">
  <ImageData WholeExtent="{extent}" Origin="{origin_text}" Spacing="{spacing_text}">
    <Piece Extent="{extent}">
      <PointData Scalars="signed_distance_mm">
        <DataArray type="Float32" Name="signed_distance_mm" format="binary">{payload}</DataArray>
      </PointData>
      <CellData/>
    </Piece>
  </ImageData>
</VTKFile>
'''
    path.write_text(xml, encoding="utf-8")


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    )


def _draw_disc(image: np.ndarray, x: float, y: float, radius: int, color: Tuple[int, int, int]):
    h, w, _ = image.shape
    xi, yi = int(round(x)), int(round(y))
    x0, x1 = max(0, xi - radius), min(w - 1, xi + radius)
    y0, y1 = max(0, yi - radius), min(h - 1, yi + radius)
    if x1 < x0 or y1 < y0:
        return
    yy, xx = np.ogrid[y0:y1 + 1, x0:x1 + 1]
    mask = (xx - xi) ** 2 + (yy - yi) ** 2 <= radius * radius
    patch = image[y0:y1 + 1, x0:x1 + 1]
    patch[mask] = color


def _draw_line(image, p0, p1, width, color):
    distance = max(1.0, float(np.linalg.norm(np.asarray(p1) - np.asarray(p0))))
    steps = int(math.ceil(distance * 1.4))
    for t in np.linspace(0.0, 1.0, steps):
        p = (1.0 - t) * np.asarray(p0) + t * np.asarray(p1)
        _draw_disc(image, p[0], p[1], max(1, width // 2), color)


def write_preview_png(path: Path, polylines: Sequence[np.ndarray], color=(46, 116, 181)) -> None:
    width, height = 1024, 768
    image = np.full((height, width, 3), 246, dtype=np.uint8)
    angle_y, angle_z = math.radians(-24.0), math.radians(12.0)
    ry = np.array(
        [[math.cos(angle_y), 0, math.sin(angle_y)], [0, 1, 0], [-math.sin(angle_y), 0, math.cos(angle_y)]]
    )
    rz = np.array(
        [[math.cos(angle_z), -math.sin(angle_z), 0], [math.sin(angle_z), math.cos(angle_z), 0], [0, 0, 1]]
    )
    transformed = [np.asarray(p) @ (rz @ ry).T for p in polylines]
    all_pts = np.vstack(transformed)
    xy = all_pts[:, [0, 1]]
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    span = np.maximum(hi - lo, 1.0)
    scale = min((width - 120) / span[0], (height - 120) / span[1])

    def project(p):
        x = 60 + (p[:, 0] - lo[0]) * scale
        y = height - 60 - (p[:, 1] - lo[1]) * scale
        return np.column_stack((x, y))

    projected = [project(p) for p in transformed]
    # Depth shadow, then colored centerline.
    for poly in projected:
        for a, b in zip(poly[:-1], poly[1:]):
            _draw_line(image, a + (5, 7), b + (5, 7), 14, (208, 211, 216))
    for poly in projected:
        for a, b in zip(poly[:-1], poly[1:]):
            _draw_line(image, a, b, 11, color)
    raw = b"".join(b"\x00" + row.tobytes() for row in image)
    png = b"\x89PNG\r\n\x1a\n"
    png += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += _png_chunk(b"IDAT", zlib.compress(raw, level=8))
    png += _png_chunk(b"IEND", b"")
    path.write_bytes(png)


def stl_stats(triangles: np.ndarray) -> Dict[str, object]:
    triangles = np.asarray(triangles, dtype=np.float64)
    normals2 = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = 0.5 * np.linalg.norm(normals2, axis=1)
    points = triangles.reshape((-1, 3))
    edge_lengths = np.concatenate(
        (
            np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
            np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
            np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
        )
    )
    return {
        "triangle_count": int(len(triangles)),
        "degenerate_triangles": int(np.count_nonzero(areas <= 1e-8)),
        "bounds_min_mm": points.min(axis=0).round(5).tolist(),
        "bounds_max_mm": points.max(axis=0).round(5).tolist(),
        "area_mm2_min_mean_max": [float(areas.min()), float(areas.mean()), float(areas.max())],
        "edge_mm_min_mean_max": [
            float(edge_lengths.min()), float(edge_lengths.mean()), float(edge_lengths.max())
        ],
    }


def _write_metadata(path: Path, metadata: Dict[str, object]) -> None:
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _copy_alias(src: Path, dst: Path) -> None:
    shutil.copyfile(src, dst)


def generate_curved_model(spec: CurvedSpec, model_dir: Path, spacing_mm: float) -> Dict[str, object]:
    points, radii = curved_centerline(spec)
    collision = sweep_tube(points, radii, sides=18)
    visual = sweep_tube(points, radii + 1.0, sides=40)
    seg_a, seg_b, seg_r = path_to_segments(points, radii)
    field, origin, dims = compute_sdf_grid(seg_a, seg_b, seg_r, spacing_mm=spacing_mm)

    collision_path = model_dir / "collision_inner.stl"
    visual_path = model_dir / "visual_wall.stl"
    centerline_path = model_dir / "Centerline model.vtk"
    write_binary_stl(collision_path, collision, spec.model_id + " collision")
    write_binary_stl(visual_path, visual, spec.model_id + " visual")
    _copy_alias(collision_path, model_dir / "Segmentation.stl")
    write_legacy_vtk_polydata(centerline_path, points, [list(range(len(points)))], radii)
    _copy_alias(centerline_path, model_dir / "centerline.vtk")
    write_vti(model_dir / "vessel_sdf.vti", field, origin, spacing_mm)
    write_preview_png(model_dir / "preview.png", [points], color=(45, 114, 181))

    metadata = {
        "model_id": spec.model_id,
        "family": "curved",
        "difficulty": spec.difficulty,
        "source_units": SOURCE_UNITS,
        "sofa_scale": SOFA_SCALE,
        "seed": spec.seed,
        "catheter_outer_diameter_mm": CATHETER_OUTER_DIAMETER_MM,
        "parameters": {
            "length_mm": spec.length_mm,
            "turns": spec.turns,
            "amplitude_mm": spec.amplitude_mm,
            "torsion_mm": spec.torsion_mm,
            "radius_min_mm": spec.radius_min_mm,
            "radius_max_mm": spec.radius_max_mm,
            "centerline_samples": len(points),
            "collision_ring_sides": 18,
            "visual_ring_sides": 40,
            "visual_wall_offset_mm": 1.0,
        },
        "topology": {"inlets": 1, "outlets": 1, "bifurcations": 0},
        "sdf": {
            "convention": "negative_inside_positive_outside",
            "scalar_name": "signed_distance_mm",
            "spacing_mm": spacing_mm,
            "origin_mm": origin.tolist(),
            "dimensions_xyz": list(dims),
        },
        "collision_validation": stl_stats(collision),
        "visual_validation": stl_stats(visual),
    }
    _write_metadata(model_dir / "metadata.json", metadata)
    return metadata


def _graph_polydata(polylines):
    point_list: List[np.ndarray] = []
    radius_list: List[float] = []
    cells: List[List[int]] = []
    node_endpoint_ids: Dict[str, int] = {}
    for poly in polylines:
        cell: List[int] = []
        for idx, (point, radius) in enumerate(zip(poly["points"], poly["radii"])):
            key = None
            if idx == 0:
                key = poly["start"]
            elif idx == len(poly["points"]) - 1:
                key = poly["end"]
            if key is not None and key in node_endpoint_ids:
                pid = node_endpoint_ids[key]
            else:
                pid = len(point_list)
                point_list.append(np.asarray(point, dtype=np.float64))
                radius_list.append(float(radius))
                if key is not None:
                    node_endpoint_ids[key] = pid
            cell.append(pid)
        cells.append(cell)
    return np.asarray(point_list), cells, np.asarray(radius_list)


def _root_to_outlet_path(polylines, outlet: str) -> Tuple[np.ndarray, np.ndarray]:
    by_end = {poly["end"]: poly for poly in polylines}
    chain = []
    current = outlet
    while current != "root":
        poly = by_end[current]
        chain.append(poly)
        current = poly["start"]
    chain.reverse()
    points, radii = [], []
    for i, poly in enumerate(chain):
        start = 0 if i == 0 else 1
        points.extend(poly["points"][start:])
        radii.extend(poly["radii"][start:])
    return np.asarray(points), np.asarray(radii)


def generate_branch_model(spec: BranchSpec, model_dir: Path, spacing_mm: float) -> Dict[str, object]:
    nodes, node_radii, polylines = branch_polylines(spec)
    seg_a, seg_b, seg_r = polylines_to_segments(polylines)
    field, origin, dims = compute_sdf_grid(seg_a, seg_b, seg_r, spacing_mm=spacing_mm)
    # Keep the SDF reasonably fine for safety-distance queries, but use a slightly
    # coarser implicit grid for STL surfaces.  This avoids feeding SOFA more than
    # one hundred thousand collision triangles per vessel without sacrificing the
    # lumen shape at the catheter's scale.
    collision_spacing_mm = max(2.0, spacing_mm)
    collision_field, collision_origin, _ = compute_sdf_grid(
        seg_a, seg_b, seg_r, spacing_mm=collision_spacing_mm
    )
    collision_closed = marching_tetrahedra(
        collision_field, collision_origin, collision_spacing_mm
    )

    # Remove inlet/outlet caps while retaining clean straight collars.
    first_poly = next(poly for poly in polylines if poly["start"] == "root")
    terminals = [
        (
            nodes["root"],
            nodes["root"] - first_poly["points"][1],
            node_radii["root"],
        )
    ]
    for outlet in BRANCH_OUTLETS:
        poly = next(poly for poly in polylines if poly["end"] == outlet)
        terminals.append((nodes[outlet], nodes[outlet] - poly["points"][-2], node_radii[outlet]))
    collision = remove_terminal_caps(collision_closed, terminals, collision_spacing_mm)

    visual_spacing_mm = max(1.4, spacing_mm)
    visual_r = seg_r + 1.0
    visual_field, visual_origin, _ = compute_sdf_grid(
        seg_a, seg_b, visual_r, spacing_mm=visual_spacing_mm, margin_mm=6.0
    )
    visual_closed = marching_tetrahedra(
        visual_field, visual_origin, visual_spacing_mm
    )
    visual_terminals = [(p, d, r + 1.0) for p, d, r in terminals]
    visual = remove_terminal_caps(visual_closed, visual_terminals, visual_spacing_mm)

    collision_path = model_dir / "collision_inner.stl"
    visual_path = model_dir / "visual_wall.stl"
    write_binary_stl(collision_path, collision, spec.model_id + " collision")
    write_binary_stl(visual_path, visual, spec.model_id + " visual")
    _copy_alias(collision_path, model_dir / "Segmentation.stl")

    graph_points, graph_cells, graph_radii = _graph_polydata(polylines)
    write_legacy_vtk_polydata(
        model_dir / "centerline_graph.vtk", graph_points, graph_cells, graph_radii
    )

    target_files = []
    default_path = None
    for index, outlet in enumerate(BRANCH_OUTLETS, start=1):
        path_points, path_radii = _root_to_outlet_path(polylines, outlet)
        target_path = model_dir / f"target_{index:02d}_centerline.vtk"
        write_legacy_vtk_polydata(
            target_path, path_points, [list(range(len(path_points)))], path_radii
        )
        target_files.append({"target_index": index, "outlet_node": outlet, "file": target_path.name})
        if index == 1:
            default_path = target_path

    assert default_path is not None
    _copy_alias(default_path, model_dir / "Centerline model.vtk")
    _copy_alias(default_path, model_dir / "centerline.vtk")
    write_vti(model_dir / "vessel_sdf.vti", field, origin, spacing_mm)
    write_preview_png(
        model_dir / "preview.png", [poly["points"] for poly in polylines], color=(203, 79, 54)
    )

    metadata = {
        "model_id": spec.model_id,
        "family": "branching",
        "difficulty": spec.difficulty,
        "source_units": SOURCE_UNITS,
        "sofa_scale": SOFA_SCALE,
        "seed": spec.seed,
        "catheter_outer_diameter_mm": CATHETER_OUTER_DIAMETER_MM,
        "parameters": {
            "x_scale": spec.x_scale,
            "z_scale_mm": spec.z_scale_mm,
            "curve_mm": spec.curve_mm,
            "trunk_radius_mm": spec.trunk_radius_mm,
            "outlet_radius_mm": spec.outlet_radius_mm,
            "nominal_angle_range_deg": list(spec.angle_range_deg),
            "collision_surface_spacing_mm": collision_spacing_mm,
            "visual_surface_spacing_mm": visual_spacing_mm,
            "visual_wall_offset_mm": 1.0,
        },
        "topology": {
            "inlets": 1,
            "outlets": 6,
            "bifurcation_levels": 3,
            "target_centerlines": target_files,
            "default_target_centerline": "target_01_centerline.vtk",
            "full_graph_centerline": "centerline_graph.vtk",
        },
        "sdf": {
            "convention": "negative_inside_positive_outside",
            "scalar_name": "signed_distance_mm",
            "spacing_mm": spacing_mm,
            "origin_mm": origin.tolist(),
            "dimensions_xyz": list(dims),
        },
        "collision_validation": stl_stats(collision),
        "visual_validation": stl_stats(visual),
    }
    _write_metadata(model_dir / "metadata.json", metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    python_root = Path(__file__).resolve().parents[1]
    project_root = python_root.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "mesh" / "train",
        help="Destination containing C01..C05 and B01..B05.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=[],
        help="Optional model IDs to generate. Default: all ten.",
    )
    parser.add_argument(
        "--spacing-mm",
        type=float,
        default=1.0,
        help=(
            "SDF spacing in millimetres. Branch collision/visual surfaces keep "
            "performance-oriented lower bounds of 2.0/1.4 mm."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    selected = set(args.models or [s.model_id for s in CURVED_SPECS + BRANCH_SPECS])
    known = {s.model_id for s in CURVED_SPECS + BRANCH_SPECS}
    unknown = sorted(selected - known)
    if unknown:
        raise SystemExit(f"Unknown model IDs: {unknown}")
    if not (0.5 <= float(args.spacing_mm) <= 2.0):
        raise SystemExit("--spacing-mm must be between 0.5 and 2.0 mm")

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "source_units": SOURCE_UNITS,
        "sofa_scale": SOFA_SCALE,
        "catheter_outer_diameter_mm": CATHETER_OUTER_DIAMETER_MM,
        "models": [],
    }
    for spec in CURVED_SPECS:
        if spec.model_id not in selected:
            continue
        model_dir = output_root / spec.model_id
        if model_dir.exists() and any(model_dir.iterdir()) and not args.overwrite:
            raise SystemExit(f"Refusing to overwrite non-empty directory: {model_dir}")
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"[GENERATE] {spec.model_id} curved -> {model_dir}", flush=True)
        meta = generate_curved_model(spec, model_dir, float(args.spacing_mm))
        manifest["models"].append(
            {"model_id": spec.model_id, "family": "curved", "metadata": f"{spec.model_id}/metadata.json"}
        )
        print(
            f"[OK] {spec.model_id}: triangles={meta['collision_validation']['triangle_count']} "
            f"sdf={meta['sdf']['dimensions_xyz']}",
            flush=True,
        )

    for spec in BRANCH_SPECS:
        if spec.model_id not in selected:
            continue
        model_dir = output_root / spec.model_id
        if model_dir.exists() and any(model_dir.iterdir()) and not args.overwrite:
            raise SystemExit(f"Refusing to overwrite non-empty directory: {model_dir}")
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"[GENERATE] {spec.model_id} branching -> {model_dir}", flush=True)
        meta = generate_branch_model(spec, model_dir, float(args.spacing_mm))
        manifest["models"].append(
            {"model_id": spec.model_id, "family": "branching", "metadata": f"{spec.model_id}/metadata.json"}
        )
        print(
            f"[OK] {spec.model_id}: triangles={meta['collision_validation']['triangle_count']} "
            f"sdf={meta['sdf']['dimensions_xyz']}",
            flush=True,
        )

    _write_metadata(output_root / "manifest.json", manifest)
    print(f"[DONE] generated {len(manifest['models'])} models in {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
