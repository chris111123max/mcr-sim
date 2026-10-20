"""Read-only STL topology and catheter-to-lumen diagnostics for acceptance runs.

This module never repairs or changes source geometry. Signed wall depth is only
reported when topology and point-classification checks support it.
"""
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
import struct

import numpy as np

_TRIMESH_CACHE = {}
_VTK_CACHE = {}


@dataclass
class STLMesh:
    path: Path
    triangles_m: np.ndarray
    vertices_m: np.ndarray
    faces: np.ndarray
    sha256: str
    binary: bool


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_stl(path):
    path = Path(path)
    raw = path.read_bytes()
    binary = False
    if len(raw) >= 84:
        count = struct.unpack_from("<I", raw, 80)[0]
        binary = 84 + count * 50 == len(raw)
    if binary:
        count = struct.unpack_from("<I", raw, 80)[0]
        packed = np.frombuffer(raw, dtype=np.dtype([
            ("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")
        ]), offset=84, count=count)
        triangles = packed["vertices"].astype(np.float64)
    else:
        text = raw.decode("ascii", errors="strict")
        values = re.findall(r"\bvertex\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)", text, re.I)
        if not values or len(values) % 3:
            raise ValueError("STL is neither a valid binary triangle file nor parseable ASCII STL")
        triangles = np.asarray(values, dtype=np.float64).reshape(-1, 3, 3)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or len(triangles) == 0:
        raise ValueError("STL contains no triangle geometry")
    if not np.all(np.isfinite(triangles)):
        raise ValueError("STL contains NaN or Inf coordinates")
    # STL assets are authored in millimetres; make the conversion explicit here.
    triangles_m = triangles * 0.001
    flat = triangles_m.reshape(-1, 3)
    span = max(float(np.ptp(flat, axis=0).max()), 1e-9)
    weld = max(span * 1e-10, 1e-12)
    quantized = np.rint(flat / weld).astype(np.int64)
    vertices, inverse = np.unique(quantized, axis=0, return_inverse=True)
    # Keep the first actual vertex for each weld key, rather than quantized coordinates.
    first = np.full(len(vertices), -1, dtype=np.int64)
    for i, idx in enumerate(inverse):
        if first[idx] < 0:
            first[idx] = i
    unique_vertices = flat[first]
    faces = inverse.reshape(-1, 3)
    return STLMesh(path, triangles_m, unique_vertices, faces, sha256_file(path), binary)


def inspect_mesh(mesh):
    triangles = mesh.triangles_m
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    scale = max(float(np.ptp(mesh.vertices_m, axis=0).max()), 1e-9)
    degenerate = area <= scale * scale * 1e-12
    edge_map = {}
    for face_id, face in enumerate(mesh.faces):
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            key = (int(min(a, b)), int(max(a, b)))
            edge_map.setdefault(key, []).append((1 if a < b else -1, face_id))
    counts = np.asarray([len(v) for v in edge_map.values()], dtype=np.int64)
    boundary_edges = int(np.count_nonzero(counts == 1))
    nonmanifold_edges = int(np.count_nonzero(counts > 2))
    inconsistent_edges = int(sum(1 for v in edge_map.values()
                                 if len(v) == 2 and v[0][0] == v[1][0]))
    face_adjacency = [[] for _ in mesh.faces]
    for linked in edge_map.values():
        if len(linked) == 2:
            a, b = linked[0][1], linked[1][1]
            face_adjacency[a].append(b)
            face_adjacency[b].append(a)
    unseen = set(range(len(mesh.faces)))
    components = 0
    while unseen:
        components += 1
        stack = [unseen.pop()]
        while stack:
            current = stack.pop()
            for neighbor in face_adjacency[current]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
    signed_volume = float(np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6.0)
    return {
        "triangle_count": int(len(triangles)),
        "vertex_count_welded": int(len(mesh.vertices_m)),
        "connected_face_components": int(components),
        "bbox_min_m": mesh.vertices_m.min(axis=0).tolist(),
        "bbox_max_m": mesh.vertices_m.max(axis=0).tolist(),
        "bbox_diagonal_m": float(np.linalg.norm(np.ptp(mesh.vertices_m, axis=0))),
        "degenerate_triangle_count": int(np.count_nonzero(degenerate)),
        "boundary_edge_count": boundary_edges,
        "nonmanifold_edge_count": nonmanifold_edges,
        "inconsistent_orientation_edge_count": inconsistent_edges,
        "watertight": boundary_edges == 0 and nonmanifold_edges == 0,
        "locally_consistent_winding": inconsistent_edges == 0,
        "signed_volume_m3": signed_volume,
        "self_intersection": {"status": "NOT_VERIFIED", "reason": "No verified self-intersection checker is part of this read-only harness."},
        "sha256": mesh.sha256,
        "stl_encoding": "binary" if mesh.binary else "ascii",
    }


def _vtk_geometry(mesh):
    cached = _VTK_CACHE.get(id(mesh))
    if cached is not None and cached[0] is mesh:
        return cached[1], None
    try:
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
    except Exception as exc:
        return None, f"VTK unavailable: {type(exc).__name__}: {exc}"
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(np.ascontiguousarray(mesh.vertices_m), deep=True))
    cells = vtk.vtkCellArray()
    for face in mesh.faces:
        cells.InsertNextCell(3)
        cells.InsertCellPoint(int(face[0]))
        cells.InsertCellPoint(int(face[1]))
        cells.InsertCellPoint(int(face[2]))
    surface = vtk.vtkPolyData()
    surface.SetPoints(points)
    surface.SetPolys(cells)
    surface.BuildCells()
    enclosed = vtk.vtkSelectEnclosedPoints()
    enclosed.SetSurfaceData(surface)
    enclosed.SetTolerance(1e-7)
    implicit_distance = vtk.vtkImplicitPolyDataDistance()
    implicit_distance.SetInput(surface)
    backend = {"vtk": vtk, "vtk_to_numpy": vtk_to_numpy, "surface": surface,
               "enclosed": enclosed, "implicit_distance": implicit_distance}
    _VTK_CACHE[id(mesh)] = (mesh, backend)
    return backend, None


def _trimesh_geometry(mesh):
    cached = _TRIMESH_CACHE.get(id(mesh))
    if cached is not None and cached[0] is mesh:
        return cached[1], None
    try:
        import trimesh
    except Exception as exc:
        return None, f"trimesh unavailable: {type(exc).__name__}: {exc}"
    tm = trimesh.Trimesh(vertices=mesh.vertices_m, faces=mesh.faces, process=False)
    _TRIMESH_CACHE[id(mesh)] = (mesh, tm)
    return tm, None


def classification_backend():
    try:
        import vtk  # noqa: F401
        return "VTK.vtkSelectEnclosedPoints + vtkImplicitPolyDataDistance"
    except Exception:
        try:
            import trimesh  # noqa: F401
            return "trimesh.contains + trimesh.nearest.on_surface"
        except Exception:
            return "NOT_AVAILABLE"


def classify_points(mesh, points_m):
    """Return containment and unsigned nearest-surface distance, or a blocker."""
    points = np.asarray(points_m, dtype=np.float64).reshape(-1, 3)
    vtk_data, vtk_reason = _vtk_geometry(mesh)
    if vtk_data is not None:
        try:
            vtk = vtk_data["vtk"]
            vtk_points = vtk.vtkPoints()
            for point in points:
                vtk_points.InsertNextPoint(float(point[0]), float(point[1]), float(point[2]))
            verts = vtk.vtkCellArray()
            for index in range(len(points)):
                verts.InsertNextCell(1)
                verts.InsertCellPoint(index)
            cloud = vtk.vtkPolyData()
            cloud.SetPoints(vtk_points)
            cloud.SetVerts(verts)
            enclosed = vtk_data["enclosed"]
            enclosed.SetInputData(cloud)
            enclosed.Update()
            selected = vtk_data["vtk_to_numpy"](
                enclosed.GetOutput().GetPointData().GetArray("SelectedPoints")
            ).astype(bool, copy=True)
            implicit = vtk_data["implicit_distance"]
            distance = np.asarray([abs(float(implicit.EvaluateFunction(
                (float(point[0]), float(point[1]), float(point[2]))))) for point in points], dtype=np.float64)
            if selected.shape != (len(points),) or not np.all(np.isfinite(distance)):
                raise ValueError("VTK returned invalid containment/distance values")
            return selected, distance, None
        except Exception as exc:
            vtk_reason = f"VTK signed geometry query failed: {type(exc).__name__}: {exc}"
    tm, trimesh_reason = _trimesh_geometry(mesh)
    if tm is None:
        return None, None, f"{vtk_reason}; {trimesh_reason}"
    try:
        inside = np.asarray(tm.contains(points), dtype=bool)
        _, distance, _ = tm.nearest.on_surface(points)
        distance = np.asarray(distance, dtype=np.float64)
        if inside.shape != (len(points),) or distance.shape != (len(points),):
            raise ValueError("geometry backend returned an unexpected result shape")
        if not np.all(np.isfinite(distance)):
            raise ValueError("nearest-surface distances are non-finite")
        return inside, distance, None
    except Exception as exc:
        return None, None, f"signed point classification unavailable: {type(exc).__name__}: {exc}"


def wall_diagnostic(mesh, mesh_report, tracking_tip_first_m, inserted_length_m,
                    catheter_radius_m=0.000665, spacing_m=0.00025, ring_samples=16):
    """Estimate outer-surface breach from inserted SOFA DOF centerline geometry.

    Linear interpolation is dense relative to catheter radius but is not the
    BeamAdapter's internal curved interpolation. The report retains that limit.
    """
    result = {
        "status": "NOT_VERIFIED", "method": "SOFA active Rigid3d DOFs; dense linear segment and 16-point radial sampling; " + classification_backend(),
        "geometry_source": "SofaBackend.state().catheter_tracking_m / IRC CurvAbs active range",
        "max_penetration_depth_mm": None, "max_penetration_position_m": None,
        "affected_inserted_length_mm": None, "sampled_centerline_points": 0,
        "sampled_tube_surface_points": 0, "per_step_classification": None,
        "numeric_tolerance_m": None, "limitations": [],
    }
    if not mesh_report.get("watertight") or not mesh_report.get("locally_consistent_winding"):
        result["limitations"].append("STL is open, nonmanifold, or winding-inconsistent; signed interior is not trusted.")
        return result
    if mesh_report.get("centerline_inside", {}).get("status") != "PASS":
        result["limitations"].append("Known centerline/inlet-inside and exterior-outside sign checks did not PASS.")
        return result
    if not np.all(np.isfinite([inserted_length_m, catheter_radius_m, spacing_m])) or inserted_length_m < 0:
        result["limitations"].append("Invalid insertion length or catheter sampling parameters.")
        return result
    tracking = np.asarray(tracking_tip_first_m, dtype=np.float64).reshape(-1, 3)
    if len(tracking) < 2 or not np.all(np.isfinite(tracking)):
        result["limitations"].append("SOFA returned fewer than two finite inserted catheter nodes.")
        return result
    # The active geometry must span the insertion coordinate within mesh precision.
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(tracking[::-1], axis=0), axis=1))]
    if arc[-1] <= 0 or abs(float(arc[-1]) - inserted_length_m) > max(spacing_m * 2.0, inserted_length_m * 0.02):
        result["limitations"].append("Active SOFA DOF arc length does not agree with IRC insertion length.")
        return result
    count = max(2, int(math.ceil(arc[-1] / spacing_m)) + 1)
    sample_arc = np.linspace(0.0, arc[-1], count)
    base_to_tip = tracking[::-1]
    center = np.column_stack([np.interp(sample_arc, arc, base_to_tip[:, axis]) for axis in range(3)])
    tangents = np.gradient(center, axis=0)
    norms = np.linalg.norm(tangents, axis=1)
    if np.any(norms <= 1e-12):
        result["limitations"].append("Degenerate tangent in sampled inserted catheter geometry.")
        return result
    tangents /= norms[:, None]
    reference = np.tile([0.0, 0.0, 1.0], (len(center), 1))
    parallel = np.abs(np.sum(tangents * reference, axis=1)) > 0.9
    reference[parallel] = [0.0, 1.0, 0.0]
    axis1 = np.cross(tangents, reference)
    axis1 /= np.linalg.norm(axis1, axis=1)[:, None]
    axis2 = np.cross(tangents, axis1)
    angles = np.arange(ring_samples, dtype=np.float64) * (2.0 * np.pi / ring_samples)
    radial = catheter_radius_m * (np.cos(angles)[None, :, None] * axis1[:, None, :] +
                                  np.sin(angles)[None, :, None] * axis2[:, None, :])
    surface = center[:, None, :] + radial
    all_points = np.concatenate([center, surface.reshape(-1, 3)], axis=0)
    inside, distance, reason = classify_points(mesh, all_points)
    if inside is None:
        result["limitations"].append(reason)
        return result
    if mesh_report.get("self_intersection", {}).get("status") != "PASS":
        result["limitations"].append("STL self-intersection has not been independently ruled out.")
    result["sampled_centerline_points"] = int(len(center))
    result["sampled_tube_surface_points"] = int(surface.shape[0] * surface.shape[1])
    result["numeric_tolerance_m"] = float(max(np.finfo(float).eps * max(np.abs(mesh.vertices_m).max(), 1.0) * 64,
                                               np.finfo(np.float32).eps * max(np.abs(mesh.vertices_m).max(), 1.0) * 8))
    center_inside = inside[:len(center)]
    surface_inside = inside[len(center):].reshape(len(center), ring_samples)
    surface_distance = distance[len(center):].reshape(len(center), ring_samples)
    outside = ~surface_inside
    tolerance = result["numeric_tolerance_m"]
    deep = outside & (surface_distance > tolerance)
    any_outside = np.any(outside, axis=1)
    intervals = np.diff(sample_arc)
    affected = float(np.sum(intervals * (any_outside[:-1] | any_outside[1:])))
    if np.any(deep):
        candidate = np.where(deep, surface_distance, -np.inf)
        row, col = np.unravel_index(int(np.argmax(candidate)), candidate.shape)
        result["max_penetration_depth_mm"] = float(candidate[row, col] * 1000.0)
        result["max_penetration_position_m"] = surface[row, col].tolist()
        result["max_penetration_centerline_sample_m"] = center[row].tolist()
        result["per_step_classification"] = "TRUE_WALL_CROSSING_CANDIDATE"
    elif np.any(outside):
        result["max_penetration_depth_mm"] = 0.0
        first_outside_row = int(np.flatnonzero(any_outside)[0])
        first_outside_col = int(np.flatnonzero(outside[first_outside_row])[0])
        result["max_penetration_position_m"] = surface[first_outside_row, first_outside_col].tolist()
        result["max_penetration_centerline_sample_m"] = center[first_outside_row].tolist()
        result["per_step_classification"] = "NUMERICAL_TOLERANCE_OR_CONTACT_REQUIRES_REVIEW"
    else:
        result["max_penetration_depth_mm"] = 0.0
        result["per_step_classification"] = "NO_SAMPLED_TUBE_SURFACE_OUTSIDE_LUMEN"
    result["affected_inserted_length_mm"] = affected * 1000.0
    result["centerline_outside_count"] = int(np.count_nonzero(~center_inside))
    result["tube_surface_outside_count"] = int(np.count_nonzero(outside))
    tip_outside = outside[-1]
    result["tip_surface_outside_count"] = int(np.count_nonzero(tip_outside))
    result["tip_max_penetration_depth_mm"] = (float(np.max(surface_distance[-1][tip_outside]) * 1000.0)
                                                if np.any(tip_outside) else 0.0)
    result["tip_classification"] = "OUTSIDE_SAMPLE" if np.any(tip_outside) else "NO_OUTSIDE_TIP_SAMPLE"
    result["limitations"].append("BeamAdapter interpolation between Rigid3d DOFs is approximated by dense straight segments; no claim is made for sub-DOF curvature.")
    if mesh_report.get("self_intersection", {}).get("status") != "PASS":
        result["max_penetration_depth_mm"] = None
        result["max_penetration_position_m"] = None
        result["max_penetration_centerline_sample_m"] = None
        result["affected_inserted_length_mm"] = None
        result["tip_max_penetration_depth_mm"] = None
        result["status"] = "NOT_VERIFIED"
    else:
        result["status"] = "PASS"
    return result
