"""Independent real-SOFA acceptance runner for mcr_sim.
This is a measurement harness only; it never patches production files, changes
SOFA parameters, generates vessels, or trains/evaluates a policy.
"""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PYTHON_ROOT = HERE.parents[1]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from wall_geometry import inspect_mesh, read_stl, classify_points, wall_diagnostic, classification_backend

ACTION_PATTERNS = {
    "natural_advance": np.array([0.0, 0.0, 0.25], dtype=np.float32),
    "magnetic_turn": np.array([0.5, 0.0, 0.0], dtype=np.float32),
    "turn_and_advance": np.array([0.5, 0.5, 0.25], dtype=np.float32),
    "moderate_retract": np.array([0.0, 0.0, -0.25], dtype=np.float32),
}


def jsonable(value):
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return repr(value)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def asset_directory(root, seed):
    root = Path(root).expanduser().resolve()
    if (root / "metadata.json").is_file():
        return root if f"seed_{seed}" in root.name else root / f"arch_type_I_seed_{seed}"
    return root / f"arch_type_I_seed_{seed}"


def _mesh_point_checks(bundle, mesh, mesh_report, inlet_probe):
    scale = max(float(np.ptp(mesh.vertices_m, axis=0).max()), 1e-9)
    outside = mesh.vertices_m.max(axis=0) + np.maximum(np.ptp(mesh.vertices_m, axis=0), scale) * 0.2
    samples = []
    for branch in bundle.branches.values():
        points = branch.coordinates_m
        if len(points) >= 3:
            samples.extend(0.5 * (points[:-1] + points[1:])[::max(1, len(points) // 32)])
    if not samples:
        return {"status": "NOT_VERIFIED", "reason": "No centerline interior samples available."}
    classified, _, reason = classify_points(mesh, np.vstack([np.asarray(samples), inlet_probe, outside]))
    if classified is None:
        return {"status": "NOT_VERIFIED", "reason": reason}
    inside = classified[:len(samples)]
    inlet_inside = bool(classified[-2])
    external = bool(not classified[-1])
    centers_inside = bool(np.all(inside))
    if not mesh_report["watertight"] or not mesh_report["locally_consistent_winding"]:
        status = "NOT_VERIFIED"
        reason = "Open/nonmanifold or orientation-inconsistent surface cannot provide trusted interior classification."
    elif centers_inside and inlet_inside and external:
        status, reason = "PASS", None
    else:
        status, reason = "FAIL", "Centerline/inlet interior or known exterior sanity point has an unexpected classification."
    return {"status": status, "centerline_midpoint_sample_count": int(len(samples)),
            "centerline_midpoints_inside": centers_inside, "inward_inlet_probe_inside": inlet_inside,
            "known_exterior_outside": external, "reason": reason}


def inspect_asset(root, seed):
    from mcr_sim.vessel_bundle import BranchData, LegacyVTKPolyline, TYPE_I_PARENTS, VesselBundle
    directory = asset_directory(root, seed)
    result = {"seed": int(seed), "directory": str(directory), "status": "FAIL", "checks": {}, "files": {}}
    required = ("collision_inner.stl", "visual_wall.stl", "centerline_graph.vtk", "metadata.json")
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        result["reason"] = "Missing required assets: " + ", ".join(missing)
        result["checks"]["asset_files"] = {"status": "FAIL", "missing": missing}
        return result, None, None
    for name in required:
        result["files"][name] = {"sha256": sha256(directory / name), "bytes": (directory / name).stat().st_size}
    try:
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        vtk = LegacyVTKPolyline(directory / "centerline_graph.vtk")
        names = tuple(metadata.get("centerline_length_mm", {}).keys())
        if len(names) != len(vtk.lines) or "radius_mm" not in vtk.point_data:
            raise ValueError("centerline_graph.vtk branch schema does not match metadata")
        branches = {}
        for name, ids in zip(names, vtk.lines):
            coordinates_mm = np.asarray(vtk.points[ids], dtype=np.float64)
            radii_mm = np.asarray(vtk.point_data["radius_mm"][ids], dtype=np.float64)
            branches[name] = BranchData(name, coordinates_mm, radii_mm,
                                        coordinates_mm * 0.001, radii_mm * 0.001)
        insertion_data = metadata["insertion"]
        insertion_mm = np.asarray(insertion_data["point_mm"], dtype=np.float64)
        insertion_direction = np.asarray(insertion_data["direction"], dtype=np.float64)
        insertion_direction /= max(float(np.linalg.norm(insertion_direction)), 1e-15)
        target_paths = {name: directory / filename for name, filename in metadata.get("target_paths", {}).items()
                        if (directory / filename).is_file()}
        sdf_path = directory / "vessel_sdf.vti"
        bundle = VesselBundle(seed=int(metadata["seed"]), directory=directory,
            collision_stl=directory / "collision_inner.stl", visual_stl=directory / "visual_wall.stl",
            centerline_graph_vtk=directory / "centerline_graph.vtk", metadata_path=directory / "metadata.json",
            metadata=metadata, branches=branches, parents=dict(TYPE_I_PARENTS),
            insertion_point_mm=insertion_mm, insertion_point_m=insertion_mm * 0.001,
            insertion_direction=insertion_direction, target_centerlines=target_paths,
            sdf_path=sdf_path if sdf_path.is_file() else None)
        metadata_ok = (bundle.seed == seed and metadata.get("arch_type") == "I" and
                       metadata.get("scale_diameter_array") == [0.85] and metadata.get("length_unit") == "mm")
        result["checks"]["metadata"] = {
            "status": "PASS" if metadata_ok else "FAIL", "seed": bundle.seed,
            "arch_type": metadata.get("arch_type"), "scale_diameter_array": metadata.get("scale_diameter_array"),
            "length_unit": metadata.get("length_unit"), "declared_validation": metadata.get("validation"),
        }
        declared = metadata.get("asset_sha256", {})
        hashes = {}
        unsafe_or_missing_hash_files = []
        for name, expected in declared.items():
            hashed_path = (directory / name).resolve()
            if hashed_path.parent != directory.resolve() or not hashed_path.is_file():
                unsafe_or_missing_hash_files.append(name)
                continue
            hashes[name] = sha256(hashed_path)
        required_hashes_present = all(name in hashes for name in
                                      ("collision_inner.stl", "visual_wall.stl", "centerline_graph.vtk"))
        hash_ok = (required_hashes_present and not unsafe_or_missing_hash_files and
                   all(declared.get(name) == value for name, value in hashes.items()))
        result["checks"]["metadata_hashes"] = {"status": "PASS" if hash_ok else "FAIL",
                                                "declared": declared, "actual": hashes,
                                                "unsafe_or_missing_files": unsafe_or_missing_hash_files}
        mm_consistency = all(np.array_equal(branch.coordinates_m, branch.coordinates_mm * 0.001)
                             for branch in bundle.branches.values())
        result["checks"]["coordinate_units"] = {
            "status": "PASS" if mm_consistency else "FAIL", "metadata_unit": "mm",
            "conversion": "coordinates_m = coordinates_mm * 0.001",
            "centerline_branch_count": len(bundle.branches)}
        gaps = {}
        for child, parent in bundle.parents.items():
            if child not in bundle.branches or parent not in bundle.branches:
                gaps[child] = None
                continue
            origin = bundle.branches[child].coordinates_mm[0]
            gaps[child] = float(np.min(np.linalg.norm(bundle.branches[parent].coordinates_mm - origin, axis=1)))
        all_coords = np.vstack([branch.coordinates_mm for branch in bundle.branches.values()])
        required_branches = {"aorta", "bct", "rcca", "rsa", "lcca", "lsa"}
        segment_counts = {name: int(np.count_nonzero(np.linalg.norm(np.diff(branch.coordinates_mm, axis=0), axis=1) <= 0.0))
                          for name, branch in bundle.branches.items()}
        branch_ok = (set(bundle.branches) >= required_branches and
                     all(len(b.coordinates_mm) >= 2 and np.all(np.isfinite(b.coordinates_mm)) and
                         np.all(np.isfinite(b.radii_mm)) and np.all(b.radii_mm > 0) and
                         segment_counts[name] == 0
                         for name, b in bundle.branches.items()))
        gap_tolerance_mm = max(float(np.ptp(all_coords, axis=0).max()) * 1e-9, 1e-6)
        connected = all(value is not None and value <= gap_tolerance_mm for value in gaps.values())
        result["checks"]["branch_topology_and_continuity"] = {
            "status": "PASS" if branch_ok and connected else "FAIL",
            "required_branches_present_and_valid": branch_ok, "zero_length_segments_by_branch": segment_counts,
            "child_parent_gaps_mm": gaps,
            "continuity_tolerance_mm": gap_tolerance_mm}
        insertion = np.asarray(bundle.insertion_point_m, dtype=float)
        direction = np.asarray(bundle.insertion_direction, dtype=float)
        direction_norm = float(np.linalg.norm(direction))
        rootline = bundle.branches.get("aorta")
        inlet_gap = None if rootline is None else float(np.min(np.linalg.norm(rootline.coordinates_m - insertion, axis=1)))
        inlet_ok = (np.all(np.isfinite(insertion)) and np.isfinite(direction_norm) and
                    abs(direction_norm - 1.0) < 1e-6 and inlet_gap is not None and
                    inlet_gap <= max(float(np.min(rootline.radii_m)), 1e-6))
        result["checks"]["inlet_pose"] = {
            "status": "PASS" if inlet_ok else "NOT_VERIFIED", "insertion_point_m": insertion.tolist(),
            "direction": direction.tolist(), "direction_unit_norm": direction_norm,
            "nearest_aorta_centerline_gap_m": inlet_gap,
            "probe_5mm_m": (insertion + direction * 0.005).tolist(),
            "reason": "The 5 mm inward probe is classified against collision_inner.stl below."}
        mesh = read_stl(bundle.collision_stl)
        mesh_report = inspect_mesh(mesh)
        mesh_report["units_assumption"] = "STL millimetres converted to metres by 0.001; STL has no embedded unit metadata."
        mesh_report["point_classification_backend"] = classification_backend()
        mesh_report["open_boundary_policy"] = {
            "status": "NOT_VERIFIED" if mesh_report["boundary_edge_count"] else "PASS",
            "reason": "No inlet/outlet boundary-loop labels are assumed. Open boundaries are never exempted from wall checks."}
        inlet_probe = insertion + direction * 0.005
        mesh_report["centerline_inside"] = _mesh_point_checks(bundle, mesh, mesh_report, inlet_probe)
        result["checks"]["inlet_pose"]["inward_probe_inside"] = mesh_report["centerline_inside"].get("inward_inlet_probe_inside")
        if mesh_report["centerline_inside"].get("status") == "FAIL":
            result["checks"]["inlet_pose"]["status"] = "FAIL"
        elif mesh_report["centerline_inside"].get("status") != "PASS":
            result["checks"]["inlet_pose"]["status"] = "NOT_VERIFIED"
        if (mesh_report["degenerate_triangle_count"] or mesh_report["nonmanifold_edge_count"] or
                not mesh_report["watertight"] or mesh_report["connected_face_components"] != 1):
            mesh_report["mesh_integrity_status"] = "FAIL"
        else:
            mesh_report["mesh_integrity_status"] = "PASS"
        if mesh_report["mesh_integrity_status"] == "FAIL" or mesh_report["centerline_inside"]["status"] == "FAIL":
            mesh_report["status"] = "FAIL"
        elif mesh_report["centerline_inside"]["status"] != "PASS":
            mesh_report["status"] = "NOT_VERIFIED"
        else:
            mesh_report["status"] = "PASS"
        result["checks"]["collision_inner_mesh"] = mesh_report
        visual = read_stl(bundle.visual_stl)
        visual_report = inspect_mesh(visual)
        visual_report["status"] = "FAIL" if (visual_report["degenerate_triangle_count"] or
            visual_report["nonmanifold_edge_count"] or not visual_report["watertight"] or
            visual_report["connected_face_components"] != 1) else "PASS"
        visual_report["units_assumption"] = "STL millimetres converted to metres by 0.001; STL has no embedded unit metadata."
        result["checks"]["visual_wall_mesh"] = visual_report
        mesh_low, mesh_high = mesh.vertices_m.min(axis=0), mesh.vertices_m.max(axis=0)
        center = np.vstack([b.coordinates_m for b in bundle.branches.values()])
        bbox_margin = np.maximum(mesh_high - mesh_low, 1e-9) * 1e-8
        bbox_ok = bool(np.all(center >= mesh_low - bbox_margin) and np.all(center <= mesh_high + bbox_margin))
        result["checks"]["centerline_mesh_frame"] = {
            "status": "PASS" if bbox_ok else "FAIL", "centerline_within_collision_mesh_bbox": bbox_ok}
        statuses = [check.get("status") for check in result["checks"].values()]
        if "FAIL" in statuses:
            result["status"] = "FAIL"
        elif "NOT_VERIFIED" in statuses:
            result["status"] = "NOT_VERIFIED"
        else:
            result["status"] = "PASS"
        return result, bundle, (mesh, mesh_report)
    except Exception as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        result["checks"].setdefault("asset_loading", {"status": "FAIL", "reason": result["reason"]})
        return result, None, None


def _read_value(obj, name):
    return np.asarray(getattr(obj, name).value).copy()


def sofa_snapshot(env, state=None, observation=None):
    backend = env.backend
    handles = backend._handles
    if handles is None:
        raise RuntimeError("SOFA scene handles are unavailable after reset")
    controller = handles.ir_controller
    all_positions = _read_value(handles.dofs, "position")
    curv = _read_value(controller, "CurvAbs").reshape(-1)
    first = _read_value(controller, "indexFirstNode").reshape(-1)
    xtip = _read_value(controller, "xtip").reshape(-1)
    field = np.asarray(handles.torque_controller.field_world_t, dtype=float).reshape(3)
    state = state if state is not None else backend.state()
    active_first_to_tip = all_positions[int(first[0]):int(first[0]) + len(curv)] if len(first) == 1 else np.empty((0, 7))
    tracking_expected_tip_first = active_first_to_tip[::-1, :3]
    tracking_match = (tracking_expected_tip_first.shape == state.catheter_tracking_m.shape and
                      np.allclose(tracking_expected_tip_first, state.catheter_tracking_m, rtol=0.0, atol=1e-12))
    curv_matches_xtip = bool(len(curv) > 0 and np.all(np.isfinite(curv)) and
                              np.all(np.diff(curv) > 0) and np.isclose(curv[0], 0.0, atol=1e-8, rtol=0.0) and
                              len(xtip) == 1 and np.isclose(curv[-1], float(xtip[0]), atol=1e-8, rtol=1e-6))
    def optional_data(obj, name):
        try:
            return np.asarray(getattr(obj, name).value).copy()
        except Exception:
            return None
    try:
        collision_positions = np.asarray(handles.instrument.collision_dofs.position.value).copy()
    except Exception:
        collision_positions = None
    collision_edges = None
    try:
        collision_edges = np.asarray(handles.instrument.CollisionModel.collisEdgeSet.edges.value).copy()
    except Exception:
        pass
    dt_value = getattr(handles.root, "dt", None)
    try:
        sofa_dt = float(dt_value.value if hasattr(dt_value, "value") else dt_value)
    except Exception:
        sofa_dt = None
    return {
        "sofa_dofs_position_shape": list(all_positions.shape),
        "physical_beam_count": int(max(len(all_positions) - 1, 0)),
        "configured_body_beams": int(env.config.instrument.discretization.body_beams),
        "configured_tip_beams": int(env.config.instrument.discretization.tip_beams),
        "sofa_dofs_position_finite": bool(np.all(np.isfinite(all_positions))),
        "sofa_dofs_positions": all_positions.tolist(),
        "sofa_dofs_velocity": jsonable(optional_data(handles.dofs, "velocity")),
        "sofa_dofs_force": jsonable(optional_data(handles.dofs, "force")),
        "collision_dofs_position_shape": None if collision_positions is None else list(collision_positions.shape),
        "collision_dofs_position_finite": None if collision_positions is None else bool(np.all(np.isfinite(collision_positions))),
        "collision_dofs_positions": jsonable(collision_positions),
        "collision_topology_edges_shape": None if collision_edges is None else list(collision_edges.shape),
        "collision_topology_edges": jsonable(collision_edges),
        "sofa_dt_s": sofa_dt,
        "sofa_python_version": getattr(backend._Sofa, "__version__", None),
        "irc_xtip": xtip.tolist(),
        "irc_indexFirstNode": first.tolist(), "irc_CurvAbs_m": curv.tolist(),
        "irc_active_range_valid": bool(len(first) == 1 and len(curv) > 0 and int(first[0]) >= 0 and int(first[0]) + len(curv) == len(all_positions)),
        "irc_xtip_matches_backend_insertion": bool(len(xtip) == 1 and np.isclose(float(xtip[0]), state.inserted_length_m, rtol=1e-6, atol=1e-8)),
        "irc_curv_abs_matches_entrance_to_xtip": curv_matches_xtip,
        "active_geometry_matches_sofa_dof_slice": bool(tracking_match),
        "tip_position_m": state.tip_m.tolist(), "tip_rotation_world": state.tip_frame_world.tolist(),
        "magnetic_field_world_t": field.tolist(), "inserted_length_m": state.inserted_length_m,
        "active_tracking_tip_first_m": state.catheter_tracking_m.tolist(),
        "geometry_source": state.extra.get("geometry_source"),
        "active_dof_indices_tip_first": jsonable(state.extra.get("inserted_dof_indices_tip_first")),
        "active_curv_abs_tip_first_m": jsonable(state.extra.get("inserted_curv_abs_m_tip_first")),
        "observation_shape": None if observation is None else list(np.asarray(observation).shape),
        "observation_finite": None if observation is None else bool(np.all(np.isfinite(observation))),
        "observation_41": None if observation is None else np.asarray(observation).tolist(),
    }


def _log_messages(backend):
    collector = getattr(backend, "_messages", None)
    raw = [] if collector is None else list(getattr(collector, "messages", []))
    rendered = [str(message) for message in raw]
    errors = [line for line in rendered if re.search(r"\b(error|exception|fatal)\b", line, re.I)]
    warnings = [line for line in rendered if re.search(r"\bwarning\b", line, re.I)]
    return {"messages": rendered, "errors": errors, "warnings": warnings,
            "error_status": "FAIL" if errors else "PASS"}


def _capture_one_step(env, backend, action, mesh_info=None):
    captured = []
    original = backend.physics_substep
    def observed_substep(field_t, insertion_delta_m):
        diagnostic = original(field_t, insertion_delta_m)
        try:
            state = backend.state()
        except Exception:
            captured.append({"physical_substep_index": int(backend.physics_substeps_executed),
                             "capture_error": traceback.format_exc(),
                             "diagnostic_measurement_status": "NOT_VERIFIED"})
            return diagnostic
        item = {
            "physical_substep_index": int(backend.physics_substeps_executed),
            "magnetic_field_world_t": jsonable(state.magnetic_field_world_t),
            "magnetic_field_direction_world": jsonable(state.magnetic_field_world_t /
                max(float(np.linalg.norm(state.magnetic_field_world_t)), 1e-30)),
            "tip_position_m": jsonable(state.tip_m), "inserted_length_m": jsonable(state.inserted_length_m),
            "simulation_error": bool(diagnostic.simulation_error)}
        if mesh_info is not None:
            try:
                item["wall"] = wall_diagnostic(mesh_info[0], mesh_info[1],
                    state.catheter_tracking_m, state.inserted_length_m, env.config.instrument.radius_m)
            except Exception:
                item["wall"] = {"status": "NOT_VERIFIED", "max_penetration_depth_mm": None,
                                "reason": traceback.format_exc()}
        captured.append(item)
        return diagnostic
    backend.physics_substep = observed_substep
    before_error = None
    try:
        before = backend.state()
    except Exception:
        before = None
        before_error = traceback.format_exc()
    start_count = backend.physics_substeps_executed
    start_time = time.perf_counter()
    try:
        observation, reward, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32))
    finally:
        backend.physics_substep = original
    elapsed = time.perf_counter() - start_time
    next_state = None
    try:
        next_state = backend.state()
    except Exception:
        pass
    displacement = None if next_state is None or before is None else float(np.linalg.norm(next_state.tip_m - before.tip_m))
    insertion_delta = None if next_state is None or before is None else float(next_state.inserted_length_m - before.inserted_length_m)
    return {
        "action": np.asarray(action, dtype=np.float32).tolist(), "elapsed_seconds": elapsed,
        "substeps_before": int(start_count), "substeps_after": int(backend.physics_substeps_executed),
        "substeps_executed": int(backend.physics_substeps_executed - start_count),
        "substep_records": captured, "observation_shape": list(np.asarray(observation).shape),
        "observation_finite": bool(np.all(np.isfinite(observation))), "observation_41": np.asarray(observation).tolist(),
        "reward": float(reward), "reward_terms": jsonable(info.get("reward_terms")),
        "terminated": bool(terminated), "truncated": bool(truncated), "info": jsonable(info),
        "transition_valid": bool(info.get("transition_valid", True)),
        "initial_state_capture_error": before_error,
        "initial_tip_m": None if before is None else before.tip_m.tolist(), "tip_displacement_m": displacement,
        "tip_jump_check": {"status": "PASS" if displacement is not None and np.isfinite(displacement) and displacement <= env.config.instrument.total_length_m else "FAIL",
                           "displacement_m": displacement, "absolute_geometric_upper_bound_m": env.config.instrument.total_length_m},
        "actual_insertion_delta_m": insertion_delta,
        "insertion_delta_check": {"status": "PASS" if insertion_delta is not None and np.isfinite(insertion_delta) and abs(insertion_delta) <= env.config.action.max_insertion_per_rl_step_m + 1e-8 else "FAIL",
                                  "configured_maximum_absolute_delta_m": env.config.action.max_insertion_per_rl_step_m},
        "next_tip_m": None if next_state is None else next_state.tip_m.tolist(),
        "next_inserted_length_m": None if next_state is None else next_state.inserted_length_m,
        "next_magnetic_field_world_t": None if next_state is None else next_state.magnetic_field_world_t.tolist(),
        "next_magnetic_field_direction_world": None if next_state is None else
            (next_state.magnetic_field_world_t / max(float(np.linalg.norm(next_state.magnetic_field_world_t)), 1e-30)).tolist(),
        "bfs_remaining_mm": info.get("path_length_mm"), "sofa_logs": _log_messages(backend),
    }


def _config(vessel_root):
    from mcr_sim.config import DEFAULT_CONFIG
    return replace(DEFAULT_CONFIG, vessel_root=Path(vessel_root))


def _sofa_object_data_snapshot(root):
    """Best-effort read-only capture before SofaBackend unloads a failed scene."""
    output = {"tree": [], "capture_errors": []}
    pending = [(root, "/root")]
    visited = 0
    while pending and visited < 300:
        node, path = pending.pop(0)
        visited += 1
        node_record = {"path": path, "objects": []}
        try:
            children = list(node.getChildren())
        except Exception as exc:
            children = []
            output["capture_errors"].append(f"{path} children: {type(exc).__name__}: {exc}")
        for child in children:
            try:
                child_name = str(child.getName())
            except Exception:
                child_name = f"child_{len(pending)}"
            pending.append((child, path.rstrip("/") + "/" + child_name))
        try:
            objects = list(node.getObjects())
        except Exception as exc:
            objects = []
            output["capture_errors"].append(f"{path} objects: {type(exc).__name__}: {exc}")
        for obj in objects:
            try:
                name = str(obj.getName())
            except Exception:
                name = repr(obj)
            try:
                component_type = str(obj.getClassName())
            except Exception:
                component_type = type(obj).__name__
            item = {"name": name, "type": component_type, "data": {}}
            try:
                fields = list(obj.getDataFields())
            except Exception:
                fields = []
            for data in fields:
                try:
                    data_name = str(data.getName())
                except Exception:
                    data_name = str(getattr(data, "name", ""))
                if not any(token in data_name.lower() for token in
                           ("position", "velocity", "force", "dt", "xtip", "curv", "indexfirst", "field", "edge")):
                    continue
                try:
                    value = np.asarray(data.value)
                    item["data"][data_name] = (value.tolist() if value.size <= 2000 else
                        {"shape": list(value.shape), "dtype": str(value.dtype), "truncated": True})
                except Exception as exc:
                    item["data"][data_name] = {"read_error": f"{type(exc).__name__}: {exc}"}
            node_record["objects"].append(item)
        output["tree"].append(node_record)
    if pending:
        output["capture_errors"].append("SOFA node traversal exceeded 300 nodes; remaining nodes omitted.")
    return output


def _run_real_case(seed, config, mesh_info=None, action=(0.0, 0.0, 0.0), stage_name="single"):

    from mcr_sim.sofa_simulation import SofaBackend
    from mcr_sim.rl_env import MagneticNavigationEnv

    class AcceptanceSofaBackend(SofaBackend):
        def close(self):
            root = getattr(self, "_root", None)
            if root is not None:
                try:
                    self._acceptance_last_closed_root_snapshot = _sofa_object_data_snapshot(root)
                except Exception:
                    self._acceptance_last_closed_root_snapshot = {"capture_error": traceback.format_exc()}
            super().close()

    backend = AcceptanceSofaBackend(config)
    env = MagneticNavigationEnv(backend, config=config, mode="train")
    record = {"seed": int(seed), "stage": stage_name, "status": "FAIL",
              "timing_seconds": {}, "statuses": {}, "errors": []}
    started = time.perf_counter()
    try:
        t0 = time.perf_counter()
        observation, reset_info = env.reset(seed=int(seed), options={"vessel_seed": int(seed)})
        record["timing_seconds"]["scene_create_init_and_reset"] = time.perf_counter() - t0
        state = backend.state()
        snapshot = sofa_snapshot(env, state, observation)
        record["reset_info"] = jsonable(reset_info)
        record["initial_snapshot"] = snapshot
        record["sofa_dt_s"] = config.sofa.dt_s
        record["configured_initial_magnetic_field_world_t"] = list(config.instrument.initial_field_t)
        record["sofa_runtime_observed"] = snapshot.get("sofa_python_version") or "NOT_OBSERVED"
        record["expected_physics_substeps_per_rl_action"] = config.sofa.physics_substeps
        record["statuses"]["SCENE_INIT"] = "PASS" if snapshot["sofa_dofs_position_finite"] and snapshot["sofa_dt_s"] is not None and np.isclose(snapshot["sofa_dt_s"], config.sofa.dt_s) else "FAIL"
        observed_sofa = snapshot.get("sofa_python_version")
        record["statuses"]["SOFA_VERSION"] = ("NOT_VERIFIED" if not observed_sofa else
            "PASS" if "21.12" in str(observed_sofa) else "FAIL")
        record["statuses"]["BEAM_MODEL"] = "PASS" if (snapshot["irc_active_range_valid"] and
            snapshot["irc_xtip_matches_backend_insertion"] and snapshot["irc_curv_abs_matches_entrance_to_xtip"] and
            snapshot["active_geometry_matches_sofa_dof_slice"] and
            snapshot["sofa_dofs_position_shape"] == [config.instrument.discretization.mechanical_dofs, 7] and
            snapshot["physical_beam_count"] == 134 and snapshot["configured_body_beams"] == 100 and
            snapshot["configured_tip_beams"] == 34) else "FAIL"
        initial_field = np.asarray(snapshot["magnetic_field_world_t"], dtype=float)
        configured_field = np.asarray(config.instrument.initial_field_t, dtype=float)
        record["statuses"]["MAGNETIC_MODEL"] = "PASS" if (np.all(np.isfinite(initial_field)) and
            np.linalg.norm(initial_field) > 0 and np.allclose(initial_field, configured_field, rtol=1e-6, atol=1e-12)) else "FAIL"
        record["statuses"]["OBSERVATION_41D"] = "PASS" if snapshot["observation_shape"] == [41] and snapshot["observation_finite"] else "FAIL"
        record["statuses"]["SOFA_LOGS_AFTER_INIT"] = _log_messages(backend)["error_status"]
        if any(record["statuses"][name] == "FAIL" for name in ("SCENE_INIT", "BEAM_MODEL", "MAGNETIC_MODEL", "OBSERVATION_41D", "SOFA_LOGS_AFTER_INIT", "SOFA_VERSION")):
            record["statuses"]["PHYSICS_22_SUBSTEPS"] = "NOT_RUN"
            record["statuses"]["NUMERICAL_STABILITY"] = "NOT_RUN"
            record["errors"].append("Reset snapshot failed a structural or finite-value check; physics action was not started.")
            record["status"] = "FAIL"
            return record
        step = _capture_one_step(env, backend, action, mesh_info)
        record["timing_seconds"]["one_rl_action"] = step["elapsed_seconds"]
        record["action_step"] = step
        record["statuses"]["PHYSICS_22_SUBSTEPS"] = "PASS" if step["substeps_executed"] == config.sofa.physics_substeps else "FAIL"
        finite = (step["observation_finite"] and
                  all(np.all(np.isfinite(np.asarray(item.get("tip_position_m", []), dtype=float))) and
                      np.all(np.isfinite(np.asarray(item.get("magnetic_field_world_t", []), dtype=float))) and
                      np.isfinite(item.get("inserted_length_m", np.nan))
                      for item in step["substep_records"] if item.get("tip_position_m") is not None))
        finite = finite and step["tip_jump_check"]["status"] == "PASS" and step["insertion_delta_check"]["status"] == "PASS"
        record["statuses"]["NUMERICAL_STABILITY"] = "PASS" if finite and step["transition_valid"] and not step["info"].get("simulation_error", False) else "FAIL"
        record["statuses"]["SOFA_LOGS_AFTER_ACTION"] = step["sofa_logs"]["error_status"]
        if mesh_info is not None:
            wall_records = [item.get("wall", {}) for item in step["substep_records"]]
            record["wall_diagnostics"] = wall_records
            record["statuses"]["WALL_DIAGNOSTICS"] = "PASS" if wall_records and all(item.get("status") == "PASS" for item in wall_records) else "NOT_VERIFIED" if any(item.get("status") == "NOT_VERIFIED" for item in wall_records) else "FAIL"
        failures = [key for key, value in record["statuses"].items() if value == "FAIL"]
        record["status"] = "FAIL" if failures else ("NOT_VERIFIED" if any(value == "NOT_VERIFIED" for value in record["statuses"].values()) else "PASS")
        return record
    except Exception as exc:
        record["errors"].append(f"{type(exc).__name__}: {exc}")
        record["traceback"] = traceback.format_exc()
        record["statuses"].setdefault("SCENE_INIT", "FAIL")
        record["statuses"].setdefault("PHYSICS_22_SUBSTEPS", "NOT_RUN")
        record["status"] = "FAIL"
        try:
            record["backend_last_exception"] = backend.last_exception
            record["sofa_logs"] = _log_messages(backend)
            record["sofa_failure_data_snapshot"] = getattr(backend, "_acceptance_last_closed_root_snapshot", None)
            if backend._handles is not None:
                record["failure_snapshot"] = sofa_snapshot(env)
        except Exception:
            record["failure_snapshot_error"] = traceback.format_exc()
        return record
    finally:
        record["timing_seconds"]["total"] = time.perf_counter() - started
        env.close()
        record["sofa_messages_after_close"] = jsonable(backend.last_messages)


def run_assets(seeds, vessel_root):
    start = time.perf_counter()
    checks = []
    for seed in seeds:
        result, _, _ = inspect_asset(vessel_root, seed)
        checks.append(result)
    status = "FAIL" if any(item["status"] == "FAIL" for item in checks) else (
        "NOT_VERIFIED" if any(item["status"] == "NOT_VERIFIED" for item in checks) else "PASS")
    return {"status": status, "elapsed_seconds": time.perf_counter() - start, "vessels": checks}


def _asset_mesh_info(vessel_root, seed):
    return inspect_asset(vessel_root, seed)


def run_short(seeds, max_steps, config, mesh_by_seed):
    output = {"status": "PASS", "seeds": {}, "max_rl_steps_per_action_pattern": int(max_steps)}
    for seed in seeds:
        per_seed = {}
        seed_failed = False
        for pattern, action in ACTION_PATTERNS.items():
            env = backend = None
            sequence = {"action_pattern": pattern, "action": action.tolist(), "steps": [], "status": "PASS"}
            try:
                from mcr_sim.sofa_simulation import SofaBackend
                from mcr_sim.rl_env import MagneticNavigationEnv
                backend = SofaBackend(config)
                env = MagneticNavigationEnv(backend, config=config, mode="train")
                obs, reset_info = env.reset(seed=int(seed), options={"vessel_seed": int(seed)})
                sequence["reset_info"] = jsonable(reset_info)
                if np.asarray(obs).shape != (41,) or not np.all(np.isfinite(obs)):
                    raise ValueError("41D reset observation is invalid")
                start = time.perf_counter()
                for index in range(max_steps):
                    step = _capture_one_step(env, backend, action, mesh_by_seed.get(seed))
                    step["rl_step_index"] = index + 1
                    sequence["steps"].append(step)
                    if (not step["transition_valid"] or step["info"].get("simulation_error", False) or
                            not step["observation_finite"] or
                            step["tip_jump_check"]["status"] != "PASS" or
                            step["insertion_delta_check"]["status"] != "PASS" or
                            step["sofa_logs"]["errors"] or
                            any("capture_error" in item for item in step["substep_records"])):
                        sequence["status"] = "FAIL"
                        sequence["stop_reason"] = "SOFA failure or invalid state; sequence stopped immediately."
                        break
                    if step["substeps_executed"] != config.sofa.physics_substeps:
                        sequence["status"] = "FAIL"
                        sequence["stop_reason"] = "The action did not complete all configured physical substeps."
                        break
                    if step["terminated"] or step["truncated"]:
                        sequence["stop_reason"] = "Episode ended normally."
                        break
                sequence["elapsed_seconds"] = time.perf_counter() - start
            except Exception as exc:
                sequence["status"] = "FAIL"
                sequence["stop_reason"] = f"{type(exc).__name__}: {exc}"
                sequence["traceback"] = traceback.format_exc()
            finally:
                if env is not None:
                    env.close()
                    sequence["sofa_messages_after_close"] = jsonable(backend.last_messages)
            per_seed[pattern] = sequence
            if sequence["status"] != "PASS":
                seed_failed = True
        output["seeds"][str(seed)] = per_seed
        if seed_failed:
            output["status"] = "FAIL"
    return output


def run_parallel(seeds, config):
    from mcr_sim.sampling import ProcessEnvironment, step_environments
    selected = [101, 202]
    if not set(selected).issubset(set(seeds)):
        return {"status": "FAIL", "reason": "Stage parallel requires both seeds 101 and 202 in --seeds."}
    envs = []
    result = {"status": "FAIL", "environment_ids": [0, 1], "seeds": selected,
              "rounds": [], "reset_isolation": "NOT_VERIFIED"}
    try:
        for _ in selected:
            envs.append(ProcessEnvironment(config, mode="train"))
        reset_data = []
        for seed, env in zip(selected, envs):
            obs, info = env.reset(seed=seed, options={"vessel_seed": seed})
            reset_data.append({"observation_shape": list(np.asarray(obs).shape),
                               "observation_finite": bool(np.all(np.isfinite(obs))),
                               "reported_seed": info.get("vessel_seed")})
        if any(item["reported_seed"] != seed or item["observation_shape"] != [41] or not item["observation_finite"]
               for seed, item in zip(selected, reset_data)):
            raise RuntimeError("Parallel reset did not return the requested seed and valid 41D observation.")
        result["worker_pids"] = [int(env.process.pid) for env in envs]
        result["distinct_worker_processes"] = len(set(result["worker_pids"])) == 2
        result["reset_observations"] = jsonable(reset_data)
        actions = [np.array([0.0, 0.0, 0.25], dtype=np.float32),
                   np.array([0.5, 0.0, 0.0], dtype=np.float32)]
        reset_info = None
        for round_index in range(2):
            if round_index == 1:
                _, reset_info = envs[0].reset(seed=99101, options={"vessel_seed": 101})
            started = time.perf_counter()
            transitions = step_environments(envs, actions)
            round_info = []
            for env_id, (transition, expected_seed) in enumerate(zip(transitions, selected)):
                observation, reward, terminated, truncated, info = transition
                round_info.append({"environment_id": env_id, "expected_seed": expected_seed,
                    "reported_seed": info.get("vessel_seed"), "observation_shape": list(np.asarray(observation).shape),
                    "observation_finite": bool(np.all(np.isfinite(observation))), "reward": float(reward),
                    "terminated": bool(terminated), "truncated": bool(truncated),
                    "transition_valid": bool(info.get("transition_valid", True)),
                    "simulation_error": bool(info.get("simulation_error", False)),
                    "physics_substeps_attempted": info.get("physics_substeps_attempted")})
            result["rounds"].append({"index": round_index + 1,
                "elapsed_seconds": time.perf_counter() - started, "results": round_info})
            if any(item["reported_seed"] != item["expected_seed"] or not item["transition_valid"] or
                   item["simulation_error"] or not item["observation_finite"] or
                   item["physics_substeps_attempted"] != config.sofa.physics_substeps for item in round_info):
                raise RuntimeError("Parallel worker result was invalid or assigned to the wrong environment.")
        last_env1 = result["rounds"][-1]["results"][1]
        result["reset_environment_0_info"] = jsonable(reset_info)
        result["reset_isolation"] = "PASS" if reset_info.get("vessel_seed") == 101 and last_env1.get("reported_seed") == 202 and last_env1.get("transition_valid") and last_env1.get("observation_finite") else "FAIL"
        result["status"] = "PASS" if result["distinct_worker_processes"] and result["reset_isolation"] == "PASS" else "FAIL"
    except Exception as exc:
        result["status"] = "FAIL"
        result["reason"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        for env in envs:
            env.close()
    return result


def source_version():
    package = PYTHON_ROOT / "mcr_sim"
    candidates = ("config.py", "sofa_simulation.py", "rl_env.py", "observation.py", "sampling.py")
    digest = hashlib.sha256()
    for name in candidates:
        path = package / name
        if path.is_file():
            digest.update(name.encode("utf-8"))
            digest.update(path.read_bytes())
    return {"version_method": "SHA256 of selected production source files (Git was not queried)",
            "source_sha256": digest.hexdigest(), "files": list(candidates)}


def _stage_record(status, started, details):
    return {"status": status, "elapsed_seconds": time.perf_counter() - started, **details}


def execute(args):
    from mcr_sim.config import DEFAULT_CONFIG
    vessel_root = Path(args.vessel_root).expanduser().resolve() if args.vessel_root else DEFAULT_CONFIG.vessel_root
    seeds = list(dict.fromkeys(int(seed) for seed in args.seeds))
    config = _config(vessel_root)
    report = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "project": "mcr_sim_wall_acceptance", "stage_requested": args.stage,
        "source_version": source_version(),
        "sofa_runtime_expected": "SOFA 21.12 + BeamAdapter; runtime version observed only during Stage B",
        "vessel_root": str(vessel_root), "requested_seeds": seeds, "max_rl_steps": int(args.max_rl_steps),
        "action_patterns": {name: action.tolist() for name, action in ACTION_PATTERNS.items()},
        "stages": {}, "failure_reasons": [],
        "safety_threshold_statement": "No clinical or physical safety threshold is asserted. Numeric tolerance is only machine/STL precision-derived.",
    }
    requested = args.stage
    if requested == "assets":
        started = time.perf_counter()
        stage = run_assets(seeds, vessel_root)
        report["stages"]["A_ASSETS"] = _stage_record(stage["status"], started, stage)
    else:
        prerequisite_seeds = [seeds[0]] if requested == "single" else ([101, 202] if requested == "parallel" else seeds)
        started = time.perf_counter()
        asset_stage = run_assets(prerequisite_seeds, vessel_root)
        report["stages"]["A_ASSETS"] = _stage_record(asset_stage["status"], started, asset_stage)
        if asset_stage["status"] == "FAIL":
            report["failure_reasons"].append("SOFA stages blocked because all required asset checks did not PASS.")
            for name in ("B_SINGLE", "C_WALL", "D_SHORT", "E_PARALLEL"):
                report["stages"].setdefault(name, {"status": "BLOCKED", "reason": "Stage A did not PASS."})
        elif requested == "single":
            seed = prerequisite_seeds[0]
            started = time.perf_counter()
            case = _run_real_case(seed, config, action=(0.0, 0.0, 0.0), stage_name="single")
            report["stages"]["B_SINGLE"] = _stage_record(case["status"], started, {"cases": [case]})
        elif requested == "wall":
            cases, statuses = [], []
            for seed in prerequisite_seeds:
                _, _, mesh_info = _asset_mesh_info(vessel_root, seed)
                case = _run_real_case(seed, config, mesh_info, action=(0.0, 0.0, 0.0), stage_name="wall")
                cases.append(case)
                statuses.append(case["status"])
                if case["status"] == "FAIL":
                    break
            report["stages"]["B_SINGLE"] = {"status": "FAIL" if "FAIL" in statuses else "PASS", "cases": cases,
                                            "scope": "One actual [0,0,0] RL action per requested seed."}
            wall_status = "FAIL" if "FAIL" in statuses else ("PASS" if all(x == "PASS" for x in statuses) else "NOT_VERIFIED")
            report["stages"]["C_WALL"] = {"status": wall_status, "cases": cases,
                                           "scope": "Read-only geometry diagnostic after every captured physical substep."}
        elif requested in ("short", "parallel"):
            prerequisite_seeds = [101, 202] if requested == "parallel" else prerequisite_seeds
            cases, mesh_by_seed = [], {}
            prereq_status = "PASS"
            for seed in prerequisite_seeds:
                asset, _, mesh_info = _asset_mesh_info(vessel_root, seed)
                mesh_by_seed[seed] = mesh_info
                if asset["status"] != "PASS" or mesh_info is None:
                    prereq_status = "NOT_VERIFIED" if asset["status"] == "NOT_VERIFIED" else "FAIL"
                    break
                case = _run_real_case(seed, config, mesh_info, action=(0.0, 0.0, 0.0), stage_name=f"{requested}_prerequisite")
                cases.append(case)
                if case["status"] == "FAIL":
                    prereq_status = "FAIL"
                    break
                if case["status"] != "PASS":
                    prereq_status = "NOT_VERIFIED"
                    break
            report["stages"]["B_SINGLE"] = {"status": prereq_status, "cases": cases}
            report["stages"]["C_WALL"] = {"status": prereq_status,
                "reason": "Stage D/E requires fully trusted signed geometry classification; NOT_VERIFIED blocks later stages."}
            if prereq_status != "PASS":
                name = "D_SHORT" if requested == "short" else "E_PARALLEL"
                report["stages"][name] = {"status": "BLOCKED", "reason": "Stage B and Stage C must both PASS first."}
            elif requested == "short":
                started = time.perf_counter()
                result = run_short(prerequisite_seeds, args.max_rl_steps, config, mesh_by_seed)
                report["stages"]["D_SHORT"] = _stage_record(result["status"], started, result)
            else:
                started = time.perf_counter()
                result = run_parallel(seeds, config)
                report["stages"]["E_PARALLEL"] = _stage_record(result["status"], started, result)
    failures = [name for name, stage in report["stages"].items() if stage.get("status") != "PASS"]
    report["failure_reasons"].extend(
        f"{name}: {stage.get('reason', stage.get('status'))}"
        for name, stage in report["stages"].items()
        if stage.get("status") in ("FAIL", "NOT_VERIFIED", "BLOCKED"))
    report["overall_status"] = "PASS" if not failures else (
        "FAIL" if any(report["stages"][name].get("status") == "FAIL" for name in failures) else "NOT_VERIFIED")
    report_dir = HERE / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = report_dir / f"acceptance_{args.stage}_{timestamp}.json"
    path.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({"overall_status": report["overall_status"], "report": str(path),
                      "stages": {name: stage.get("status") for name, stage in report["stages"].items()}},
                     ensure_ascii=False, indent=2))
    return 0 if report["overall_status"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("assets", "single", "wall", "short", "parallel"), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303])
    parser.add_argument("--max-rl-steps", type=int, default=10)
    parser.add_argument("--vessel-root", type=Path, default=None)
    args = parser.parse_args()
    if not args.seeds or args.max_rl_steps < 1 or args.max_rl_steps > 10:
        parser.error("--seeds must be nonempty and --max-rl-steps must be between 1 and 10")
    try:
        return execute(args)
    except Exception as exc:
        failure = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                   "stage_requested": args.stage, "requested_seeds": args.seeds,
                   "vessel_root": None if args.vessel_root is None else str(args.vessel_root),
                   "source_version": source_version(), "overall_status": "FAIL",
                   "failure_reasons": [f"{type(exc).__name__}: {exc}", traceback.format_exc()]}
        report_dir = HERE / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        path = report_dir / f"acceptance_{args.stage}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_fatal.json"
        path.write_text(json.dumps(failure, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"overall_status": "FAIL", "report": str(path),
                          "reason": failure["failure_reasons"][0]}, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
