"""Dense active-Beam geometry and clearance-only independent SDF audit.

Only a small number of near-penetration samples need gradient checks.  The
constraint planner and the independent clearance auditor never need a normal
for every 10-um sample, so avoid six whole-grid interpolation passes.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from mcr_sim.vessel_assets import sim_points_to_asset_source


def query_clearance_only(env, points):
    """Exact SDF clearance convention of the production Beam adapter.

    Do not replace non-finite grid samples with an invented finite clearance:
    the caller must handle out-of-grid states explicitly.
    """
    source = sim_points_to_asset_source(
        np.asarray(points, dtype=np.float64),
        env.asset_T_env_sim,
        env.asset_offset_sim,
        env.asset_source_to_sim_scale,
    )
    signed = np.asarray(env.sdf_grid.sample(source), dtype=np.float64)
    return -signed * float(env.asset_source_to_sim_scale) - float(env.catheter_radius)


def dense_check(env, q):
    geometry = env._beam_adapter._geometry()
    parts, tags, locations, bounds = [], [], [], []
    elements = geometry["active_elements"]
    tip_first = max(0, len(elements) - int(getattr(env.mcr_controller_sofa.instrument, "num_elem_tip", 3)))
    for k, element in enumerate(elements):
        a, b = q[np.asarray(element["nodes"], dtype=int)]
        length = float(element["rest_length_m"])
        p0, p3 = a[:3], b[:3]
        p1 = p0 + Rotation.from_quat(a[3:]).apply([length / 3, 0, 0])
        p2 = p3 + Rotation.from_quat(b[3:]).apply([-length / 3, 0, 0])
        linear = bool(np.dot(p1-p0, p2-p1) < 0 and np.linalg.norm(p3-p0) < 0.4 * length)
        bound = float(np.linalg.norm(p3-p0) if linear else
                      3 * max(np.linalg.norm(p1-p0), np.linalg.norm(p2-p1), np.linalg.norm(p3-p2)))
        n = max(1, int(np.ceil(max(length, bound) / 1e-5)))
        t = np.linspace(0, 1, n+1)
        u = 1-t
        pts = (u[:, None]*p0+t[:, None]*p3 if linear else
               u[:, None]**3*p0+3*(u*u*t)[:, None]*p1+
               3*(u*t*t)[:, None]*p2+t[:, None]**3*p3)
        parts.append(pts)
        tags.extend([k >= tip_first]*len(pts))
        locations.extend((k, float(v)) for v in t)
        bounds.append(bound/n)
    points = np.concatenate(parts)
    clearance = query_clearance_only(env, points)
    valid = np.isfinite(clearance)
    mask = np.asarray(tags, dtype=bool)
    negative = valid & (clearance < 0)
    invalid = int((~valid).sum())
    # The original full-grid audit spent six extra interpolations per point to
    # compute gradients that are not used by the contact/clearance solver.
    # Retain gradient validity checks only for genuinely penetrating samples.
    negative_invalid_normals = 0
    if np.any(negative):
        source = sim_points_to_asset_source(
            points[negative], env.asset_T_env_sim, env.asset_offset_sim,
            env.asset_source_to_sim_scale,
        )
        gradients = np.asarray(env.sdf_grid.gradient(source), dtype=np.float64)
        normals_valid = (np.isfinite(gradients).all(axis=1) &
                         (np.linalg.norm(gradients, axis=1) > 1e-9))
        negative_invalid_normals = int((~normals_valid).sum())
    worst = int(np.argmin(np.where(valid, clearance, np.inf))) if np.any(valid) else None
    # If any active Beam point is outside the SDF domain, the *whole-Beam*
    # clearance is unknown, not a fabricated large (or zero) penetration.
    minimum = float(clearance[worst]*1000) if invalid == 0 and worst is not None else None
    def part_min(part):
        subset = valid & part
        return float(clearance[subset].min()*1000) if invalid == 0 and subset.any() else None
    worst_element, worst_t = locations[worst] if worst is not None else (None, None)
    return {
        "clearance_mm": minimum,
        "body_clearance_mm": part_min(~mask),
        "tip_clearance_mm": part_min(mask),
        "negative_samples": int(negative.sum()),
        "body_negative_samples": int((negative & ~mask).sum()),
        "tip_negative_samples": int((negative & mask).sum()),
        "sample_count": int(len(clearance)),
        "sdf_out_of_grid": bool(invalid),
        "out_of_grid_samples": invalid,
        "invalid_normal_count": None,  # no longer evaluate unused gradients on every point
        "negative_invalid_normal_count": negative_invalid_normals,
        "max_spacing_bound_mm": float(max(bounds)*1000),
        "covered_active_elements": len(elements),
        "coverage": "all physically deployed Beam elements; inactive tail outside inlet excluded",
        "radius_subtracted_m": float(env.catheter_radius),
        "worst_element": elements[worst_element] if worst_element is not None else None,
        "worst_t": worst_t,
        "worst_point_m": points[worst].tolist() if worst is not None else None,
        "strict_nonpenetration": bool(minimum is not None and minimum >= 0),
        "aux_margin_0p100": bool(minimum is not None and minimum >= 0.100),
    }
