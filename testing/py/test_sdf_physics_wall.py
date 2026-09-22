from pathlib import Path

import numpy as np

from mcr_sim.paths import TRAIN_MESH_DIR
from mcr_sim.sdf_physics_wall import (
    compute_sdf_wall_forces,
    wall_force_magnitude,
)
from mcr_sim.vessel_assets import load_signed_distance_grid


def test_wall_force_law_matches_v15_2b_specification():
    clearances = np.asarray(
        [0.0003, 0.0002, 0.0001, 0.0, -0.0001, -0.0002, -0.0010]
    )
    expected_mn = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0])
    np.testing.assert_allclose(
        wall_force_magnitude(clearances) * 1000.0,
        expected_mn,
        rtol=0.0,
        atol=1e-12,
    )


def test_b02_near_wall_force_points_toward_more_negative_lumen_sdf():
    sdf_path = Path(TRAIN_MESH_DIR) / "B02" / "vessel_sdf.vti"
    grid = load_signed_distance_grid(sdf_path)
    values = np.asarray(grid.values, dtype=np.float64)

    # Select an interior voxel whose catheter-surface clearance lies inside the
    # 0.30 mm activation buffer for a 0.665 mm radius catheter.
    mask = (values > -0.95) & (values < -0.70)
    mask[[0, -1], :, :] = False
    mask[:, [0, -1], :] = False
    mask[:, :, [0, -1]] = False
    candidates = np.argwhere(mask)
    assert len(candidates) > 0

    chosen_source = None
    chosen_gradient = None
    for z, y, x in candidates:
        point = grid.origin + grid.spacing * np.asarray([x, y, z], dtype=float)
        gradient = np.asarray(grid.gradient(point), dtype=np.float64)
        if np.all(np.isfinite(gradient)) and np.linalg.norm(gradient) > 0.5:
            chosen_source = point
            chosen_gradient = gradient
            break
    assert chosen_source is not None

    outward_source = chosen_gradient / np.linalg.norm(chosen_gradient)
    inward_source = -outward_source
    inside_source = chosen_source + inward_source * (0.25 * np.min(grid.spacing))
    assert float(grid.sample(inside_source)) < float(grid.sample(chosen_source))

    scale = 0.001
    forces, clearance, inward, magnitude, valid = compute_sdf_wall_forces(
        chosen_source[None, :] * scale,
        sdf_grid=grid,
        asset_T_env_sim=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        asset_offset_sim=[0.0, 0.0, 0.0],
        asset_source_to_sim_scale=scale,
    )
    direction_to_lumen_inside = inside_source * scale - chosen_source * scale

    assert bool(valid[0])
    assert float(clearance[0]) < 0.0003
    assert 0.0 < float(magnitude[0]) <= 0.010
    assert float(np.dot(forces[0], direction_to_lumen_inside)) > 0.0
    assert float(np.dot(forces[0], inward[0])) > 0.0
