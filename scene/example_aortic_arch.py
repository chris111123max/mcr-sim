from pathlib import Path
import os
import sys

# This file lives in python/scene/.  Resolve the Git/Python root from the scene
# file itself so package imports do not depend on the caller's working directory.
SCENE_DIR = Path(__file__).resolve().parent
PYTHON_ROOT = SCENE_DIR.parent

if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

import numpy as np
from splib3.numerics import Quat, Vec3
from scipy.spatial.transform import Rotation as R

from mcr_sim import (
    mcr_centerline,
    mcr_controller_sofa,
    mcr_emns,
    mcr_environment,
    mcr_instrument,
    mcr_magnet,
    mcr_simulator,
    sdf_physics_wall,
)
from mcr_sim.paths import DEFAULT_CALIBRATION_PATH, TEST_MESH_DIR, TRAIN_MESH_DIR
from mcr_sim.training_config import (
    ENTRY_TANGENT_POINTS,
    FRICTION_COEFFICIENT,
    INITIAL_ORIENTATION_MAX_ANGLE_DEG,
    SOFA_TIME_STEP_S,
    START_WINDOW_DISTANCE_M,
    TARGET_WINDOW_DISTANCE_M,
    VESSEL_TRIANGLE_PROXIMITY_M,
)

# ============================================================
# Paths
# ============================================================
# Calibration file for eMNS
cal_path = str(DEFAULT_CALIBRATION_PATH)

# ============================================================
# Parameters instrument, magnet, beams
# ============================================================
young_modulus_body = 100e6 #原值170e6
young_modulus_tip = 21e6   # 原来 21e6；降低尖端杨氏模量，让 tip 更软、更容易转弯
length_body = 0.5           # (m)
length_tip = 0.034        # 原来 0.0034 m；
outer_diam = 0.00133        # (m)
inner_diam = 0.0008         # (m)

length_init = 0.35

# Fast GUI/testing version. Original visualization value was 600.
nume_nodes_viz = 600
num_elem_body = 30
num_elem_tip = 3

# ============================================================
# Transforms
# ============================================================
T_sim_mns = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

rot_env_sim = [-0.7071068, 0, 0, 0.7071068]
transl_env_sim = [0.0, -0.45, 0.0]
T_env_sim = [transl_env_sim[0], transl_env_sim[1], transl_env_sim[2], -0.7071068, 0, 0, 0.7071068]

# ============================================================
# Initial mCR pose
# ============================================================
# ROS 版本原始 baseline：
#   T_start_env = [-0.075, -0.001, -0.020, 0.0, -0.3826834, 0.0, 0.9238795]
#
# 当前策略：
#   1. 先用 baseline 作为初始姿态基准；
#   2. 读取当前血管中心线 start_point / target_point；
#   3. 对指定模型，把 local +X 轴对齐入口中心线切线方向；
#   4. 再把初始平移覆盖到当前中心线 start_point。
T_start_env = [-0.075, -0.001, -0.020, 0.0, -0.3826834, 0.0, 0.9238795]


def _normalize_vec(vec, eps=1e-9):
    vec = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return None
    return vec / norm


def _rotation_between_vectors(source_vec, target_vec):
    """Minimal rotation that maps source_vec to target_vec."""
    source = _normalize_vec(source_vec)
    target = _normalize_vec(target_vec)
    if source is None or target is None:
        return R.identity()

    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))

    if dot > 1.0 - 1e-8:
        return R.identity()

    if dot < -1.0 + 1e-8:
        axis = np.cross(source, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-8:
            axis = np.cross(source, [0.0, 1.0, 0.0])
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        return R.from_rotvec(np.pi * axis)

    axis = np.cross(source, target)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-12:
        return R.identity()

    axis = axis / axis_norm
    angle = float(np.arccos(dot))
    return R.from_rotvec(angle * axis)


def align_start_pose_minus_y_to_centerline_tangent(t_start_env_value, entry_tangent_sim):
    """把 mcR 初始姿态的 local +X 轴对齐中心线入口切线。

    注意：函数名沿用旧名 minus_y，但当前实际对齐轴是 local +X：
        local_forward_axis = [1, 0, 0]
    这是根据当前调试结果保留的设置。
    """
    entry_tangent_sim = _normalize_vec(entry_tangent_sim)
    if entry_tangent_sim is None:
        return {"ok": False, "reason": "degenerate_entry_tangent"}

    r_env_to_sim = R.from_quat(rot_env_sim)
    r_env_current = R.from_quat(t_start_env_value[3:7])
    r_sim_current = r_env_to_sim * r_env_current

    # 当前认为 mcR 的真实前进方向为 local +X。
    local_forward_axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    current_forward_sim = r_sim_current.apply(local_forward_axis)

    before_dot = float(np.dot(_normalize_vec(current_forward_sim), entry_tangent_sim))
    delta_sim = _rotation_between_vectors(current_forward_sim, entry_tangent_sim)
    r_sim_new = delta_sim * r_sim_current
    new_forward_sim = r_sim_new.apply(local_forward_axis)
    after_dot = float(np.dot(_normalize_vec(new_forward_sim), entry_tangent_sim))

    r_env_new = r_env_to_sim.inv() * r_sim_new
    q_env_new = r_env_new.as_quat()

    t_start_env_value[3] = float(q_env_new[0])
    t_start_env_value[4] = float(q_env_new[1])
    t_start_env_value[5] = float(q_env_new[2])
    t_start_env_value[6] = float(q_env_new[3])

    return {
        "ok": True,
        "before_dot": before_dot,
        "after_dot": after_dot,
        "old_forward_sim": current_forward_sim,
        "new_forward_sim": new_forward_sim,
        "entry_tangent_sim": entry_tangent_sim,
        "q_env_new": q_env_new,
    }


def build_t_start_sim_from_env(t_start_env_value):
    trans_start_env = Vec3(t_start_env_value[0], t_start_env_value[1], t_start_env_value[2])
    r = R.from_quat(rot_env_sim)
    trans_start_env = r.apply(trans_start_env)

    quat_start = Quat(rot_env_sim)
    qrot = Quat(t_start_env_value[3], t_start_env_value[4], t_start_env_value[5], t_start_env_value[6])
    quat_start.rotateFromQuat(qrot)

    return [
        trans_start_env[0] + transl_env_sim[0],
        trans_start_env[1] + transl_env_sim[1],
        trans_start_env[2] + transl_env_sim[2],
        quat_start[0],
        quat_start[1],
        quat_start[2],
        quat_start[3],
    ]


def resolve_training_task(kwargs):
    """Select mesh/STL and centerline VTK for non-ROS SOFA GUI testing.

    Extra Y-vessel support:
      - Y001 / Y002 / Y003 still work as before.
      - Y003_left / Y003_right also work, while reusing mesh/test/Y003.
      - If a Y folder contains files such as
            V003_left_Centerline model.vtk
            V003_right_Centerline model.vtk
        they are found automatically.
      - You can also force a centerline filename by:
            MCR_CENTERLINE_FILE="V003_left_Centerline model.vtk"
    """
    import random

    artificial_models = [f"C{i:02d}" for i in range(1, 6)] + [
        f"B{i:02d}" for i in range(1, 6)
    ]
    random_models = list(artificial_models)

    supported_models = [
        "0207",
        "0210",
        "V1",
        "0207_left",
        "0207_right",
        "0021",
        "0028",
        "0038",
        "0230",
        "0231",
        "0237",
        "aorta6",
        "aorta7",
        "S1",
        "Y001",
        "Y002",
        "Y003",
        "Y001_left",
        "Y001_right",
        "Y002_left",
        "Y002_right",
        "Y003_left",
        "Y003_right",
    ] + artificial_models

    configured_asset_root = kwargs.get("asset_root")
    asset_roots = (
        [Path(configured_asset_root).expanduser().resolve()]
        if configured_asset_root
        else [TRAIN_MESH_DIR, TEST_MESH_DIR]
    )

    def _resolve_asset(model_name, stl_candidates, centerline_candidates):
        model_dirs = [base_dir / model_name for base_dir in asset_roots]

        environment_stl = None
        centerline_vtk = None

        # Only VTK centerline files are supported here.  Do not auto-detect VTP.
        centerline_candidates = [
            name for name in centerline_candidates
            if str(name).lower().endswith(".vtk")
        ]

        for model_dir in model_dirs:
            for stl_name in stl_candidates:
                path = model_dir / stl_name
                if path.is_file():
                    environment_stl = path
                    break
            if environment_stl is not None:
                break

        for model_dir in model_dirs:
            for vtk_name in centerline_candidates:
                path = model_dir / vtk_name
                if path.is_file():
                    centerline_vtk = path
                    break
            if centerline_vtk is not None:
                break

        if environment_stl is None:
            raise FileNotFoundError(
                f"No STL found for model={model_name}. Tried: {stl_candidates} "
                f"under {[str(path) for path in asset_roots]}"
            )
        if centerline_vtk is None:
            raise FileNotFoundError(
                f"No centerline VTK found for model={model_name}. "
                f"Tried: {centerline_candidates} under {[str(path) for path in asset_roots]}"
            )

        return environment_stl, centerline_vtk

    def _y_base_and_branch(model_name):
        """Map Y003_left/Y003_right to base folder Y003 + branch name."""
        name = str(model_name)
        for suffix in ("_left", "_right"):
            if name.endswith(suffix):
                return name[: -len(suffix)], suffix[1:]
        return name, None

    def _y_centerline_candidates(base_model, branch=None):
        """Candidate centerline names for manually split Y-type centerlines."""
        base_model = str(base_model)
        v_code = base_model.replace("Y", "V", 1) if base_model.startswith("Y") else base_model

        # Highest priority: manually selected file name.
        forced_file = str(
            kwargs.get(
                "centerline_file",
                os.environ.get("MCR_CENTERLINE_FILE", ""),
            )
            or ""
        ).strip()

        candidates = []
        if forced_file:
            if forced_file.lower().endswith(".vtk"):
                candidates.append(forced_file)
            else:
                print(f"[WARN] Ignoring non-VTK centerline file: {forced_file}")

        # If user explicitly asks Y003_left / Y003_right, prefer that branch.
        if branch in ("left", "right"):
            candidates.extend(
                [
                    f"{v_code}_{branch}_Centerline model.vtk",
                    f"{base_model}_{branch}_Centerline model.vtk",
                    f"{v_code}_{branch}_Centerline.vtk",
                    f"{base_model}_{branch}_Centerline.vtk",
                    f"{v_code}_{branch}_centerline.vtk",
                    f"{base_model}_{branch}_centerline.vtk",
                    f"{branch}_Centerline model.vtk",
                    f"{branch}_Centerline.vtk",
                    f"{branch}_centerline.vtk",
                ]
            )

        # Generic names and then common manually-created branch names.
        candidates.extend(
            [
                "Centerline model.vtk",
                f"{v_code}_left_Centerline model.vtk",
                f"{v_code}_right_Centerline model.vtk",
                f"{base_model}_left_Centerline model.vtk",
                f"{base_model}_right_Centerline model.vtk",
                "left_Centerline model.vtk",
                "right_Centerline model.vtk",
                "left_centerline.vtk",
                "right_centerline.vtk",
            ]
        )

        # De-duplicate while preserving priority.
        out = []
        seen = set()
        for item in candidates:
            if item and item not in seen:
                seen.add(item)
                out.append(item)
        return out

    def _generic_centerline_candidates(model_name):
        forced_file = str(
            kwargs.get("centerline_file", os.environ.get("MCR_CENTERLINE_FILE", "")) or ""
        ).strip()
        candidates = []
        if forced_file:
            if forced_file.lower().endswith(".vtk"):
                candidates.append(forced_file)
            else:
                print(f"[WARN] Ignoring non-VTK centerline file: {forced_file}")

        # Branching vessels expose six root-to-outlet tasks.  The environment
        # normally supplies an explicit centerline_file from its balanced route
        # sampler.  Keep this fallback for GUI/legacy callers only.
        if str(model_name).startswith("B"):
            target_names = [f"target_{i:02d}_centerline.vtk" for i in range(1, 7)]
            random.shuffle(target_names)
            candidates.extend(target_names)
        candidates.extend(
            [
                "Centerline model.vtk",
                "centerline.vtk",
                f"{model_name}_centerline.vtk",
                f"{model_name}_Centerline model.vtk",
            ]
        )
        return candidates

    chosen_model = kwargs.get("force_model", os.environ.get("MCR_FORCE_MODEL", ""))
    if chosen_model in (None, ""):
        chosen_model = random.choice(random_models)
    chosen_model = str(chosen_model)

    y_base_model, y_branch = _y_base_and_branch(chosen_model)
    chosen_dir_exists = any((base_dir / chosen_model).is_dir() for base_dir in asset_roots)
    y_base_dir_exists = chosen_model.startswith("Y") and any(
        (base_dir / y_base_model).is_dir() for base_dir in asset_roots
    )

    if chosen_model not in supported_models and not chosen_dir_exists and not y_base_dir_exists:
        print(f"[WARN] Unsupported force_model={chosen_model}, fallback to random {random_models}.")
        chosen_model = random.choice(random_models)
        y_base_model, y_branch = _y_base_and_branch(chosen_model)

    if chosen_model == "0207":
        environment_stl = TRAIN_MESH_DIR / "0207" / "0207.stl"
        centerlines = [
            TRAIN_MESH_DIR / "0207" / "0207_left_centerline.vtk",
            TRAIN_MESH_DIR / "0207" / "0207_right_centerline.vtk",
        ]
        centerline_vtk = random.choice(centerlines)
    elif chosen_model == "0207_left":
        environment_stl = TRAIN_MESH_DIR / "0207" / "0207.stl"
        centerline_vtk = TRAIN_MESH_DIR / "0207" / "0207_left_centerline.vtk"
    elif chosen_model == "0207_right":
        environment_stl = TRAIN_MESH_DIR / "0207" / "0207.stl"
        centerline_vtk = TRAIN_MESH_DIR / "0207" / "0207_right_centerline.vtk"
    elif chosen_model == "V1":
        environment_stl, centerline_vtk = _resolve_asset(
            "V1",
            stl_candidates=["J2-Naviworks.stl", "Segmentation.stl"],
            centerline_candidates=["Centerline model.vtk", "Centerline mode_未外延.vtk"],
        )
    elif chosen_model == "0210":
        environment_stl, centerline_vtk = _resolve_asset(
            "0210",
            stl_candidates=["0210.stl", "Segmentation.stl"],
            centerline_candidates=["Centerline model.vtk", "Centerline model未外延.vtk"],
        )
    elif chosen_model == "0021":
        environment_stl, centerline_vtk = _resolve_asset(
            "0021",
            stl_candidates=["Segmentation.stl", "0021.stl"],
            centerline_candidates=["Centerline model.vtk"],
        )
    elif chosen_model == "0028":
        environment_stl, centerline_vtk = _resolve_asset(
            "0028",
            stl_candidates=["Segmentation.stl", "0028.stl"],
            centerline_candidates=["Centerline model.vtk"],
        )
    elif chosen_model == "0038":
        environment_stl, centerline_vtk = _resolve_asset(
            "0038",
            stl_candidates=["Segmentation.stl", "0038.stl"],
            centerline_candidates=["Centerline model.vtk"],
        )
    elif chosen_model == "0230":
        environment_stl, centerline_vtk = _resolve_asset(
            "0230",
            stl_candidates=["Segmentation.stl", "0230.stl"],
            centerline_candidates=["Centerline model.vtk"],
        )
    elif chosen_model == "0231":
        environment_stl, centerline_vtk = _resolve_asset(
            "0231",
            stl_candidates=["Segmentation.stl", "0231.stl"],
            centerline_candidates=["Centerline model.vtk"],
        )
    elif chosen_model == "0237":
        environment_stl, centerline_vtk = _resolve_asset(
            "0237",
            stl_candidates=["Segmentation.stl", "0237.stl"],
            centerline_candidates=["Centerline model.vtk"],
        )
    elif chosen_model == "aorta6":
        environment_stl, centerline_vtk = _resolve_asset(
            "aorta6",
            stl_candidates=["Segmentation.stl", "aorta6.stl"],
            centerline_candidates=[
                "Centerline model.vtk",
                "centerline.vtk",
                "aorta6_centerline.vtk",
                "aorta6_Centerline model.vtk",
            ],
        )
    elif chosen_model == "aorta7":
        environment_stl, centerline_vtk = _resolve_asset(
            "aorta7",
            stl_candidates=["Segmentation.stl", "aorta7.stl"],
            centerline_candidates=[
                "Centerline model.vtk",
                "centerline.vtk",
                "aorta7_centerline.vtk",
                "aorta7_Centerline model.vtk",
            ],
        )

    elif chosen_model == "S1":
        environment_stl, centerline_vtk = _resolve_asset(
            "S1",
            stl_candidates=["Segmentation.stl", "S1.stl"],
            centerline_candidates=[
                "Centerline model.vtk",
                "centerline.vtk",
                "S1_centerline.vtk",
                "S1_Centerline model.vtk",
            ],
        )

    elif chosen_model.startswith("Y"):
        # For Y003_left/Y003_right, use the base folder Y003.
        # For Y003, use folder Y003 directly.
        environment_stl, centerline_vtk = _resolve_asset(
            y_base_model,
            stl_candidates=[
                "Segmentation.stl",
                f"{y_base_model}.stl",
                f"{y_base_model.replace('Y', 'V', 1)}.stl",
            ],
            centerline_candidates=_y_centerline_candidates(y_base_model, branch=y_branch),
        )
    elif chosen_dir_exists:
        # Generic test/train folder support.
        # This enables folders such as:
        #   mesh/test/aorta1/Segmentation.stl
        #   mesh/test/aorta1/Centerline model.vtk
        environment_stl, centerline_vtk = _resolve_asset(
            chosen_model,
            stl_candidates=[
                "Segmentation.stl",
                f"{chosen_model}.stl",
            ],
            centerline_candidates=_generic_centerline_candidates(chosen_model),
        )
    else:  # V1
        environment_stl, centerline_vtk = _resolve_asset(
            "V1",
            stl_candidates=["J2-Naviworks.stl", "Segmentation.stl"],
            centerline_candidates=["Centerline model.vtk", "Centerline mode_未外延.vtk"],
        )

    centerline_str = str(centerline_vtk)
    centerline_name = Path(centerline_vtk).stem.lower()
    target_route_id = "default"
    if centerline_name.startswith("target_") and centerline_name.endswith("_centerline"):
        target_route_id = centerline_name[: -len("_centerline")]
    if "0207_left" in centerline_str:
        task_id = "0207_left"
    elif "0207_right" in centerline_str:
        task_id = "0207_right"
    else:
        task_id = str(chosen_model)

    visual_stl_path = Path(environment_stl).parent / "visual_wall.stl"
    visual_stl = str(visual_stl_path) if visual_stl_path.is_file() else None

    asset_dir = Path(environment_stl).parent
    sdf_vti_path = asset_dir / "vessel_sdf.vti"
    metadata_json_path = asset_dir / "metadata.json"
    centerline_graph_path = asset_dir / "centerline_graph.vtk"
    is_generated_artificial = (
        len(str(chosen_model)) == 3
        and str(chosen_model)[0].upper() in ("B", "C")
        and str(chosen_model)[1:].isdigit()
    )
    if is_generated_artificial:
        required_assets = [sdf_vti_path, metadata_json_path]
        if str(chosen_model).upper().startswith("B"):
            required_assets.append(centerline_graph_path)
        missing_assets = [str(path) for path in required_assets if not path.is_file()]
        if missing_assets:
            raise FileNotFoundError(
                "Generated vessel is missing required multi-model assets: "
                + ", ".join(missing_assets)
            )

    return {
        "chosen_model": chosen_model,
        "environment_stl": str(environment_stl),
        "visual_stl": visual_stl,
        "centerline_vtk": str(centerline_vtk),
        "centerline_graph_vtk": (
            str(centerline_graph_path) if centerline_graph_path.is_file() else None
        ),
        "sdf_vti": str(sdf_vti_path) if sdf_vti_path.is_file() else None,
        "metadata_json": (
            str(metadata_json_path) if metadata_json_path.is_file() else None
        ),
        "task_id": task_id,
        "target_route_id": target_route_id,
    }


def _sample_uniform_ball(radius, rng=None):
    """Sample a 3-D offset uniformly inside a ball with the given radius."""
    rng = rng if rng is not None else np.random.default_rng()
    direction = rng.normal(size=3)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        direction = direction / norm
    r = float(radius) * float(rng.random() ** (1.0 / 3.0))
    return (r * direction).astype(np.float64)


def _sample_direction_within_cone(base_dir, max_angle_rad, rng=None):
    """Return a unit vector whose angular deviation from base_dir is <= max_angle_rad."""
    rng = rng if rng is not None else np.random.default_rng()
    base = _normalize_vec(base_dir)
    if base is None:
        return None

    # Build a random perpendicular direction.
    rand = rng.normal(size=3)
    perp = rand - float(np.dot(rand, base)) * base
    perp_norm = float(np.linalg.norm(perp))
    if perp_norm < 1e-12:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(ref, base))) > 0.9:
            ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        perp = ref - float(np.dot(ref, base)) * base
        perp_norm = float(np.linalg.norm(perp))
    perp = perp / (perp_norm + 1e-12)

    # Uniform-ish over the cone cap; exact uniformity is not critical for domain randomization.
    max_angle_rad = max(0.0, float(max_angle_rad))
    cos_min = float(np.cos(max_angle_rad))
    cos_theta = float(rng.uniform(cos_min, 1.0))
    sin_theta = float(np.sqrt(max(0.0, 1.0 - cos_theta * cos_theta)))
    out = cos_theta * base + sin_theta * perp
    return out / (float(np.linalg.norm(out)) + 1e-12)


def _sample_endpoint_window_start_target(
    centerline_data,
    nominal_start_point_sim,
    nominal_target_point_sim,
    start_window_distance_m=START_WINDOW_DISTANCE_M,
    target_window_distance_m=TARGET_WINDOW_DISTANCE_M,
    rng=None,
):
    """Sample start/target from fixed arc-length endpoint windows.

    Scheme-1 initialization:
      - start is sampled inside ``start_window_distance_m`` from the entry;
      - target is sampled inside ``target_window_distance_m`` from the outlet;
      - no off-center ball perturbation is added.

    The helper determines the centerline direction from the nominal start/target,
    so it also works if a VTK happens to be stored in the reverse order.
    """
    rng = rng if rng is not None else np.random.default_rng()
    pts = np.asarray(centerline_data.points_sim, dtype=np.float64)

    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 2:
        raise ValueError("Centerline points must be an Nx3 array with at least two points")

    n = int(pts.shape[0])
    start_window_distance_m = max(0.0, float(start_window_distance_m))
    target_window_distance_m = max(0.0, float(target_window_distance_m))
    cumulative = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    )
    total_length = float(cumulative[-1])

    nominal_start = np.asarray(nominal_start_point_sim, dtype=np.float64).reshape(3)
    nominal_target = np.asarray(nominal_target_point_sim, dtype=np.float64).reshape(3)

    # Decide which end of the stored centerline is the entry side.
    forward_cost = float(np.linalg.norm(pts[0] - nominal_start) + np.linalg.norm(pts[-1] - nominal_target))
    reverse_cost = float(np.linalg.norm(pts[-1] - nominal_start) + np.linalg.norm(pts[0] - nominal_target))
    stored_forward = bool(forward_cost <= reverse_cost)

    if stored_forward:
        start_arc = cumulative
        target_arc = total_length - cumulative
    else:
        start_arc = total_length - cumulative
        target_arc = cumulative

    start_candidates = np.flatnonzero(
        start_arc <= start_window_distance_m + 1e-12
    ).astype(np.int64)
    target_candidates = np.flatnonzero(
        target_arc <= target_window_distance_m + 1e-12
    ).astype(np.int64)
    if start_candidates.size == 0:
        start_candidates = np.asarray([0 if stored_forward else n - 1], dtype=np.int64)
    if target_candidates.size == 0:
        target_candidates = np.asarray([n - 1 if stored_forward else 0], dtype=np.int64)

    start_idx = int(rng.choice(start_candidates))
    target_idx = int(rng.choice(target_candidates))

    # If an extremely short centerline makes the two windows overlap, force a
    # valid start-before-target ordering in the chosen traversal direction.
    if stored_forward and target_idx <= start_idx and n > 1:
        target_idx = n - 1
    if (not stored_forward) and target_idx >= start_idx and n > 1:
        target_idx = 0

    start_point_sim = pts[start_idx].copy()
    target_point_sim = pts[target_idx].copy()

    return {
        "start_point_sim": start_point_sim,
        "target_point_sim": target_point_sim,
        "start_idx": start_idx,
        "target_idx": target_idx,
        "stored_forward": stored_forward,
        "start_candidates": start_candidates.tolist(),
        "target_candidates": target_candidates.tolist(),
        "start_window_distance_m": start_window_distance_m,
        "target_window_distance_m": target_window_distance_m,
    }


def _get_centerline_shortest_path_indices(
    centerline_data,
    start_index,
    target_index,
):
    """Return centerline indices along the shortest topology path."""
    import heapq

    pts = np.asarray(centerline_data.points_sim, dtype=np.float64)
    edges = list(getattr(centerline_data, "edges", []) or [])
    n = int(len(pts))

    start_index = int(start_index)
    target_index = int(target_index)

    if not (0 <= start_index < n and 0 <= target_index < n):
        raise ValueError("Centerline start/target index is out of range")

    if start_index == target_index:
        return [start_index]

    adjacency = [[] for _ in range(n)]

    for edge in edges:
        if len(edge) < 2:
            continue

        a = int(edge[0])
        b = int(edge[1])

        if not (0 <= a < n and 0 <= b < n):
            continue

        weight = float(np.linalg.norm(pts[b] - pts[a]))
        adjacency[a].append((b, weight))
        adjacency[b].append((a, weight))

    # No usable topology: fall back to point-array order.
    if not any(adjacency):
        direction = 1 if target_index >= start_index else -1
        return list(range(
            start_index,
            target_index + direction,
            direction,
        ))

    distances = [float("inf")] * n
    previous = [-1] * n

    distances[start_index] = 0.0
    queue = [(0.0, start_index)]

    while queue:
        dist_u, u = heapq.heappop(queue)

        if dist_u != distances[u]:
            continue

        if u == target_index:
            break

        for v, weight in adjacency[u]:
            new_dist = dist_u + weight

            if new_dist < distances[v]:
                distances[v] = new_dist
                previous[v] = u
                heapq.heappush(queue, (new_dist, v))

    if not np.isfinite(distances[target_index]):
        direction = 1 if target_index >= start_index else -1
        return list(range(
            start_index,
            target_index + direction,
            direction,
        ))

    path_indices = []
    current = target_index

    while current >= 0:
        path_indices.append(current)

        if current == start_index:
            break

        current = previous[current]

    if not path_indices or path_indices[-1] != start_index:
        direction = 1 if target_index >= start_index else -1
        return list(range(
            start_index,
            target_index + direction,
            direction,
        ))

    path_indices.reverse()
    return path_indices


def _select_centerline_midpoint_between_endpoints(
    centerline_data,
    start_index,
    target_index,
):
    """Select the existing centerline point nearest 50% route arc length."""
    pts = np.asarray(centerline_data.points_sim, dtype=np.float64)

    path_indices = _get_centerline_shortest_path_indices(
        centerline_data=centerline_data,
        start_index=start_index,
        target_index=target_index,
    )

    path_points = pts[np.asarray(path_indices, dtype=np.int64)]

    if len(path_indices) < 2:
        cumulative = np.asarray([0.0], dtype=np.float64)
        midpoint_local_idx = 0
    else:
        segment_lengths = np.linalg.norm(
            np.diff(path_points, axis=0),
            axis=1,
        )
        cumulative = np.concatenate(
            ([0.0], np.cumsum(segment_lengths))
        )

        half_length = 0.5 * float(cumulative[-1])
        midpoint_local_idx = int(
            np.argmin(np.abs(cumulative - half_length))
        )

    midpoint_index = int(path_indices[midpoint_local_idx])

    return {
        "point": pts[midpoint_index].copy(),
        "index": midpoint_index,
        "path_indices": path_indices,
        "path_length": float(cumulative[-1]),
        "midpoint_progress": float(cumulative[midpoint_local_idx]),
    }


def _compute_entry_tangent_toward_target(
    centerline_data,
    start_point_sim,
    target_point_sim,
    tangent_step=5,
):
    """Compute the centerline tangent from start toward target."""
    pts = np.asarray(centerline_data.points_sim, dtype=np.float64)

    start_np = np.asarray(
        start_point_sim,
        dtype=np.float64,
    ).reshape(3)

    target_np = np.asarray(
        target_point_sim,
        dtype=np.float64,
    ).reshape(3)

    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
        raise ValueError(
            "Centerline points must be an Nx3 array "
            "with at least two points"
        )

    start_idx = int(np.argmin(
        np.linalg.norm(pts - start_np[None, :], axis=1)
    ))

    target_idx = int(np.argmin(
        np.linalg.norm(pts - target_np[None, :], axis=1)
    ))

    path_indices = _get_centerline_shortest_path_indices(
        centerline_data=centerline_data,
        start_index=start_idx,
        target_index=target_idx,
    )

    if len(path_indices) < 2:
        raise ValueError(
            "Could not find a valid centerline path "
            "from start to target"
        )

    step = int(np.clip(
        int(tangent_step),
        1,
        len(path_indices) - 1,
    ))

    next_idx = int(path_indices[step])
    tangent = pts[next_idx] - pts[start_idx]
    tangent_norm = float(np.linalg.norm(tangent))

    if tangent_norm < 1e-9:
        next_idx = int(path_indices[1])
        tangent = pts[next_idx] - pts[start_idx]
        tangent_norm = float(np.linalg.norm(tangent))

    if tangent_norm < 1e-9:
        raise ValueError("Degenerate centerline tangent toward target")

    return tangent / tangent_norm, start_idx, next_idx



def _compute_entry_tangent_from_centerline(centerline_data, start_point_sim, tangent_step=5):
    """Compute the local entry tangent from the selected randomized start point.

    For scheme-1 endpoint-window randomization, the start point may be any one of
    the first several centerline points instead of exactly pts[0]. Therefore the
    tangent is computed from the nearest selected start index toward the target
    side, not always from the absolute endpoint.
    """
    pts = np.asarray(centerline_data.points_sim, dtype=np.float64)
    start_np = np.asarray(start_point_sim, dtype=np.float64).reshape(3)

    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("Centerline points must be an Nx3 array")
    if pts.shape[0] < 2:
        raise ValueError("Centerline must contain at least two points")

    n = int(pts.shape[0])
    step = int(np.clip(int(tangent_step), 1, n - 1))

    nearest_idx = int(np.argmin(np.linalg.norm(pts - start_np[None, :], axis=1)))

    # Decide whether this start is closer to the first or last end. The tangent
    # should point from the selected start point into the vessel.
    dist_to_first = float(np.linalg.norm(pts[0] - start_np))
    dist_to_last = float(np.linalg.norm(pts[-1] - start_np))

    start_idx = nearest_idx
    if dist_to_first <= dist_to_last:
        next_idx = int(min(start_idx + step, n - 1))
        if next_idx == start_idx:
            next_idx = int(min(start_idx + 1, n - 1))
    else:
        next_idx = int(max(start_idx - step, 0))
        if next_idx == start_idx:
            next_idx = int(max(start_idx - 1, 0))

    entry_tangent = pts[next_idx] - pts[start_idx]
    norm = float(np.linalg.norm(entry_tangent))
    if norm < 1e-9:
        raise ValueError("Degenerate entry tangent: selected points are too close")

    entry_tangent = entry_tangent / norm
    return entry_tangent, start_idx, next_idx

def createScene(root_node, image_shape=None, debug_rendering=True, positioning_camera=False, **kwargs):
    """Build SOFA scene using non-ROS modules with ROS-version initial pose."""

    # Scene construction used to print large pose arrays and mesh diagnostics
    # on every reset.  Keep those diagnostics available for GUI/preflight work,
    # while headless training stays quiet unless explicitly requested.
    import builtins
    scene_verbose = bool(
        kwargs.get("verbose_scene", bool(debug_rendering or positioning_camera))
    )

    def print(*values, **print_kwargs):
        if scene_verbose:
            builtins.print(*values, **print_kwargs)

    task_cfg = resolve_training_task(kwargs)
    chosen_model = task_cfg["chosen_model"]
    environment_stl = task_cfg["environment_stl"]
    visual_stl = task_cfg.get("visual_stl")
    centerline_vtk = task_cfg["centerline_vtk"]
    centerline_graph_vtk = task_cfg.get("centerline_graph_vtk")
    sdf_vti = task_cfg.get("sdf_vti")
    metadata_json = task_cfg.get("metadata_json")
    task_id = task_cfg["task_id"]
    target_route_id = task_cfg.get("target_route_id", "default")

    print("[example_aortic_arch_nonros] chosen_model    =", chosen_model)
    print("[example_aortic_arch_nonros] task_id         =", task_id)
    print("[example_aortic_arch_nonros] environment_stl =", environment_stl)
    print("[example_aortic_arch_nonros] visual_stl      =", visual_stl)
    print("[example_aortic_arch_nonros] centerline_vtk  =", centerline_vtk)
    print("[example_aortic_arch_nonros] centerline_graph=", centerline_graph_vtk)
    print("[example_aortic_arch_nonros] vessel_sdf.vti  =", sdf_vti)
    print("[example_aortic_arch_nonros] metadata.json   =", metadata_json)

    if not Path(environment_stl).is_file():
        raise FileNotFoundError(f"Vessel STL not found: {environment_stl}")
    if not Path(centerline_vtk).is_file():
        raise FileNotFoundError(f"Centerline VTK not found: {centerline_vtk}")

    # When enabled, single-vessel start/target/orientation randomization is handled
    # by MCREnv.reset() inside the already-created SOFA scene. createScene then keeps
    # a nominal pose so the same vessel STL/collision scene does not need to be
    # unloaded/reloaded every episode.
    soft_randomize_single_vessel = bool(kwargs.get("soft_randomize_single_vessel", False))

    centerline_point_frame = "env"

    # Default scale for original/train vessels.
    centerline_scale = 0.0004

    # Vessels loaded from mesh/test are enlarged before entering SOFA.
    # Default test scale: 0.003.
    # Test aorta vessels use 0.005, except aorta1 which remains 0.003.
    # The same scale is used for both STL and centerline/radius, so they stay aligned.
    try:
        environment_stl_path = Path(environment_stl).resolve()
        test_mesh_root = TEST_MESH_DIR.resolve()
        is_test_mesh = (environment_stl_path == test_mesh_root) or (test_mesh_root in environment_stl_path.parents)
    except Exception:
        is_test_mesh = str(environment_stl).startswith(str(TEST_MESH_DIR))

    task_name_lower = str(task_id).lower()
    is_artificial_model = (
        len(task_name_lower) == 3
        and task_name_lower[0] in ("b", "c", "v")
        and task_name_lower[1:].isdigit()
    )
    if is_artificial_model:
        # Artificial generator source units are millimetres.
        centerline_scale = 0.001
    elif task_name_lower == "s1":
        centerline_scale = 0.002
    elif is_test_mesh and task_name_lower.startswith("y"):
        centerline_scale = 0.002
    elif is_test_mesh and task_name_lower in ("aorta6", "aorta7"):
        centerline_scale = 0.0025
    elif is_test_mesh and task_name_lower == "aorta3":
        centerline_scale = 0.01
    elif is_test_mesh and task_name_lower.startswith("aorta") and task_name_lower != "aorta1":
        centerline_scale = 0.008
    elif is_test_mesh:
        centerline_scale = 0.003

    # Scale the STL, centerline coordinates, and centerline radii together.
    # A fixed factor is used by the GUI preflight viewer; training supplies a
    # bounded range and samples once whenever the SOFA scene is constructed.
    fixed_scale_factor = kwargs.get("vessel_scale_factor", None)
    if fixed_scale_factor is not None:
        vessel_scale_factor = float(fixed_scale_factor)
    else:
        # Generic/test scenes remain nominal unless training explicitly passes
        # the shrink-only randomization range.
        scale_min = float(kwargs.get("vessel_scale_min", 1.0))
        scale_max = float(kwargs.get("vessel_scale_max", 1.0))
        if not (0.5 <= scale_min <= scale_max <= 1.0):
            raise ValueError(
                "vessel_scale_min/max must satisfy 0.5 <= min <= max <= 1.0"
            )
        vessel_scale_factor = float(np.random.uniform(scale_min, scale_max))
    if not (0.5 <= vessel_scale_factor <= 1.0):
        raise ValueError("vessel_scale_factor must be between 0.5 and 1.0")
    centerline_scale *= vessel_scale_factor

    print(
        "[MODEL_SCALE]",
        "task_id=", task_id,
        "is_test_mesh=", bool(is_test_mesh),
        "centerline_scale=", centerline_scale,
        "vessel_scale_factor=", vessel_scale_factor,
    )

    centerline_offset_sim = [0.0, 0.0, 0.0]

    # Parameter magnet (aligned with sofa_env)
    magnet_length = 4e-3
    magnet_id = 0.86e-3
    magnet_od = 1.33e-3
    magnet_remanence = 1.45

    if image_shape is None:
        viewport_height, viewport_width = 600, 800
    else:
        viewport_height = int(image_shape[0])
        viewport_width = int(image_shape[1])

    # Only create heavy visual pipeline in debug mode.
    if debug_rendering or positioning_camera:
        try:
            root_node.addObject("RequiredPlugin", name="Sofa.GL.Component.Shader")
        except Exception:
            pass
        try:
            root_node.addObject("RequiredPlugin", name="Sofa.Component.Visual")
        except Exception:
            pass
        try:
            root_node.addObject("RequiredPlugin", name="Sofa.GL.Component.Rendering3D")
        except Exception:
            pass

        root_node.addObject("VisualStyle", displayFlags="showVisualModels showBehaviorModels showCollisionModels")
        root_node.addObject("BackgroundSetting", color=[0.3, 0.4, 0.5, 1.0])
    else:
        root_node.addObject("VisualStyle", displayFlags="hideVisualModels hideBehaviorModels hideCollisionModels")

    # Always create a camera object so the environment can initialize rendering
    # even when debug_rendering is False. When not in debug mode the camera is
    # created with `activated=False` to avoid heavy visual pipeline setup.
    camera = root_node.addObject(
        "InteractiveCamera",
        name="camera",
        position=[0.00, -0.90, 0.35],
        lookAt=[0.00, -0.45, 0.00],
        fieldOfView=75,
        zNear=0.0001,
        zFar=20.0,
        widthViewport=viewport_width,
        heightViewport=viewport_height,
        activated=True,
    )

    sim_friction_coef = float(
        os.environ.get("MCR_FRICTION_COEF", str(FRICTION_COEFFICIENT))
    )
    mcr_simulator.Simulator(
        root_node=root_node,
        friction_coef=sim_friction_coef,
        verbose=scene_verbose,
    )
    print("[SIM_FRICTION] friction_coef =", sim_friction_coef)

    # SOFA dt for training/testing.
    # Default is 0.01 s. This matches mcr_simulator.Simulator(dt=0.01)
    # and avoids the previous GUI-only override to 0.002 s.
    sofa_dt = float(os.environ.get("MCR_SOFA_DT", str(SOFA_TIME_STEP_S)))
    old_sofa_dt = float(root_node.dt.value)
    root_node.dt.value = sofa_dt
    sofa_steps_per_0p1s = int(round(0.1 / sofa_dt)) if sofa_dt > 0.0 else 1
    print(
        "[SOFA_DT]",
        "old_dt=", old_sofa_dt,
        "new_dt=", float(root_node.dt.value),
        "steps_per_0.1s=", sofa_steps_per_0p1s,
    )

    if debug_rendering or positioning_camera:
        root_node.addObject("LightManager", ambient=(1.0, 1.0, 1.0, 1.0))
        root_node.addObject("DirectionalLight", direction=[1, 1, -1], color=[1.0, 1.0, 1.0, 1.0])
        root_node.addObject("DirectionalLight", direction=[-1, 1, -1], color=[1.0, 1.0, 1.0, 1.0])
        root_node.addObject("DirectionalLight", direction=[0, -1, 1], color=[0.8, 0.8, 0.8, 1.0])
        root_node.addObject("DirectionalLight", direction=[0, 0, -1], color=[1.0, 1.0, 1.0, 1.0])

    navion = mcr_emns.EMNS(name="Navion", calibration_path=cal_path)

    # Default vessel alpha: if not provided by caller, prefer a higher alpha
    # so vessels are visible when GUI is launched from different entrypoints.
    vessel_alpha = float(kwargs.get("vessel_alpha", 0.8))
    vessel_alpha = max(0.0, min(1.0, vessel_alpha))

    # 目前保持原有规则：0021 不翻转法向，其它模型沿用 True。
    # 如果某些 test/Segmentation.stl 出现碰撞内外异常，可再把对应 task_id 加入 no_flip_models。
    no_flip_models = {"0028", "0038", "0231", "0237"}
    y_flip_models = ("Y001", "Y002", "Y003", "Y004", "Y005")
    flip_normals_value = True if str(task_id).startswith(y_flip_models) else (
        False if (task_id in no_flip_models or str(task_id).startswith("Y")) else True
    )
    # Keep vessel Line/PointCollisionModel optional for speed, but make the
    # TriangleCollisionModel slightly more conservative because centerline escape
    # is no longer used as a training-time safety termination.
    # Training default: 0.0002 m = 0.2 mm vessel triangle proximity.
    vessel_collision_proximity = float(
        kwargs.get("vessel_collision_proximity", VESSEL_TRIANGLE_PROXIMITY_M)
    )
    vessel_triangle_collision_proximity = float(
        kwargs.get(
            "vessel_triangle_collision_proximity",
            os.environ.get("MCR_VESSEL_TRIANGLE_PROXIMITY", vessel_collision_proximity),
        )
    )
    vessel_line_point_collision_proximity = float(
        kwargs.get(
            "vessel_line_point_collision_proximity",
            os.environ.get("MCR_VESSEL_LINE_POINT_PROXIMITY", vessel_collision_proximity),
        )
    )
    use_vessel_line_point_collision = bool(
        kwargs.get(
            "use_vessel_line_point_collision",
            str(os.environ.get("MCR_USE_VESSEL_LINE_POINT_COLLISION", "0")).lower()
            in ("1", "true", "yes", "y", "on"),
        )
    )

    print(
        "[VESSEL_COLLISION]",
        "task_id=", task_id,
        "flip_normals=", flip_normals_value,
        "collision_proximity=", vessel_collision_proximity,
        "triangle_proximity=", vessel_triangle_collision_proximity,
        "line_point_proximity=", vessel_line_point_collision_proximity,
        "use_line_point_collision=", use_vessel_line_point_collision,
    )

    environment = mcr_environment.Environment(
        root_node=root_node,
        environment_stl=environment_stl,
        visual_stl=visual_stl,
        name="aortic_arch",
        T_env_sim=T_env_sim,
        flip_normals=flip_normals_value,
        color=[0.9, 0.2, 0.2, vessel_alpha],
        scale=centerline_scale,
        visual=(True if (debug_rendering or positioning_camera) else False),
        collision_proximity=vessel_collision_proximity,
        triangle_collision_proximity=vessel_triangle_collision_proximity,
        line_point_collision_proximity=vessel_line_point_collision_proximity,
        use_line_point_collision=use_vessel_line_point_collision,
        verbose=scene_verbose,
    )
    
    t_start_env_runtime = list(T_start_env)
    t_start_sim_runtime = build_t_start_sim_from_env(t_start_env_runtime)
    target_point_sim = None
    centerline_data = None
    centerline_graph_data = None

    try:
        centerline_data = mcr_centerline.load_centerline_data(
            vtk_path=centerline_vtk,
            T_env_sim=T_env_sim,
            point_frame=centerline_point_frame,
            scale=centerline_scale,
            offset_sim=centerline_offset_sim,
            verbose=scene_verbose,
        )

        if centerline_graph_vtk:
            centerline_graph_data = mcr_centerline.load_centerline_data(
                vtk_path=centerline_graph_vtk,
                T_env_sim=T_env_sim,
                point_frame=centerline_point_frame,
                scale=centerline_scale,
                offset_sim=centerline_offset_sim,
                verbose=scene_verbose,
            )

        endpoints = mcr_centerline.get_start_target_by_y(centerline_data)
        if is_artificial_model:
            # Generated path VTKs are deliberately ordered inlet -> target.
            # Do not infer direction from world Y: T_env_sim rotates source Y
            # into another simulation axis, and some branch outlets have lower
            # source Z than the inlet.
            generated_points = np.asarray(centerline_data.points_sim, dtype=np.float64)
            if len(generated_points) < 2:
                raise ValueError(f"Artificial centerline has fewer than 2 points: {centerline_vtk}")
            centerline_data.start_index = 0
            centerline_data.target_index = len(generated_points) - 1
            endpoints = {
                "start_index": 0,
                "target_index": len(generated_points) - 1,
                "start_point_sim": generated_points[0],
                "target_point_sim": generated_points[-1],
            }
        start_point_sim = endpoints["start_point_sim"]
        target_point_sim = endpoints["target_point_sim"]

        # ============================================================
        # aorta6 special test start point
        # ============================================================
        # Only for aorta6 testing:
        # move the initial catheter insertion point to a later centerline
        # index while keeping the same target point.
        #
        # This changes only the initial position. The existing entry tangent
        # alignment below will automatically recompute the catheter orientation
        # according to this new starting point.
        if str(task_id).lower() == "aorta6":
            aorta6_start_index = int(kwargs.get("aorta6_start_index", 700))
            points_sim_np = np.asarray(centerline_data.points_sim, dtype=np.float64)

            if 0 <= aorta6_start_index < len(points_sim_np):
                start_point_sim = points_sim_np[aorta6_start_index].copy()

                print("[AORTA6_SPECIAL_START]")
                print("  enabled=True")
                print("  start_index =", aorta6_start_index)
                print("  start_point_sim =", start_point_sim)
                print("  original_endpoint_start =", endpoints["start_point_sim"])
                print("  target_point_sim =", target_point_sim)
            else:
                print(
                    "[AORTA6_SPECIAL_START][WARN] invalid index:",
                    aorta6_start_index,
                    "centerline_size=",
                    len(points_sim_np),
                    "fallback to original endpoint",
                )

        # Scheme-1 start/target randomization for generalization:
        # sample within fixed arc-length windows at the two centerline ends.
        # No spatial ball perturbation is added, so the initial
        # task stays exactly on the vessel centerline and avoids wall-biased starts.
        #
        # In single-vessel soft-reset mode, MCREnv.reset() handles randomization
        # without rebuilding the scene; createScene keeps the nominal pose here.
        randomize_start_target = (
            bool(kwargs.get("randomize_start_target", False))
            and (not soft_randomize_single_vessel)
            and str(task_id).lower() not in ("aorta6", "aorta7")
        )
        # Derive Generator state from the worker's seeded NumPy stream so
        # distributed runs are different across environments yet reproducible.
        rng = np.random.default_rng(np.random.randint(0, 2**32 - 1))
        nominal_start_point_sim = np.asarray(start_point_sim, dtype=np.float64).copy()
        nominal_target_point_sim = np.asarray(target_point_sim, dtype=np.float64).copy()
        sampled_start_idx = None
        sampled_target_idx = None
        if randomize_start_target:
            start_window_distance_m = float(
                kwargs.get("start_window_distance_m", START_WINDOW_DISTANCE_M)
            )
            target_window_distance_m = float(
                kwargs.get("target_window_distance_m", TARGET_WINDOW_DISTANCE_M)
            )
            endpoint_sample = _sample_endpoint_window_start_target(
                centerline_data=centerline_data,
                nominal_start_point_sim=nominal_start_point_sim,
                nominal_target_point_sim=nominal_target_point_sim,
                start_window_distance_m=start_window_distance_m,
                target_window_distance_m=target_window_distance_m,
                rng=rng,
            )
            start_point_sim = np.asarray(endpoint_sample["start_point_sim"], dtype=np.float64)
            target_point_sim = np.asarray(endpoint_sample["target_point_sim"], dtype=np.float64)
            sampled_start_idx = int(endpoint_sample["start_idx"])
            sampled_target_idx = int(endpoint_sample["target_idx"])
            print("[START_TARGET_ENDPOINT_WINDOW_RANDOMIZATION]")
            print("  start_window_mm      =", start_window_distance_m * 1000.0)
            print("  target_window_mm     =", target_window_distance_m * 1000.0)
            print("  stored_forward      =", endpoint_sample["stored_forward"])
            print("  start_candidates    =", endpoint_sample["start_candidates"])
            print("  target_candidates   =", endpoint_sample["target_candidates"])
            print("  sampled_start_idx   =", sampled_start_idx)
            print("  sampled_target_idx  =", sampled_target_idx)
            print("  nominal_start_sim   =", nominal_start_point_sim)
            print("  sampled_start_sim   =", start_point_sim)
            print("  nominal_target_sim  =", nominal_target_point_sim)
            print("  sampled_target_sim  =", target_point_sim)

        # 启用入口切线方向对齐初始姿态。
        # 0207 会随机选择左/右中心线，因此也加入这里，避免 force-model 0207 时仍保持 baseline 姿态。
        alignment_models = {
            "0207",
            "0207_left",
            "0207_right",
            "0210",
            "V1",
            "0021",
            "0028",
            "0038",
            "0230",
            "0231",
            "0237",
            "S1",
        }
        if start_point_sim is not None and (
            chosen_model in alignment_models
            or is_artificial_model
            or str(chosen_model).startswith("Y")
            or str(chosen_model).lower().startswith("aorta")
        ):
            try:
                entry_tangent, start_idx, next_idx = _compute_entry_tangent_toward_target(
                    centerline_data=centerline_data,
                    start_point_sim=start_point_sim,
                    target_point_sim=target_point_sim,
                    tangent_step=int(kwargs.get("entry_tangent_points", ENTRY_TANGENT_POINTS)),
                )

                # Randomize the initial mcR forward direction within a cone around
                # the centerline entry direction. This improves sim-to-sim robustness
                # without changing the nominal insertion point distribution.
                randomize_initial_orientation = bool(kwargs.get("randomize_initial_orientation", False)) and (not soft_randomize_single_vessel)
                max_init_angle_deg = float(
                    kwargs.get(
                        "initial_orientation_max_angle_deg",
                        INITIAL_ORIENTATION_MAX_ANGLE_DEG,
                    )
                )
                randomized_entry_tangent = entry_tangent
                sampled_angle_deg = 0.0
                if randomize_initial_orientation and max_init_angle_deg > 0.0:
                    randomized_entry_tangent = _sample_direction_within_cone(
                        entry_tangent,
                        np.deg2rad(max_init_angle_deg),
                        rng=rng,
                    )
                    sampled_angle_deg = float(np.degrees(np.arccos(np.clip(np.dot(entry_tangent, randomized_entry_tangent), -1.0, 1.0))))

                align_info = align_start_pose_minus_y_to_centerline_tangent(
                    t_start_env_runtime,
                    randomized_entry_tangent,
                )
                print("[INITIAL_MCR_CENTERLINE_ALIGNMENT]")
                print("  task_id             =", task_id)
                print("  start_idx           =", start_idx)
                print("  next_idx            =", next_idx)
                print("  start_point_sim     =", start_point_sim)
                print("  target_point_sim    =", target_point_sim)
                print("  entry_tangent_sim   =", entry_tangent)
                print("  randomized_tangent  =", randomized_entry_tangent)
                print("  sampled_angle_deg   =", sampled_angle_deg)
                print("  align_ok            =", align_info.get("ok", False))
                print("  before_forward_dot  =", align_info.get("before_dot", None))
                print("  after_forward_dot   =", align_info.get("after_dot", None))
                print("  new_forward_sim     =", align_info.get("new_forward_sim", None))
                print("  q_env_aligned       =", t_start_env_runtime[3:7])
            except Exception:
                import traceback

                traceback.print_exc()

        print("[INITIAL_MCR_POSE]")
        print("  task_id             =", task_id)
        print("  start_point_sim     =", start_point_sim)
        print("  target_point_sim    =", target_point_sim)
        print("  final_T_start_env_quat =", t_start_env_runtime[3:7])

        # 覆盖初始平移到中心线入口点；姿态使用上面可能已对齐后的 quaternion。
        # 不做 aorta2 tip reset 补偿；所有模型都直接使用中心线 start_point_sim。
        if start_point_sim is not None:
            desired_start_point_sim = np.asarray(start_point_sim, dtype=np.float64).reshape(3)

            r_env_to_sim = R.from_quat(rot_env_sim)
            start_point_env = r_env_to_sim.inv().apply(
                [
                    desired_start_point_sim[0] - transl_env_sim[0],
                    desired_start_point_sim[1] - transl_env_sim[1],
                    desired_start_point_sim[2] - transl_env_sim[2],
                ]
            )
            t_start_env_runtime[0] = float(start_point_env[0])
            t_start_env_runtime[1] = float(start_point_env[1])
            t_start_env_runtime[2] = float(start_point_env[2])
            t_start_sim_runtime = build_t_start_sim_from_env(t_start_env_runtime)

            t_start_sim_np = np.asarray(t_start_sim_runtime[:3], dtype=np.float64).reshape(3)
            print("[CREATE_SCENE_START_CHECK]")
            print("  desired_centerline_start_sim =", desired_start_point_sim)
            print("  actual_T_start_sim_xyz       =", t_start_sim_np)
            print("  Tstart_to_desired_error_mm   =", float(np.linalg.norm(t_start_sim_np - desired_start_point_sim) * 1000.0))

        root_node.centerline_data = centerline_data.to_python()
        root_node.centerline_data["t_start_env_runtime"] = t_start_env_runtime
        root_node.centerline_data["chosen_model"] = chosen_model
        root_node.centerline_data["environment_stl"] = environment_stl
        root_node.centerline_data["centerline_vtk"] = centerline_vtk
        root_node.centerline_data["task_id"] = task_id
        root_node.centerline_data["target_route_id"] = target_route_id
        root_node.centerline_data["curriculum_stage"] = "gui_nonros_centerline_aligned_pose"
        root_node.centerline_data["soft_randomize_single_vessel"] = bool(soft_randomize_single_vessel)

        if debug_rendering or positioning_camera:
            mcr_centerline.add_centerline_to_sofa(
                root_node=root_node,
                centerline_data=centerline_data,
                node_name="Centerline",
                line_color=[0.0, 1.0, 0.0, 1.0],
                target_color=[1.0, 1.0, 0.0, 1.0],
                target_point_sim=target_point_sim,
                show_target=True,
                target_scale=0.03,
            )
    except Exception as centerline_error:
        import traceback

        traceback.print_exc()
        root_node.centerline_data = {
            "vtk_path": centerline_vtk,
            "points_raw": [],
            "points_sim": [],
            "edges": [],
            "target_point_sim": None,
            "error": str(centerline_error),
            "chosen_model": chosen_model,
            "environment_stl": environment_stl,
            "centerline_vtk": centerline_vtk,
            "task_id": task_id,
            "curriculum_stage": "gui_nonros_centerline_aligned_pose",
        }

    magnet = mcr_magnet.Magnet(
        length=magnet_length,
        outer_diam=magnet_od,
        inner_diam=magnet_id,
        remanence=magnet_remanence,
    )

    # Magnetic tip: use the distal 3 tip elements.
    # Because MagController maps index 0/1/2 to the last/second-last/third-last
    # mechanical nodes, this makes the actual distal tip three segments magnetic.
    magnets = [0.0 for _ in range(num_elem_tip)]
    # Magnetic tip: only the distal-most tip segment is magnetic.
    # Keep num_elem_tip=3 as mechanical tip segmentation, but only magnets[0]
    # receives magnetic torque; the other two tip elements remain passive.
    magnetic_tip_elems = min(1, int(num_elem_tip))
    for i in range(magnetic_tip_elems):
        magnets[i] = magnet
    print(
        "[MAGNETIC_TIP]",
        "magnetic_tip_elems=", magnetic_tip_elems,
        "num_elem_tip=", num_elem_tip,
        "magnetic_indices_from_tip=", list(range(magnetic_tip_elems)),
    )
    instrument = mcr_instrument.Instrument(
        name="mcr",
        root_node=root_node,
        length_body=length_body,
        length_tip=length_tip,
        outer_diam=outer_diam,
        inner_diam=inner_diam,
        young_modulus_body=young_modulus_body,
        young_modulus_tip=young_modulus_tip,
        magnets=magnets,
        num_elem_body=num_elem_body,
        num_elem_tip=num_elem_tip,
        nume_nodes_viz=nume_nodes_viz,
        T_start_sim=t_start_sim_runtime,
        color=[0.2, 0.8, 1.0, 1.0],
        verbose=scene_verbose,
    )

    try:
        mo_pos = np.asarray(instrument.MO.position.array(), dtype=np.float64)
        start_np = np.asarray(t_start_sim_runtime[:3], dtype=np.float64).reshape(3)

        print("[INSTRUMENT_MO_CHECK]")
        print("  T_start_sim_xyz     =", start_np)
        print("  MO.shape            =", mo_pos.shape)

        if mo_pos.ndim == 2 and mo_pos.shape[0] > 0 and mo_pos.shape[1] >= 3:
            p0 = mo_pos[0, :3]
            pmid = mo_pos[len(mo_pos)//2, :3]
            plast = mo_pos[-1, :3]

            print("  MO[0] xyz           =", p0)
            print("  MO[mid] xyz         =", pmid)
            print("  MO[-1] xyz          =", plast)
            print("  dist MO[0]-start mm =", float(np.linalg.norm(p0 - start_np) * 1000.0))
            print("  dist MO[-1]-start mm=", float(np.linalg.norm(plast - start_np) * 1000.0))
            print("  body length mm      =", float(np.linalg.norm(plast - p0) * 1000.0))
    except Exception as e:
        print("[INSTRUMENT_MO_CHECK][WARN]", e)

    # V15.2-B: apply near-wall SDF repulsion through a SOFA force field on
    # inserted mechanical nodes.  Native triangle/line/point contact remains
    # enabled and continues to provide the primary hard collision constraint.
    sdf_wall_controller = None
    if sdf_vti:
        sdf_wall_controller = sdf_physics_wall.SDFPhysicsWallController(
            name="SDFPhysicsWallController",
            instrument=instrument,
            sdf_vti=sdf_vti,
            asset_T_env_sim=T_env_sim,
            asset_offset_sim=centerline_offset_sim,
            asset_source_to_sim_scale=centerline_scale,
            catheter_radius_m=outer_diam / 2.0,
            enabled=kwargs.get("sdf_physics_wall_enabled", None),
            activation_clearance_m=kwargs.get(
                "sdf_wall_activation_clearance_m", None
            ),
            stiffness_n_per_m=kwargs.get("sdf_wall_stiffness_n_per_m", None),
            max_force_n=kwargs.get("sdf_wall_max_force_n", None),
            verbose=scene_verbose,
        )
        root_node.addObject(sdf_wall_controller)

    controller_sofa = mcr_controller_sofa.ControllerSofa(
        root_node=root_node,
        e_mns=navion,
        instrument=instrument,
        T_sim_mns=T_sim_mns,
    )
    root_node.addObject(controller_sofa)

    scene_creation_result = {
        "mcr_controller_sofa": controller_sofa,
        "mcr_environment": environment,
        "sdf_physics_wall_controller": sdf_wall_controller,
        "camera": camera,
        "target_position": target_point_sim,
        "centerline_points": centerline_data.points_sim if centerline_data is not None else None,
        "centerline_radius": centerline_data.radius_sim if centerline_data is not None else None,
        "centerline_graph_points": (
            centerline_graph_data.points_sim
            if centerline_graph_data is not None
            else None
        ),
        "centerline_graph_edges": (
            centerline_graph_data.edges
            if centerline_graph_data is not None
            else None
        ),
        "centerline_graph_radius": (
            centerline_graph_data.radius_sim
            if centerline_graph_data is not None
            else None
        ),
        "chosen_model": chosen_model,
        "environment_stl": environment_stl,
        "visual_stl": visual_stl,
        "centerline_vtk": centerline_vtk,
        "centerline_graph_vtk": centerline_graph_vtk,
        "sdf_vti": sdf_vti,
        "metadata_json": metadata_json,
        "task_id": task_id,
        "target_route_id": target_route_id,
        "curriculum_stage": "gui_nonros_centerline_aligned_pose",
        "vessel_scale_factor": float(vessel_scale_factor),
        "asset_source_to_sim_scale": float(centerline_scale),
        "asset_T_env_sim": list(T_env_sim),
        "asset_offset_sim": list(centerline_offset_sim),
        "is_artificial_model": bool(is_artificial_model),
        "soft_randomize_single_vessel": bool(soft_randomize_single_vessel),
        "nominal_start_position": start_point_sim,
        "nominal_target_position": target_point_sim,
    }
    return scene_creation_result
