#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS test scene aligned with the training scene `example_aortic_arch.py`.

What is aligned with training:
1. The same train mesh set is used: 0207 and V1 by default.
2. 0207 randomly selects left/right centerline, exactly like training.
3. `force_model` is preserved for fixed testing: force_model="0207", "0210", or "V1".
4. `task_id`, `chosen_model`, `centerline_vtk`, and `environment_stl` are written into root_node.centerline_data.
5. Centerline loading, start-point overwrite, target-point selection, scale, and transform logic are kept consistent with training.
6. Visualization is lightweight unless debug_rendering or positioning_camera is enabled.

What is NOT changed:
1. ROS node/topic names are preserved.
2. Published message types are preserved.
3. Published data order on /sofa/mcr_state remains [tip_x, tip_y, tip_z, B_x, B_y, B_z].
4. Controller and SOFA-to-ROS communication flow remain unchanged.
"""

import random
import sys
import time
from pathlib import Path

# This file lives in python/scene/.  mesh/ and calib/ are siblings of the
# python/ Git root in the complete server project.
SCENE_DIR = Path(__file__).resolve().parent
PYTHON_ROOT = SCENE_DIR.parent
PROJECT_ROOT = PYTHON_ROOT.parent

if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

import Sofa
import rospy
from std_msgs.msg import Float64MultiArray

from splib3.numerics import Quat, Vec3
from scipy.spatial.transform import Rotation as R

from mcr_sim_ros import (
    mcr_centerline_ros,
    mcr_controller_sofa_ros,
    mcr_emns_ros,
    mcr_environment_ros,
    mcr_instrument_ros,
    mcr_magnet_ros,
    mcr_simulator_ros,
)

# ============================================================
# Paths
# ============================================================
TRAIN_MESH_DIR = PROJECT_ROOT / "mesh" / "train"
TEST_MESH_DIR = PROJECT_ROOT / "mesh" / "test"

# Calibration file for eMNS
cal_path = str(PROJECT_ROOT / "calib" / "Navion_2_Calibration_24-02-2020.yaml")

# ============================================================
# Parameters instrument, magnet, beams
# ============================================================
young_modulus_body = 170e6
young_modulus_tip = 21e6
length_body = 0.5
length_tip = 0.034
outer_diam = 0.00133
inner_diam = 0.0008

length_init = 0.35

magnet_length = 4e-3
magnet_id = 0.86e-3
magnet_od = 1.33e-3
magnet_remanence = 1.45

# Keep the ROS test scene closer to the fast training scene.
# If you need denser visualization for recording, set debug_rendering=True and change this back to 600.
nume_nodes_viz = 100
num_elem_body = 30
num_elem_tip = 3

# ============================================================
# Transforms
# ============================================================
T_sim_mns = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

rot_env_sim = [-0.7071068, 0, 0, 0.7071068]
transl_env_sim = [0.0, -0.45, 0.0]
T_env_sim = [transl_env_sim[0], transl_env_sim[1], transl_env_sim[2], -0.7071068, 0, 0, 0.7071068]

# Starting pose in environment frame.
# Orientation is kept as sofa_env baseline; translation can be overwritten by centerline start point.
T_start_env = [-0.075, -0.001, -0.020, 0.0, -0.3826834, 0.0, 0.9238795]


def _normalize_vec(vec, eps=1e-9):
    """Return a normalized numpy vector. If degenerate, return None."""
    import numpy as np

    vec = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return None
    return vec / norm


def _rotation_between_vectors(source_vec, target_vec):
    """Minimal rotation that maps source_vec to target_vec."""
    import numpy as np

    source = _normalize_vec(source_vec)
    target = _normalize_vec(target_vec)
    if source is None or target is None:
        return R.identity()

    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))

    if dot > 1.0 - 1e-8:
        return R.identity()

    if dot < -1.0 + 1e-8:
        # 180-degree case: choose any stable axis orthogonal to source.
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
    """Align catheter local -Y axis to the centerline entry tangent in sim frame.

    The previous diagnostics showed the catheter advancing direction corresponds
    best to local -Y.  The start pose stored in T_start_env is an environment-frame
    pose.  build_t_start_sim_from_env later composes it with rot_env_sim, so here
    we solve:

        R_sim_new = delta_sim * R_sim_current
        R_env_new = R_env_sim^{-1} * R_sim_new

    and write R_env_new back into t_start_env_value[3:7].
    """
    import numpy as np

    entry_tangent_sim = _normalize_vec(entry_tangent_sim)
    if entry_tangent_sim is None:
        return {
            "ok": False,
            "reason": "degenerate_entry_tangent",
        }

    r_env_to_sim = R.from_quat(rot_env_sim)
    r_env_current = R.from_quat(t_start_env_value[3:7])
    r_sim_current = r_env_to_sim * r_env_current

    local_forward_axis = np.asarray([0.0, -1.0, 0.0], dtype=np.float64)
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
    """Use exactly the same task-selection logic as the training scene."""
    train_dir = TRAIN_MESH_DIR

    # Default training set: 0207 Y-type and V1 aortic type.
    random_models = ["0207", "V1"]

    # 0210 is kept only for forced testing/debugging, not default random testing.
    supported_models = ["0207", "0210", "V1", "0207_left", "0207_right","0021"]

    chosen_model = kwargs.get("force_model", None)
    if chosen_model not in supported_models:
        chosen_model = random.choice(random_models)

    if chosen_model == "0207":
        environment_stl = train_dir / "0207" / "0207.stl"
        centerlines = [
            train_dir / "0207" / "0207_left_centerline.vtk",
            train_dir / "0207" / "0207_right_centerline.vtk",
        ]
        centerline_vtk = random.choice(centerlines)
    elif chosen_model == "0207_left":
        environment_stl = train_dir / "0207" / "0207.stl"
        centerline_vtk = train_dir / "0207" / "0207_left_centerline.vtk"
    elif chosen_model == "0207_right":
        environment_stl = train_dir / "0207" / "0207.stl"
        centerline_vtk = train_dir / "0207" / "0207_right_centerline.vtk"
    elif chosen_model == "0210":
        environment_stl = train_dir / "0210" / "0210.stl"
        centerline_vtk = train_dir / "0210" / "Centerline model.vtk"
    elif chosen_model == "0021":
        environment_stl = TEST_MESH_DIR / "0021" / "Segmentation.stl"
        centerline_vtk = TEST_MESH_DIR / "0021" / "Centerline model.vtk"
    else:  # V1
        environment_stl = train_dir / "V1" / "J2-Naviworks.stl"
        centerline_vtk = train_dir / "V1" / "Centerline model.vtk"

    centerline_str = str(centerline_vtk)
    if "0207_left" in centerline_str:
        task_id = "0207_left"
    elif "0207_right" in centerline_str:
        task_id = "0207_right"
    elif chosen_model == "0210":
        task_id = "0210"
    elif chosen_model == "0021":
        task_id = "0021"
    elif chosen_model == "V1":
        task_id = "V1"
    else:
        task_id = str(chosen_model)

    return {
        "chosen_model": chosen_model,
        "environment_stl": str(environment_stl),
        "centerline_vtk": str(centerline_vtk),
        "task_id": task_id,
    }


class SofaRosPublisher(Sofa.Core.Controller):
    def __init__(self, root, instrument, controller_sofa, *args, **kwargs):
        kwargs["listening"] = True
        kwargs["name"] = "SofaRosPublisher"
        super().__init__(*args, **kwargs)

        self.root = root
        self.instrument = instrument
        self.controller_sofa = controller_sofa

        self.pub = None
        try:
            if not rospy.core.is_initialized():
                rospy.init_node("sofa_state_publisher", anonymous=True, disable_signals=True)
            from geometry_msgs.msg import PointStamped

            self.pub_tip = rospy.Publisher("/sofa/mcr_tip_sofa", PointStamped, queue_size=1)
            self.pub_tip_world = rospy.Publisher("/sofa/mcr_tip_world", PointStamped, queue_size=1)
            self.pub_b_target = rospy.Publisher("/sofa/mcr_B_target", PointStamped, queue_size=1)
            self.pub = rospy.Publisher("/sofa/mcr_state", Float64MultiArray, queue_size=1)
            rospy.loginfo("[SOFA] ROS publisher initialized: tip_sofa, tip_world, B_target and mcr_state")
        except Exception as e:
            print(f"\n>>>>>>>> [致命错误] ROS 初始化失败 (请检查 roscore 是否运行): {e} <<<<<<<<\n")

        self.publish_hz = 20.0
        self.publish_period = 1.0 / self.publish_hz
        self.last_pub_time = 0.0
        self.tip_source = None
        self.tip_source_name = None

    def init(self):
        try:
            self._find_tip_source_once()
        except Exception as e:
            print(f"\n>>>>>>>> [致命错误] SOFA init() 寻找末端失败: {e} <<<<<<<<\n")

    def _extract_last_point(self, obj):
        try:
            pos = obj.position.value
            if len(pos) > 0 and len(pos[-1]) >= 3:
                return [float(pos[-1][0]), float(pos[-1][1]), float(pos[-1][2])]
        except Exception:
            pass
        return None

    def _find_tip_source_once(self):
        candidate_names = [
            "DOFs", "dofs", "MO", "mo", "MechanicalObject",
            "mstate", "state", "instrumentMO", "rodMO",
        ]

        for name in candidate_names:
            try:
                cand = getattr(self.instrument, name, None)
                tip = self._extract_last_point(cand)
                if tip is not None:
                    self.tip_source = cand
                    self.tip_source_name = f"instrument.{name}"
                    rospy.loginfo(f"[SOFA] tip source selected: {self.tip_source_name}")
                    return
            except Exception:
                pass

        if self.controller_sofa is not None:
            try:
                ctrl_inst = getattr(self.controller_sofa, "instrument", None)
                if ctrl_inst is not None:
                    for name in candidate_names:
                        try:
                            cand = getattr(ctrl_inst, name, None)
                            tip = self._extract_last_point(cand)
                            if tip is not None:
                                self.tip_source = cand
                                self.tip_source_name = f"controller_sofa.instrument.{name}"
                                rospy.loginfo(f"[SOFA] tip source selected: {self.tip_source_name}")
                                return
                        except Exception:
                            pass
            except Exception:
                pass

        rospy.logwarn("[SOFA] Could not auto-find tip source. Will publish zeros.")

    def get_tip_position(self):
        if self.tip_source is not None:
            tip = self._extract_last_point(self.tip_source)
            if tip is not None:
                return tip
        return [0.0, 0.0, 0.0]

    def get_target_B(self):
        try:
            value = self.controller_sofa.get_mag_field_des()
            if value is not None and len(value) >= 3:
                return [float(value[0]), float(value[1]), float(value[2])]
        except Exception:
            pass
        return [0.0, 0.0, 0.0]

    def publish_state(self):
        if self.pub is None:
            print("[SOFA->ROS] 发送被拦截！原因：ROS 节点未成功初始化 (roscore没开？)")
            return

        tip = self.get_tip_position()
        B = self.get_target_B()

        from geometry_msgs.msg import PointStamped

        msg_tip = PointStamped()
        msg_tip.header.stamp = rospy.Time.now()
        msg_tip.header.frame_id = "env"
        msg_tip.point.x, msg_tip.point.y, msg_tip.point.z = tip[0], tip[1], tip[2]
        self.pub_tip.publish(msg_tip)

        # Preserve the existing calibrated communication behavior:
        # SOFA coordinates are passed as executor world coordinates.
        msg_tip_world = PointStamped()
        msg_tip_world.header.stamp = msg_tip.header.stamp
        msg_tip_world.header.frame_id = "world"
        msg_tip_world.point.x, msg_tip_world.point.y, msg_tip_world.point.z = tip[0], tip[1], tip[2]
        self.pub_tip_world.publish(msg_tip_world)

        msg_b = PointStamped()
        msg_b.header.stamp = rospy.Time.now()
        msg_b.header.frame_id = "world"
        msg_b.point.x, msg_b.point.y, msg_b.point.z = B[0], B[1], B[2]
        self.pub_b_target.publish(msg_b)

        msg = Float64MultiArray()
        msg.data = [tip[0], tip[1], tip[2], B[0], B[1], B[2]]
        self.pub.publish(msg)
        print(f"[SOFA->ROS] tip={tip}, B={B}")

    def onAnimateBeginEvent(self, event):
        import traceback

        try:
            now = time.time()
            if now - self.last_pub_time >= self.publish_period:
                self.publish_state()
                self.last_pub_time = now
        except Exception as e:
            print(f"\n>>>>>>>> [致命错误] 数据发布时崩溃: {e} <<<<<<<<\n")
            print(traceback.format_exc())
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n")


def createScene(root_node, image_shape=None, debug_rendering=False, positioning_camera=False, **kwargs):
    """Build ROS SOFA scene aligned with the training file, while preserving ROS communication."""

    task_cfg = resolve_training_task(kwargs)
    chosen_model = task_cfg["chosen_model"]
    environment_stl = task_cfg["environment_stl"]
    centerline_vtk = task_cfg["centerline_vtk"]
    task_id = task_cfg["task_id"]

    centerline_point_frame = "env"
    centerline_scale = 0.0005
    centerline_offset_sim = [0.0, 0.0, 0.0]

    print("[example_aortic_arch_ros_aligned] chosen_model    =", chosen_model)
    print("[example_aortic_arch_ros_aligned] task_id         =", task_id)
    print("[example_aortic_arch_ros_aligned] environment_stl =", environment_stl)
    print("[example_aortic_arch_ros_aligned] centerline_vtk  =", centerline_vtk)

    if not Path(environment_stl).is_file():
        raise FileNotFoundError(f"Vessel STL not found: {environment_stl}")
    if not Path(centerline_vtk).is_file():
        raise FileNotFoundError(f"Centerline VTK not found: {centerline_vtk}")

    if image_shape is None:
        viewport_height, viewport_width = 600, 800
    else:
        viewport_height = int(image_shape[0])
        viewport_width = int(image_shape[1])

    # Match training: keep rendering light unless explicitly needed.
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

        root_node.addObject("VisualStyle", displayFlags="showVisualModels hideBehaviorModels hideCollisionModels")
        root_node.addObject("BackgroundSetting", color=[0.3, 0.4, 0.5, 1.0])
    else:
        root_node.addObject("VisualStyle", displayFlags="hideVisualModels hideBehaviorModels hideCollisionModels")

    camera = None
    if debug_rendering or positioning_camera:
        camera = root_node.addObject(
            "InteractiveCamera",
            name="camera",
            position=[0.0, -0.45, 0.85],
            lookAt=[0.0, -0.45, 0.0],
            fieldOfView=85,
            zNear=0.01,
            zFar=10.0,
            widthViewport=viewport_width,
            heightViewport=viewport_height,
            activated=True,
        )

    mcr_simulator_ros.Simulator(root_node=root_node)

    if debug_rendering or positioning_camera:
        root_node.addObject("LightManager", ambient=(1.0, 1.0, 1.0, 1.0))
        root_node.addObject("DirectionalLight", name="light1", direction=[1, 1, -1], color=[1.0, 1.0, 1.0, 1.0])
        root_node.addObject("DirectionalLight", name="light2", direction=[-1, 1, -1], color=[1.0, 1.0, 1.0, 1.0])
        root_node.addObject("DirectionalLight", name="light3", direction=[0, -1, 1], color=[0.8, 0.8, 0.8, 1.0])
        root_node.addObject("DirectionalLight", name="light4", direction=[0, 0, -1], color=[1.0, 1.0, 1.0, 1.0])

    navion = mcr_emns_ros.EMNS(name="Navion", calibration_path=cal_path)

    vessel_alpha = float(kwargs.get("vessel_alpha", 0.35))
    vessel_alpha = max(0.0, min(1.0, vessel_alpha))

    environment = mcr_environment_ros.Environment(
        root_node=root_node,
        environment_stl=environment_stl,
        name="aortic_arch",
        T_env_sim=T_env_sim,
        flip_normals=(False if task_id == "0021" else True),
        color=[0.9, 0.2, 0.2, vessel_alpha],
        scale=centerline_scale,
    )

    t_start_env_runtime = list(T_start_env)
    t_start_sim_runtime = build_t_start_sim_from_env(t_start_env_runtime)
    target_point_sim = None
    centerline_data = None

    try:
        centerline_data = mcr_centerline_ros.load_centerline_data(
            vtk_path=centerline_vtk,
            T_env_sim=T_env_sim,
            point_frame=centerline_point_frame,
            scale=centerline_scale,
            offset_sim=centerline_offset_sim,
        )

        endpoints = mcr_centerline_ros.get_start_target_by_y(centerline_data)
        start_point_sim = endpoints["start_point_sim"]
        target_point_sim = endpoints["target_point_sim"]

        if task_id == "0021":
            import numpy as np

            # 0021 使用几何方向，不再强制反转中心线。
            # 真正需要修的是初始姿态：让导管推进轴对齐入口中心线切线。
            pts = np.asarray(centerline_data.points_sim, dtype=np.float32)
            start_np = np.asarray(start_point_sim, dtype=np.float32)
            target_np = np.asarray(target_point_sim, dtype=np.float32)

            # 找到 start_point_sim 在当前中心线中的最近点。
            dists_to_start = np.linalg.norm(pts - start_np[None, :], axis=1)
            start_idx = int(np.argmin(dists_to_start))

            # 根据 start_idx 在中心线的位置，选取朝向中心线内部的点估计入口切线。
            tangent_step = 5
            if start_idx <= len(pts) // 2:
                next_idx = min(start_idx + tangent_step, len(pts) - 1)
            else:
                next_idx = max(start_idx - tangent_step, 0)

            entry_tangent = pts[next_idx] - pts[start_idx]
            entry_tangent = entry_tangent / (np.linalg.norm(entry_tangent) + 1e-9)

            # 对齐前诊断：当前固定姿态三根局部轴在 sim 坐标系中的方向。
            q_env_before = np.asarray(t_start_env_runtime[3:7], dtype=np.float32)
            rot_env_before = R.from_quat(q_env_before)

            axis_x_env = rot_env_before.apply([1.0, 0.0, 0.0])
            axis_y_env = rot_env_before.apply([0.0, 1.0, 0.0])
            axis_z_env = rot_env_before.apply([0.0, 0.0, 1.0])
            axis_minus_y_env = rot_env_before.apply([0.0, -1.0, 0.0])

            r_env_to_sim = R.from_quat(rot_env_sim)
            axis_x_sim = r_env_to_sim.apply(axis_x_env)
            axis_y_sim = r_env_to_sim.apply(axis_y_env)
            axis_z_sim = r_env_to_sim.apply(axis_z_env)
            axis_minus_y_sim = r_env_to_sim.apply(axis_minus_y_env)

            dot_x = float(np.dot(axis_x_sim, entry_tangent))
            dot_y = float(np.dot(axis_y_sim, entry_tangent))
            dot_z = float(np.dot(axis_z_sim, entry_tangent))
            dot_minus_y = float(np.dot(axis_minus_y_sim, entry_tangent))

            align_info = align_start_pose_minus_y_to_centerline_tangent(
                t_start_env_runtime,
                entry_tangent,
            )

            print("[0021_GEOMETRIC_DIRECTION_AND_ENTRY_ALIGNMENT]")
            print("  force_reverse       = False")
            print("  start_idx           =", start_idx)
            print("  next_idx            =", next_idx)
            print("  start_point_sim     =", start_point_sim)
            print("  target_point_sim    =", target_point_sim)
            print("  start_target_mm     =", float(np.linalg.norm(target_np - start_np) * 1000.0))
            print("  entry_tangent_sim   =", entry_tangent)
            print("  before axis_x_sim   =", axis_x_sim, "dot=", dot_x)
            print("  before axis_y_sim   =", axis_y_sim, "dot=", dot_y)
            print("  before axis_z_sim   =", axis_z_sim, "dot=", dot_z)
            print("  before -axis_y_sim  =", axis_minus_y_sim, "dot=", dot_minus_y)
            print("  align_ok            =", align_info.get("ok", False))
            print("  before_forward_dot  =", align_info.get("before_dot", None))
            print("  after_forward_dot   =", align_info.get("after_dot", None))
            print("  new_forward_sim     =", align_info.get("new_forward_sim", None))
            print("  q_env_aligned       =", t_start_env_runtime[3:7])

        if start_point_sim is not None:
            r_env_to_sim = R.from_quat(rot_env_sim)
            start_point_env = r_env_to_sim.inv().apply(
                [
                    start_point_sim[0] - transl_env_sim[0],
                    start_point_sim[1] - transl_env_sim[1],
                    start_point_sim[2] - transl_env_sim[2],
                ]
            )
            t_start_env_runtime[0] = float(start_point_env[0])
            t_start_env_runtime[1] = float(start_point_env[1])
            t_start_env_runtime[2] = float(start_point_env[2])
            t_start_sim_runtime = build_t_start_sim_from_env(t_start_env_runtime)

        root_node.centerline_data = centerline_data.to_python()
        root_node.centerline_data["t_start_env_runtime"] = t_start_env_runtime
        root_node.centerline_data["chosen_model"] = chosen_model
        root_node.centerline_data["environment_stl"] = environment_stl
        root_node.centerline_data["centerline_vtk"] = centerline_vtk
        root_node.centerline_data["task_id"] = task_id
        root_node.centerline_data["curriculum_stage"] = "test_ros_aligned"

        if debug_rendering or positioning_camera:
            mcr_centerline_ros.add_centerline_to_sofa(
                root_node=root_node,
                centerline_data=centerline_data,
                node_name="Centerline",
                line_color=[0.0, 1.0, 0.0, 1.0],
                target_color=[1.0, 1.0, 0.0, 1.0],
                target_point_sim=target_point_sim,
                show_target=True,
                target_scale=0.01,
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
            "curriculum_stage": "test_ros_aligned",
        }

    magnet = mcr_magnet_ros.Magnet(
        length=magnet_length,
        outer_diam=magnet_od,
        inner_diam=magnet_id,
        remanence=magnet_remanence,
    )

    magnets = [0.0 for _ in range(num_elem_tip)]
    magnets[0] = magnet
    magnets[1] = magnet

    instrument = mcr_instrument_ros.Instrument(
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
    )

    controller_sofa = mcr_controller_sofa_ros.ControllerSofa(
        root_node=root_node,
        e_mns=navion,
        instrument=instrument,
        T_sim_mns=T_sim_mns,
    )
    root_node.addObject(controller_sofa)

    sofa_ros_publisher = SofaRosPublisher(
        root=root_node,
        instrument=instrument,
        controller_sofa=controller_sofa,
    )
    root_node.addObject(sofa_ros_publisher)
    root_node.sofa_ros_publisher_ref = sofa_ros_publisher

    scene_creation_result = {
        "mcr_controller_sofa_ros": controller_sofa,
        "mcr_environment_ros": environment,
        "mcr_controller_sofa": controller_sofa,
        "mcr_environment": environment,
        "camera": camera,
        "target_position": target_point_sim,
        "centerline_points": centerline_data.points_sim if centerline_data is not None else None,
        "chosen_model": chosen_model,
        "centerline_vtk": centerline_vtk,
        "environment_stl": environment_stl,
        "task_id": task_id,
        "curriculum_stage": "test_ros_aligned",
    }
    return scene_creation_result
