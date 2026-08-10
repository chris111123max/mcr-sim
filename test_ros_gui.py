#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
联合测试 GUI 版：SOFA 官方 GUI + 已训练 SAC 模型 + PyBullet/MoveIt 磁场执行器闭环测试。

启动方式示例：
    runSofa /home/chen/SOFAA/projects/mCR_simulator-master/python/联合测试_gui版.py

启动前需要先运行：
    1) roscore
    2) PyBullet / MoveIt 磁场执行器节点，也就是原来监听：
       /magnetic/execute_request
       并发布：
       /magnetic/execute_result
       的那个闭环执行节点。

本文件不会修改 ROS 通信 topic 和数据格式。
"""

import os
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import Sofa
import rospy
from scipy.interpolate import splprep, splev
from stable_baselines3 import SAC
from std_msgs.msg import Float64MultiArray

# ============================================================
# 用户配置区
# ============================================================
PROJECT_ROOT = Path("/home/chen/SOFAA/projects/mCR_simulator-master")
PYTHON_DIR = PROJECT_ROOT / "python"

# 默认测试血管。可改成："V1", "0207", "0207_left", "0207_right", "0210", "0021"
FORCE_MODEL = "0021"

# 你的 2mm 模型。需要时直接改这里。
MODEL_PATH = (
    PROJECT_ROOT
    / "python"
    / "runs_tri"
    / "centerline_light_2mm_from_3mm_aortic_2mm_20260517_184506"
    / "models"
    / "sac_mcr_2mm_V1_y_noS_ckpt_3800000_steps.zip"
)

TARGET_DISTANCE_THRESHOLD = 0.002
MAX_EPISODE_STEPS = 2048
POLICY_DETERMINISTIC = True
DEVICE = "auto"

# 与 mcr_rl_env_ros.py 中保持一致
ACTION_SMOOTHING_ALPHA = 0.2
PROGRESS_CLIP = 0.003
TERMINAL_ENTER_THRESHOLD = 0.010
TERMINAL_EXIT_THRESHOLD = 0.012
TERMINAL_BAD_STEPS_LIMIT = 30

DESIRED_INITIAL_TIP_WORLD = np.asarray([0.75, 0.20, 0.70], dtype=np.float32)
DYNAMIC_TIP_WORLD_ALIGN = True

# 控制打印频率，避免 GUI 太卡
PRINT_EVERY_N_STEPS = 1

# SOFA GUI 可视化参数
DEBUG_RENDERING = True
POSITIONING_CAMERA = True
VESSEL_ALPHA = 0.25

# 中心线特征，与训练保持一致：原 24 维 + 1 + 8*3 = 49 维
NUM_CENTERLINE_LOOKAHEAD_POINTS = 8
CENTERLINE_LOOKAHEAD_OFFSETS = (3, 6, 10, 15, 21, 28, 36, 45)
CENTERLINE_FEATURE_DIM = 1 + 3 * NUM_CENTERLINE_LOOKAHEAD_POINTS

# 历史约束中心线投影参数
CENTERLINE_PROJECTION_SEARCH_BACK = 5
CENTERLINE_PROJECTION_SEARCH_FORWARD = 8
CENTERLINE_MAX_BACKTRACK = 0.0005
CENTERLINE_MAX_FORWARD_JUMP = 0.010

# GUI 下为了看清，默认不因为到达目标而立刻停止；只打印 success。
STOP_ON_SUCCESS = False


if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import example_aortic_arch_ros as base_scene


class GuiPolicyClosedLoopController(Sofa.Core.Controller):
    """Run SAC policy inside SOFA official GUI animation loop.

    This controller mirrors the important inference logic in run_trained_mcr_ros.py
    and mcr_rl_env_ros.py, but it is executed by SOFA GUI callbacks instead of a
    Gym while-loop.
    """

    def __init__(self, root_node, scene_result, model_path, *args, **kwargs):
        kwargs["listening"] = True
        kwargs["name"] = "GuiPolicyClosedLoopController"
        super().__init__(*args, **kwargs)

        self.root_node = root_node
        self.scene_result = scene_result
        self.controller_sofa = scene_result["mcr_controller_sofa_ros"]
        self.environment = scene_result["mcr_environment_ros"]
        self.target_position = np.asarray(scene_result["target_position"], dtype=np.float32)
        self.task_id = scene_result.get("task_id", "unknown")
        self.chosen_model = scene_result.get("chosen_model", "unknown")
        self.centerline_vtk = scene_result.get("centerline_vtk", "unknown")

        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"SAC model not found: {self.model_path}")

        print("=" * 100)
        print("SOFA GUI joint closed-loop policy test")
        print(f"Model:       {self.model_path}")
        print(f"Force model: {FORCE_MODEL}")
        print(f"Task id:     {self.task_id}")
        print(f"Centerline:  {self.centerline_vtk}")
        print("=" * 100)

        self.model = SAC.load(str(self.model_path), env=None, device=DEVICE)

        self.action_smoothing_alpha = ACTION_SMOOTHING_ALPHA
        self._last_smoothed_action = np.zeros(3, dtype=np.float32)
        self._prev_smoothed_action = np.zeros(3, dtype=np.float32)
        self._ideal_B = np.asarray(self.controller_sofa.get_mag_field_des(), dtype=np.float32).copy()

        self.step_in_episode = 0
        self.global_step = 0
        self.episode_reward = 0.0
        self.episode_idx = 1
        self.min_dist_this_episode = np.inf

        self.episode_success = False
        self.episode_success_10mm = False
        self.episode_success_6mm = False
        self.episode_success_3mm = False
        self.episode_success_2mm = False

        self.reached_10mm_once = False
        self.reached_6mm_once = False
        self.reached_3mm_once = False
        self.entered_terminal_zone_once = False
        self.was_in_terminal_zone_prev_step = False
        self.terminal_bad_steps = 0
        self.terminal_escape_failed = False

        self.reward_amount_dict = defaultdict(float)
        self.reward_amount_dict.update({
            "tip_pos_distance_to_dest_pos": 0.0,
            "delta_tip_pos_distance_to_dest_pos": 80.0,
            "centerline_progress_distance": 260.0,
            "workspace_constraint_violation": -6.0,
            "step_penalty": -0.001,
            "terminal_log_progress_bonus": 200.0,
            "terminal_escape_penalty": -3.0,
            "terminal_recapture_bonus": 20.0,
            "terminal_escape_failure_penalty": -100.0,
            "milestone_10mm": 50.0,
            "milestone_6mm": 150.0,
            "milestone_3mm": 400.0,
            "successful_task": 4000.0,
            "out_of_bounds": -100.0,
            "action_smoothness": -0.02,
        })
        self.reward_features = {"tip_pos_distance_to_dest_pos": self._get_distance_tip_to_dest()}

        self.centerline_points = None
        self.centerline_cumlength = None
        self.previous_centerline_progress = None
        self.previous_centerline_projection_index = -1
        self.current_centerline_progress = 0.0
        self.current_centerline_delta_progress = 0.0
        self.current_centerline_distance = 0.0
        self.current_centerline_progress_ratio = 0.0
        self.centerline_projection_rejected_jump = False
        self.centerline_projection_rejected_reason = ""
        self.using_centerline_reward = False
        self.using_euclidean_terminal_reward = False

        raw_centerline = scene_result.get("centerline_points", None)
        if raw_centerline is not None:
            self.centerline_points = self._resample_centerline_points_1mm(np.asarray(raw_centerline, dtype=np.float32))
            self._init_centerline_direction_and_cumlength()

        self.exit_plane_normal = self._compute_exit_plane_normal()
        self.cartesian_scaling_factor = self._compute_cartesian_scaling_factor()

        self.tip_world_offset = np.zeros(3, dtype=np.float32)
        self._update_dynamic_tip_world_offset()

        self._latest_execute_result = None
        self._execute_result_step_id = -1
        self._magnetic_step_id = 0
        self._execute_timeout = 3.0
        self._latest_executor_info = {
            "success": False,
            "rel_error": 0.0,
            "moved": False,
            "I_pair": [0.0, 0.0],
        }
        self._init_ros_magnetic_executor()

    # ============================================================
    # SOFA callbacks
    # ============================================================
    def onAnimateBeginEvent(self, event):
        """Before each SOFA step: policy -> B_target -> PyBullet -> B_actual."""
        if rospy.is_shutdown():
            return

        if self.step_in_episode >= MAX_EPISODE_STEPS:
            print("[GUI_POLICY] Max episode steps reached. Continue displaying scene without policy update.")
            return

        try:
            # Restore ideal B before computing the next incremental action.
            self.controller_sofa.set_mag_field_des(self._ideal_B)

            obs = self._get_observation()
            action, _ = self.model.predict(obs, deterministic=POLICY_DETERMINISTIC)
            action = self._sanitize_action(action)

            previous_smoothed_action = self._last_smoothed_action.copy()
            smoothed_action = (
                self.action_smoothing_alpha * action
                + (1.0 - self.action_smoothing_alpha) * previous_smoothed_action
            )
            self._prev_smoothed_action = previous_smoothed_action.copy()
            self._last_smoothed_action = smoothed_action.copy()

            self._do_action(smoothed_action)
            B_target = np.asarray(self.controller_sofa.get_mag_field_des(), dtype=np.float32).copy()
            self._ideal_B = B_target.copy()

            tip_world = self._get_tip_world_for_magnetic_executor()
            result = self._request_magnetic_execution(tip_world, B_target)
            self._latest_executor_info = result

            # This B_actual will be used by the SOFA physics step that follows this callback.
            self.controller_sofa.set_mag_field_des(result["B_actual"])

        except Exception as exc:
            import traceback

            print("\n[GUI_POLICY][ERROR] onAnimateBeginEvent failed:", exc)
            print(traceback.format_exc())

    def onAnimateEndEvent(self, event):
        """After SOFA physics step: restore ideal B, compute reward/logging."""
        try:
            # Keep controller internal target consistent for the next policy action.
            self.controller_sofa.set_mag_field_des(self._ideal_B)

            reward, info = self._compute_reward_and_info()
            self.step_in_episode += 1
            self.global_step += 1
            self.episode_reward += float(reward)

            if self.global_step % PRINT_EVERY_N_STEPS == 0:
                self._print_step_log(reward, info)

            if STOP_ON_SUCCESS and self.episode_success:
                print("[GUI_POLICY] Target reached. Policy updates are stopped; GUI remains open.")
                self.step_in_episode = MAX_EPISODE_STEPS

        except Exception as exc:
            import traceback

            print("\n[GUI_POLICY][ERROR] onAnimateEndEvent failed:", exc)
            print(traceback.format_exc())

    # ============================================================
    # Observation / action
    # ============================================================
    def _sanitize_action(self, action):
        action_np = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_np.shape[0] < 3:
            action_np = np.pad(action_np, (0, 3 - action_np.shape[0]))
        action_np = action_np[:3]
        action_np = np.nan_to_num(action_np, nan=0.0, posinf=1.0, neginf=-1.0)
        return np.clip(action_np, -1.0, 1.0).astype(np.float32)

    def _get_observation(self):
        tip_pose = np.asarray(self.controller_sofa.get_pos_quat_catheter_tip(), dtype=np.float32)
        tip_pos = tip_pose[0:3]

        offset_vec = tip_pos - self.target_position
        current_dist = np.linalg.norm(offset_vec)
        axial = np.dot(offset_vec, self.exit_plane_normal) * self.exit_plane_normal
        lateral_vec = offset_vec - axial
        lateral_error = np.linalg.norm(lateral_vec)
        prev_action = self._last_smoothed_action.astype(np.float32)
        centerline_features = self._get_centerline_light_features(tip_pos)

        obs = np.concatenate([
            tip_pose.astype(np.float32),
            np.asarray(self.controller_sofa.get_mag_field_des(), dtype=np.float32),
            offset_vec.astype(np.float32),
            np.array([current_dist], dtype=np.float32),
            lateral_vec.astype(np.float32),
            np.array([lateral_error], dtype=np.float32),
            self.exit_plane_normal.astype(np.float32),
            prev_action.astype(np.float32),
            centerline_features.astype(np.float32),
        ]).astype(np.float32)

        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def _do_action(self, action):
        self.controller_sofa.rotateZ(float(action[0]))
        self.controller_sofa.rotateX(float(action[1]))
        self.controller_sofa.insertRetract(float(action[2]))

    # ============================================================
    # Reward / info / logging
    # ============================================================
    def _compute_reward_and_info(self):
        previous_reward_features = self.reward_features.copy()
        reward_features = self._get_reward_features(previous_reward_features)
        self.reward_features = reward_features.copy()

        reward = 0.0
        reward_info = {}
        for key, feature_value in reward_features.items():
            value = self.reward_amount_dict[key] * feature_value
            if "distance" in key:
                value = value * self.cartesian_scaling_factor
            if not np.isfinite(value):
                value = 0.0
            reward_info[f"reward_{key}"] = value
            reward += value

        info = {}
        info.update(reward_features)
        info.update(reward_info)
        info["reward"] = float(reward)
        info["min_dist_to_goal"] = float(self.min_dist_this_episode)
        info["current_dist_to_goal"] = float(self._get_distance_tip_to_dest())
        info["centerline_delta_progress"] = float(self.current_centerline_delta_progress)
        info["centerline_progress_ratio"] = float(self.current_centerline_progress_ratio)
        info["centerline_distance"] = float(self.current_centerline_distance)
        info["centerline_projection_index"] = int(self.previous_centerline_projection_index)
        info["centerline_projection_rejected_jump"] = bool(self.centerline_projection_rejected_jump)
        info["centerline_projection_rejected_reason"] = str(self.centerline_projection_rejected_reason)
        info["using_centerline_reward"] = bool(self.using_centerline_reward)
        info["using_euclidean_terminal_reward"] = bool(self.using_euclidean_terminal_reward)
        info["success_10mm"] = bool(self.episode_success_10mm)
        info["success_6mm"] = bool(self.episode_success_6mm)
        info["success_3mm"] = bool(self.episode_success_3mm)
        info["success_2mm"] = bool(self.episode_success_2mm)
        info["magnetic_executor_success"] = self._latest_executor_info.get("success", False)
        info["magnetic_executor_rel_error"] = self._latest_executor_info.get("rel_error", 0.0)
        info["magnetic_executor_moved"] = self._latest_executor_info.get("moved", False)
        i_pair = self._latest_executor_info.get("I_pair", [0.0, 0.0])
        info["magnetic_executor_I_L"] = float(i_pair[0])
        info["magnetic_executor_I_R"] = float(i_pair[1])
        return float(reward), info

    def _get_reward_features(self, previous_reward_features):
        reward_features = {}

        current_dist = float(self._get_distance_tip_to_dest())
        previous_dist = float(previous_reward_features.get("tip_pos_distance_to_dest_pos", current_dist))
        delta_dist = previous_dist - current_dist
        clipped_delta_dist = float(np.clip(delta_dist, -PROGRESS_CLIP, PROGRESS_CLIP))

        self.min_dist_this_episode = min(self.min_dist_this_episode, current_dist)
        if self.min_dist_this_episode <= 0.010:
            self.episode_success_10mm = True
        if self.min_dist_this_episode <= 0.006:
            self.episode_success_6mm = True
        if self.min_dist_this_episode <= 0.003:
            self.episode_success_3mm = True
        if self.min_dist_this_episode <= 0.002:
            self.episode_success_2mm = True

        tip_pos = np.asarray(self.controller_sofa.get_pos_quat_catheter_tip()[0:3], dtype=np.float32)
        centerline_progress, centerline_seg_idx, centerline_dist = self._get_centerline_arc_progress(tip_pos)

        previous_centerline_progress = self.previous_centerline_progress
        if previous_centerline_progress is None or centerline_seg_idx < 0:
            delta_centerline_progress = 0.0
        else:
            delta_centerline_progress = centerline_progress - previous_centerline_progress
        delta_centerline_progress = float(np.clip(delta_centerline_progress, -PROGRESS_CLIP, PROGRESS_CLIP))

        self.previous_centerline_progress = centerline_progress
        self.previous_centerline_projection_index = centerline_seg_idx
        self.current_centerline_progress = float(centerline_progress)
        self.current_centerline_delta_progress = float(delta_centerline_progress)
        self.current_centerline_distance = float(centerline_dist)
        total_len = float(self.centerline_cumlength[-1]) if self.centerline_cumlength is not None else 0.0
        self.current_centerline_progress_ratio = float(centerline_progress / (total_len + 1e-9)) if total_len > 1e-9 else 0.0

        terminal_zone_active = bool(current_dist <= TERMINAL_ENTER_THRESHOLD or previous_dist <= TERMINAL_ENTER_THRESHOLD)
        self.using_centerline_reward = not terminal_zone_active
        self.using_euclidean_terminal_reward = terminal_zone_active

        reward_features["tip_pos_distance_to_dest_pos"] = current_dist
        reward_features["centerline_progress_distance"] = delta_centerline_progress if not terminal_zone_active else 0.0
        reward_features["delta_tip_pos_distance_to_dest_pos"] = clipped_delta_dist if terminal_zone_active else 0.0
        reward_features["workspace_constraint_violation"] = getattr(self.controller_sofa, "invalid_action", 0.0)
        reward_features["successful_task"] = 0.0
        reward_features["step_penalty"] = 1.0
        reward_features["terminal_escape_penalty"] = 0.0
        reward_features["terminal_recapture_bonus"] = 0.0
        reward_features["terminal_escape_failure_penalty"] = 0.0
        reward_features["milestone_10mm"] = 0.0
        reward_features["milestone_6mm"] = 0.0
        reward_features["milestone_3mm"] = 0.0

        action_delta = self._last_smoothed_action - self._prev_smoothed_action
        reward_features["action_smoothness"] = float(np.sum(np.square(action_delta)))

        if current_dist <= 0.010 and not self.reached_10mm_once:
            self.reached_10mm_once = True
            reward_features["milestone_10mm"] = 1.0
        if current_dist <= 0.006 and not self.reached_6mm_once:
            self.reached_6mm_once = True
            reward_features["milestone_6mm"] = 1.0
        if current_dist <= 0.003 and not self.reached_3mm_once:
            self.reached_3mm_once = True
            reward_features["milestone_3mm"] = 1.0

        if not self.entered_terminal_zone_once:
            if current_dist <= TERMINAL_ENTER_THRESHOLD:
                self.entered_terminal_zone_once = True
                self.was_in_terminal_zone_prev_step = True
        else:
            if current_dist > TERMINAL_EXIT_THRESHOLD:
                self.terminal_bad_steps += 1
                reward_features["terminal_escape_penalty"] = 1.0
                self.was_in_terminal_zone_prev_step = False
                if self.terminal_bad_steps >= TERMINAL_BAD_STEPS_LIMIT:
                    self.terminal_escape_failed = True
                    reward_features["terminal_escape_failure_penalty"] = 1.0
            elif current_dist <= TERMINAL_ENTER_THRESHOLD:
                self.terminal_bad_steps = 0
                if not self.was_in_terminal_zone_prev_step:
                    reward_features["terminal_recapture_bonus"] = 1.0
                self.was_in_terminal_zone_prev_step = True

        if terminal_zone_active:
            epsilon = 0.0001
            reward_features["terminal_log_progress_bonus"] = float(
                np.log(previous_dist + epsilon) - np.log(current_dist + epsilon)
            )
        else:
            reward_features["terminal_log_progress_bonus"] = 0.0

        if current_dist <= TARGET_DISTANCE_THRESHOLD:
            reward_features["successful_task"] = 1.0
            self.episode_success = True

        return reward_features

    def _print_step_log(self, reward, info):
        def yesno(v):
            return "YES" if bool(v) else "NO"

        cur_d = info["current_dist_to_goal"] * 1000.0
        min_d = info["min_dist_to_goal"] * 1000.0
        cl_d = info["centerline_delta_progress"] * 1000.0
        cl_dist = info["centerline_distance"] * 1000.0
        rel_err = info["magnetic_executor_rel_error"] * 100.0
        i_l = info["magnetic_executor_I_L"]
        i_r = info["magnetic_executor_I_R"]
        reject = info.get("centerline_projection_rejected_reason", "")

        print(
            f"[GUI_POLICY] Ep={self.episode_idx} Step={self.step_in_episode} Global={self.global_step} | "
            f"R={float(reward):.3f} SumR={self.episode_reward:.3f} | "
            f"curD={cur_d:.2f}mm minD={min_d:.2f}mm | "
            f"CL_d={cl_d:.3f}mm CL_r={info['centerline_progress_ratio']:.3f} "
            f"CL_dist={cl_dist:.2f}mm Reject={reject if reject else 'NO'} "
            f"UseCL={yesno(info['using_centerline_reward'])} "
            f"UseTerm={yesno(info['using_euclidean_terminal_reward'])} | "
            f"10mm={yesno(info['success_10mm'])} 6mm={yesno(info['success_6mm'])} "
            f"3mm={yesno(info['success_3mm'])} 2mm={yesno(info['success_2mm'])} | "
            f"Exec={yesno(info['magnetic_executor_success'])} RelErr={rel_err:.2f}% "
            f"Moved={yesno(info['magnetic_executor_moved'])} I=[{i_l:.3f},{i_r:.3f}]"
        )

    # ============================================================
    # Centerline utilities
    # ============================================================
    def _init_centerline_direction_and_cumlength(self):
        if self.centerline_points is None or len(self.centerline_points) < 2:
            return
        init_tip = np.asarray(self.controller_sofa.get_pos_quat_catheter_tip()[0:3], dtype=np.float32)
        p0 = self.centerline_points[0]
        p1 = self.centerline_points[-1]
        forward_cost = float(np.linalg.norm(p0 - init_tip) + np.linalg.norm(p1 - self.target_position))
        reverse_cost = float(np.linalg.norm(p1 - init_tip) + np.linalg.norm(p0 - self.target_position))
        if reverse_cost + 1e-9 < forward_cost:
            self.centerline_points = self.centerline_points[::-1].copy()
            print("[GUI_POLICY][CENTERLINE] reversed_for_progress=True")
        else:
            print("[GUI_POLICY][CENTERLINE] reversed_for_progress=False")

        seg_lengths = np.linalg.norm(np.diff(self.centerline_points, axis=0), axis=1)
        self.centerline_cumlength = np.concatenate(([0.0], np.cumsum(seg_lengths))).astype(np.float32)
        print(
            "[GUI_POLICY][CENTERLINE] total_len_mm=",
            float(self.centerline_cumlength[-1] * 1000.0),
            "start_to_target_mm=",
            float(np.linalg.norm(self.centerline_points[0] - self.target_position) * 1000.0),
            "end_to_target_mm=",
            float(np.linalg.norm(self.centerline_points[-1] - self.target_position) * 1000.0),
        )

    def _resample_centerline_points_1mm(self, points):
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
            return points

        segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        total_length = float(np.sum(segment_lengths))
        if total_length < 1e-9:
            return points

        target_spacing = 0.001
        try:
            k = min(3, len(points) - 1)
            tck, _ = splprep([points[:, 0], points[:, 1], points[:, 2]], s=0.0, k=k)
            expected_count = max(2, int(np.ceil(total_length / target_spacing)) + 1)
            dense_count = max(expected_count * 5, len(points) * 10)
            dense_u = np.linspace(0.0, 1.0, dense_count)
            dense_xyz = np.asarray(splev(dense_u, tck), dtype=np.float32).T
            dense_seg = np.linalg.norm(np.diff(dense_xyz, axis=0), axis=1)
            dense_cum = np.concatenate(([0.0], np.cumsum(dense_seg)))
            dense_total = float(dense_cum[-1])
            target_dist = np.arange(0.0, dense_total, target_spacing, dtype=np.float32)
            if target_dist.size == 0 or target_dist[-1] < dense_total:
                target_dist = np.append(target_dist, dense_total)
            x = np.interp(target_dist, dense_cum, dense_xyz[:, 0])
            y = np.interp(target_dist, dense_cum, dense_xyz[:, 1])
            z = np.interp(target_dist, dense_cum, dense_xyz[:, 2])
            return np.stack([x, y, z], axis=1).astype(np.float32)
        except Exception:
            cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
            target_dist = np.arange(0.0, cumulative[-1], target_spacing, dtype=np.float32)
            if target_dist.size == 0 or target_dist[-1] < cumulative[-1]:
                target_dist = np.append(target_dist, cumulative[-1])
            x = np.interp(target_dist, cumulative, points[:, 0])
            y = np.interp(target_dist, cumulative, points[:, 1])
            z = np.interp(target_dist, cumulative, points[:, 2])
            return np.stack([x, y, z], axis=1).astype(np.float32)

    def _get_centerline_light_features(self, tip_pos):
        features = np.zeros(CENTERLINE_FEATURE_DIM, dtype=np.float32)
        if self.centerline_points is None or self.centerline_cumlength is None:
            return features

        progress, seg_idx, _, _, _ = self._get_centerline_projection_state(tip_pos)
        total_len = float(self.centerline_cumlength[-1])
        if total_len < 1e-9 or seg_idx < 0:
            return features
        features[0] = float(np.clip(progress / total_len, 0.0, 1.0))
        features[1:] = self._get_lookahead_relative_vectors(tip_pos)
        return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def _get_centerline_projection_state(self, tip_pos):
        if self.centerline_points is None or self.centerline_cumlength is None or len(self.centerline_points) < 2:
            return 0.0, -1, 0.0, np.asarray(tip_pos, dtype=np.float32), np.zeros(3, dtype=np.float32)

        points = np.asarray(self.centerline_points, dtype=np.float32)
        num_segments = len(points) - 1
        tip_pos = np.asarray(tip_pos, dtype=np.float32)

        prev_idx = int(self.previous_centerline_projection_index)
        if 0 <= prev_idx < num_segments:
            search_start = max(0, prev_idx - CENTERLINE_PROJECTION_SEARCH_BACK)
            search_end = min(num_segments, prev_idx + CENTERLINE_PROJECTION_SEARCH_FORWARD + 1)
            candidate_indices = np.arange(search_start, search_end, dtype=np.int32)
        else:
            candidate_indices = np.arange(0, num_segments, dtype=np.int32)

        seg_start = points[candidate_indices]
        seg_end = points[candidate_indices + 1]
        seg_vec = seg_end - seg_start
        tip_vec = tip_pos[None, :] - seg_start
        seg_len_sq = np.sum(seg_vec * seg_vec, axis=1)
        seg_len_sq = np.maximum(seg_len_sq, 1e-12)
        t = np.sum(tip_vec * seg_vec, axis=1) / seg_len_sq
        t = np.clip(t, 0.0, 1.0)
        proj = seg_start + t[:, None] * seg_vec
        dists = np.linalg.norm(proj - tip_pos[None, :], axis=1)

        best_local_idx = int(np.argmin(dists))
        best_seg_idx = int(candidate_indices[best_local_idx])
        best_t = float(t[best_local_idx])
        closest_dist = float(dists[best_local_idx])
        best_proj = proj[best_local_idx]
        seg = points[best_seg_idx + 1] - points[best_seg_idx]
        seg_length = float(np.linalg.norm(seg))
        tangent = seg / (seg_length + 1e-9)
        progress = float(self.centerline_cumlength[best_seg_idx] + best_t * seg_length)
        return progress, best_seg_idx, closest_dist, best_proj.astype(np.float32), tangent.astype(np.float32)

    def _get_centerline_arc_progress(self, tip_pos):
        progress, best_seg_idx, closest_dist, _, _ = self._get_centerline_projection_state(tip_pos)
        previous_progress = self.previous_centerline_progress
        previous_idx = int(self.previous_centerline_projection_index)
        self.centerline_projection_rejected_jump = False
        self.centerline_projection_rejected_reason = ""

        if previous_progress is not None and best_seg_idx >= 0 and previous_idx >= 0 and np.isfinite(progress):
            previous_progress_f = float(previous_progress)
            if progress < previous_progress_f - CENTERLINE_MAX_BACKTRACK:
                self.centerline_projection_rejected_jump = True
                self.centerline_projection_rejected_reason = "backward"
                progress = previous_progress_f
                best_seg_idx = previous_idx
            elif progress > previous_progress_f + CENTERLINE_MAX_FORWARD_JUMP:
                self.centerline_projection_rejected_jump = True
                self.centerline_projection_rejected_reason = "forward"
                progress = previous_progress_f
                best_seg_idx = previous_idx

        return float(progress), int(best_seg_idx), float(closest_dist)

    def _get_lookahead_relative_vectors(self, tip_pos):
        dim = 3 * NUM_CENTERLINE_LOOKAHEAD_POINTS
        if self.centerline_points is None or len(self.centerline_points) == 0:
            return np.zeros(dim, dtype=np.float32)

        _, seg_idx, _, _, _ = self._get_centerline_projection_state(tip_pos)
        closest_idx = max(int(seg_idx), 0)
        last_idx = len(self.centerline_points) - 1
        lookahead_vectors = []
        for offset in CENTERLINE_LOOKAHEAD_OFFSETS:
            idx = min(closest_idx + int(offset), last_idx)
            lookahead_vectors.append(self.centerline_points[idx] - tip_pos)
        out = np.asarray(lookahead_vectors, dtype=np.float32).reshape(-1)
        if out.shape[0] != dim:
            out = np.pad(out, (0, max(0, dim - out.shape[0])))[:dim]
        return out.astype(np.float32)

    # ============================================================
    # Geometry / ROS helpers
    # ============================================================
    def _compute_exit_plane_normal(self):
        if self.centerline_points is None or self.target_position is None or len(self.centerline_points) == 0:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)
        distances_to_target = np.linalg.norm(self.centerline_points - self.target_position, axis=1)
        idx_1mm = int(np.argmin(np.abs(distances_to_target - 0.001)))
        point_1mm = self.centerline_points[idx_1mm]
        direction_vec = self.target_position - point_1mm
        norm = float(np.linalg.norm(direction_vec))
        if norm > 1e-9:
            return (direction_vec / norm).astype(np.float32)
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def _compute_cartesian_scaling_factor(self):
        try:
            vessel_positions = np.asarray(self.environment.get_vessel_tree_positions(), dtype=np.float32)
            if vessel_positions.ndim == 2 and vessel_positions.shape[1] == 3 and vessel_positions.shape[0] > 1:
                bbox_diag = float(np.linalg.norm(np.min(vessel_positions, axis=0) - np.max(vessel_positions, axis=0)))
                if np.isfinite(bbox_diag) and bbox_diag > 1e-9:
                    print("[GUI_POLICY] cartesian_scaling_factor=", 1.0 / bbox_diag, "bbox_diag=", bbox_diag)
                    return 1.0 / bbox_diag
        except Exception:
            pass
        print("[GUI_POLICY][WARN] fallback cartesian_scaling_factor=1.0")
        return 1.0

    def _get_distance_tip_to_dest(self):
        tip = np.asarray(self.controller_sofa.get_pos_quat_catheter_tip()[0:3], dtype=np.float32)
        return float(np.linalg.norm(self.target_position - tip))

    def _update_dynamic_tip_world_offset(self):
        tip_sofa = np.asarray(self.controller_sofa.get_pos_quat_catheter_tip()[0:3], dtype=np.float32)
        if DYNAMIC_TIP_WORLD_ALIGN:
            self.tip_world_offset = DESIRED_INITIAL_TIP_WORLD - tip_sofa
        else:
            self.tip_world_offset = np.zeros(3, dtype=np.float32)
        print(
            "[GUI_POLICY] dynamic tip-world align | "
            f"initial_tip_sofa={tip_sofa} | "
            f"desired_initial_tip_world={DESIRED_INITIAL_TIP_WORLD} | "
            f"tip_world_offset={self.tip_world_offset} | "
            f"initial_tip_world={tip_sofa + self.tip_world_offset}"
        )

    def _get_tip_world_for_magnetic_executor(self):
        tip_sofa = np.asarray(self.controller_sofa.get_pos_quat_catheter_tip()[0:3], dtype=np.float32)
        return tip_sofa + self.tip_world_offset

    def _init_ros_magnetic_executor(self):
        try:
            if not rospy.core.is_initialized():
                rospy.init_node("sofa_gui_policy_runner", anonymous=True, disable_signals=True)
        except Exception:
            pass
        self._execute_request_pub = rospy.Publisher("/magnetic/execute_request", Float64MultiArray, queue_size=1)
        rospy.Subscriber("/magnetic/execute_result", Float64MultiArray, self._execute_result_callback, queue_size=1)
        time.sleep(0.2)
        print("[GUI_POLICY] ROS magnetic executor client initialized.")

    def _execute_result_callback(self, msg):
        if len(msg.data) < 9:
            return
        step_id = int(msg.data[0])
        result = {
            "step_id": step_id,
            "success": bool(msg.data[1] > 0.5),
            "B_actual": np.array([msg.data[2], msg.data[3], msg.data[4]], dtype=np.float32),
            "I_pair": np.array([msg.data[5], msg.data[6]], dtype=np.float32),
            "rel_error": float(msg.data[7]),
            "moved": bool(msg.data[8] > 0.5),
        }
        self._latest_execute_result = result
        self._execute_result_step_id = step_id

    def _request_magnetic_execution(self, tip_world, B_target):
        self._magnetic_step_id += 1
        step_id = self._magnetic_step_id
        self._latest_execute_result = None

        msg = Float64MultiArray(data=[
            float(step_id),
            float(tip_world[0]),
            float(tip_world[1]),
            float(tip_world[2]),
            float(B_target[0]),
            float(B_target[1]),
            float(B_target[2]),
        ])
        self._execute_request_pub.publish(msg)

        start = time.time()
        while time.time() - start < self._execute_timeout:
            if self._latest_execute_result is not None and self._latest_execute_result.get("step_id") == step_id:
                return self._latest_execute_result
            rospy.sleep(0.005)

        raise TimeoutError(f"Wait /magnetic/execute_result timeout step_id={step_id}")


def createScene(root_node):
    """SOFA official GUI scene entry."""
    if not rospy.core.is_initialized():
        rospy.init_node("sofa_gui_joint_test", anonymous=True, disable_signals=True)

    scene_result = base_scene.createScene(
        root_node,
        debug_rendering=DEBUG_RENDERING,
        positioning_camera=POSITIONING_CAMERA,
        force_model=FORCE_MODEL,
        vessel_alpha=VESSEL_ALPHA,
    )

    policy_controller = GuiPolicyClosedLoopController(
        root_node=root_node,
        scene_result=scene_result,
        model_path=MODEL_PATH,
    )
    root_node.addObject(policy_controller)
    root_node.gui_policy_controller_ref = policy_controller

    return scene_result
