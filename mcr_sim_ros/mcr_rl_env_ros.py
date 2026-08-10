from typing import Union, Tuple, Optional, Any, Dict
from pathlib import Path
from enum import Enum, unique
import gymnasium.spaces as spaces
import numpy as np
from collections import defaultdict
from scipy.interpolate import splprep, splev

from mcr_sim_ros.rl_core_ros.base_ros import SofaEnv, RenderMode, RenderFramework
from mcr_sim_ros.mcr_controller_sofa_ros import ControllerSofa
from std_msgs.msg import Float64MultiArray
import rospy
import time

HERE = Path(__file__).resolve().parent
FLAT_SCENE_DESCRIPTION_FILE_PATH = HERE / "scene_description_2d.py"
AORTIC_SCENE_DESCRIPTION_FILE_PATH = HERE.parent / "scene" / "example_aortic_arch_ros.py"
FLAT_CATHETER_DESTINATION_EXIT_POINT = np.array([0.101129, 0.0238015, 0.002])
AORTIC_CATHETER_DESTINATION_EXIT_POINT = np.array([-0.0101583, -0.180636, 0.0345185])


@unique
class ObservationType(Enum):
    RGB = 0
    STATE = 1


@unique
class ActionType(Enum):
    DISCRETE = 0
    CONTINUOUS = 1


@unique
class EnvType(Enum):
    FLAT = 0
    AORTIC = 1


class MCREnv(SofaEnv):
    """Magnetic Continuum Robot Environment, ROS strict-sync version.

    Lightweight centerline-guided version with terminal-zone switch.

    Key logic:
    1. Outside 10 mm: use centerline arc-length progress as the main dense reward.
    2. Inside 10 mm: disable centerline progress reward and switch back to Euclidean
       distance refinement, because final 3/2 mm success is judged by tip-target distance.
    3. Observation only adds two centerline-related abilities:
       - centerline progress ratio: 1-D;
       - multiple lookahead vectors: 8 x 3-D.
       Final state dimension = original 24-D + 1 + 24 = 49-D.
    4. Centerline direction is checked and corrected in _init_sim so positive progress
       always means moving from start to target.
    5. NaN fail-safe is kept to prevent SOFA numerical divergence from poisoning replay buffer.
    6. ROS strict closed-loop execution is preserved:
       action -> B_target -> /magnetic/execute_request -> B_actual -> SOFA animate.
    7. SOFA/world axes are assumed aligned. Only a per-episode translation is applied
       before sending tip_world to the PyBullet magnetic executor.
    """

    def __init__(
        self,
        image_shape: Tuple[int, int] = (400, 400),
        create_scene_kwargs: Optional[dict] = None,
        observation_type: ObservationType = ObservationType.STATE,
        action_type: ActionType = ActionType.CONTINUOUS,
        time_step: float = 0.1,
        frame_skip: int = 1,
        settle_steps: int = 20,
        render_mode: RenderMode = RenderMode.HUMAN,
        render_framework: RenderFramework = RenderFramework.PYGLET,
        reward_amount_dict={
            # 欧氏绝对距离不作为 shaping，只作为日志/成功判定。
            "tip_pos_distance_to_dest_pos": 0.0,

            # 欧氏距离进步奖励仅在 10 mm 终端区内启用；终端区外 feature 被置 0。
            "delta_tip_pos_distance_to_dest_pos": 80.0,

            # 中心线弧长进度奖励仅在 10 mm 之外启用；进入 10 mm 后 feature 被置 0。
            "centerline_progress_distance": 260.0,

            "workspace_constraint_violation": -6.0,
            "step_penalty": -0.001,

            # 终端区精修：10 mm 内启用，配合欧氏距离进步压到 3/2 mm。
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
        },
        target_position: Optional[np.ndarray] = None,
        env_type: EnvType = EnvType.FLAT,
        target_distance_threshold: float = 0.010,
        num_catheter_tracking_points: int = 4,
        max_episode_steps: int = 1000,
    ):
        if not isinstance(create_scene_kwargs, dict):
            create_scene_kwargs = {}
        create_scene_kwargs["image_shape"] = image_shape

        self.target_distance_threshold = target_distance_threshold
        self.num_catheter_tracking_points = num_catheter_tracking_points
        self.max_episode_steps = int(max_episode_steps)
        self._elapsed_steps = 0

        # ------------------------------------------------------------
        # ROS/PyBullet execution coordinate alignment
        # ------------------------------------------------------------
        # 你前面已经确认 SOFA 和 world 坐标轴方向一致，因此这里不做旋转，
        # 只做每个 episode 的平移对齐：
        #     tip_world = tip_sofa + tip_world_offset
        # 每次 reset 后，把初始 tip_sofa 对齐到 PyBullet 可达、安全位置
        # desired_initial_tip_world = [0.75, 0.20, 0.70]。
        self.desired_initial_tip_world = np.asarray(
            create_scene_kwargs.get("desired_initial_tip_world", [0.75, 0.20, 0.70]),
            dtype=np.float32,
        )
        self.dynamic_tip_world_align = bool(
            create_scene_kwargs.get("dynamic_tip_world_align", True)
        )
        self.tip_world_offset = np.asarray(
            create_scene_kwargs.get("tip_world_offset", [0.0, 0.0, 0.0]),
            dtype=np.float32,
        )
        self.initial_tip_sofa_for_world_align = None
        self.debug_ros_sync = bool(create_scene_kwargs.get("debug_ros_sync", False))

        self.env_type = env_type
        if self.env_type == EnvType.FLAT:
            self.target_position = target_position if target_position is not None else FLAT_CATHETER_DESTINATION_EXIT_POINT
            self.scene_path = FLAT_SCENE_DESCRIPTION_FILE_PATH
        elif self.env_type == EnvType.AORTIC:
            self.target_position = target_position if target_position is not None else AORTIC_CATHETER_DESTINATION_EXIT_POINT
            self.scene_path = AORTIC_SCENE_DESCRIPTION_FILE_PATH
        else:
            raise ValueError(f"Unsupported env_type: {self.env_type}")

        super().__init__(
            scene_path=self.scene_path,
            time_step=time_step,
            frame_skip=frame_skip,
            render_mode=render_mode,
            render_framework=render_framework,
            create_scene_kwargs=create_scene_kwargs,
        )

        self.observation_type = observation_type
        self._settle_steps = settle_steps

        # 原始 24 维 + centerline progress ratio 1 维 + 8 个领航点 24 维 = 49 维。
        self.num_centerline_lookahead_points = 8
        self.centerline_lookahead_offsets = (3, 6, 10, 15, 21, 28, 36, 45)
        self.centerline_feature_dim = 1 + 3 * self.num_centerline_lookahead_points

        if self.observation_type == ObservationType.STATE:
            observations_size = 24 + self.centerline_feature_dim
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(observations_size,),
                dtype=np.float32,
            )
        elif self.observation_type == ObservationType.RGB:
            self.observation_space = spaces.Box(low=0, high=255, shape=image_shape + (3,), dtype=np.uint8)

        action_dimensionality = 3
        self.action_type = action_type
        if action_type == ActionType.CONTINUOUS:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dimensionality,), dtype=np.float32)
        else:
            raise NotImplementedError("Only continuous action space is implemented.")

        self.reward_info = {}
        self.reward_features = {}
        self.reward_amount_dict = defaultdict(float)
        self.reward_amount_dict.update(reward_amount_dict)

        self.previous_closest_idx = None
        self.previous_centerline_progress = None
        self.previous_centerline_projection_index = -1
        self.centerline_cumlength = None
        self.centerline_points = None

        self.episode_success = False
        self.episode_success_2mm = False
        self.min_dist_this_episode = np.inf
        self.episode_success_10mm = False
        self.episode_success_6mm = False
        self.episode_success_3mm = False
        self.action_smoothing_alpha = 0.2

        self._ros_sync_initialized = False
        self._magnetic_step_id = 0
        self._latest_execute_result = None
        self._execute_result_step_id = -1
        self._execute_timeout = 3.0
        self._latest_executor_info = {
            "success": False,
            "rel_error": 0.0,
            "moved": False,
            "I_pair": [0.0, 0.0],
        }
        self._init_magnetic_executor_ros()

        self.progress_clip = 0.003

        # ------------------------------------------------------------
        # Centerline projection tracking
        # ------------------------------------------------------------
        # 原先每一步都在整条中心线上做“全局最近点投影”，在主动脉弓、
        # S 型、回弯血管中容易发生跳段：tip 稍微偏离中心线后，最近点
        # 可能从中段跳回起点附近，导致 CL_r 突然归零、奖励异常。
        #
        # 这里加入历史约束：
        #   1) 第一步仍然允许全局搜索；
        #   2) 后续只在上一帧投影 segment 附近的局部窗口搜索；
        #   3) 若 progress 大幅回退，则拒绝该投影跳变，保持上一帧进度；
        #   4) 若 progress 单步大幅前跳，也拒绝该投影跳变，避免 CL_r 突然跳到 0.7/0.8。
        #
        # 注意：中心线已经在 _resample_centerline_points_1mm() 中重采样为约 1mm 间距。
        # 因此 search_back=5、search_forward=8 大致表示只允许在上一帧附近
        # 后退 5mm、前进 8mm 的局部窗口内投影。
        self.centerline_projection_search_back = int(
            create_scene_kwargs.get("centerline_projection_search_back", 5)
        )
        self.centerline_projection_search_forward = int(
            create_scene_kwargs.get("centerline_projection_search_forward", 8)
        )
        self.centerline_max_backtrack = float(
            create_scene_kwargs.get("centerline_max_backtrack", 0.0005)
        )
        self.centerline_max_forward_jump = float(
            create_scene_kwargs.get("centerline_max_forward_jump", 0.003)
        )
        self.centerline_projection_rejected_jump = False
        self.centerline_projection_rejected_reason = ""

        # 10 mm 是奖励逻辑切换边界：外部中心线，内部欧氏距离精修。
        self.terminal_enter_threshold = 0.010
        self.terminal_exit_threshold = 0.012
        self.terminal_bad_steps_limit = 30
        self.non_finite_failure = False

    def reset(
        self,
        seed: Union[int, np.random.SeedSequence, None] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Union[np.ndarray, None], Dict]:
        # Force a full reload of the scene to pick a random environment model.
        if self._initialized:
            if hasattr(self, "sofa_simulation") and self._sofa_root_node is not None:
                self.sofa_simulation.unload(self._sofa_root_node)
            self._initialized = False

        super().reset(seed)
        self._elapsed_steps = 0
        self.episode_success = False
        self.episode_success_2mm = False
        self.min_dist_this_episode = np.inf
        self.episode_success_10mm = False
        self.episode_success_6mm = False
        self.episode_success_3mm = False

        self.previous_centerline_progress = None
        self.previous_centerline_projection_index = -1
        self.current_centerline_progress = 0.0
        self.current_centerline_delta_progress = 0.0
        self.current_centerline_distance = 0.0
        self.current_centerline_progress_ratio = 0.0
        self.using_centerline_reward = False
        self.using_euclidean_terminal_reward = False
        self.non_finite_failure = False

        self.entered_terminal_zone_once = False
        self.terminal_bad_steps = 0
        self.was_in_terminal_zone_prev_step = False
        self.terminal_recovered = False
        self.terminal_escape_failed = False
        self.terminal_failed_due_to_timeout = False

        self.reached_10mm_once = False
        self.reached_6mm_once = False
        self.reached_3mm_once = False

        self.mcr_controller_sofa_ros.reset()

        self.reward_info = {}
        self.reward_features = {}
        self.reward_features["tip_pos_distance_to_dest_pos"] = self._get_distance_tip_to_dest()

        self._last_smoothed_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self._prev_smoothed_action = np.zeros(self.action_space.shape, dtype=np.float32)

        for _ in range(self._settle_steps):
            self.sofa_simulation.animate(self._sofa_root_node, self._sofa_root_node.getDt())

        # 不同血管/分支的初始 tip_sofa 不同，因此每次 reset 后都重新计算
        # SOFA->world 的平移，使初始 tip_world 固定到 [0.75, 0.20, 0.70]。
        self._update_dynamic_tip_world_offset()

        return self._get_observation(image_observation=self._maybe_update_rgb_buffer()), {}

    def step(self, action: Any) -> Tuple[Union[np.ndarray, dict], float, bool, bool, dict]:
        action_np = np.array(action, dtype=np.float32)
        action_np = np.nan_to_num(action_np, nan=0.0, posinf=1.0, neginf=-1.0)
        action_np = np.clip(action_np, -1.0, 1.0).astype(np.float32)

        previous_smoothed_action = getattr(self, "_last_smoothed_action", np.zeros_like(action_np))
        smoothed_action = self.action_smoothing_alpha * action_np + (1.0 - self.action_smoothing_alpha) * previous_smoothed_action
        self._prev_smoothed_action = previous_smoothed_action.copy()
        self._last_smoothed_action = smoothed_action.copy()

        if getattr(self, "_ros_sync_initialized", False):
            self._do_action(smoothed_action)
            B_target = self.mcr_controller_sofa_ros.get_mag_field_des().copy()
            tip_world = self._get_tip_world_for_magnetic_executor()
            result = self._request_magnetic_execution(tip_world, B_target)
            self._latest_executor_info = result

            ideal_B = B_target.copy()
            self.mcr_controller_sofa_ros.set_mag_field_des(result["B_actual"])

            image_observation = None
            for _ in range(self.frame_skip):
                image_observation = self.sofa_simulation.animate(self._sofa_root_node, self._sofa_root_node.getDt())

            # 保持控制器内部 ideal_B 状态，避免下一步动作累计逻辑受 B_actual 覆盖影响。
            self.mcr_controller_sofa_ros.set_mag_field_des(ideal_B)
        else:
            image_observation = super().step(smoothed_action)

        self._elapsed_steps += 1
        observation = self._get_observation(image_observation)
        reward = self._get_reward()

        terminated = self._get_done()
        non_finite_failure = False

        if self.observation_type == ObservationType.STATE and not np.all(np.isfinite(observation)):
            non_finite_failure = True
            observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        if not np.isfinite(reward):
            non_finite_failure = True
            reward = -100.0

        if non_finite_failure:
            self.non_finite_failure = True
            terminated = True

        if (
            getattr(self, "is_out_of_bounds", False)
            or getattr(self, "terminal_escape_failed", False)
            or getattr(self, "non_finite_failure", False)
        ):
            terminated = True

        truncated = (self._elapsed_steps >= self.max_episode_steps) and (not terminated)

        info = self._get_info(terminated=terminated, truncated=truncated)
        if truncated:
            info["TimeLimit.truncated"] = True

        return observation, reward, terminated, truncated, info

    def _get_observation(self, image_observation: Union[np.ndarray, None]) -> Union[np.ndarray, dict]:
        if self.observation_type == ObservationType.RGB:
            return image_observation
        elif self.observation_type == ObservationType.STATE:
            tip_pose = self.mcr_controller_sofa_ros.get_pos_quat_catheter_tip()
            tip_pos = tip_pose[0:3]

            offset_vec = tip_pos - self.target_position
            current_dist = np.linalg.norm(offset_vec)
            exit_normal = getattr(self, "exit_plane_normal", np.array([0.0, 0.0, 1.0], dtype=np.float32))
            axial = np.dot(offset_vec, exit_normal) * exit_normal
            lateral_vec = offset_vec - axial
            lateral_error = np.linalg.norm(lateral_vec)
            prev_action = getattr(self, "_last_smoothed_action", np.zeros(3, dtype=np.float32))

            centerline_features = self._get_centerline_light_features(tip_pos)

            obs = {
                "position-quaternion-catheter-tip": tip_pose,
                "magnetic-field-des": self.mcr_controller_sofa_ros.get_mag_field_des(),
                "offset_vec": offset_vec.astype(np.float32),
                "current_dist": np.array([current_dist], dtype=np.float32),
                "lateral_vec": lateral_vec.astype(np.float32),
                "lateral_error": np.array([lateral_error], dtype=np.float32),
                "exit_normal": exit_normal.astype(np.float32),
                "prev_action": prev_action.astype(np.float32),
                "centerline_light_features": centerline_features.astype(np.float32),
            }
            observation = np.concatenate(tuple(obs.values())).astype(np.float32)
            return np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            return {}

    def _get_reward_features(self, previous_reward_features: dict) -> dict:
        reward_features = {}

        current_dist = float(self._get_distance_tip_to_dest())
        if not np.isfinite(current_dist):
            current_dist = 1e3
            self.non_finite_failure = True
        self.is_out_of_bounds = False

        self.min_dist_this_episode = min(self.min_dist_this_episode, current_dist)
        if self.min_dist_this_episode <= 0.010:
            self.episode_success_10mm = True
        if self.min_dist_this_episode <= 0.006:
            self.episode_success_6mm = True
        if self.min_dist_this_episode <= 0.003:
            self.episode_success_3mm = True
        if self.min_dist_this_episode <= 0.002:
            self.episode_success_2mm = True

        previous_dist = previous_reward_features.get("tip_pos_distance_to_dest_pos", current_dist)
        delta_dist = previous_dist - current_dist
        clipped_delta_dist = float(np.clip(delta_dist, -self.progress_clip, self.progress_clip))

        tip_pos = self.mcr_controller_sofa_ros.get_pos_quat_catheter_tip()[0:3]
        centerline_progress, centerline_seg_idx, centerline_dist = self._get_centerline_arc_progress(tip_pos)

        previous_centerline_progress = getattr(self, "previous_centerline_progress", None)
        if previous_centerline_progress is None or centerline_seg_idx < 0:
            delta_centerline_progress = 0.0
        else:
            # Centerline is ordered start -> target in _init_sim.
            delta_centerline_progress = centerline_progress - previous_centerline_progress

        delta_centerline_progress = float(np.clip(delta_centerline_progress, -self.progress_clip, self.progress_clip))

        self.previous_centerline_progress = centerline_progress
        self.previous_centerline_projection_index = centerline_seg_idx
        self.current_centerline_distance = float(centerline_dist)
        self.current_centerline_progress = float(centerline_progress)
        self.current_centerline_delta_progress = float(delta_centerline_progress)
        total_len = float(self.centerline_cumlength[-1]) if getattr(self, "centerline_cumlength", None) is not None else 0.0
        self.current_centerline_progress_ratio = float(centerline_progress / (total_len + 1e-9)) if total_len > 1e-9 else 0.0

        # 关键切换逻辑：
        # 10 mm 外：中心线进度奖励生效，欧氏距离进步奖励关闭。
        # 10 mm 内：中心线进度奖励关闭，启用欧氏距离进步 + terminal_log_progress_bonus 做精修。
        terminal_zone_active = bool(current_dist <= self.terminal_enter_threshold or previous_dist <= self.terminal_enter_threshold)
        self.using_centerline_reward = not terminal_zone_active
        self.using_euclidean_terminal_reward = terminal_zone_active

        reward_features["tip_pos_distance_to_dest_pos"] = current_dist
        reward_features["centerline_progress_distance"] = delta_centerline_progress if not terminal_zone_active else 0.0
        reward_features["delta_tip_pos_distance_to_dest_pos"] = clipped_delta_dist if terminal_zone_active else 0.0

        reward_features["workspace_constraint_violation"] = self.mcr_controller_sofa_ros.invalid_action
        reward_features["successful_task"] = 0.0
        reward_features["step_penalty"] = 1.0
        reward_features["terminal_escape_penalty"] = 0.0
        reward_features["terminal_recapture_bonus"] = 0.0
        reward_features["terminal_escape_failure_penalty"] = 0.0
        reward_features["milestone_10mm"] = 0.0
        reward_features["milestone_6mm"] = 0.0
        reward_features["milestone_3mm"] = 0.0

        action_delta = getattr(self, "_last_smoothed_action", np.zeros(3, dtype=np.float32)) - getattr(
            self, "_prev_smoothed_action", np.zeros(3, dtype=np.float32)
        )
        reward_features["action_smoothness"] = float(np.sum(np.square(action_delta)))

        if current_dist <= 0.010 and not getattr(self, "reached_10mm_once", False):
            self.reached_10mm_once = True
            reward_features["milestone_10mm"] = 1.0

        if current_dist <= 0.006 and not getattr(self, "reached_6mm_once", False):
            self.reached_6mm_once = True
            reward_features["milestone_6mm"] = 1.0

        if current_dist <= 0.003 and not getattr(self, "reached_3mm_once", False):
            self.reached_3mm_once = True
            reward_features["milestone_3mm"] = 1.0

        if not self.entered_terminal_zone_once:
            if current_dist <= self.terminal_enter_threshold:
                self.entered_terminal_zone_once = True
                self.was_in_terminal_zone_prev_step = True
        else:
            if current_dist > self.terminal_exit_threshold:
                self.terminal_bad_steps += 1
                reward_features["terminal_escape_penalty"] = 1.0
                self.was_in_terminal_zone_prev_step = False

                if self.terminal_bad_steps >= self.terminal_bad_steps_limit:
                    self.terminal_escape_failed = True
                    reward_features["terminal_escape_failure_penalty"] = 1.0

            elif current_dist <= self.terminal_enter_threshold:
                self.terminal_bad_steps = 0
                if not self.was_in_terminal_zone_prev_step:
                    self.terminal_recovered = True
                    reward_features["terminal_recapture_bonus"] = 1.0
                self.was_in_terminal_zone_prev_step = True
            else:
                pass

        if terminal_zone_active:
            epsilon = 0.0001
            delta_log = np.log(previous_dist + epsilon) - np.log(current_dist + epsilon)
            reward_features["terminal_log_progress_bonus"] = float(delta_log)
        else:
            reward_features["terminal_log_progress_bonus"] = 0.0

        if current_dist <= self.target_distance_threshold:
            reward_features["successful_task"] = 1.0
            self.episode_success = True
            self.episode_success_2mm = bool(current_dist <= 0.002 + 1e-12)
            self.is_out_of_bounds = True

        return reward_features

    def _get_reward(self) -> float:
        reward = 0.0
        self.reward_info = {}
        reward_features = self._get_reward_features(previous_reward_features=self.reward_features)
        self.reward_features = reward_features.copy()

        for key, value in reward_features.items():
            value = self.reward_amount_dict[key] * value
            if "distance" in key:
                value = value * self.cartesian_scaling_factor
            if not np.isfinite(value):
                print(f"[WARN] Non-finite reward term: {key}, feature={reward_features.get(key)}, value={value}")
                value = 0.0
            self.reward_info[f"reward_{key}"] = value
            reward += self.reward_info[f"reward_{key}"]

        if not np.isfinite(reward):
            print("[WARN] Non-finite total reward, fallback to -100.")
            reward = -100.0
            self.non_finite_failure = True

        self.reward_info["reward"] = reward
        return float(reward)

    def _get_done(self) -> bool:
        return getattr(self, "episode_success", False)

    def _get_info(self, terminated: bool = False, truncated: bool = False) -> dict:
        self.info = {}
        self.episode_info = {}

        current_dist = float(self._get_distance_tip_to_dest())
        min_dist = float(getattr(self, "min_dist_this_episode", current_dist))

        task_id = getattr(self, "task_id", "unknown")
        chosen_model = getattr(self, "chosen_model", "unknown")
        centerline_vtk = getattr(self, "centerline_vtk", "unknown")

        done_by_target = bool(getattr(self, "episode_success", False))
        done_by_timeout = bool(truncated)
        done_by_terminal_escape = bool(getattr(self, "terminal_escape_failed", False))
        done_by_non_finite = bool(getattr(self, "non_finite_failure", False))

        if done_by_target:
            terminal_reason = "target"
        elif done_by_timeout:
            terminal_reason = "timeout"
        elif done_by_terminal_escape:
            terminal_reason = "terminal_escape"
        elif done_by_non_finite:
            terminal_reason = "non_finite"
        elif terminated:
            terminal_reason = "other"
        else:
            terminal_reason = "not_done"

        self.info["success_10mm"] = bool(getattr(self, "episode_success_10mm", False))
        self.info["success_6mm"] = bool(getattr(self, "episode_success_6mm", False))
        self.info["success_3mm"] = bool(getattr(self, "episode_success_3mm", False))
        self.info["success_2mm"] = bool(getattr(self, "episode_success_2mm", False))

        if getattr(self, "_latest_executor_info", None) is not None:
            self.info["magnetic_executor_success"] = self._latest_executor_info.get("success", False)
            self.info["magnetic_executor_rel_error"] = self._latest_executor_info.get("rel_error", 0.0)
            self.info["magnetic_executor_moved"] = self._latest_executor_info.get("moved", False)
            I_pair = self._latest_executor_info.get("I_pair", [0.0, 0.0])
            self.info["magnetic_executor_I_L"] = I_pair[0]
            self.info["magnetic_executor_I_R"] = I_pair[1]

        self.info["min_dist_to_goal"] = min_dist
        self.info["current_dist_to_goal"] = current_dist
        self.info["final_dist_to_goal"] = current_dist if (terminated or truncated) else np.nan
        self.info["target_distance_threshold"] = float(self.target_distance_threshold)

        self.info["task_id"] = task_id
        self.info["chosen_model"] = chosen_model
        self.info["centerline_vtk"] = centerline_vtk

        self.info["terminal_reason"] = terminal_reason
        self.info["done_by_target"] = done_by_target
        self.info["done_by_timeout"] = done_by_timeout
        self.info["done_by_terminal_escape"] = done_by_terminal_escape
        self.info["done_by_non_finite"] = done_by_non_finite

        self.info["centerline_progress"] = float(getattr(self, "current_centerline_progress", np.nan))
        self.info["centerline_delta_progress"] = float(getattr(self, "current_centerline_delta_progress", np.nan))
        self.info["centerline_distance"] = float(getattr(self, "current_centerline_distance", np.nan))
        self.info["centerline_progress_ratio"] = float(getattr(self, "current_centerline_progress_ratio", np.nan))
        self.info["centerline_projection_index"] = int(getattr(self, "previous_centerline_projection_index", -1))
        self.info["centerline_projection_rejected_jump"] = bool(getattr(self, "centerline_projection_rejected_jump", False))
        self.info["centerline_projection_rejected_reason"] = str(getattr(self, "centerline_projection_rejected_reason", ""))
        self.info["centerline_reversed_for_progress"] = bool(getattr(self, "centerline_reversed_for_progress", False))
        self.info["using_centerline_reward"] = bool(getattr(self, "using_centerline_reward", False))
        self.info["using_euclidean_terminal_reward"] = bool(getattr(self, "using_euclidean_terminal_reward", False))
        self.info["non_finite_failure"] = bool(getattr(self, "non_finite_failure", False))

        return {**self.info, **self.reward_info, **self.episode_info, **self.reward_features}

    def _get_centerline_light_features(self, tip_pos: np.ndarray) -> np.ndarray:
        """Return lightweight centerline features.

        Layout:
            [progress_ratio, lookahead_1(3), ..., lookahead_8(3)]

        Only two centerline abilities are intentionally exposed:
            1. where the tip is along the path;
            2. how the vessel bends ahead.
        """
        features = np.zeros(self.centerline_feature_dim, dtype=np.float32)
        if self.centerline_points is None or len(self.centerline_points) < 2:
            return features
        if getattr(self, "centerline_cumlength", None) is None:
            return features

        progress, seg_idx, _, _, _ = self._get_centerline_projection_state(tip_pos)
        total_len = float(self.centerline_cumlength[-1])
        if total_len < 1e-9 or seg_idx < 0:
            return features

        features[0] = float(np.clip(progress / total_len, 0.0, 1.0))
        features[1:] = self._get_lookahead_relative_vectors(tip_pos)
        return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def _get_centerline_projection_state(self, tip_pos: np.ndarray):
        """Project tip position onto centerline with local-history constraint.

        The original implementation used a global nearest projection over the
        whole centerline every step. For curved or loop-like vessels, that can
        jump to a spatially nearby but topologically wrong segment. This function
        keeps the projection local around the previous segment after the first
        valid projection.
        """
        if self.centerline_points is None or len(self.centerline_points) < 2:
            return 0.0, -1, 0.0, np.asarray(tip_pos, dtype=np.float32), np.zeros(3, dtype=np.float32)
        if getattr(self, "centerline_cumlength", None) is None:
            return 0.0, -1, 0.0, np.asarray(tip_pos, dtype=np.float32), np.zeros(3, dtype=np.float32)

        points = np.asarray(self.centerline_points, dtype=np.float32)
        num_segments = len(points) - 1
        if num_segments <= 0:
            return 0.0, -1, 0.0, np.asarray(tip_pos, dtype=np.float32), np.zeros(3, dtype=np.float32)

        tip_pos = np.asarray(tip_pos, dtype=np.float32)

        prev_idx = int(getattr(self, "previous_centerline_projection_index", -1))
        if 0 <= prev_idx < num_segments:
            search_back = int(getattr(self, "centerline_projection_search_back", 5))
            search_forward = int(getattr(self, "centerline_projection_search_forward", 8))
            search_start = max(0, prev_idx - search_back)
            search_end = min(num_segments, prev_idx + search_forward + 1)
            candidate_indices = np.arange(search_start, search_end, dtype=np.int32)
        else:
            candidate_indices = np.arange(0, num_segments, dtype=np.int32)

        if candidate_indices.size == 0:
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

    def _get_centerline_arc_progress(self, tip_pos: np.ndarray) -> Tuple[float, int, float]:
        progress, best_seg_idx, closest_dist, _, _ = self._get_centerline_projection_state(tip_pos)

        previous_progress = getattr(self, "previous_centerline_progress", None)
        previous_idx = int(getattr(self, "previous_centerline_projection_index", -1))
        self.centerline_projection_rejected_jump = False
        self.centerline_projection_rejected_reason = ""

        # Reject projection jumps caused by nearest-point projection switching
        # to a topologically wrong centerline segment.
        #
        # 1) Large backward jumps: CL_r suddenly falls back toward 0.
        # 2) Large forward jumps: CL_r suddenly jumps to a far later segment,
        #    for example 0.17 -> 0.78 in one step.
        #
        # Small backtracking and small forward progress are still allowed so the
        # controller can recover naturally and the reward remains smooth.
        if (
            previous_progress is not None
            and best_seg_idx >= 0
            and previous_idx >= 0
            and np.isfinite(progress)
        ):
            previous_progress_f = float(previous_progress)
            max_backtrack = float(getattr(self, "centerline_max_backtrack", 0.0005))
            max_forward_jump = float(getattr(self, "centerline_max_forward_jump", 0.003))

            if progress < previous_progress_f - max_backtrack:
                self.centerline_projection_rejected_jump = True
                self.centerline_projection_rejected_reason = "backward"
                progress = previous_progress_f
                best_seg_idx = int(previous_idx)
                # closest_dist remains the local projection distance for logging.

            elif progress > previous_progress_f + max_forward_jump:
                self.centerline_projection_rejected_jump = True
                self.centerline_projection_rejected_reason = "forward"
                progress = previous_progress_f
                best_seg_idx = int(previous_idx)
                # closest_dist remains the local projection distance for logging.

        return float(progress), int(best_seg_idx), float(closest_dist)

    def _get_closest_centerline_index_and_distance(self, tip_pos: np.ndarray) -> Tuple[int, float]:
        if self.centerline_points is None or len(self.centerline_points) == 0:
            return -1, 0.0
        distances = np.linalg.norm(self.centerline_points - tip_pos, axis=1)
        closest_idx = int(np.argmin(distances))
        return closest_idx, float(distances[closest_idx])

    def _get_lookahead_relative_vectors(self, tip_pos: np.ndarray) -> np.ndarray:
        dim = 3 * self.num_centerline_lookahead_points
        if self.centerline_points is None or len(self.centerline_points) == 0:
            return np.zeros(dim, dtype=np.float32)

        # Use the same history-constrained projection used by centerline reward,
        # instead of global closest point, so observation does not jump to a
        # topologically wrong branch/segment in curved vessels.
        _, seg_idx, _, _, _ = self._get_centerline_projection_state(tip_pos)
        if seg_idx < 0:
            closest_idx, _ = self._get_closest_centerline_index_and_distance(tip_pos)
        else:
            closest_idx = int(seg_idx)

        if closest_idx < 0:
            return np.zeros(dim, dtype=np.float32)

        lookahead_vectors = []
        last_idx = len(self.centerline_points) - 1
        for offset in self.centerline_lookahead_offsets:
            idx = min(closest_idx + int(offset), last_idx)
            lookahead_vectors.append(self.centerline_points[idx] - tip_pos)

        out = np.asarray(lookahead_vectors, dtype=np.float32).reshape(-1)
        if out.shape[0] != dim:
            out = np.pad(out, (0, max(0, dim - out.shape[0])))[:dim]
        return out.astype(np.float32)

    def _resample_centerline_points_1mm(self, points: np.ndarray) -> np.ndarray:
        if points is None:
            return None

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

            if dense_total < 1e-9:
                return points

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

    def _get_distance_tip_to_dest(self):
        tip = np.asarray(self.mcr_controller_sofa_ros.get_pos_quat_catheter_tip()[0:3], dtype=np.float32)
        dist = np.linalg.norm(self.target_position - tip)
        if not np.isfinite(dist):
            self.non_finite_failure = True
            return float(1e3)
        return float(dist)

    def _do_action(self, action: np.ndarray) -> None:
        self.mcr_controller_sofa_ros.rotateZ(action[0])
        self.mcr_controller_sofa_ros.rotateX(action[1])
        self.mcr_controller_sofa_ros.insertRetract(action[2])

    def _init_sim(self):
        super()._init_sim()
        self.mcr_controller_sofa_ros: ControllerSofa = self.scene_creation_result.get(
            "mcr_controller_sofa_ros",
            self.scene_creation_result.get("mcr_controller_sofa", None),
        )
        self.mcr_environment_ros = self.scene_creation_result.get(
            "mcr_environment_ros",
            self.scene_creation_result.get("mcr_environment", None),
        )

        if self.mcr_controller_sofa_ros is None:
            raise KeyError("scene_creation_result missing mcr_controller_sofa_ros")
        if self.mcr_environment_ros is None:
            raise KeyError("scene_creation_result missing mcr_environment_ros")
        self.chosen_model = self.scene_creation_result.get("chosen_model", "unknown")
        self.centerline_vtk = self.scene_creation_result.get("centerline_vtk", "unknown")
        self.task_id = self.scene_creation_result.get("task_id", str(self.chosen_model))

        raw_centerline = self.scene_creation_result.get("centerline_points", None)
        if raw_centerline is not None:
            self.centerline_points = self._resample_centerline_points_1mm(np.array(raw_centerline, dtype=np.float32))
        else:
            self.centerline_points = None

        scene_target = self.scene_creation_result.get("target_position", None)
        if scene_target is not None:
            self.target_position = np.asarray(scene_target, dtype=np.float32)

        # Centerline direction check: order centerline as start -> target.
        # 优先用“起点接近初始 tip、终点接近 target”的联合代价判断，避免仅靠 target 判断反向。
        self.centerline_reversed_for_progress = False
        if self.centerline_points is not None and self.target_position is not None and len(self.centerline_points) >= 2:
            try:
                init_tip = np.asarray(self.mcr_controller_sofa_ros.get_pos_quat_catheter_tip()[0:3], dtype=np.float32)
                p0 = self.centerline_points[0]
                p1 = self.centerline_points[-1]
                forward_cost = float(np.linalg.norm(p0 - init_tip) + np.linalg.norm(p1 - self.target_position))
                reverse_cost = float(np.linalg.norm(p1 - init_tip) + np.linalg.norm(p0 - self.target_position))
                self.centerline_reversed_for_progress = bool(reverse_cost + 1e-9 < forward_cost)
            except Exception:
                d_first = float(np.linalg.norm(self.centerline_points[0] - self.target_position))
                d_last = float(np.linalg.norm(self.centerline_points[-1] - self.target_position))
                self.centerline_reversed_for_progress = bool(d_first < d_last)

            if self.centerline_reversed_for_progress:
                self.centerline_points = self.centerline_points[::-1].copy()

            print(
                "[CENTERLINE] reversed_for_progress=", self.centerline_reversed_for_progress,
                "start_to_target_dist=", float(np.linalg.norm(self.centerline_points[0] - self.target_position)),
                "end_to_target_dist=", float(np.linalg.norm(self.centerline_points[-1] - self.target_position)),
            )

        if self.centerline_points is not None and len(self.centerline_points) >= 2:
            seg_lengths = np.linalg.norm(np.diff(self.centerline_points, axis=0), axis=1)
            self.centerline_cumlength = np.concatenate(([0.0], np.cumsum(seg_lengths))).astype(np.float32)
        else:
            self.centerline_cumlength = None

        self.exit_plane_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if self.centerline_points is not None and self.target_position is not None and len(self.centerline_points) > 0:
            distances_to_target = np.linalg.norm(self.centerline_points - self.target_position, axis=1)
            diffs_from_1mm = np.abs(distances_to_target - 0.001)
            idx_1mm = int(np.argmin(diffs_from_1mm))
            point_1mm = self.centerline_points[idx_1mm]

            direction_vec = self.target_position - point_1mm
            norm = np.linalg.norm(direction_vec)
            if norm > 1e-9:
                self.exit_plane_normal = (direction_vec / norm).astype(np.float32)

        vessel_positions = np.asarray(self.mcr_environment_ros.get_vessel_tree_positions(), dtype=np.float32)
        bbox_diag = np.nan
        if (
            vessel_positions.ndim == 2
            and vessel_positions.shape[1] == 3
            and vessel_positions.shape[0] > 1
            and np.all(np.isfinite(vessel_positions))
        ):
            bbox_diag = float(np.linalg.norm(np.min(vessel_positions, axis=0) - np.max(vessel_positions, axis=0)))

        if not np.isfinite(bbox_diag) or bbox_diag < 1e-9:
            print(f"[WARN] Invalid vessel bbox_diag={bbox_diag}, fallback cartesian_scaling_factor=1.0")
            self.cartesian_scaling_factor = 1.0
        else:
            self.cartesian_scaling_factor = 1.0 / bbox_diag

        print("[MCREnv] cartesian_scaling_factor =", self.cartesian_scaling_factor, "bbox_diag =", bbox_diag)

    def _init_magnetic_executor_ros(self):
        try:
            if not rospy.core.is_initialized():
                rospy.init_node("mcr_rl_env_ros", anonymous=True, disable_signals=True)
        except Exception:
            pass
        self._execute_request_pub = rospy.Publisher("/magnetic/execute_request", Float64MultiArray, queue_size=1)
        rospy.Subscriber("/magnetic/execute_result", Float64MultiArray, self._execute_result_callback, queue_size=1)
        time.sleep(0.2)
        self._ros_sync_initialized = True

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
        data = [
            float(step_id),
            float(tip_world[0]),
            float(tip_world[1]),
            float(tip_world[2]),
            float(B_target[0]),
            float(B_target[1]),
            float(B_target[2]),
        ]
        msg = Float64MultiArray(data=data)
        self._execute_request_pub.publish(msg)

        start = time.time()
        while time.time() - start < self._execute_timeout:
            if self._latest_execute_result is not None and self._latest_execute_result.get("step_id") == step_id:
                return self._latest_execute_result
            rospy.sleep(0.005)

        raise TimeoutError(f"Wait /magnetic/execute_result timeout step_id={step_id}")

    def _update_dynamic_tip_world_offset(self):
        """Update per-episode SOFA->world translation for the magnetic executor.

        Assumption: SOFA and world coordinate axes are aligned, so the mapping is
        a pure translation:
            tip_world = tip_sofa + tip_world_offset

        The offset is recomputed after every reset because different vessels or
        centerline branches can produce different initial tip_sofa positions.
        """
        tip_sofa = np.asarray(
            self.mcr_controller_sofa_ros.get_pos_quat_catheter_tip()[0:3],
            dtype=np.float32,
        )
        self.initial_tip_sofa_for_world_align = tip_sofa.copy()

        if getattr(self, "dynamic_tip_world_align", True):
            self.tip_world_offset = self.desired_initial_tip_world - tip_sofa
        else:
            self.tip_world_offset = np.asarray(self.tip_world_offset, dtype=np.float32)

        initial_tip_world = tip_sofa + self.tip_world_offset
        print(
            "[MCREnvROS] dynamic tip-world align | "
            f"task_id={getattr(self, 'task_id', 'unknown')} | "
            f"chosen_model={getattr(self, 'chosen_model', 'unknown')} | "
            f"initial_tip_sofa={tip_sofa} | "
            f"desired_initial_tip_world={self.desired_initial_tip_world} | "
            f"tip_world_offset={self.tip_world_offset} | "
            f"initial_tip_world={initial_tip_world}"
        )

    def _get_tip_world_for_magnetic_executor(self):
        tip_sofa = np.asarray(self.mcr_controller_sofa_ros.get_pos_quat_catheter_tip()[0:3], dtype=np.float32)
        tip_world = tip_sofa + self.tip_world_offset

        if getattr(self, "debug_ros_sync", False):
            print(f"[MCREnvROS] tip_sofa={tip_sofa} tip_world_sent={tip_world}")

        return tip_world


if __name__ == "__main__":
    env = MCREnv(env_type=EnvType.AORTIC)
    env.reset()
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(reward)
        if terminated:
            break
    env.close()
