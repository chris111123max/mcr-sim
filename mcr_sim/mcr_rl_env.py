from typing import Union, Tuple, Optional, Any, Dict
from pathlib import Path
from enum import Enum, unique
from collections import defaultdict, deque

import gymnasium.spaces as spaces
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial.transform import Rotation as R

from .mcr_controller_sofa import ControllerSofa
from .paths import SCENE_DIR
from .rl_core.base import SofaEnv, RenderMode, RenderFramework

MCR_SIM_DIR = Path(__file__).resolve().parent
FLAT_SCENE_DESCRIPTION_FILE_PATH = MCR_SIM_DIR / "scene_description_2d.py"
AORTIC_SCENE_DESCRIPTION_FILE_PATH = SCENE_DIR / "example_aortic_arch.py"
FLAT_CATHETER_DESTINATION_EXIT_POINT = np.array([0.101129, 0.0238015, 0.002], dtype=np.float32)
AORTIC_CATHETER_DESTINATION_EXIT_POINT = np.array([-0.0101583, -0.180636, 0.0345185], dtype=np.float32)


#这个是1950万步的环境，小分叉容易穿模
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
    """Minimal waypoint-based mCR RL environment.

    Training logic kept in this file:
      1. Sequential Euclidean waypoints sampled from the centerline.
      2. Binary intermediate-waypoint approach reward and stronger final-target approach reward.
      3. One-shot intermediate-waypoint bonus, final target success bonus, out-of-vessel terminal penalty.
      4. Timeout terminal penalty when max_episode_steps is reached.
      5. Actor and critic use the same tip-local observation.
      6. Body multi-point safety is intentionally not included in this version.
    """

    def __init__(
        self,
        image_shape: Tuple[int, int] = (400, 400),
        create_scene_kwargs: Optional[dict] = None,
        observation_type: ObservationType = ObservationType.STATE,
        action_type: ActionType = ActionType.CONTINUOUS,
        time_step: float = 0.01,
        frame_skip: int = 1,
        settle_steps: int = 8,
        render_mode: RenderMode = RenderMode.HUMAN,
        render_framework: RenderFramework = RenderFramework.PYGLET,
        reward_amount_dict={
            "waypoint_approach": 20.0,           # intermediate phase: +20 if closer to active waypoint, -20 if farther
            "waypoint_reached": 100.0,           # one-shot bonus when an intermediate waypoint is reached
            "target_approach": 80.0,             # final target phase: +80 if closer to target, -80 if farther
            "successful_task": 4000.0,           # final target bonus when the 2mm target condition is satisfied
            "out_of_vessel_penalty": -500.0,    # terminal penalty for tip out-of-vessel
            "timeout_penalty": -500.0,           # terminal penalty when max_episode_steps is reached
            "step_penalty": -0.5,                # per-step time cost
        },
        target_position: Optional[np.ndarray] = None,
        env_type: EnvType = EnvType.FLAT,
        target_distance_threshold: float = 0.002,
        num_catheter_tracking_points: int = 4,
        max_episode_steps: int = 1000,
    ):
        if not isinstance(create_scene_kwargs, dict):
            create_scene_kwargs = {}
        create_scene_kwargs["image_shape"] = image_shape
        create_scene_kwargs.setdefault("insert_substep_max", 1)

        self.target_distance_threshold = float(target_distance_threshold)
        self.num_catheter_tracking_points = int(num_catheter_tracking_points)
        self.max_episode_steps = int(max_episode_steps)
        self._elapsed_steps = 0
        self.env_type = env_type

        if self.env_type == EnvType.FLAT:
            self.target_position = target_position if target_position is not None else FLAT_CATHETER_DESTINATION_EXIT_POINT.copy()
            self.scene_path = FLAT_SCENE_DESCRIPTION_FILE_PATH
        elif self.env_type == EnvType.AORTIC:
            self.target_position = target_position if target_position is not None else AORTIC_CATHETER_DESTINATION_EXIT_POINT.copy()
            self.scene_path = AORTIC_SCENE_DESCRIPTION_FILE_PATH
        else:
            raise ValueError(f"Unsupported env_type: {self.env_type}")
        self.target_position = np.asarray(self.target_position, dtype=np.float32).reshape(3)

        super().__init__(
            scene_path=self.scene_path,
            time_step=time_step,
            frame_skip=frame_skip,
            render_mode=render_mode,
            render_framework=render_framework,
            create_scene_kwargs=create_scene_kwargs,
        )

        self.observation_type = observation_type
        self._settle_steps = int(settle_steps)

        # Scales and safety parameters.
        self.magnetic_field_observation_scale = float(create_scene_kwargs.get("magnetic_field_observation_scale", 0.10))
        self.radius_observation_scale = float(create_scene_kwargs.get("radius_observation_scale", 0.005))
        self.default_local_radius = float(create_scene_kwargs.get("default_local_radius", 0.005))
        self.catheter_radius = float(create_scene_kwargs.get("catheter_radius", 0.000665))
        self.out_of_vessel_safety_ratio = float(create_scene_kwargs.get("out_of_vessel_safety_ratio", 1.10))
        self.out_of_vessel_fallback_distance = float(create_scene_kwargs.get("out_of_vessel_fallback_distance", 0.012))
        self.local_field_action_angle = float(create_scene_kwargs.get("local_field_action_angle", 2.0 * np.pi / 180.0))

        # Insertion safety shield is disabled in this version.
        # The third action component is passed through directly after clipping
        # to [-1, 1]. Near-wall insertion reduction, insertion blocking, and
        # forced retraction after out-of-vessel detection are all removed.

        # Insertion exploration bias.
        # SAC random exploration is zero-mean by default, which easily leads to
        # insert/retract cancellation at the vessel entry. A small positive bias
        # helps the agent experience waypoint approach rewards early, while still
        # allowing retraction through negative raw_insert actions.
        self.insert_bias = float(create_scene_kwargs.get("insert_bias", 0.0))
        self.insert_negative_limit = float(create_scene_kwargs.get("insert_negative_limit", -1.0))

        # Waypoint task parameters.
        self.waypoint_spacing = float(create_scene_kwargs.get("waypoint_spacing", 0.005))       # 5 mm
        self.waypoint_reach_threshold = float(create_scene_kwargs.get("waypoint_reach_threshold", 0.003))  # 3 mm for intermediate waypoints
        self.pre_target_waypoint_offset = float(create_scene_kwargs.get("pre_target_waypoint_offset", 0.001))  # 1 mm before final target
        self.waypoint_observation_scale = float(create_scene_kwargs.get("waypoint_observation_scale", 0.010))
        self.progress_clip = float(create_scene_kwargs.get("waypoint_progress_clip", 0.003))     # cap per-step delta to +/-3 mm

        # Adjacent-waypoint handoff.
        # When the next ordered waypoint is clearly closer than the active one
        # for several consecutive steps, advance exactly one waypoint. This is
        # a recovery mechanism for a missed 3 mm waypoint sphere.
        # It uses only Euclidean distances to adjacent waypoints:
        # no centerline progress and no cross-section gate.
        self.waypoint_handoff_margin = float(
            create_scene_kwargs.get("waypoint_handoff_margin", 0.0005)
        )  # next waypoint must be at least 0.5 mm closer
        self.waypoint_handoff_confirm_steps = max(
            1,
            int(create_scene_kwargs.get("waypoint_handoff_confirm_steps", 2)),
        )

        # Actor observation: 24-D current geometry + 4 * 7-D action-response history = 52-D.
        self.vessel_section_feature_dim = 4
        self.actor_current_geometry_dim = 24
        self.actor_dynamic_step_dim = 7
        self.actor_history_steps = max(1, int(create_scene_kwargs.get("actor_history_steps", 4)))
        self.actor_dynamic_history_dim = self.actor_dynamic_step_dim * self.actor_history_steps
        self.actor_observation_dim = self.actor_current_geometry_dim + self.actor_dynamic_history_dim
        self._actor_dynamic_history = deque(maxlen=self.actor_history_steps)

        # Uniform multi-vessel sampling. No priority sampling.
        self.training_models = [f"C{i:02d}" for i in range(1, 6)] + [
            f"B{i:02d}" for i in range(1, 6)
        ]

        if self.observation_type == ObservationType.STATE:
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.actor_observation_dim,),
                dtype=np.float32,
            )
        elif self.observation_type == ObservationType.RGB:
            self.observation_space = spaces.Box(low=0, high=255, shape=image_shape + (3,), dtype=np.uint8)
        else:
            raise ValueError(f"Unsupported observation_type: {self.observation_type}")

        self.action_type = action_type
        if self.action_type == ActionType.CONTINUOUS:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        else:
            raise NotImplementedError("Only continuous action space is implemented.")

        self.reward_amount_dict = defaultdict(float)
        self.reward_amount_dict.update(reward_amount_dict)
        self.reward_info = {}
        self.reward_features = {}

        # Centerline / waypoint buffers.
        self.centerline_points = None
        self.centerline_cumlength = None
        self.centerline_radius = None
        self.centerline_reversed_for_progress = False
        self.waypoint_points = None
        self.waypoint_progress = None
        self.current_waypoint_idx = 0
        self.current_waypoint_progress_ratio = 0.0
        self.current_waypoint_distance = np.nan
        self.current_waypoint_approach_delta = 0.0
        self.current_waypoint_reached_this_step = False
        self.current_waypoint_reached_count_episode = 0
        self.current_waypoint_is_final = False
        self.current_target_reached_this_step = False
        self.previous_waypoint_idx = None
        self.previous_waypoint_distance = None
        self.current_waypoint_handoff_counter = 0
        self.current_waypoint_handoff_this_step = False
        self.current_waypoint_handoff_count_episode = 0

        # Safety state.
        self.current_centerline_local_radius = np.nan
        self.current_centerline_safety_ratio = np.nan
        self.current_centerline_safety_margin = np.nan
        self.current_centerline_offset_N_over_radius = np.nan
        self.current_centerline_offset_B_over_radius = np.nan
        self.current_centerline_local_radius_norm = np.nan
        self.current_centerline_progress = np.nan
        self.current_centerline_projection = np.zeros(3, dtype=np.float32)
        self.current_centerline_tangent = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.current_centerline_radial_offset = np.nan
        self.current_tip_centerline_offset_norm = np.nan
        self.previous_tip_centerline_radial_offset = None
        self.current_out_of_vessel = False
        self.out_of_vessel_this_episode = False
        self.out_of_vessel_failure = False
        self.max_safety_ratio_this_episode = 0.0
        self.min_safety_margin_this_episode = np.inf

        # Episode state.
        self.episode_success = False
        self.episode_success_2mm = False
        self.min_dist_this_episode = np.inf
        self.non_finite_failure = False
        self.is_out_of_bounds = False

        # Action diagnostics used by observation/info.
        self.current_raw_insert = 0.0
        self.current_effective_insert = 0.0

        # Low-level action rate limiter.
        # Prevents rapid magnetic command reversal in high-curvature sections.
        self.max_action_delta = float(
            create_scene_kwargs.get("max_action_delta", 0.30)
        )

        self._last_smoothed_action = np.zeros(3, dtype=np.float32)
        self._prev_smoothed_action = np.zeros(3, dtype=np.float32)
        self._last_actor_tip_pos = None

        # Task sampling: uniform over active vessels unless --force-model is used.
        self._explicit_force_model = str(create_scene_kwargs.get("force_model", "") or "").strip()
        self._sampler_rng = np.random.default_rng()
        self.current_sampling_model = self._explicit_force_model if self._explicit_force_model else None

        # Single-vessel soft reset.
        self.soft_randomize_single_vessel = bool(create_scene_kwargs.get("soft_randomize_single_vessel", True))
        self._soft_reset_rng = np.random.default_rng()
        self._soft_reset_base_start_sim = None
        self._soft_reset_info_printed = False
        self._soft_reset_warning_printed = False

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _seed_to_int(seed) -> Optional[int]:
        if seed is None:
            return None
        try:
            if isinstance(seed, np.random.SeedSequence):
                return int(seed.entropy)
            return int(seed)
        except Exception:
            return None

    @staticmethod
    def _unit_vector(vec: np.ndarray, eps: float = 1e-9) -> Optional[np.ndarray]:
        vec = np.asarray(vec, dtype=np.float64).reshape(3)
        n = float(np.linalg.norm(vec))
        if n < eps:
            return None
        return vec / n

    @staticmethod
    def _rotation_between_vectors(source_vec: np.ndarray, target_vec: np.ndarray) -> R:
        source = MCREnv._unit_vector(source_vec)
        target = MCREnv._unit_vector(target_vec)
        if source is None or target is None:
            return R.identity()
        dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
        if dot > 1.0 - 1e-8:
            return R.identity()
        if dot < -1.0 + 1e-8:
            axis = np.cross(source, np.array([1.0, 0.0, 0.0], dtype=np.float64))
            if float(np.linalg.norm(axis)) < 1e-8:
                axis = np.cross(source, np.array([0.0, 1.0, 0.0], dtype=np.float64))
            axis = axis / (float(np.linalg.norm(axis)) + 1e-12)
            return R.from_rotvec(np.pi * axis)
        axis = np.cross(source, target)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-12:
            return R.identity()
        axis = axis / axis_norm
        return R.from_rotvec(float(np.arccos(dot)) * axis)

    @staticmethod
    def _sample_direction_within_cone(base_dir: np.ndarray, max_angle_rad: float, rng: np.random.Generator) -> Optional[np.ndarray]:
        base = MCREnv._unit_vector(base_dir)
        if base is None:
            return None
        max_angle_rad = max(0.0, float(max_angle_rad))
        if max_angle_rad <= 1e-12:
            return base.astype(np.float64)
        rand = rng.normal(size=3)
        perp = rand - float(np.dot(rand, base)) * base
        if float(np.linalg.norm(perp)) < 1e-12:
            ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if abs(float(np.dot(ref, base))) > 0.9:
                ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            perp = ref - float(np.dot(ref, base)) * base
        perp = perp / (float(np.linalg.norm(perp)) + 1e-12)
        cos_theta = float(rng.uniform(float(np.cos(max_angle_rad)), 1.0))
        sin_theta = float(np.sqrt(max(0.0, 1.0 - cos_theta * cos_theta)))
        out = cos_theta * base + sin_theta * perp
        return out / (float(np.linalg.norm(out)) + 1e-12)

    @staticmethod
    def _assign_sofa_data_value(obj, attr_name: str, value) -> bool:
        try:
            data = getattr(obj, attr_name)
        except Exception:
            return False
        try:
            data.value = value
            return True
        except Exception:
            pass
        try:
            setattr(obj, attr_name, value)
            return True
        except Exception:
            return False

    def _quat_rotate_vector(self, quat_xyzw: np.ndarray, vec: np.ndarray) -> np.ndarray:
        q = np.asarray(quat_xyzw, dtype=np.float32)
        v = np.asarray(vec, dtype=np.float32)
        if q.shape[0] != 4:
            return v.astype(np.float32)
        q_norm = float(np.linalg.norm(q))
        if q_norm < 1e-9:
            return v.astype(np.float32)
        q = q / q_norm
        t = 2.0 * np.cross(q[:3], v)
        return (v + float(q[3]) * t + np.cross(q[:3], t)).astype(np.float32)

    def _build_tip_local_frame(self, quat_xyzw: np.ndarray) -> np.ndarray:
        try:
            x_axis = self._quat_rotate_vector(quat_xyzw, np.array([1.0, 0.0, 0.0], dtype=np.float32))
            y_axis = self._quat_rotate_vector(quat_xyzw, np.array([0.0, 1.0, 0.0], dtype=np.float32))
            x_axis = x_axis / (float(np.linalg.norm(x_axis)) + 1e-9)
            y_axis = y_axis - float(np.dot(y_axis, x_axis)) * x_axis
            y_axis = y_axis / (float(np.linalg.norm(y_axis)) + 1e-9)
            z_axis = np.cross(x_axis, y_axis)
            z_axis = z_axis / (float(np.linalg.norm(z_axis)) + 1e-9)
            return np.stack([x_axis, y_axis, z_axis], axis=1).astype(np.float32)
        except Exception:
            return np.eye(3, dtype=np.float32)

    def _build_local_centerline_frame(self, tangent: np.ndarray) -> np.ndarray:
        t = np.asarray(tangent, dtype=np.float32).reshape(3)
        t = t / (float(np.linalg.norm(t)) + 1e-9)
        if float(np.linalg.norm(t)) < 1e-6:
            return np.eye(3, dtype=np.float32)
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if abs(float(np.dot(t, ref))) > 0.90:
            ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        n = np.cross(ref, t)
        n = n / (float(np.linalg.norm(n)) + 1e-9)
        b = np.cross(t, n)
        b = b / (float(np.linalg.norm(b)) + 1e-9)
        return np.stack([t, n, b], axis=1).astype(np.float32)

    def _world_vec_to_local(self, v_world: np.ndarray, frame: np.ndarray) -> np.ndarray:
        v = np.asarray(v_world, dtype=np.float32).reshape(3)
        f = np.asarray(frame, dtype=np.float32)
        return np.array([np.dot(v, f[:, 0]), np.dot(v, f[:, 1]), np.dot(v, f[:, 2])], dtype=np.float32)

    # ------------------------------------------------------------------
    # Reset / sampling
    # ------------------------------------------------------------------
    def _sample_next_training_model(self, seed: Union[int, np.random.SeedSequence, None] = None) -> None:
        """Uniformly sample one training vessel unless a fixed vessel is requested."""
        if self._explicit_force_model:
            self.create_scene_kwargs["force_model"] = self._explicit_force_model
            self.current_sampling_model = self._explicit_force_model
            return

        seed_value = self._seed_to_int(seed)
        if seed_value is not None:
            self._sampler_rng = np.random.default_rng(seed_value)

        chosen = str(self._sampler_rng.choice(list(self.training_models)))
        self.create_scene_kwargs["force_model"] = chosen
        self.current_sampling_model = chosen

    def _capture_soft_reset_reference_pose(self) -> None:
        if getattr(self, "_soft_reset_base_start_sim", None) is not None:
            return
        base_pose = None
        try:
            base_pose = np.asarray(self.mcr_controller_sofa.instrument.IRC.startingPos.value, dtype=np.float64).reshape(-1)
        except Exception:
            pass
        if base_pose is None or base_pose.shape[0] < 7 or not np.all(np.isfinite(base_pose[:7])):
            try:
                rest = np.asarray(self.mcr_controller_sofa.instrument.MO.rest_position.value, dtype=np.float64)
                if rest.ndim == 2 and rest.shape[1] >= 7 and rest.shape[0] > 0:
                    base_pose = rest[0, :7].copy()
            except Exception:
                base_pose = None
        if base_pose is not None and len(base_pose) >= 7 and np.all(np.isfinite(base_pose[:7])):
            self._soft_reset_base_start_sim = np.asarray(base_pose[:7], dtype=np.float64).copy()

    def _apply_instrument_start_pose_sim(self, t_start_sim: np.ndarray) -> bool:
        t_start_sim = np.asarray(t_start_sim, dtype=np.float64).reshape(7)
        if not np.all(np.isfinite(t_start_sim)):
            return False
        try:
            instrument = self.mcr_controller_sofa.instrument
            controller = self.mcr_controller_sofa
        except Exception:
            return False

        ok_any = False
        try:
            ok_any = self._assign_sofa_data_value(instrument.IRC, "startingPos", t_start_sim.tolist()) or ok_any
            instrument.IRC.xtip[0] = 0.0
            controller.pending_insert_delta = 0.0
            controller.invalid_action = False
            ok_any = True
        except Exception:
            pass

        try:
            n = int(len(instrument.MO.position.array()))
        except Exception:
            try:
                n = int(len(instrument.MO.rest_position.value))
            except Exception:
                n = 0
        if n > 0:
            poses = np.tile(t_start_sim[None, :], (n, 1)).astype(np.float64).tolist()
            ok_any = self._assign_sofa_data_value(instrument.MO, "rest_position", poses) or ok_any
            ok_any = self._assign_sofa_data_value(instrument.MO, "position", poses) or ok_any
            ok_any = self._assign_sofa_data_value(instrument.MO, "free_position", poses) or ok_any
            try:
                instrument.MO.velocity.value = np.zeros((n, 6), dtype=np.float64).tolist()
                ok_any = True
            except Exception:
                pass
        return bool(ok_any)

    def _soft_randomize_single_vessel_scene(self, seed: Union[int, np.random.SeedSequence, None] = None) -> None:
        if not bool(getattr(self, "soft_randomize_single_vessel", True)):
            return
        if not bool(getattr(self, "_explicit_force_model", "")):
            return
        randomize_start_target = bool(self.create_scene_kwargs.get("randomize_start_target", True))
        randomize_initial_orientation = bool(self.create_scene_kwargs.get("randomize_initial_orientation", True))
        if not (randomize_start_target or randomize_initial_orientation):
            return

        seed_value = self._seed_to_int(seed)
        if seed_value is not None:
            self._soft_reset_rng = np.random.default_rng(seed_value)
        rng = self._soft_reset_rng

        points = getattr(self, "centerline_points", None)
        if points is None or len(points) < 2:
            return
        points = np.asarray(points, dtype=np.float64)
        n_pts = int(len(points))

        self._capture_soft_reset_reference_pose()
        base_start_sim = getattr(self, "_soft_reset_base_start_sim", None)
        if base_start_sim is None or len(base_start_sim) < 7:
            if not self._soft_reset_warning_printed:
                print("[SOFT_RANDOM_RESET][WARN] Could not capture base start pose; target randomization only.")
                self._soft_reset_warning_printed = True
            base_start_sim = None

        start_idx = 0
        target_idx = n_pts - 1

        is_aorta6 = (
            str(getattr(self, "_explicit_force_model", "")).lower()
            == "aorta6"
            or str(getattr(self, "task_id", "")).lower()
            == "aorta6"
        )

        if is_aorta6:
            # centerline_points has already been oriented from the episode
            # start side toward the target in _init_sim(). Start from the
            # resampled point nearest 50% of the route and keep the final
            # point as the target.
            start_fraction = float(
                self.create_scene_kwargs.get(
                    "aorta6_start_fraction",
                    0.50,
                )
            )
            start_fraction = float(np.clip(
                start_fraction,
                0.0,
                0.95,
            ))

            start_idx = int(np.clip(
                round((n_pts - 1) * start_fraction),
                0,
                n_pts - 2,
            ))
            target_idx = n_pts - 1

            print(
                "[AORTA6_SOFT_MIDPOINT_START]",
                "start_fraction=", start_fraction,
                "start_idx=", start_idx,
                "target_idx=", target_idx,
                "num_points=", n_pts,
            )

        elif randomize_start_target:
            start_window = int(np.clip(self.create_scene_kwargs.get("start_window_points", self.create_scene_kwargs.get("start_target_window_points", 5)), 1, n_pts))
            target_window = int(np.clip(self.create_scene_kwargs.get("target_window_points", self.create_scene_kwargs.get("start_target_window_points", 5)), 1, n_pts))
            start_idx = int(rng.choice(np.arange(0, start_window, dtype=np.int64)))
            target_idx = int(rng.choice(np.arange(max(0, n_pts - target_window), n_pts, dtype=np.int64)))
            if target_idx <= start_idx and n_pts > 1:
                target_idx = n_pts - 1

        start_point = points[start_idx].copy()
        target_point = points[target_idx].copy()
        self.target_position = np.asarray(target_point, dtype=np.float32)
        self.current_soft_start_position = np.asarray(start_point, dtype=np.float32)
        self.current_soft_target_position = np.asarray(target_point, dtype=np.float32)
        self.current_soft_start_idx = int(start_idx)
        self.current_soft_target_idx = int(target_idx)

        if base_start_sim is not None:
            step = int(np.clip(self.create_scene_kwargs.get("entry_tangent_points", 5), 1, n_pts - 1))
            next_idx = int(np.clip(start_idx + step, 0, n_pts - 1))
            if next_idx == start_idx:
                next_idx = int(np.clip(start_idx + 1, 0, n_pts - 1))
            entry_tangent = self._unit_vector(points[next_idx] - points[start_idx])
            if entry_tangent is not None:
                desired_tangent = entry_tangent
                if randomize_initial_orientation:
                    max_angle_deg = float(self.create_scene_kwargs.get("initial_orientation_max_angle_deg", 20.0))
                    sampled = self._sample_direction_within_cone(entry_tangent, np.deg2rad(max_angle_deg), rng)
                    if sampled is not None:
                        desired_tangent = sampled
                base_rot = R.from_quat(np.asarray(base_start_sim[3:7], dtype=np.float64))
                base_forward = base_rot.apply(np.array([1.0, 0.0, 0.0], dtype=np.float64))
                new_rot = self._rotation_between_vectors(base_forward, desired_tangent) * base_rot
                t_start_sim = np.asarray(base_start_sim, dtype=np.float64).copy()
                t_start_sim[0:3] = np.asarray(start_point, dtype=np.float64)
                t_start_sim[3:7] = new_rot.as_quat()
                self._apply_instrument_start_pose_sim(t_start_sim)

        self._build_waypoint_sequence()

    def reset(self, seed: Union[int, np.random.SeedSequence, None] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Union[np.ndarray, None], Dict]:
        self._sample_next_training_model(seed=seed)

        single_vessel_mode = bool(getattr(self, "_explicit_force_model", ""))
        randomization_requested = bool(self.create_scene_kwargs.get("randomize_start_target", True) or self.create_scene_kwargs.get("randomize_initial_orientation", True))
        soft_randomization_available = bool(single_vessel_mode and getattr(self, "soft_randomize_single_vessel", True))
        randomized_scene_each_episode = bool(randomization_requested and not soft_randomization_available)
        need_full_reload = bool(self._initialized)

        if single_vessel_mode and self._initialized and not randomized_scene_each_episode:
            # Fixed-vessel training can corrupt the reused SOFA scene after
            # severe contact/non-finite failures. Reload every episode for a
            # clean mechanical state.
            need_full_reload = True
        elif not self._initialized:
            need_full_reload = True
        else:
            need_full_reload = True

        if self._initialized and need_full_reload:
            if hasattr(self, "sofa_simulation") and self._sofa_root_node is not None:
                self.sofa_simulation.unload(self._sofa_root_node)
            self._initialized = False
        if not self._initialized:
            super().reset(seed)

        self._elapsed_steps = 0
        self.episode_success = False
        self.episode_success_2mm = False
        self.min_dist_this_episode = np.inf
        self.non_finite_failure = False
        self.is_out_of_bounds = False
        self.current_out_of_vessel = False
        self.out_of_vessel_this_episode = False
        self.out_of_vessel_failure = False
        self.max_safety_ratio_this_episode = 0.0
        self.min_safety_margin_this_episode = np.inf
        self.current_centerline_radial_offset = np.nan
        self.current_tip_centerline_offset_norm = np.nan
        self.previous_tip_centerline_radial_offset = None
        self.current_waypoint_idx = 0
        self.current_waypoint_progress_ratio = 0.0
        self.current_waypoint_distance = np.nan
        self.current_waypoint_approach_delta = 0.0
        self.current_waypoint_reached_this_step = False
        self.current_waypoint_reached_count_episode = 0
        self.current_waypoint_is_final = False
        self.current_target_reached_this_step = False
        self.previous_waypoint_idx = None
        self.previous_waypoint_distance = None
        self.current_waypoint_handoff_counter = 0
        self.current_waypoint_handoff_this_step = False
        self.current_waypoint_handoff_count_episode = 0

        self.current_raw_insert = 0.0
        self.current_effective_insert = 0.0
        self._last_smoothed_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self._prev_smoothed_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self._last_actor_tip_pos = None
        self._reset_actor_history()
        self.reward_info = {}
        self.reward_features = {}

        self.mcr_controller_sofa.reset()
        if bool(single_vessel_mode and getattr(self, "soft_randomize_single_vessel", True)):
            self._soft_randomize_single_vessel_scene(seed=seed)

        self.sofa_simulation.animate(self._sofa_root_node, 0.05)

        try:
            tip_pose = np.asarray(self.mcr_controller_sofa.get_pos_quat_catheter_tip(), dtype=np.float64)
            tip_pos = tip_pose[:3]
            scene_start = getattr(self, "current_scene_start_position", None)
            if scene_start is not None:
                scene_start = np.asarray(scene_start, dtype=np.float64).reshape(3)
                print("[REAL_TIP_AFTER_RESET_CHECK]")
                print("  task_id             =", getattr(self, "task_id", "unknown"))
                print("  scene_start_position=", scene_start)
                print("  real_tip_after_reset=", tip_pos)
                print("  tip_start_error_mm  =", float(np.linalg.norm(tip_pos - scene_start) * 1000.0))
        except Exception as e:
            print("[REAL_TIP_AFTER_RESET_CHECK][WARN]", e)

        self._initialize_waypoint_progress_from_current_tip()
        return self._get_observation(image_observation=self._maybe_update_rgb_buffer()), {}

    # ------------------------------------------------------------------
    # Step / observation / reward
    # ------------------------------------------------------------------
    def step(self, action: Any) -> Tuple[Union[np.ndarray, dict], float, bool, bool, dict]:
        action_np = np.array(action, dtype=np.float32)
        action_np = np.nan_to_num(action_np, nan=0.0, posinf=1.0, neginf=-1.0)
        action_np = np.clip(action_np, -1.0, 1.0).astype(np.float32)

        # Action rate limit:
        # a_filtered = a_previous + clip(a_raw-a_previous, +/-delta)
        previous_action = self._last_smoothed_action.copy()
        delta = action_np - previous_action
        delta = np.clip(
            delta,
            -float(self.max_action_delta),
            float(self.max_action_delta),
        )
        action_np = np.clip(
            previous_action + delta,
            -1.0,
            1.0,
        ).astype(np.float32)

        self._prev_smoothed_action = previous_action
        self._last_smoothed_action = action_np.copy()

        image_observation = super().step(action_np)
        self._elapsed_steps += 1

        reward = self._get_reward()
        observation = self._get_observation(image_observation)

        non_finite_failure = False
        if self.observation_type == ObservationType.STATE:
            obs_values = observation.values() if isinstance(observation, dict) else [observation]
            if any(not np.all(np.isfinite(v)) for v in obs_values):
                non_finite_failure = True
                if isinstance(observation, dict):
                    observation = {k: np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32) for k, v in observation.items()}
                else:
                    observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        if not np.isfinite(reward):
            reward = -100.0
            non_finite_failure = True
        if non_finite_failure:
            self.non_finite_failure = True

        terminated = bool(self.episode_success or self.out_of_vessel_failure or self.non_finite_failure)
        truncated = (self._elapsed_steps >= self.max_episode_steps) and (not terminated)

        if truncated:
            timeout_penalty = float(self.reward_amount_dict["timeout_penalty"])
            if np.isfinite(timeout_penalty):
                reward += timeout_penalty
                self.reward_features["timeout_penalty"] = 1.0
                self.reward_info["timeout_penalty"] = 1.0
                self.reward_info["reward_timeout_penalty"] = timeout_penalty
                self.reward_info["reward"] = float(reward)

        info = self._get_info(terminated=terminated, truncated=truncated)
        if truncated:
            info["TimeLimit.truncated"] = True
        return observation, float(reward), terminated, truncated, info

    def _reset_actor_history(self) -> None:
        self._actor_dynamic_history = deque(maxlen=int(getattr(self, "actor_history_steps", 4)))

    def _push_actor_dynamic_history(self, dynamic_step_obs: np.ndarray) -> None:
        dynamic_step_obs = np.asarray(dynamic_step_obs, dtype=np.float32).reshape(-1)
        expected = int(getattr(self, "actor_dynamic_step_dim", 7))
        if dynamic_step_obs.shape[0] != expected:
            fixed = np.zeros(expected, dtype=np.float32)
            fixed[: min(expected, dynamic_step_obs.shape[0])] = dynamic_step_obs[: min(expected, dynamic_step_obs.shape[0])]
            dynamic_step_obs = fixed
        self._actor_dynamic_history.append(np.nan_to_num(dynamic_step_obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))

    def _get_actor_dynamic_history_observation(self) -> np.ndarray:
        steps = int(getattr(self, "actor_history_steps", 4))
        dim = int(getattr(self, "actor_dynamic_step_dim", 7))
        history = list(getattr(self, "_actor_dynamic_history", []))
        chunks = [np.zeros(dim, dtype=np.float32) for _ in range(max(0, steps - len(history)))] + history
        return np.concatenate(chunks[-steps:]).astype(np.float32)

    def _build_actor_current_geometry_observation(
        self,
        tip_forward_local: np.ndarray,
        magnetic_field_norm: np.ndarray,
        waypoint_vector_local: np.ndarray,
        next_waypoint_vector_local: np.ndarray,
        centerline_correction_vec_local: np.ndarray,
        centerline_tangent_local: np.ndarray,
        waypoint_distance_norm: np.ndarray,
        waypoint_progress_ratio: float,
        vessel_section_features: np.ndarray,
    ) -> np.ndarray:
        obs = np.concatenate(
            [
                np.asarray(tip_forward_local, dtype=np.float32).reshape(3),
                np.asarray(magnetic_field_norm, dtype=np.float32).reshape(3),
                np.asarray(waypoint_vector_local, dtype=np.float32).reshape(3),
                np.asarray(next_waypoint_vector_local, dtype=np.float32).reshape(3),
                np.asarray(centerline_correction_vec_local, dtype=np.float32).reshape(3),
                np.asarray(centerline_tangent_local, dtype=np.float32).reshape(3),
                np.asarray(waypoint_distance_norm, dtype=np.float32).reshape(1),
                np.array([np.clip(float(waypoint_progress_ratio), 0.0, 1.0)], dtype=np.float32),
                np.asarray(vessel_section_features, dtype=np.float32).reshape(self.vessel_section_feature_dim),
            ]
        ).astype(np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def _build_actor_dynamic_step_observation(self, prev_action: np.ndarray, tip_delta_local: np.ndarray, progress_delta_norm: np.ndarray) -> np.ndarray:
        obs = np.concatenate(
            [
                np.asarray(prev_action, dtype=np.float32).reshape(3),
                np.asarray(tip_delta_local, dtype=np.float32).reshape(3),
                np.asarray(progress_delta_norm, dtype=np.float32).reshape(1),
            ]
        ).astype(np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def _get_actor_observation(self, current_geometry_obs: np.ndarray) -> np.ndarray:
        obs = np.concatenate([np.asarray(current_geometry_obs, dtype=np.float32).reshape(-1), self._get_actor_dynamic_history_observation()]).astype(np.float32)
        expected = int(self.actor_observation_dim)
        if obs.shape[0] != expected:
            fixed = np.zeros(expected, dtype=np.float32)
            fixed[: min(expected, obs.shape[0])] = obs[: min(expected, obs.shape[0])]
            obs = fixed
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def _get_observation(self, image_observation: Union[np.ndarray, None]) -> Union[np.ndarray, dict]:
        if self.observation_type == ObservationType.RGB:
            return image_observation
        if self.observation_type != ObservationType.STATE:
            return {}

        tip_pose = np.asarray(self.mcr_controller_sofa.get_pos_quat_catheter_tip(), dtype=np.float32)
        tip_pos = tip_pose[0:3]
        tip_quat = tip_pose[3:7]
        frame = self._build_tip_local_frame(tip_quat)

        # Update safety features from centerline; reward does not use centerline progress.
        self._update_vessel_safety_state(tip_pos)

        tip_forward_world = self._quat_rotate_vector(tip_quat, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        tip_forward_local = self._world_vec_to_local(tip_forward_world, frame)
        tip_forward_local = tip_forward_local / (float(np.linalg.norm(tip_forward_local)) + 1e-9)

        last_tip_pos = getattr(self, "_last_actor_tip_pos", None)
        tip_delta_world = np.zeros(3, dtype=np.float32) if last_tip_pos is None else np.asarray(tip_pos, dtype=np.float32) - np.asarray(last_tip_pos, dtype=np.float32)
        tip_delta_local = np.clip(self._world_vec_to_local(tip_delta_world, frame) / 0.001, -5.0, 5.0).astype(np.float32)

        mag_field = np.asarray(self.mcr_controller_sofa.get_mag_field_des(), dtype=np.float32)
        magnetic_field_norm = np.clip(self._world_vec_to_local(mag_field, frame) / max(self.magnetic_field_observation_scale, 1e-9), -2.0, 2.0).astype(np.float32)

        wp_scale = max(float(self.waypoint_observation_scale), 1e-9)
        wp_points = getattr(self, "waypoint_points", None)
        if wp_points is not None and len(wp_points) > 0:
            wp_idx = int(np.clip(self.current_waypoint_idx, 0, len(wp_points) - 1))
            next_idx = int(np.clip(wp_idx + 1, 0, len(wp_points) - 1))
            wp_vec_world = np.asarray(wp_points[wp_idx], dtype=np.float32) - tip_pos
            next_vec_world = np.asarray(wp_points[next_idx], dtype=np.float32) - tip_pos
            wp_distance = float(np.linalg.norm(wp_vec_world))
            waypoint_progress_ratio = float(wp_idx / max(1, len(wp_points) - 1))
        else:
            wp_vec_world = np.asarray(self.target_position, dtype=np.float32) - tip_pos
            next_vec_world = wp_vec_world.copy()
            wp_distance = float(np.linalg.norm(wp_vec_world))
            waypoint_progress_ratio = 0.0

        waypoint_vector_local = np.clip(self._world_vec_to_local(wp_vec_world, frame) / wp_scale, -5.0, 5.0).astype(np.float32)
        next_waypoint_vector_local = np.clip(self._world_vec_to_local(next_vec_world, frame) / wp_scale, -5.0, 5.0).astype(np.float32)
        waypoint_distance_norm = np.array([np.clip(wp_distance / wp_scale, 0.0, 10.0)], dtype=np.float32)

        centerline_proj = np.asarray(getattr(self, "current_centerline_projection", tip_pos), dtype=np.float32).reshape(3)
        centerline_tangent_world = np.asarray(getattr(self, "current_centerline_tangent", np.array([1.0, 0.0, 0.0], dtype=np.float32)), dtype=np.float32).reshape(3)
        centerline_tangent_world = centerline_tangent_world / (float(np.linalg.norm(centerline_tangent_world)) + 1e-9)
        centerline_correction_world = centerline_proj - tip_pos
        centerline_correction_vec_local = np.clip(self._world_vec_to_local(centerline_correction_world, frame) / wp_scale, -5.0, 5.0).astype(np.float32)
        centerline_tangent_local = self._world_vec_to_local(centerline_tangent_world, frame)
        centerline_tangent_local = (centerline_tangent_local / (float(np.linalg.norm(centerline_tangent_local)) + 1e-9)).astype(np.float32)

        prev_action = np.asarray(getattr(self, "_last_smoothed_action", np.zeros(3, dtype=np.float32)), dtype=np.float32).reshape(3).copy()
        prev_action[2] = float(getattr(self, "current_effective_insert", prev_action[2]))

        vessel_section_features = self._get_vessel_section_features(tip_pos)
        actor_current_geometry = self._build_actor_current_geometry_observation(
            tip_forward_local=tip_forward_local,
            magnetic_field_norm=magnetic_field_norm,
            waypoint_vector_local=waypoint_vector_local,
            next_waypoint_vector_local=next_waypoint_vector_local,
            centerline_correction_vec_local=centerline_correction_vec_local,
            centerline_tangent_local=centerline_tangent_local,
            waypoint_distance_norm=waypoint_distance_norm,
            waypoint_progress_ratio=waypoint_progress_ratio,
            vessel_section_features=vessel_section_features,
        )
        actor_dynamic_step = self._build_actor_dynamic_step_observation(
            prev_action=prev_action,
            tip_delta_local=tip_delta_local,
            progress_delta_norm=np.array([np.clip(float(self.current_waypoint_approach_delta) / 0.001, -5.0, 5.0)], dtype=np.float32),
        )
        self._push_actor_dynamic_history(actor_dynamic_step)
        self._last_actor_tip_pos = np.asarray(tip_pos, dtype=np.float32).copy()

        return self._get_actor_observation(actor_current_geometry)

    def _get_reward_features(self, previous_reward_features: dict) -> dict:
        current_final_dist = float(self._get_distance_tip_to_dest())
        if not np.isfinite(current_final_dist):
            current_final_dist = 1e3
            self.non_finite_failure = True
        self.min_dist_this_episode = min(float(self.min_dist_this_episode), current_final_dist)

        try:
            tip_pos = np.asarray(self.mcr_controller_sofa.get_pos_quat_catheter_tip()[0:3], dtype=np.float32)
        except Exception:
            tip_pos = np.zeros(3, dtype=np.float32)

        self._update_vessel_safety_state(tip_pos)
        valid_inside_vessel = not bool(self.current_out_of_vessel)
        reached = self._update_waypoint_progress(tip_pos, valid_inside_vessel=valid_inside_vessel)

        wp_points = getattr(self, "waypoint_points", None)
        if wp_points is not None and len(wp_points) > 0:
            in_final_target_phase = int(self.current_waypoint_idx) >= len(wp_points) - 1
        else:
            in_final_target_phase = True

        # Reward terms are intentionally separated by phase:
        #   - intermediate waypoint phase: waypoint_approach and waypoint_reached;
        #   - final target phase: target_approach and successful_task;
        #   - out-of-vessel terminal penalty;
        #   - timeout terminal penalty is added in step() after truncated is known.
        # Tip-centerline approach/keep/near-wall reward terms are disabled, but
        # centerline safety features are still computed for observation, logging,
        # waypoint gating, and out-of-vessel termination.
        # Binary approach feature:
        #   +1.0 if the active point distance is smaller than in the previous step;
        #   -1.0 if the active point distance is larger than in the previous step;
        #    0.0 if the distance is unchanged up to a tiny numerical tolerance.
        approach_delta = float(self.current_waypoint_approach_delta)
        if approach_delta > 1e-9:
            approach_feature = 1.0
        elif approach_delta < -1e-9:
            approach_feature = -1.0
        else:
            approach_feature = 0.0

        reward_features = {
            "waypoint_approach": 0.0 if in_final_target_phase else approach_feature,
            "waypoint_reached": 1.0 if reached else 0.0,
            "target_approach": approach_feature if in_final_target_phase else 0.0,
            "out_of_vessel_penalty": 1.0 if self.current_out_of_vessel else 0.0,
            "timeout_penalty": 0.0,
            "step_penalty": 1.0,
            "successful_task": 0.0,
        }

        if self.current_out_of_vessel:
            self.out_of_vessel_this_episode = True
            self.out_of_vessel_failure = True

        final_close = bool(current_final_dist <= float(self.target_distance_threshold))
        self.current_target_reached_this_step = bool(valid_inside_vessel and in_final_target_phase and final_close)
        if self.current_target_reached_this_step:
            reward_features["successful_task"] = 1.0
            self.episode_success = True
            self.episode_success_2mm = bool(current_final_dist <= 0.002 + 1e-12)
            self.is_out_of_bounds = True

        return {k: float(v) for k, v in reward_features.items()}

    def _get_reward(self) -> float:
        reward = 0.0
        self.reward_info = {}
        self.reward_features = self._get_reward_features(previous_reward_features=self.reward_features).copy()
        for key, feature in self.reward_features.items():
            value = float(self.reward_amount_dict[key]) * float(feature)
            if not np.isfinite(value):
                value = 0.0
                self.non_finite_failure = True
            self.reward_info[f"reward_{key}"] = value
            reward += value
        if not np.isfinite(reward):
            reward = -100.0
            self.non_finite_failure = True
        self.reward_info["reward"] = float(reward)
        return float(reward)

    def _get_done(self) -> bool:
        return bool(
            getattr(self, "episode_success", False)
            or getattr(self, "out_of_vessel_failure", False)
            or getattr(self, "non_finite_failure", False)
        )

    def _get_info(self, terminated: bool = False, truncated: bool = False) -> dict:
        current_dist = float(self._get_distance_tip_to_dest())
        terminal_reason = "target" if self.episode_success else "timeout" if truncated else "out_of_vessel" if self.out_of_vessel_failure else "non_finite" if self.non_finite_failure else "other" if terminated else "not_done"
        chosen_model = str(getattr(self, "chosen_model", "unknown"))
        info = {
            "task_id": str(getattr(self, "task_id", "unknown")),
            "chosen_model": chosen_model,
            "sampling_model": str(getattr(self, "current_sampling_model", chosen_model)),
            "success_2mm": bool(self.episode_success_2mm),
            "done_by_target": bool(self.episode_success),
            "done_by_timeout": bool(truncated),
            "done_by_out_of_vessel": bool(self.out_of_vessel_failure),
            "done_by_non_finite": bool(self.non_finite_failure),
            "terminal_reason": terminal_reason,
            "min_dist_to_goal": float(self.min_dist_this_episode),
            "current_dist_to_goal": current_dist,
            "final_dist_to_goal": current_dist if (terminated or truncated) else np.nan,
            "target_distance_threshold": float(self.target_distance_threshold),
            "waypoint_idx": int(self.current_waypoint_idx),
            "waypoint_num": int(len(self.waypoint_points)) if self.waypoint_points is not None else 0,
            "waypoint_progress_ratio": float(self.current_waypoint_progress_ratio),
            "waypoint_distance": float(self.current_waypoint_distance),
            "waypoint_approach_delta": float(self.current_waypoint_approach_delta),
            "waypoint_reached_this_step": bool(self.current_waypoint_reached_this_step),
            "waypoint_reached_count_episode": int(self.current_waypoint_reached_count_episode),
            "waypoint_handoff_counter": int(
                getattr(self, "current_waypoint_handoff_counter", 0)
            ),
            "waypoint_handoff_this_step": bool(
                getattr(self, "current_waypoint_handoff_this_step", False)
            ),
            "waypoint_handoff_count_episode": int(
                getattr(self, "current_waypoint_handoff_count_episode", 0)
            ),
            "waypoint_is_final": bool(getattr(self, "current_waypoint_is_final", False)),
            "target_reached_this_step": bool(getattr(self, "current_target_reached_this_step", False)),
            "target_approach_delta": float(self.current_waypoint_approach_delta) if bool(getattr(self, "current_waypoint_is_final", False)) else 0.0,
            "waypoint_spacing": float(self.waypoint_spacing),
            "waypoint_reach_threshold": float(self.waypoint_reach_threshold),
            "pre_target_waypoint_offset": float(getattr(self, "pre_target_waypoint_offset", 0.001)),
            "out_of_vessel": bool(self.current_out_of_vessel),
            "out_of_vessel_this_episode": bool(self.out_of_vessel_this_episode),
            "out_of_vessel_safety_ratio": float(self.out_of_vessel_safety_ratio),
            "centerline_local_radius": float(self.current_centerline_local_radius),
            "centerline_safety_ratio": float(self.current_centerline_safety_ratio),
            "centerline_safety_margin": float(self.current_centerline_safety_margin),
            "centerline_offset_N_over_radius": float(self.current_centerline_offset_N_over_radius),
            "centerline_offset_B_over_radius": float(self.current_centerline_offset_B_over_radius),
            "centerline_local_radius_norm": float(self.current_centerline_local_radius_norm),
            "tip_centerline_radial_offset": float(self.current_centerline_radial_offset),
            "tip_centerline_offset_norm": float(self.current_tip_centerline_offset_norm),
            "centerline_progress": float(self.current_centerline_progress),
            "centerline_safety_ratio_max_episode": float(self.max_safety_ratio_this_episode),
            "centerline_safety_margin_min_episode": float(self.min_safety_margin_this_episode),
            "raw_insert": float(self.current_raw_insert),
            "effective_insert": float(self.current_effective_insert),
            "insert_bias": float(getattr(self, "insert_bias", 0.0)),
            "insert_negative_limit": float(getattr(self, "insert_negative_limit", -1.0)),
            "actor_obs_dim": int(self.actor_observation_dim),
        }
        return {**info, **self.reward_info, **self.reward_features}

    # ------------------------------------------------------------------
    # Centerline / waypoint / safety
    # ------------------------------------------------------------------
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
            dense_u = np.linspace(0.0, 1.0, max(expected_count * 5, len(points) * 10))
            dense_xyz = np.asarray(splev(dense_u, tck), dtype=np.float32).T
            dense_seg = np.linalg.norm(np.diff(dense_xyz, axis=0), axis=1)
            dense_cum = np.concatenate(([0.0], np.cumsum(dense_seg)))
            dense_total = float(dense_cum[-1])
            if dense_total < 1e-9:
                return points
            target_dist = np.arange(0.0, dense_total, target_spacing, dtype=np.float32)
            if target_dist.size == 0 or target_dist[-1] < dense_total:
                target_dist = np.append(target_dist, dense_total)
            return np.stack(
                [
                    np.interp(target_dist, dense_cum, dense_xyz[:, 0]),
                    np.interp(target_dist, dense_cum, dense_xyz[:, 1]),
                    np.interp(target_dist, dense_cum, dense_xyz[:, 2]),
                ],
                axis=1,
            ).astype(np.float32)
        except Exception:
            return points

    def _resample_centerline_radius_1mm(self, raw_points: np.ndarray, raw_radius, resampled_points: np.ndarray):
        if raw_points is None or raw_radius is None or resampled_points is None:
            return None
        raw_points = np.asarray(raw_points, dtype=np.float32)
        raw_radius = np.asarray(raw_radius, dtype=np.float32).reshape(-1)
        resampled_points = np.asarray(resampled_points, dtype=np.float32)
        if raw_points.ndim != 2 or raw_points.shape[1] != 3 or resampled_points.ndim != 2 or resampled_points.shape[1] != 3:
            return None
        if len(raw_points) < 2 or len(raw_radius) != len(raw_points) or len(resampled_points) < 2 or not np.all(np.isfinite(raw_radius)):
            return None
        raw_cum = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(raw_points, axis=0), axis=1)))).astype(np.float32)
        res_cum = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(resampled_points, axis=0), axis=1)))).astype(np.float32)
        if float(raw_cum[-1]) < 1e-9 or float(res_cum[-1]) < 1e-9:
            return None
        target_raw_dist = res_cum / (float(res_cum[-1]) + 1e-12) * float(raw_cum[-1])
        return np.interp(target_raw_dist, raw_cum, raw_radius).astype(np.float32)

    def _get_centerline_projection_state(self, tip_pos: np.ndarray):
        if self.centerline_points is None or len(self.centerline_points) < 2 or self.centerline_cumlength is None:
            return 0.0, -1, 0.0, np.asarray(tip_pos, dtype=np.float32), np.array([1.0, 0.0, 0.0], dtype=np.float32)
        points = np.asarray(self.centerline_points, dtype=np.float32)
        seg_start = points[:-1]
        seg_end = points[1:]
        seg_vec = seg_end - seg_start
        tip_pos = np.asarray(tip_pos, dtype=np.float32).reshape(3)
        seg_len_sq = np.maximum(np.sum(seg_vec * seg_vec, axis=1), 1e-12)
        t = np.clip(np.sum((tip_pos[None, :] - seg_start) * seg_vec, axis=1) / seg_len_sq, 0.0, 1.0)
        proj = seg_start + t[:, None] * seg_vec
        dists = np.linalg.norm(proj - tip_pos[None, :], axis=1)
        idx = int(np.argmin(dists))
        seg_length = float(np.linalg.norm(seg_vec[idx]))
        tangent = seg_vec[idx] / (seg_length + 1e-9)
        progress = float(self.centerline_cumlength[idx] + float(t[idx]) * seg_length)
        return progress, idx, float(dists[idx]), proj[idx].astype(np.float32), tangent.astype(np.float32)

    def _interpolate_centerline_point_at_progress(self, query_progress: float) -> np.ndarray:
        points = getattr(self, "centerline_points", None)
        cum = getattr(self, "centerline_cumlength", None)
        if points is None or cum is None or len(points) == 0 or len(cum) == 0:
            return np.zeros(3, dtype=np.float32)
        points = np.asarray(points, dtype=np.float32)
        cum = np.asarray(cum, dtype=np.float32)
        q = float(np.clip(float(query_progress), float(cum[0]), float(cum[-1])))
        return np.array([np.interp(q, cum, points[:, 0]), np.interp(q, cum, points[:, 1]), np.interp(q, cum, points[:, 2])], dtype=np.float32)

    def _raw_project_point_to_centerline_progress(self, point: np.ndarray) -> Tuple[float, int, float]:
        progress, idx, dist, _, _ = self._get_centerline_projection_state(point)
        return progress, idx, dist

    def _get_current_local_radius(self, centerline_seg_idx: int):
        if self.centerline_radius is None or self.centerline_points is None:
            return None
        radius = np.asarray(self.centerline_radius, dtype=np.float32).reshape(-1)
        if len(radius) == 0:
            return None
        idx = int(np.clip(centerline_seg_idx, 0, len(radius) - 1))
        local_radius = float(radius[idx])
        return local_radius if np.isfinite(local_radius) and local_radius > 1e-6 else None

    def _update_vessel_safety_state(self, tip_pos: np.ndarray) -> None:
        progress, seg_idx, centerline_dist, centerline_proj, tangent = self._get_centerline_projection_state(tip_pos)
        local_radius = self._get_current_local_radius(seg_idx)
        if local_radius is None:
            local_radius = float(self.default_local_radius)
        local_radius = max(float(local_radius), 1e-9)
        catheter_radius = float(self.catheter_radius)

        frame = self._build_local_centerline_frame(tangent)
        offset_world = np.asarray(tip_pos, dtype=np.float32) - np.asarray(centerline_proj, dtype=np.float32)
        offset_local = self._world_vec_to_local(offset_world, frame)
        radial_offset = float(
            np.sqrt(
                offset_local[1] ** 2
                +
                offset_local[2] ** 2
            )
        )

        # -------------------------------------------------
        # catheter-aware safety ratio
        #
        # radial_offset:
        #   tip中心距离centerline距离
        #
        # local_radius:
        #   vessel radius
        #
        # catheter_radius:
        #   catheter自身半径
        #
        # 可用空间:
        #   vessel radius - catheter radius
        #
        # -------------------------------------------------

        effective_radius = max(
            local_radius - catheter_radius,
            1e-6
        )

        safety_ratio = float(
            radial_offset / effective_radius
        )

        safety_margin = float(
            1.0 - safety_ratio
        )

        tangent = np.asarray(tangent, dtype=np.float32).reshape(3)
        tangent = tangent / (float(np.linalg.norm(tangent)) + 1e-9)
        self.current_centerline_progress = float(progress)
        self.current_centerline_projection = np.asarray(centerline_proj, dtype=np.float32).reshape(3)
        self.current_centerline_tangent = tangent.astype(np.float32)
        self.current_centerline_local_radius = local_radius
        self.current_centerline_safety_ratio = safety_ratio
        self.current_centerline_safety_margin = safety_margin
        self.current_centerline_radial_offset = radial_offset
        self.current_tip_centerline_offset_norm = float(radial_offset / local_radius)
        self.current_centerline_offset_N_over_radius = float(offset_local[1] / local_radius)
        self.current_centerline_offset_B_over_radius = float(offset_local[2] / local_radius)
        self.current_centerline_local_radius_norm = float(local_radius / max(float(self.radius_observation_scale), 1e-9))
        self.max_safety_ratio_this_episode = max(float(self.max_safety_ratio_this_episode), safety_ratio)
        self.min_safety_margin_this_episode = min(float(self.min_safety_margin_this_episode), safety_margin)
        self.current_out_of_vessel = bool(safety_ratio >= float(self.out_of_vessel_safety_ratio))
        if not np.isfinite(safety_ratio):
            self.current_out_of_vessel = bool(np.isfinite(centerline_dist) and centerline_dist >= float(self.out_of_vessel_fallback_distance))

    def _get_vessel_section_features(self, tip_pos: np.ndarray) -> np.ndarray:
        self._update_vessel_safety_state(tip_pos)
        return np.array(
            [
                np.clip(self.current_centerline_offset_N_over_radius, -3.0, 3.0),
                np.clip(self.current_centerline_offset_B_over_radius, -3.0, 3.0),
                np.clip(self.current_centerline_local_radius_norm, 0.0, 5.0),
                np.clip(self.current_centerline_safety_margin, -3.0, 1.0),
            ],
            dtype=np.float32,
        )

    def _get_distance_tip_to_dest(self):
        tip = np.asarray(self.mcr_controller_sofa.get_pos_quat_catheter_tip()[0:3], dtype=np.float32)
        dist = float(np.linalg.norm(np.asarray(self.target_position, dtype=np.float32) - tip))
        if not np.isfinite(dist):
            self.non_finite_failure = True
            return float(1e3)
        return dist

    def _build_waypoint_sequence(self) -> None:
        self.waypoint_points = None
        self.waypoint_progress = None
        if self.centerline_points is None or self.centerline_cumlength is None or len(self.centerline_points) < 2:
            return
        total_len = float(self.centerline_cumlength[-1])
        if not np.isfinite(total_len) or total_len <= 1e-9:
            return
        try:
            target_progress, _, _ = self._raw_project_point_to_centerline_progress(self.target_position)
            target_progress = float(np.clip(target_progress, float(self.centerline_cumlength[0]), total_len))
        except Exception:
            target_progress = total_len

        # Start waypoint generation from the actual episode start, not always
        # from centerline_cumlength[0]. This is essential when start-point
        # randomization places the catheter at P1/P2/... instead of P0.
        start_progress = float(self.centerline_cumlength[0])
        start_reference = getattr(self, "current_soft_start_position", None)
        if start_reference is None:
            start_reference = getattr(self, "current_scene_start_position", None)
        if start_reference is not None:
            try:
                start_reference = np.asarray(start_reference, dtype=np.float32).reshape(3)
                if np.all(np.isfinite(start_reference)):
                    start_progress, _, _ = self._raw_project_point_to_centerline_progress(start_reference)
                    start_progress = float(np.clip(start_progress, float(self.centerline_cumlength[0]), total_len))
            except Exception:
                start_progress = float(self.centerline_cumlength[0])

        # Avoid degenerate/reversed route if a sampled target is accidentally
        # not ahead of the sampled start.
        if target_progress <= start_progress + 1e-7:
            start_progress = float(self.centerline_cumlength[0])

        self.current_waypoint_start_progress = float(start_progress)
        self.current_waypoint_target_progress = float(target_progress)

        spacing = max(float(self.waypoint_spacing), 1e-6)
        pre_target_offset = max(float(getattr(self, "pre_target_waypoint_offset", 0.001)), 0.0)
        eps = 1e-7

        is_aorta6 = (
            str(getattr(self, "task_id", "")).lower() == "aorta6"
            or str(getattr(self, "_explicit_force_model", "")).lower() == "aorta6"
        )

        # Normally waypoint generation starts at the physical episode start.
        # For aorta6, exclude the physical start itself and make the first
        # waypoint one waypoint-spacing farther toward the target.
        waypoint_generation_start = float(start_progress)
        if is_aorta6:
            waypoint_generation_start = float(
                min(start_progress + spacing, target_progress)
            )

        # Keep the last intermediate waypoint at pre_target_offset before the
        # final target. The final target itself is appended separately.
        if (
            target_progress - waypoint_generation_start
            > pre_target_offset + eps
        ):
            pre_target_progress = float(
                np.clip(
                    target_progress - pre_target_offset,
                    waypoint_generation_start,
                    target_progress,
                )
            )

            waypoint_progress = np.arange(
                waypoint_generation_start,
                pre_target_progress,
                spacing,
                dtype=np.float32,
            )

            if (
                waypoint_progress.size == 0
                or abs(
                    float(waypoint_progress[-1])
                    - pre_target_progress
                ) > eps
            ):
                waypoint_progress = np.append(
                    waypoint_progress,
                    np.float32(pre_target_progress),
                )
            else:
                waypoint_progress[-1] = np.float32(
                    pre_target_progress
                )
        else:
            # For aorta6, never put start_progress back into the waypoint list.
            # If the remaining route is very short, use the target directly.
            if is_aorta6:
                waypoint_progress = np.asarray(
                    [],
                    dtype=np.float32,
                )
            else:
                waypoint_progress = np.asarray(
                    [start_progress],
                    dtype=np.float32,
                )

        # Always append the actual target as the final task point.
        if (
            waypoint_progress.size == 0
            or abs(
                float(waypoint_progress[-1])
                - target_progress
            ) > eps
        ):
            waypoint_progress = np.append(
                waypoint_progress,
                np.float32(target_progress),
            )
        else:
            waypoint_progress[-1] = np.float32(
                target_progress
            )

        # Preserve the original fallback for other vessels. For aorta6, a
        # one-point sequence containing only the target is valid and must not
        # reintroduce the physical start as waypoint[0].
        if waypoint_progress.size < 2 and not is_aorta6:
            waypoint_progress = np.asarray(
                [start_progress, target_progress],
                dtype=np.float32,
            )

        if is_aorta6:
            first_progress = float(waypoint_progress[0])
            print(
                "[AORTA6_WAYPOINT_AFTER_START]",
                "physical_start_progress_mm=",
                float(start_progress) * 1000.0,
                "first_waypoint_progress_mm=",
                first_progress * 1000.0,
                "distance_after_start_mm=",
                max(0.0, first_progress - float(start_progress)) * 1000.0,
                "waypoint_num=",
                int(len(waypoint_progress)),
            )

        waypoint_points = np.asarray([self._interpolate_centerline_point_at_progress(float(q)) for q in waypoint_progress], dtype=np.float32)
        if self.target_position is not None and len(waypoint_points) > 0:
            target_np = np.asarray(self.target_position, dtype=np.float32).reshape(3)
            if np.all(np.isfinite(target_np)):
                waypoint_points[-1] = target_np
        self.waypoint_progress = waypoint_progress.astype(np.float32)
        self.waypoint_points = waypoint_points.astype(np.float32)
        self.current_waypoint_idx = 0
        self.current_waypoint_progress_ratio = 0.0
        self.current_waypoint_distance = np.nan
        self.current_waypoint_approach_delta = 0.0
        self.current_waypoint_reached_this_step = False
        self.current_waypoint_reached_count_episode = 0
        self.current_waypoint_is_final = False
        self.current_target_reached_this_step = False
        self.previous_waypoint_idx = None
        self.previous_waypoint_distance = None
        self.current_waypoint_handoff_counter = 0
        self.current_waypoint_handoff_this_step = False
        self.current_waypoint_handoff_count_episode = 0

    def _initialize_waypoint_progress_from_current_tip(self) -> None:
        """Start every episode from the first waypoint.

        Do not initialize the active waypoint by centerline projection. This keeps
        the waypoint task strictly sequential: a waypoint is counted only after
        the tip enters its reach threshold.
        """
        try:
            tip = np.asarray(self.mcr_controller_sofa.get_pos_quat_catheter_tip()[0:3], dtype=np.float32)
        except Exception:
            tip = np.zeros(3, dtype=np.float32)

        points = getattr(self, "waypoint_points", None)
        if points is None or len(points) == 0:
            self.current_waypoint_idx = 0
            self.current_waypoint_progress_ratio = 0.0
            self.current_waypoint_distance = float(np.linalg.norm(np.asarray(self.target_position, dtype=np.float32) - tip))
            self.current_waypoint_approach_delta = 0.0
            self.current_waypoint_reached_this_step = False
            self.current_waypoint_reached_count_episode = 0
            self.current_waypoint_is_final = False
            self.current_target_reached_this_step = False
            self.previous_waypoint_idx = 0
            self.previous_waypoint_distance = self.current_waypoint_distance
            self.current_waypoint_handoff_counter = 0
            self.current_waypoint_handoff_this_step = False
            self.current_waypoint_handoff_count_episode = 0
            return

        idx = 0
        self.current_waypoint_idx = idx
        self.current_waypoint_progress_ratio = 0.0
        self.current_waypoint_distance = float(np.linalg.norm(np.asarray(points[idx], dtype=np.float32) - tip))
        self.current_waypoint_approach_delta = 0.0
        self.current_waypoint_reached_this_step = False
        self.current_waypoint_reached_count_episode = 0
        self.current_waypoint_is_final = False
        self.current_target_reached_this_step = False
        self.previous_waypoint_idx = idx
        self.previous_waypoint_distance = self.current_waypoint_distance
        self.current_waypoint_handoff_counter = 0
        self.current_waypoint_handoff_this_step = False
        self.current_waypoint_handoff_count_episode = 0

    def _update_waypoint_progress(self, tip_pos: np.ndarray, valid_inside_vessel: bool = True) -> bool:
        """Update ordered Euclidean waypoints with adjacent-point handoff.

        Normal hit:
            current waypoint distance <= waypoint_reach_threshold.
            Advance one waypoint and keep the +100 waypoint bonus.

        Missed-point handoff:
            the next ordered waypoint is clearly closer than the current one
            for waypoint_handoff_confirm_steps consecutive steps.
            Advance exactly one waypoint, but do not award the +100 bonus.

        The handoff uses only current/next waypoint Euclidean distances. It does
        not use centerline progress or a cross-section gate.
        """
        points = getattr(self, "waypoint_points", None)
        self.current_waypoint_handoff_this_step = False

        if points is None or len(points) == 0:
            target_dist = float(
                np.linalg.norm(
                    np.asarray(self.target_position, dtype=np.float32)
                    - np.asarray(tip_pos, dtype=np.float32)
                )
            )
            self.current_waypoint_distance = target_dist
            self.current_waypoint_approach_delta = 0.0
            self.current_waypoint_is_final = True
            self.current_waypoint_reached_this_step = False
            self.current_target_reached_this_step = bool(
                valid_inside_vessel
                and target_dist <= float(self.target_distance_threshold)
            )
            self.current_waypoint_handoff_counter = 0
            return False

        points = np.asarray(points, dtype=np.float32)
        tip_pos = np.asarray(tip_pos, dtype=np.float32).reshape(3)
        idx = int(np.clip(self.current_waypoint_idx, 0, len(points) - 1))
        last_idx = int(len(points) - 1)
        is_final = bool(idx >= last_idx)
        self.current_waypoint_is_final = is_final

        active_point = np.asarray(
            self.target_position if is_final else points[idx],
            dtype=np.float32,
        ).reshape(3)
        current_dist = float(np.linalg.norm(active_point - tip_pos))

        prev_dist = self.previous_waypoint_distance
        if (
            self.previous_waypoint_idx is None
            or int(self.previous_waypoint_idx) != idx
            or prev_dist is None
            or not np.isfinite(float(prev_dist))
        ):
            prev_dist = current_dist

        self.current_waypoint_approach_delta = float(
            np.clip(
                float(prev_dist) - current_dist,
                -self.progress_clip,
                self.progress_clip,
            )
        )

        if is_final:
            self.current_waypoint_idx = last_idx
            self.current_waypoint_distance = float(current_dist)
            self.current_waypoint_progress_ratio = 1.0
            self.current_waypoint_reached_this_step = False
            self.current_target_reached_this_step = bool(
                valid_inside_vessel
                and current_dist <= float(self.target_distance_threshold)
            )
            self.previous_waypoint_idx = int(self.current_waypoint_idx)
            self.previous_waypoint_distance = float(current_dist)
            self.current_waypoint_handoff_counter = 0
            return False

        reached_by_radius = bool(
            valid_inside_vessel
            and current_dist <= float(self.waypoint_reach_threshold)
        )

        # Compare only the adjacent next waypoint.
        next_point = np.asarray(
            self.target_position if idx + 1 >= last_idx else points[idx + 1],
            dtype=np.float32,
        ).reshape(3)
        next_dist = float(np.linalg.norm(next_point - tip_pos))

        next_is_clearly_closer = bool(
            valid_inside_vessel
            and not reached_by_radius
            and next_dist + float(self.waypoint_handoff_margin) < current_dist
        )

        if next_is_clearly_closer:
            self.current_waypoint_handoff_counter = int(
                getattr(self, "current_waypoint_handoff_counter", 0)
            ) + 1
        else:
            self.current_waypoint_handoff_counter = 0

        handoff = bool(
            self.current_waypoint_handoff_counter
            >= int(self.waypoint_handoff_confirm_steps)
        )
        advance = bool(reached_by_radius or handoff)

        # Only a true 3 mm hit is counted/rewarded as waypoint_reached.
        self.current_waypoint_reached_this_step = bool(reached_by_radius)
        self.current_target_reached_this_step = False

        if advance:
            if reached_by_radius:
                self.current_waypoint_reached_count_episode += 1
            else:
                self.current_waypoint_handoff_this_step = True
                self.current_waypoint_handoff_count_episode += 1

            # Advance at most one waypoint per environment step.
            idx += 1
            self.current_waypoint_idx = int(idx)
            is_final = bool(idx >= last_idx)
            self.current_waypoint_is_final = is_final

            active_point = np.asarray(
                self.target_position if is_final else points[idx],
                dtype=np.float32,
            ).reshape(3)
            current_dist = float(np.linalg.norm(active_point - tip_pos))

            # Distances before/after a target switch are not comparable.
            # Reset the switch-step delta to avoid a fake +20/-20 reward.
            if handoff or is_final:
                self.current_waypoint_approach_delta = 0.0

            self.current_waypoint_handoff_counter = 0
        else:
            self.current_waypoint_idx = int(idx)

        self.current_waypoint_distance = float(current_dist)
        self.current_waypoint_progress_ratio = float(
            self.current_waypoint_idx / max(1, last_idx)
        )
        self.previous_waypoint_idx = int(self.current_waypoint_idx)
        self.previous_waypoint_distance = float(current_dist)

        # Return True only for a real threshold hit, so handoff gets no +100.
        return bool(reached_by_radius)

    # ------------------------------------------------------------------
    # Action application and scene initialization
    # ------------------------------------------------------------------

    def _apply_insert_safety_shield(self, raw_insert: float) -> float:
        """Aggressive high-curvature insertion damping test.

        Temporary experiment:
        - safety ratio <= 0.6: original insertion command
        - safety ratio > 0.6: insertion magnitude reduced to 10%

        Purpose:
        verify whether excessive insertion in sharp turns causes
        accumulated bending energy and late-stage oscillation.
        """
        raw_insert = float(np.clip(raw_insert, -1.0, 1.0))

        self.current_raw_insert = raw_insert

        ratio = float(getattr(self, "current_centerline_safety_ratio", np.nan))

        if np.isfinite(ratio) and ratio > 0.6:
            effective_insert = raw_insert * 0.1
        else:
            effective_insert = raw_insert

        effective_insert = float(
            np.clip(
                effective_insert,
                float(self.insert_negative_limit),
                1.0,
            )
        )

        self.current_effective_insert = effective_insert
        return self.current_effective_insert



    def _apply_local_magnetic_action(self, rot_n: float, rot_b: float) -> None:
        try:
            tip_pose = np.asarray(self.mcr_controller_sofa.get_pos_quat_catheter_tip(), dtype=np.float32)
            frame = self._build_tip_local_frame(tip_pose[3:7])
            axis_n = np.asarray(frame[:, 1], dtype=np.float32)
            axis_b = np.asarray(frame[:, 2], dtype=np.float32)
            axis_n = axis_n / (float(np.linalg.norm(axis_n)) + 1e-9)
            axis_b = axis_b / (float(np.linalg.norm(axis_b)) + 1e-9)
            field = np.asarray(self.mcr_controller_sofa.get_mag_field_des(), dtype=np.float32)
            old_mag = float(np.linalg.norm(field))
            angle_scale = float(self.local_field_action_angle)
            if abs(float(rot_n)) > 1e-8:
                field = R.from_rotvec(float(rot_n) * angle_scale * axis_n).apply(field).astype(np.float32)
            if abs(float(rot_b)) > 1e-8:
                field = R.from_rotvec(float(rot_b) * angle_scale * axis_b).apply(field).astype(np.float32)
            new_mag = float(np.linalg.norm(field))
            if old_mag > 1e-9 and new_mag > 1e-9:
                field = field / new_mag * old_mag
            self.mcr_controller_sofa.mag_controller.field_des = field
        except Exception:
            self.mcr_controller_sofa.rotateZ(float(rot_n))
            self.mcr_controller_sofa.rotateX(float(rot_b))


    def _do_action(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        action = np.clip(action, -1.0, 1.0)

        rot_n = float(action[0]) if action.shape[0] > 0 else 0.0
        rot_b = float(action[1]) if action.shape[0] > 1 else 0.0
        raw_insert = float(action[2]) if action.shape[0] > 2 else 0.0

        effective_insert = self._apply_insert_safety_shield(raw_insert)
        self._apply_local_magnetic_action(rot_n, rot_b)
        self.mcr_controller_sofa.insertRetract(effective_insert)

    def _init_sim(self):
        super()._init_sim()
        self.mcr_controller_sofa: ControllerSofa = self.scene_creation_result["mcr_controller_sofa"]
        self.mcr_environment = self.scene_creation_result["mcr_environment"]
        self.chosen_model = self.scene_creation_result.get("chosen_model", "unknown")
        self.centerline_vtk = self.scene_creation_result.get("centerline_vtk", "unknown")
        self.task_id = self.scene_creation_result.get("task_id", str(self.chosen_model))

        raw_centerline = self.scene_creation_result.get("centerline_points", None)
        raw_centerline_radius = self.scene_creation_result.get("centerline_radius", None)
        if raw_centerline is not None:
            raw_centerline_arr = np.asarray(raw_centerline, dtype=np.float32)
            self.centerline_points = self._resample_centerline_points_1mm(raw_centerline_arr)
            raw_radius_arr = None
            try:
                raw_radius_arr = None if raw_centerline_radius is None else np.asarray(raw_centerline_radius, dtype=np.float32).reshape(-1)
            except Exception:
                raw_radius_arr = None
            if raw_radius_arr is not None and len(raw_radius_arr) == len(raw_centerline_arr) and np.all(np.isfinite(raw_radius_arr)):
                self.centerline_radius = self._resample_centerline_radius_1mm(raw_centerline_arr, raw_radius_arr, self.centerline_points)
            else:
                self.centerline_radius = None
        else:
            self.centerline_points = None
            self.centerline_radius = None

        scene_target = self.scene_creation_result.get("target_position", None)
        if scene_target is not None:
            self.target_position = np.asarray(scene_target, dtype=np.float32).reshape(3)

        # Scene creation may already randomize the entry point when not using
        # soft single-vessel randomization. Keep that sampled start so the
        # waypoint sequence can begin from the actual catheter start rather
        # than from the absolute centerline endpoint. Reset soft-start here to
        # avoid stale start references after full scene reloads; soft reset will
        # set it again before rebuilding waypoints.
        self.current_scene_start_position = None
        self.current_soft_start_position = None
        scene_start = self.scene_creation_result.get("nominal_start_position", None)
        if scene_start is not None:
            try:
                scene_start_np = np.asarray(scene_start, dtype=np.float32).reshape(3)
                if np.all(np.isfinite(scene_start_np)):
                    self.current_scene_start_position = scene_start_np.copy()
            except Exception:
                self.current_scene_start_position = None

        self.centerline_reversed_for_progress = False
        if self.centerline_points is not None and self.target_position is not None and len(self.centerline_points) >= 2:
            try:
                init_tip = np.asarray(self.mcr_controller_sofa.get_pos_quat_catheter_tip()[0:3], dtype=np.float32)
                p0 = self.centerline_points[0]
                p1 = self.centerline_points[-1]
                forward_cost = float(np.linalg.norm(p0 - init_tip) + np.linalg.norm(p1 - self.target_position))
                reverse_cost = float(np.linalg.norm(p1 - init_tip) + np.linalg.norm(p0 - self.target_position))
                self.centerline_reversed_for_progress = bool(reverse_cost + 1e-9 < forward_cost)
            except Exception:
                self.centerline_reversed_for_progress = bool(np.linalg.norm(self.centerline_points[0] - self.target_position) < np.linalg.norm(self.centerline_points[-1] - self.target_position))
            if self.centerline_reversed_for_progress:
                self.centerline_points = self.centerline_points[::-1].copy()
                if self.centerline_radius is not None:
                    self.centerline_radius = self.centerline_radius[::-1].copy()

        if self.centerline_points is not None and len(self.centerline_points) >= 2:
            seg_lengths = np.linalg.norm(np.diff(self.centerline_points, axis=0), axis=1)
            self.centerline_cumlength = np.concatenate(([0.0], np.cumsum(seg_lengths))).astype(np.float32)
        else:
            self.centerline_cumlength = None

        self._build_waypoint_sequence()
        self._capture_soft_reset_reference_pose()

        vessel_positions = np.asarray(self.mcr_environment.get_vessel_tree_positions(), dtype=np.float32)
        if vessel_positions.ndim == 2 and vessel_positions.shape[1] == 3 and vessel_positions.shape[0] > 1 and np.all(np.isfinite(vessel_positions)):
            bbox_diag = float(np.linalg.norm(np.min(vessel_positions, axis=0) - np.max(vessel_positions, axis=0)))
            self.cartesian_scaling_factor = 1.0 / bbox_diag if np.isfinite(bbox_diag) and bbox_diag > 1e-9 else 1.0
        else:
            self.cartesian_scaling_factor = 1.0


if __name__ == "__main__":
    env = MCREnv(env_type=EnvType.AORTIC)
    env.reset()
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(reward)
        if terminated or truncated:
            break
    env.close()
