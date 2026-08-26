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
from .vessel_assets import load_signed_distance_grid, load_vessel_metadata
from .training_config import (
    ACTOR_HISTORY_STEPS,
    ACTOR_CURRENT_GEOMETRY_DIM,
    ACTOR_DYNAMIC_STEP_DIM,
    CATHETER_RADIUS_M,
    CENTERLINE_LOOKAHEAD_DISTANCES_M,
    ENTRY_TANGENT_POINTS,
    FRAME_SKIP,
    INITIAL_ORIENTATION_MAX_ANGLE_DEG,
    LOCAL_FIELD_ACTION_ANGLE_RAD,
    MAX_ACTION_DELTA,
    MAX_EPISODE_STEPS,
    NO_PROGRESS_CONFIRM_STEPS,
    NO_PROGRESS_GRACE_STEPS,
    NO_PROGRESS_MIN_NET_APPROACH_M,
    NO_PROGRESS_WINDOW_STEPS,
    OUT_OF_VESSEL_FALLBACK_DISTANCE_M,
    OUT_OF_VESSEL_SAFETY_RATIO,
    RADIUS_OBSERVATION_SCALE_M,
    REWARD_OUT_OF_VESSEL,
    REWARD_OFF_TARGET_BRANCH,
    REWARD_NO_PROGRESS,
    REWARD_NO_PROGRESS_TERMINAL,
    REWARD_NON_FINITE,
    REWARD_PROGRESS_NORMALIZATION_M,
    REWARD_ROUTE_PROGRESS,
    REWARD_RETRACTION,
    REWARD_STEP,
    REWARD_SUCCESS,
    REWARD_TIMEOUT,
    REWARD_WALL_PENETRATION,
    REWARD_WALL_PROXIMITY,
    REWARD_WRONG_BRANCH,
    SETTLE_STEPS,
    SOFA_TIME_STEP_S,
    SDF_CLEARANCE_OBSERVATION_SCALE_M,
    SDF_BODY_WARNING_MARGIN_M,
    SDF_FORWARD_PROBE_DISTANCES_M,
    SDF_NEAR_WALL_MARGIN_M,
    SDF_OUTSIDE_CENTER_TOLERANCE_M,
    SDF_OUTSIDE_CONFIRM_STEPS,
    SDF_SAMPLE_STEP_FRACTION,
    START_WINDOW_DISTANCE_M,
    TARGET_THRESHOLD_M,
    TARGET_WINDOW_DISTANCE_M,
    TIP_NEAR_WALL_GRACE_STEPS,
    TIP_NEAR_WALL_RAMP_STEPS,
    ROUTE_GUIDANCE_LOOKAHEAD_DISTANCES_M,
    ROUTE_GUIDANCE_OBSERVATION_SCALE_M,
    ROUTE_PROJECTION_AMBIGUITY_TOLERANCE_M,
    ROUTE_PROJECTION_BACKWARD_WINDOW_M,
    ROUTE_PROJECTION_FORWARD_WINDOW_M,
    ROUTE_PROJECTION_MAX_PROGRESS_STEP_M,
    ROUTE_SUCCESS_PROGRESS_MARGIN_M,
    WRONG_BRANCH_CONFIRM_STEPS,
    WRONG_BRANCH_DISTANCE_MARGIN_M,
    WRONG_BRANCH_OBSERVATION_SCALE_M,
    TRAINING_CURRICULUM_ENABLED,
    TRAINING_CURRICULUM_DR_FRACTIONS,
    TRAINING_CURRICULUM_MODELS,
    TRAINING_CURRICULUM_TARGET_FRACTIONS,
    VESSEL_SCALE_MAX,
    VESSEL_SCALE_MIN,
    VESSEL_SECTION_FEATURE_DIM,
    body_sdf_risk_features,
)
from .route_tracking import normalized_route_progress, project_to_route

MCR_SIM_DIR = Path(__file__).resolve().parent
FLAT_SCENE_DESCRIPTION_FILE_PATH = MCR_SIM_DIR / "scene_description_2d.py"
AORTIC_SCENE_DESCRIPTION_FILE_PATH = SCENE_DIR / "example_aortic_arch.py"
FLAT_CATHETER_DESTINATION_EXIT_POINT = np.array([0.101129, 0.0238015, 0.002], dtype=np.float32)
AORTIC_CATHETER_DESTINATION_EXIT_POINT = np.array([-0.0101583, -0.180636, 0.0345185], dtype=np.float32)


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
    """Multi-asset continuous-route mCR RL environment.

    Training logic kept in this file:
      1. The selected target centerline supplies jump-safe progress and moving guidance.
      2. The VTI SDF supplies catheter-surface clearance and true lumen escape.
      3. The complete branching graph separates a wrong route from vessel escape.
      4. Collision remains a SOFA Triangle versus catheter Line/Point solve.
      5. Metadata validates that the runtime asset bundle is self-consistent.
      6. Actor and critic use the same tip-local observation and short history.
    """

    def __init__(
        self,
        image_shape: Tuple[int, int] = (400, 400),
        create_scene_kwargs: Optional[dict] = None,
        observation_type: ObservationType = ObservationType.STATE,
        action_type: ActionType = ActionType.CONTINUOUS,
        time_step: float = SOFA_TIME_STEP_S,
        frame_skip: int = FRAME_SKIP,
        settle_steps: int = SETTLE_STEPS,
        render_mode: RenderMode = RenderMode.HUMAN,
        render_framework: RenderFramework = RenderFramework.PYGLET,
        reward_amount_dict: Optional[dict] = None,
        target_position: Optional[np.ndarray] = None,
        env_type: EnvType = EnvType.FLAT,
        target_distance_threshold: float = TARGET_THRESHOLD_M,
        num_catheter_tracking_points: int = 4,
        max_episode_steps: int = MAX_EPISODE_STEPS,
    ):
        if not isinstance(create_scene_kwargs, dict):
            create_scene_kwargs = {}
        create_scene_kwargs["image_shape"] = image_shape
        # Keep the user-requested DR envelope immutable, then expose only a
        # stage-dependent fraction of it during curriculum training.  A forced
        # vessel (validation/GUI) always uses the complete requested envelope.
        self._curriculum_full_dr = {
            "vessel_scale_min": float(create_scene_kwargs.get("vessel_scale_min", VESSEL_SCALE_MIN)),
            "vessel_scale_max": float(create_scene_kwargs.get("vessel_scale_max", VESSEL_SCALE_MAX)),
            "start_window_distance_m": float(
                create_scene_kwargs.get("start_window_distance_m", START_WINDOW_DISTANCE_M)
            ),
            "target_window_distance_m": float(
                create_scene_kwargs.get("target_window_distance_m", TARGET_WINDOW_DISTANCE_M)
            ),
            "initial_orientation_max_angle_deg": float(
                create_scene_kwargs.get(
                    "initial_orientation_max_angle_deg",
                    INITIAL_ORIENTATION_MAX_ANGLE_DEG,
                )
            ),
        }
        initial_curriculum_enabled = bool(
            create_scene_kwargs.get(
                "training_curriculum_enabled",
                TRAINING_CURRICULUM_ENABLED,
            )
        )
        initial_force_model = str(create_scene_kwargs.get("force_model", "") or "").strip()
        initial_stage = min(
            max(int(create_scene_kwargs.get("training_curriculum_stage", 0)), 0),
            len(TRAINING_CURRICULUM_MODELS) - 1,
        )
        if initial_curriculum_enabled and not initial_force_model:
            fraction = float(TRAINING_CURRICULUM_DR_FRACTIONS[initial_stage])
            create_scene_kwargs["vessel_scale_min"] = 1.0 - fraction * (
                1.0 - self._curriculum_full_dr["vessel_scale_min"]
            )
            create_scene_kwargs["vessel_scale_max"] = 1.0 - fraction * (
                1.0 - self._curriculum_full_dr["vessel_scale_max"]
            )
            create_scene_kwargs["start_window_distance_m"] = fraction * self._curriculum_full_dr[
                "start_window_distance_m"
            ]
            create_scene_kwargs["target_window_distance_m"] = fraction * self._curriculum_full_dr[
                "target_window_distance_m"
            ]
            create_scene_kwargs["initial_orientation_max_angle_deg"] = fraction * self._curriculum_full_dr[
                "initial_orientation_max_angle_deg"
            ]
        self.scene_verbose = bool(create_scene_kwargs.get("verbose_scene", False))
        if reward_amount_dict is None:
            reward_amount_dict = {
                "route_progress": REWARD_ROUTE_PROGRESS,
                "wall_proximity_penalty": REWARD_WALL_PROXIMITY,
                "wall_penetration_penalty": REWARD_WALL_PENETRATION,
                "off_target_branch_penalty": REWARD_OFF_TARGET_BRANCH,
                "retraction_penalty": REWARD_RETRACTION,
                "no_progress_penalty": REWARD_NO_PROGRESS,
                "wrong_branch_penalty": REWARD_WRONG_BRANCH,
                "successful_task": REWARD_SUCCESS,
                "out_of_vessel_penalty": REWARD_OUT_OF_VESSEL,
                "non_finite_penalty": REWARD_NON_FINITE,
                "timeout_penalty": REWARD_TIMEOUT,
                "no_progress_terminal_penalty": REWARD_NO_PROGRESS_TERMINAL,
                "step_penalty": REWARD_STEP,
            }
        elif "route_progress" not in reward_amount_dict:
            # Read old experiment dictionaries without retaining waypoint
            # navigation semantics.
            reward_amount_dict = dict(reward_amount_dict)
            reward_amount_dict["route_progress"] = float(
                reward_amount_dict.pop(
                    "waypoint_approach",
                    reward_amount_dict.pop("target_approach", REWARD_ROUTE_PROGRESS),
                )
            )
            reward_amount_dict.pop("waypoint_reached", None)

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
        self.radius_observation_scale = float(create_scene_kwargs.get("radius_observation_scale", RADIUS_OBSERVATION_SCALE_M))
        self.default_local_radius = float(create_scene_kwargs.get("default_local_radius", 0.005))
        self.catheter_radius = float(create_scene_kwargs.get("catheter_radius", CATHETER_RADIUS_M))
        self.out_of_vessel_safety_ratio = float(create_scene_kwargs.get("out_of_vessel_safety_ratio", OUT_OF_VESSEL_SAFETY_RATIO))
        self.out_of_vessel_fallback_distance = float(create_scene_kwargs.get("out_of_vessel_fallback_distance", OUT_OF_VESSEL_FALLBACK_DISTANCE_M))
        self.local_field_action_angle = float(create_scene_kwargs.get("local_field_action_angle", LOCAL_FIELD_ACTION_ANGLE_RAD))
        self.sdf_clearance_observation_scale = float(
            create_scene_kwargs.get(
                "sdf_clearance_observation_scale",
                SDF_CLEARANCE_OBSERVATION_SCALE_M,
            )
        )
        self.sdf_near_wall_margin = float(
            create_scene_kwargs.get("sdf_near_wall_margin", SDF_NEAR_WALL_MARGIN_M)
        )
        self.sdf_outside_center_tolerance = float(
            create_scene_kwargs.get(
                "sdf_outside_center_tolerance",
                SDF_OUTSIDE_CENTER_TOLERANCE_M,
            )
        )
        self.sdf_body_warning_margin = float(
            create_scene_kwargs.get(
                "sdf_body_warning_margin",
                SDF_BODY_WARNING_MARGIN_M,
            )
        )
        self.sdf_outside_confirm_steps = max(
            1,
            int(
                create_scene_kwargs.get(
                    "sdf_outside_confirm_steps",
                    SDF_OUTSIDE_CONFIRM_STEPS,
                )
            ),
        )
        self.sdf_sample_step_fraction = float(
            create_scene_kwargs.get(
                "sdf_sample_step_fraction",
                SDF_SAMPLE_STEP_FRACTION,
            )
        )
        self.sdf_forward_probe_distances = tuple(
            float(value)
            for value in create_scene_kwargs.get(
                "sdf_forward_probe_distances",
                SDF_FORWARD_PROBE_DISTANCES_M,
            )
        )
        self.centerline_lookahead_distances = tuple(
            float(value)
            for value in create_scene_kwargs.get(
                "centerline_lookahead_distances",
                CENTERLINE_LOOKAHEAD_DISTANCES_M,
            )
        )
        if not (0.0 < self.sdf_sample_step_fraction <= 1.0):
            raise ValueError("sdf_sample_step_fraction must be in (0, 1].")
        if (
            len(self.sdf_forward_probe_distances) != 3
            or any(value <= 0.0 for value in self.sdf_forward_probe_distances)
        ):
            raise ValueError(
                "Exactly three positive sdf_forward_probe_distances are required."
            )
        if (
            len(self.centerline_lookahead_distances) != 3
            or any(value <= 0.0 for value in self.centerline_lookahead_distances)
        ):
            raise ValueError(
                "Exactly three positive centerline_lookahead_distances are required."
            )
        self.tip_near_wall_grace_steps = max(
            0,
            int(
                create_scene_kwargs.get(
                    "tip_near_wall_grace_steps",
                    TIP_NEAR_WALL_GRACE_STEPS,
                )
            ),
        )
        self.tip_near_wall_ramp_steps = max(
            1,
            int(
                create_scene_kwargs.get(
                    "tip_near_wall_ramp_steps",
                    TIP_NEAR_WALL_RAMP_STEPS,
                )
            ),
        )
        self.wrong_branch_distance_margin = float(
            create_scene_kwargs.get(
                "wrong_branch_distance_margin",
                WRONG_BRANCH_DISTANCE_MARGIN_M,
            )
        )
        self.wrong_branch_observation_scale = float(
            create_scene_kwargs.get(
                "wrong_branch_observation_scale",
                WRONG_BRANCH_OBSERVATION_SCALE_M,
            )
        )
        self.wrong_branch_confirm_steps = max(
            1,
            int(
                create_scene_kwargs.get(
                    "wrong_branch_confirm_steps",
                    WRONG_BRANCH_CONFIRM_STEPS,
                )
            ),
        )

        # Insertion safety shield is disabled in this version.
        # The third action component is passed through directly after clipping
        # to [-1, 1]. Near-wall insertion reduction, insertion blocking, and
        # forced retraction after out-of-vessel detection are all removed.

        # Retraction remains available across the full action range.
        self.insert_negative_limit = float(create_scene_kwargs.get("insert_negative_limit", -1.0))

        # Continuous selected-route navigation. Guidance points move with the
        # tracked arc length; projection locality and the physical step gate
        # prevent jumps across spatially adjacent bends or branches.
        self.route_guidance_lookahead_distances = tuple(
            float(value)
            for value in create_scene_kwargs.get(
                "route_guidance_lookahead_distances",
                ROUTE_GUIDANCE_LOOKAHEAD_DISTANCES_M,
            )
        )
        if len(self.route_guidance_lookahead_distances) != 2 or any(
            value <= 0.0 for value in self.route_guidance_lookahead_distances
        ):
            raise ValueError("Exactly two positive route guidance distances are required.")
        self.route_guidance_observation_scale = float(
            create_scene_kwargs.get(
                "route_guidance_observation_scale",
                ROUTE_GUIDANCE_OBSERVATION_SCALE_M,
            )
        )
        self.route_projection_backward_window = float(
            create_scene_kwargs.get(
                "route_projection_backward_window",
                ROUTE_PROJECTION_BACKWARD_WINDOW_M,
            )
        )
        self.route_projection_forward_window = float(
            create_scene_kwargs.get(
                "route_projection_forward_window",
                ROUTE_PROJECTION_FORWARD_WINDOW_M,
            )
        )
        self.route_projection_ambiguity_tolerance = float(
            create_scene_kwargs.get(
                "route_projection_ambiguity_tolerance",
                ROUTE_PROJECTION_AMBIGUITY_TOLERANCE_M,
            )
        )
        self.route_projection_max_progress_step = float(
            create_scene_kwargs.get(
                "route_projection_max_progress_step",
                ROUTE_PROJECTION_MAX_PROGRESS_STEP_M,
            )
        )
        self.route_success_progress_margin = float(
            create_scene_kwargs.get(
                "route_success_progress_margin",
                ROUTE_SUCCESS_PROGRESS_MARGIN_M,
            )
        )
        self.reward_progress_normalization = float(
            create_scene_kwargs.get(
                "reward_progress_normalization",
                REWARD_PROGRESS_NORMALIZATION_M,
            )
        )
        self.no_progress_window_steps = max(
            1,
            int(create_scene_kwargs.get("no_progress_window_steps", NO_PROGRESS_WINDOW_STEPS)),
        )
        self.no_progress_grace_steps = max(
            self.no_progress_window_steps,
            int(create_scene_kwargs.get("no_progress_grace_steps", NO_PROGRESS_GRACE_STEPS)),
        )
        self.no_progress_confirm_steps = max(
            1,
            int(create_scene_kwargs.get("no_progress_confirm_steps", NO_PROGRESS_CONFIRM_STEPS)),
        )
        self.no_progress_min_net_approach = float(
            create_scene_kwargs.get(
                "no_progress_min_net_approach",
                NO_PROGRESS_MIN_NET_APPROACH_M,
            )
        )

        # Actor observation: 50-D current geometry + 4 * 7-D action-response
        # history = 78-D.  The 30 vessel features retain the prior route/tip
        # signals and add the worst shaft point (arc position, relative local
        # position, local inward normal), moving route guidance, and future
        # route tangents. PPO, RecurrentPPO and SAC receive this identical state.
        self.vessel_section_feature_dim = VESSEL_SECTION_FEATURE_DIM
        self.actor_current_geometry_dim = ACTOR_CURRENT_GEOMETRY_DIM
        self.actor_dynamic_step_dim = ACTOR_DYNAMIC_STEP_DIM
        self.actor_history_steps = max(1, int(create_scene_kwargs.get("actor_history_steps", ACTOR_HISTORY_STEPS)))
        self.actor_dynamic_history_dim = self.actor_dynamic_step_dim * self.actor_history_steps
        self.actor_observation_dim = self.actor_current_geometry_dim + self.actor_dynamic_history_dim
        self._actor_dynamic_history = deque(maxlen=self.actor_history_steps)

        # Training-only curriculum.  A forced model (including every validation
        # vessel) bypasses this pool entirely.
        self.training_curriculum_enabled = bool(
            create_scene_kwargs.get(
                "training_curriculum_enabled",
                TRAINING_CURRICULUM_ENABLED,
            )
        )
        requested_stage = int(create_scene_kwargs.get("training_curriculum_stage", 0))
        self.curriculum_stage = min(
            max(requested_stage, 0), len(TRAINING_CURRICULUM_MODELS) - 1
        )
        self.training_models = list(
            TRAINING_CURRICULUM_MODELS[
                self.curriculum_stage if self.training_curriculum_enabled else -1
            ]
        )
        self.curriculum_target_fraction = float(
            TRAINING_CURRICULUM_TARGET_FRACTIONS[self.curriculum_stage]
            if self.training_curriculum_enabled and not initial_force_model
            else 1.0
        )
        uniform_probability = 1.0 / max(len(self.training_models), 1)
        self.training_model_sampling_weights = {
            model_id: uniform_probability for model_id in self.training_models
        }
        if self.training_curriculum_enabled and not initial_force_model:
            self._apply_curriculum_domain_randomization(self.curriculum_stage)
        else:
            self.curriculum_dr_profile = {
                "fraction": 1.0,
                **self._curriculum_full_dr,
            }

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
        self.episode_reward_totals = defaultdict(float)
        self.previous_route_potential = 0.0
        self.current_route_potential = 0.0
        self.current_route_potential_delta = 0.0

        # Centerline and continuous selected-route buffers.
        self.centerline_points = None
        self.centerline_cumlength = None
        self.centerline_radius = None
        self.centerline_graph_points = None
        self.centerline_graph_edges = None
        self.centerline_graph_radius = None
        self.centerline_reversed_for_progress = False
        self.current_route_start_progress = 0.0
        self.current_route_target_progress = 0.0
        self.previous_route_progress = np.nan
        self.current_route_progress = np.nan
        self.current_route_progress_delta = 0.0
        self.current_route_progress_ratio = 0.0
        self.current_route_projection_segment = -1
        self.current_route_projection_distance = np.nan
        self.current_route_projection_jump_rejected = False
        self.route_projection_jump_rejections_episode = 0
        self.current_route_guidance_points = np.zeros((2, 3), dtype=np.float32)
        self._route_projection_cache_step = -1
        self._route_projection_cache_tip = None
        self._route_projection_cache_value = None
        self.current_target_reached_this_step = False

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
        self.current_sdf_tip_signed_distance = np.nan
        self.current_sdf_max_signed_distance = np.nan
        self.current_sdf_body_min_surface_clearance = np.nan
        self.current_sdf_surface_clearance = np.nan
        self.current_sdf_body_warning_feature = 0.0
        self.current_sdf_body_outside_depth_feature = 0.0
        self.current_sdf_near_wall = False
        self.current_sdf_penetrating = False
        self.current_sdf_sample_count = 0
        self.current_sdf_inserted_length = 0.0
        self.current_sdf_worst_point_sim = np.zeros(3, dtype=np.float64)
        self.current_sdf_inward_world = np.zeros(3, dtype=np.float64)
        self.current_sdf_worst_inward_world = np.zeros(3, dtype=np.float64)
        self.current_sdf_worst_arc_fraction = 0.0
        self.current_sdf_forward_clearances = np.full(
            len(self.sdf_forward_probe_distances),
            np.nan,
            dtype=np.float64,
        )
        self._sdf_geometry_cache_step = -1
        self._sdf_geometry_cache_tip = None
        self.sdf_outside_counter = 0
        self.sdf_outside_confirmed = False
        self.min_sdf_surface_clearance_this_episode = np.inf
        self.min_sdf_body_surface_clearance_this_episode = np.inf
        self.max_sdf_signed_distance_this_episode = -np.inf
        self.sdf_wall_contact_steps_episode = 0
        self.sdf_tip_near_wall_counter = 0
        self.sdf_tip_near_wall_counter_max_episode = 0
        self.sdf_tip_near_wall_steps_episode = 0
        self.max_sdf_penetration_depth_this_episode = 0.0
        self.sdf_penetration_integral_this_episode = 0.0
        self.sdf_penetration_this_episode = False
        self.current_graph_distance = np.nan
        self.current_route_graph_distance_gap = 0.0
        self.current_off_target_branch_feature = 0.0
        self.current_wrong_branch = False
        self.wrong_branch_counter = 0
        self.wrong_branch_failure = False
        self.wrong_branch_this_episode = False
        self.sdf_grid = None
        self.vessel_metadata = None
        self.collision_triangle_count = 0
        self.asset_source_to_sim_scale = np.nan
        self.asset_T_env_sim = None
        self.asset_offset_sim = np.zeros(3, dtype=np.float64)

        # Episode state.
        self.episode_success = False
        self.episode_success_2mm = False
        self.episode_safe_success = False
        self.episode_contact_free_success = False
        self.min_dist_this_episode = np.inf
        self.non_finite_failure = False
        self.is_out_of_bounds = False

        # Action diagnostics used by observation/info.
        self.current_raw_insert = 0.0
        self.current_effective_insert = 0.0
        self.insert_action_sum_episode = 0.0
        self.insert_positive_steps_episode = 0
        self.insert_negative_steps_episode = 0
        self.insert_near_zero_steps_episode = 0
        self.max_inserted_length_episode = 0.0

        # Continuous route stagnation diagnostics/termination.
        self.no_progress_counter = 0
        self.no_progress_failure = False
        self.no_progress_net_approach = 0.0
        self.no_progress_feature = 0.0
        self._no_progress_deltas = deque(maxlen=self.no_progress_window_steps)

        # Low-level action rate limiter.
        # Prevents rapid magnetic command reversal in high-curvature sections.
        self.max_action_delta = float(
            create_scene_kwargs.get("max_action_delta", MAX_ACTION_DELTA)
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
        """Sample from the active curriculum pool unless a vessel is forced."""
        if self._explicit_force_model:
            self.create_scene_kwargs["force_model"] = self._explicit_force_model
            self.current_sampling_model = self._explicit_force_model
            return

        seed_value = self._seed_to_int(seed)
        if seed_value is not None:
            self._sampler_rng = np.random.default_rng(seed_value)

        models = list(self.training_models)
        probabilities = np.asarray(
            [self.training_model_sampling_weights.get(model_id, 0.0) for model_id in models],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(probabilities)) or float(np.sum(probabilities)) <= 0.0:
            probabilities = np.full(len(models), 1.0 / len(models), dtype=np.float64)
        else:
            probabilities = probabilities / float(np.sum(probabilities))
        chosen = str(self._sampler_rng.choice(models, p=probabilities))
        self.create_scene_kwargs["force_model"] = chosen
        self.current_sampling_model = chosen

    def _apply_curriculum_domain_randomization(self, stage: int) -> dict:
        """Apply the stage fraction to the original user-requested DR range."""

        stage = min(max(int(stage), 0), len(TRAINING_CURRICULUM_DR_FRACTIONS) - 1)
        fraction = float(TRAINING_CURRICULUM_DR_FRACTIONS[stage])
        full = self._curriculum_full_dr
        profile = {
            "fraction": fraction,
            "vessel_scale_min": 1.0 - fraction * (1.0 - full["vessel_scale_min"]),
            "vessel_scale_max": 1.0 - fraction * (1.0 - full["vessel_scale_max"]),
            "start_window_distance_m": fraction * full["start_window_distance_m"],
            "target_window_distance_m": fraction * full["target_window_distance_m"],
            "initial_orientation_max_angle_deg": fraction * full[
                "initial_orientation_max_angle_deg"
            ],
        }
        self.create_scene_kwargs.update(
            {key: value for key, value in profile.items() if key != "fraction"}
        )
        self.curriculum_dr_profile = profile
        return dict(profile)

    def get_curriculum_domain_randomization_profile(self) -> dict:
        return dict(getattr(self, "curriculum_dr_profile", {}))

    def get_curriculum_target_fraction(self) -> float:
        return float(getattr(self, "curriculum_target_fraction", 1.0))

    def set_training_model_sampling_weights(self, weights: dict) -> dict:
        """Set normalized probabilities for the current vessel pool."""

        models = list(self.training_models)
        values = np.asarray(
            [max(float((weights or {}).get(model_id, 0.0)), 0.0) for model_id in models],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)) or float(np.sum(values)) <= 0.0:
            values = np.ones(len(models), dtype=np.float64)
        values = values / float(np.sum(values))
        self.training_model_sampling_weights = {
            model_id: float(probability)
            for model_id, probability in zip(models, values)
        }
        return dict(self.training_model_sampling_weights)

    def set_curriculum_stage(self, stage: int) -> int:
        """Set the latched training geometry stage for future episode resets."""

        if not self.training_curriculum_enabled or self._explicit_force_model:
            return int(self.curriculum_stage)
        stage = min(max(int(stage), 0), len(TRAINING_CURRICULUM_MODELS) - 1)
        self.curriculum_stage = stage
        self.training_models = list(TRAINING_CURRICULUM_MODELS[stage])
        self.curriculum_target_fraction = float(
            TRAINING_CURRICULUM_TARGET_FRACTIONS[stage]
        )
        self._apply_curriculum_domain_randomization(stage)
        self.set_training_model_sampling_weights({})
        return int(self.curriculum_stage)

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

            if self.scene_verbose:
                print(
                    "[AORTA6_SOFT_MIDPOINT_START]",
                    "start_fraction=", start_fraction,
                    "start_idx=", start_idx,
                    "target_idx=", target_idx,
                    "num_points=", n_pts,
                )

        elif randomize_start_target:
            segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
            cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
            total_length = float(cumulative[-1])
            start_window_distance = max(
                0.0,
                float(
                    self.create_scene_kwargs.get(
                        "start_window_distance_m",
                        START_WINDOW_DISTANCE_M,
                    )
                ),
            )
            target_window_distance = max(
                0.0,
                float(
                    self.create_scene_kwargs.get(
                        "target_window_distance_m",
                        TARGET_WINDOW_DISTANCE_M,
                    )
                ),
            )
            start_candidates = np.flatnonzero(
                cumulative <= start_window_distance + 1e-12
            )
            target_candidates = np.flatnonzero(
                total_length - cumulative <= target_window_distance + 1e-12
            )
            if start_candidates.size == 0:
                start_candidates = np.asarray([0], dtype=np.int64)
            if target_candidates.size == 0:
                target_candidates = np.asarray([n_pts - 1], dtype=np.int64)
            start_idx = int(rng.choice(start_candidates))
            target_idx = int(rng.choice(target_candidates))
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
            step = int(np.clip(self.create_scene_kwargs.get("entry_tangent_points", ENTRY_TANGENT_POINTS), 1, n_pts - 1))
            next_idx = int(np.clip(start_idx + step, 0, n_pts - 1))
            if next_idx == start_idx:
                next_idx = int(np.clip(start_idx + 1, 0, n_pts - 1))
            entry_tangent = self._unit_vector(points[next_idx] - points[start_idx])
            if entry_tangent is not None:
                desired_tangent = entry_tangent
                if randomize_initial_orientation:
                    max_angle_deg = float(
                        self.create_scene_kwargs.get(
                            "initial_orientation_max_angle_deg",
                            INITIAL_ORIENTATION_MAX_ANGLE_DEG,
                        )
                    )
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

        self._configure_continuous_route()

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
        self.episode_safe_success = False
        self.episode_contact_free_success = False
        self.min_dist_this_episode = np.inf
        self.non_finite_failure = False
        self.is_out_of_bounds = False
        self.current_out_of_vessel = False
        self.out_of_vessel_this_episode = False
        self.out_of_vessel_failure = False
        self.max_safety_ratio_this_episode = 0.0
        self.min_safety_margin_this_episode = np.inf
        self.current_sdf_tip_signed_distance = np.nan
        self.current_sdf_max_signed_distance = np.nan
        self.current_sdf_body_min_surface_clearance = np.nan
        self.current_sdf_surface_clearance = np.nan
        self.current_sdf_body_warning_feature = 0.0
        self.current_sdf_body_outside_depth_feature = 0.0
        self.current_sdf_near_wall = False
        self.current_sdf_penetrating = False
        self.current_sdf_sample_count = 0
        self.current_sdf_inserted_length = 0.0
        self.current_sdf_worst_point_sim = np.zeros(3, dtype=np.float64)
        self.current_sdf_inward_world = np.zeros(3, dtype=np.float64)
        self.current_sdf_worst_inward_world = np.zeros(3, dtype=np.float64)
        self.current_sdf_worst_arc_fraction = 0.0
        self.current_sdf_forward_clearances = np.full(
            len(self.sdf_forward_probe_distances),
            np.nan,
            dtype=np.float64,
        )
        self._sdf_geometry_cache_step = -1
        self._sdf_geometry_cache_tip = None
        self.sdf_outside_counter = 0
        self.sdf_outside_confirmed = False
        self.min_sdf_surface_clearance_this_episode = np.inf
        self.min_sdf_body_surface_clearance_this_episode = np.inf
        self.max_sdf_signed_distance_this_episode = -np.inf
        self.sdf_wall_contact_steps_episode = 0
        self.sdf_tip_near_wall_counter = 0
        self.sdf_tip_near_wall_counter_max_episode = 0
        self.sdf_tip_near_wall_steps_episode = 0
        self.max_sdf_penetration_depth_this_episode = 0.0
        self.sdf_penetration_integral_this_episode = 0.0
        self.sdf_penetration_this_episode = False
        self.current_graph_distance = np.nan
        self.current_route_graph_distance_gap = 0.0
        self.current_off_target_branch_feature = 0.0
        self.current_wrong_branch = False
        self.wrong_branch_counter = 0
        self.wrong_branch_failure = False
        self.wrong_branch_this_episode = False
        self.current_centerline_radial_offset = np.nan
        self.current_tip_centerline_offset_norm = np.nan
        self.previous_tip_centerline_radial_offset = None
        self.previous_route_progress = np.nan
        self.current_route_progress = np.nan
        self.current_route_progress_delta = 0.0
        self.current_route_progress_ratio = 0.0
        self.current_route_projection_segment = -1
        self.current_route_projection_distance = np.nan
        self.current_route_projection_jump_rejected = False
        self.route_projection_jump_rejections_episode = 0
        self.current_route_guidance_points = np.zeros((2, 3), dtype=np.float32)
        self._route_projection_cache_step = -1
        self._route_projection_cache_tip = None
        self._route_projection_cache_value = None
        self.current_target_reached_this_step = False

        self.current_raw_insert = 0.0
        self.current_effective_insert = 0.0
        self.insert_action_sum_episode = 0.0
        self.insert_positive_steps_episode = 0
        self.insert_negative_steps_episode = 0
        self.insert_near_zero_steps_episode = 0
        self.max_inserted_length_episode = 0.0
        self.no_progress_counter = 0
        self.no_progress_failure = False
        self.no_progress_net_approach = 0.0
        self.no_progress_feature = 0.0
        self._no_progress_deltas = deque(maxlen=self.no_progress_window_steps)
        self._last_smoothed_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self._prev_smoothed_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self._last_actor_tip_pos = None
        self._reset_actor_history()
        self.reward_info = {}
        self.reward_features = {}
        self.episode_reward_totals = defaultdict(float)
        self.previous_route_potential = 0.0
        self.current_route_potential = 0.0
        self.current_route_potential_delta = 0.0

        self.mcr_controller_sofa.reset()
        if bool(single_vessel_mode and getattr(self, "soft_randomize_single_vessel", True)):
            self._soft_randomize_single_vessel_scene(seed=seed)

        self.sofa_simulation.animate(self._sofa_root_node, 0.05)

        if self.scene_verbose:
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

        self._initialize_continuous_route_from_current_tip()
        self.current_route_potential = self._continuous_route_potential()
        self.previous_route_potential = self.current_route_potential
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
            non_finite_penalty = float(self.reward_amount_dict["non_finite_penalty"])
            if np.isfinite(non_finite_penalty):
                reward += non_finite_penalty
                self.reward_features["non_finite_penalty"] = 1.0
                self.reward_info["non_finite_penalty"] = 1.0
                self.reward_info["reward_non_finite_penalty"] = non_finite_penalty
                self.reward_info["reward"] = float(reward)
                self.episode_reward_totals["non_finite_penalty"] += non_finite_penalty

        terminated = bool(
            self.episode_success
            or self.out_of_vessel_failure
            or self.wrong_branch_failure
            or self.no_progress_failure
            or self.non_finite_failure
        )
        truncated = (self._elapsed_steps >= self.max_episode_steps) and (not terminated)

        if truncated:
            timeout_penalty = float(self.reward_amount_dict["timeout_penalty"])
            if np.isfinite(timeout_penalty):
                reward += timeout_penalty
                self.reward_features["timeout_penalty"] = 1.0
                self.reward_info["timeout_penalty"] = 1.0
                self.reward_info["reward_timeout_penalty"] = timeout_penalty
                self.reward_info["reward"] = float(reward)
                self.episode_reward_totals["timeout_penalty"] += timeout_penalty

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
        near_guidance_vector_local: np.ndarray,
        far_guidance_vector_local: np.ndarray,
        centerline_correction_vec_local: np.ndarray,
        centerline_tangent_local: np.ndarray,
        guidance_distance_norm: np.ndarray,
        route_progress_ratio: float,
        vessel_section_features: np.ndarray,
    ) -> np.ndarray:
        obs = np.concatenate(
            [
                np.asarray(tip_forward_local, dtype=np.float32).reshape(3),
                np.asarray(magnetic_field_norm, dtype=np.float32).reshape(3),
                np.asarray(near_guidance_vector_local, dtype=np.float32).reshape(3),
                np.asarray(far_guidance_vector_local, dtype=np.float32).reshape(3),
                np.asarray(centerline_correction_vec_local, dtype=np.float32).reshape(3),
                np.asarray(centerline_tangent_local, dtype=np.float32).reshape(3),
                np.asarray(guidance_distance_norm, dtype=np.float32).reshape(1),
                np.array([np.clip(float(route_progress_ratio), 0.0, 1.0)], dtype=np.float32),
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

        # Refresh selected-route geometry plus direct SDF/graph safety features.
        self._update_vessel_safety_state(tip_pos)

        tip_forward_world = self._quat_rotate_vector(tip_quat, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        tip_forward_local = self._world_vec_to_local(tip_forward_world, frame)
        tip_forward_local = tip_forward_local / (float(np.linalg.norm(tip_forward_local)) + 1e-9)

        last_tip_pos = getattr(self, "_last_actor_tip_pos", None)
        tip_delta_world = np.zeros(3, dtype=np.float32) if last_tip_pos is None else np.asarray(tip_pos, dtype=np.float32) - np.asarray(last_tip_pos, dtype=np.float32)
        tip_delta_local = np.clip(self._world_vec_to_local(tip_delta_world, frame) / 0.001, -5.0, 5.0).astype(np.float32)

        mag_field = np.asarray(self.mcr_controller_sofa.get_mag_field_des(), dtype=np.float32)
        magnetic_field_norm = np.clip(self._world_vec_to_local(mag_field, frame) / max(self.magnetic_field_observation_scale, 1e-9), -2.0, 2.0).astype(np.float32)

        guidance_scale = max(float(self.route_guidance_observation_scale), 1e-9)
        route_progress = float(getattr(self, "current_route_progress", np.nan))
        target_progress = float(getattr(self, "current_route_target_progress", np.nan))
        if np.isfinite(route_progress) and np.isfinite(target_progress):
            guidance_points = np.asarray(
                [
                    self._interpolate_centerline_point_at_progress(
                        min(route_progress + distance, target_progress)
                    )
                    for distance in self.route_guidance_lookahead_distances
                ],
                dtype=np.float32,
            )
        else:
            guidance_points = np.repeat(
                np.asarray(self.target_position, dtype=np.float32).reshape(1, 3),
                2,
                axis=0,
            )
        self.current_route_guidance_points = guidance_points.copy()
        near_guidance_world = guidance_points[0] - tip_pos
        far_guidance_world = guidance_points[1] - tip_pos
        guidance_distance = float(np.linalg.norm(near_guidance_world))
        near_guidance_vector_local = np.clip(
            self._world_vec_to_local(near_guidance_world, frame) / guidance_scale,
            -5.0,
            5.0,
        ).astype(np.float32)
        far_guidance_vector_local = np.clip(
            self._world_vec_to_local(far_guidance_world, frame) / guidance_scale,
            -5.0,
            5.0,
        ).astype(np.float32)
        guidance_distance_norm = np.array(
            [np.clip(guidance_distance / guidance_scale, 0.0, 10.0)],
            dtype=np.float32,
        )

        centerline_proj = np.asarray(getattr(self, "current_centerline_projection", tip_pos), dtype=np.float32).reshape(3)
        centerline_tangent_world = np.asarray(getattr(self, "current_centerline_tangent", np.array([1.0, 0.0, 0.0], dtype=np.float32)), dtype=np.float32).reshape(3)
        centerline_tangent_world = centerline_tangent_world / (float(np.linalg.norm(centerline_tangent_world)) + 1e-9)
        centerline_correction_world = centerline_proj - tip_pos
        centerline_correction_vec_local = np.clip(
            self._world_vec_to_local(centerline_correction_world, frame)
            / guidance_scale,
            -5.0,
            5.0,
        ).astype(np.float32)
        centerline_tangent_local = self._world_vec_to_local(centerline_tangent_world, frame)
        centerline_tangent_local = (centerline_tangent_local / (float(np.linalg.norm(centerline_tangent_local)) + 1e-9)).astype(np.float32)

        prev_action = np.asarray(getattr(self, "_last_smoothed_action", np.zeros(3, dtype=np.float32)), dtype=np.float32).reshape(3).copy()
        prev_action[2] = float(getattr(self, "current_effective_insert", prev_action[2]))

        vessel_section_features = self._get_vessel_section_features(
            tip_pos=tip_pos,
            tip_frame=frame,
            tip_forward_world=tip_forward_world,
        )
        actor_current_geometry = self._build_actor_current_geometry_observation(
            tip_forward_local=tip_forward_local,
            magnetic_field_norm=magnetic_field_norm,
            near_guidance_vector_local=near_guidance_vector_local,
            far_guidance_vector_local=far_guidance_vector_local,
            centerline_correction_vec_local=centerline_correction_vec_local,
            centerline_tangent_local=centerline_tangent_local,
            guidance_distance_norm=guidance_distance_norm,
            route_progress_ratio=float(self.current_route_progress_ratio),
            vessel_section_features=vessel_section_features,
        )
        actor_dynamic_step = self._build_actor_dynamic_step_observation(
            prev_action=prev_action,
            tip_delta_local=tip_delta_local,
            progress_delta_norm=np.array([np.clip(float(self.current_route_progress_delta) / 0.001, -5.0, 5.0)], dtype=np.float32),
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

        self._update_vessel_safety_state(
            tip_pos,
            advance_failure_counters=True,
        )
        sdf_center_outside = bool(
            self.sdf_grid is not None
            and np.isfinite(self.current_sdf_max_signed_distance)
            and self.current_sdf_max_signed_distance
            > float(self.sdf_outside_center_tolerance)
        )
        valid_inside_vessel = not bool(
            self.current_out_of_vessel
            or self.current_wrong_branch
            or sdf_center_outside
        )
        # Net continuous arc-length progress detects stagnation without a
        # waypoint-switch discontinuity. Backward corrections remain negative,
        # so oscillation cannot look productive.
        approach_delta = float(self.current_route_progress_delta)
        self._no_progress_deltas.append(approach_delta)
        self.no_progress_net_approach = float(sum(self._no_progress_deltas))
        eligible = bool(
            self._elapsed_steps >= self.no_progress_grace_steps
            and len(self._no_progress_deltas) >= self.no_progress_window_steps
        )
        if eligible:
            threshold = max(float(self.no_progress_min_net_approach), 1e-9)
            self.no_progress_feature = float(
                np.clip(
                    (threshold - self.no_progress_net_approach) / threshold,
                    0.0,
                    1.0,
                )
            )
            if self.no_progress_net_approach <= 0.1 * threshold:
                self.no_progress_counter += 1
            else:
                self.no_progress_counter = 0
        else:
            self.no_progress_feature = 0.0
            self.no_progress_counter = 0
        self.no_progress_failure = bool(
            self.no_progress_counter >= self.no_progress_confirm_steps
        )

        try:
            inserted_length = float(self.mcr_controller_sofa._getXTipValue())
        except Exception:
            inserted_length = float(getattr(self, "current_sdf_inserted_length", 0.0))
        if np.isfinite(inserted_length):
            self.max_inserted_length_episode = max(
                float(self.max_inserted_length_episode),
                max(0.0, inserted_length),
            )

        # Continuous route completion is the single navigation potential.
        # Out-of-vessel terminal penalty remains separate; timeout is added in
        # step() after truncated is known. Centreline geometry is retained for
        # selected-route navigation. Wall risk and
        # out-of-vessel termination come from the VTI SDF whenever it is present.
        # Potential-difference shaping telescopes over the whole trajectory.
        # Corrections restore their earlier negative credit when the tip moves
        # forward again, while oscillation has zero net progress reward.
        if valid_inside_vessel:
            self.current_route_potential = self._continuous_route_potential()
            self.current_route_potential_delta = float(
                self.current_route_potential - self.previous_route_potential
            )
            self.previous_route_potential = self.current_route_potential
        else:
            self.current_route_potential_delta = 0.0
        approach_feature = self.current_route_potential_delta

        if self.sdf_grid is not None and np.isfinite(self.current_sdf_surface_clearance):
            # Tip clearance keeps the original contact shaping.  Whole-body
            # samples add a bounded pre-terminal warning, while ordinary shaft
            # contact remains legal and can still slide along the wall.
            tip_clearance = float(self.current_sdf_surface_clearance)
            base_near_wall_feature = float(
                np.clip(
                    (float(self.sdf_near_wall_margin) - max(tip_clearance, 0.0))
                    / max(float(self.sdf_near_wall_margin), 1e-9),
                    0.0,
                    1.0,
                )
            )
            persistence = max(
                0,
                int(self.sdf_tip_near_wall_counter)
                - int(self.tip_near_wall_grace_steps),
            )
            persistence_scale = float(
                np.clip(
                    persistence / float(self.tip_near_wall_ramp_steps),
                    0.0,
                    1.0,
                )
            )
            tip_near_wall_feature = base_near_wall_feature * persistence_scale
            tip_penetration_feature = float(
                np.clip(
                    max(-tip_clearance, 0.0)
                    / max(float(self.catheter_radius), 1e-9),
                    0.0,
                    1.0,
                )
            )
            # The same whole-body SDF sample that drives termination now gives
            # dense warning before the three-step terminal confirmation.  Shaft
            # contact is still legal: the stronger feature starts only when a
            # sampled centre crosses the wall.
            near_wall_feature = max(
                tip_near_wall_feature,
                float(self.current_sdf_body_warning_feature),
            )
            penetration_feature = max(
                tip_penetration_feature,
                float(self.current_sdf_body_outside_depth_feature),
            )
        else:
            # Legacy vessels without VTI keep navigation rewards but do not
            # invent a dense SDF wall term from the selected route centreline.
            near_wall_feature = 0.0
            penetration_feature = 0.0

        reward_features = {
            "route_progress": approach_feature,
            "wall_proximity_penalty": near_wall_feature,
            "wall_penetration_penalty": penetration_feature,
            "off_target_branch_penalty": float(self.current_off_target_branch_feature),
            "retraction_penalty": float(max(-self.current_effective_insert, 0.0)),
            "no_progress_penalty": float(self.no_progress_feature),
            "wrong_branch_penalty": 1.0 if self.current_wrong_branch else 0.0,
            "out_of_vessel_penalty": 1.0 if self.current_out_of_vessel else 0.0,
            "non_finite_penalty": 0.0,
            "timeout_penalty": 0.0,
            "no_progress_terminal_penalty": 1.0 if self.no_progress_failure else 0.0,
            "step_penalty": 1.0,
            "successful_task": 0.0,
        }

        if self.current_out_of_vessel:
            self.out_of_vessel_this_episode = True
            self.out_of_vessel_failure = True
        if self.current_wrong_branch:
            self.wrong_branch_this_episode = True
            self.wrong_branch_failure = True

        final_close = bool(current_final_dist <= float(self.target_distance_threshold))
        route_ready = bool(
            np.isfinite(self.current_route_progress)
            and self.current_route_progress
            >= self.current_route_target_progress
            - float(self.route_success_progress_margin)
        )
        self.current_target_reached_this_step = bool(
            valid_inside_vessel and route_ready and final_close
        )
        if self.current_target_reached_this_step:
            reward_features["successful_task"] = 1.0
            self.episode_success = True
            self.episode_success_2mm = bool(current_final_dist <= 0.002 + 1e-12)
            current_clearance_safe = bool(
                self.sdf_grid is None
                or (
                    np.isfinite(self.current_sdf_surface_clearance)
                    and self.current_sdf_surface_clearance >= 0.0
                )
            )
            self.episode_safe_success = current_clearance_safe
            self.episode_contact_free_success = bool(
                current_clearance_safe
                and not self.sdf_penetration_this_episode
            )
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
            self.episode_reward_totals[key] += value
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
            or getattr(self, "wrong_branch_failure", False)
            or getattr(self, "no_progress_failure", False)
            or getattr(self, "non_finite_failure", False)
        )

    def _get_info(self, terminated: bool = False, truncated: bool = False) -> dict:
        current_dist = float(self._get_distance_tip_to_dest())
        terminal_reason = (
            "target"
            if self.episode_success
            else "timeout"
            if truncated
            else "out_of_vessel"
            if self.out_of_vessel_failure
            else "wrong_branch"
            if self.wrong_branch_failure
            else "no_progress"
            if self.no_progress_failure
            else "non_finite"
            if self.non_finite_failure
            else "other"
            if terminated
            else "not_done"
        )
        chosen_model = str(getattr(self, "chosen_model", "unknown"))
        info = {
            "task_id": str(getattr(self, "task_id", "unknown")),
            "chosen_model": chosen_model,
            "sampling_model": str(getattr(self, "current_sampling_model", chosen_model)),
            "sampling_probability": float(
                getattr(self, "training_model_sampling_weights", {}).get(
                    str(getattr(self, "current_sampling_model", chosen_model)),
                    1.0,
                )
            ),
            "vessel_family": str(getattr(self, "asset_model_family", "unknown")),
            "vessel_difficulty": str(getattr(self, "asset_difficulty", "unknown")),
            "collision_triangle_count": int(
                getattr(self, "collision_triangle_count", 0)
            ),
            "centerline_vtk": str(getattr(self, "centerline_vtk", "unknown")),
            "success_2mm": bool(self.episode_success_2mm),
            "safe_success": bool(self.episode_safe_success),
            "contact_free_success": bool(self.episode_contact_free_success),
            "done_by_target": bool(self.episode_success),
            "done_by_timeout": bool(truncated),
            "done_by_out_of_vessel": bool(self.out_of_vessel_failure),
            "done_by_wrong_branch": bool(self.wrong_branch_failure),
            "done_by_no_progress": bool(self.no_progress_failure),
            "done_by_non_finite": bool(self.non_finite_failure),
            "terminal_reason": terminal_reason,
            "min_dist_to_goal": float(self.min_dist_this_episode),
            "current_dist_to_goal": current_dist,
            "final_dist_to_goal": current_dist if (terminated or truncated) else np.nan,
            "target_distance_threshold": float(self.target_distance_threshold),
            "vessel_scale_factor": float(getattr(self, "vessel_scale_factor", 1.0)),
            "route_progress": float(self.current_route_progress),
            "route_start_progress": float(self.current_route_start_progress),
            "route_target_progress": float(self.current_route_target_progress),
            "route_progress_delta": float(self.current_route_progress_delta),
            "route_progress_ratio": float(self.current_route_progress_ratio),
            "route_projection_segment": int(self.current_route_projection_segment),
            "route_projection_distance": float(self.current_route_projection_distance),
            "route_projection_jump_rejected": bool(
                self.current_route_projection_jump_rejected
            ),
            "route_projection_jump_rejections_episode": int(
                self.route_projection_jump_rejections_episode
            ),
            "route_guidance_points": np.asarray(
                self.current_route_guidance_points,
                dtype=np.float64,
            ).reshape((2, 3)).tolist(),
            "reward_progress_normalization": float(self.reward_progress_normalization),
            "reward_route_length": float(self.reward_progress_normalization),
            "route_potential": float(self.current_route_potential),
            "route_potential_delta": float(self.current_route_potential_delta),
            "curriculum_stage": int(self.curriculum_stage),
            "curriculum_target_fraction": float(
                getattr(self, "curriculum_target_fraction", 1.0)
            ),
            "curriculum_dr_fraction": float(
                getattr(self, "curriculum_dr_profile", {}).get("fraction", 1.0)
            ),
            "target_reached_this_step": bool(getattr(self, "current_target_reached_this_step", False)),
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
            "sdf_available": bool(self.sdf_grid is not None),
            "sdf_tip_signed_distance": float(self.current_sdf_tip_signed_distance)
            if np.isfinite(self.current_sdf_tip_signed_distance)
            else np.nan,
            "sdf_max_signed_distance": float(self.current_sdf_max_signed_distance)
            if np.isfinite(self.current_sdf_max_signed_distance)
            else np.nan,
            "sdf_surface_clearance": float(self.current_sdf_surface_clearance)
            if np.isfinite(self.current_sdf_surface_clearance)
            else np.nan,
            "sdf_tip_surface_clearance": float(
                self.current_sdf_surface_clearance
            )
            if np.isfinite(self.current_sdf_surface_clearance)
            else np.nan,
            "sdf_body_min_surface_clearance": float(
                self.current_sdf_body_min_surface_clearance
            )
            if np.isfinite(self.current_sdf_body_min_surface_clearance)
            else np.nan,
            "sdf_body_warning_feature": float(
                self.current_sdf_body_warning_feature
            ),
            "sdf_body_outside_depth_feature": float(
                self.current_sdf_body_outside_depth_feature
            ),
            "sdf_surface_clearance_min_episode": float(
                self.min_sdf_surface_clearance_this_episode
            )
            if np.isfinite(self.min_sdf_surface_clearance_this_episode)
            else np.nan,
            "sdf_body_surface_clearance_min_episode": float(
                self.min_sdf_body_surface_clearance_this_episode
            )
            if np.isfinite(self.min_sdf_body_surface_clearance_this_episode)
            else np.nan,
            "sdf_signed_distance_max_episode": float(
                self.max_sdf_signed_distance_this_episode
            )
            if np.isfinite(self.max_sdf_signed_distance_this_episode)
            else np.nan,
            "sdf_near_wall": bool(self.current_sdf_near_wall),
            "sdf_penetrating": bool(self.current_sdf_penetrating),
            "sdf_center_outside_candidate": bool(
                self.sdf_grid is not None
                and np.isfinite(self.current_sdf_max_signed_distance)
                and self.current_sdf_max_signed_distance
                > float(self.sdf_outside_center_tolerance)
            ),
            "sdf_sample_count": int(self.current_sdf_sample_count),
            "sdf_inserted_length": float(self.current_sdf_inserted_length),
            "sdf_worst_point_sim": np.asarray(
                self.current_sdf_worst_point_sim,
                dtype=np.float64,
            ).reshape(3).tolist(),
            "sdf_worst_arc_fraction": float(self.current_sdf_worst_arc_fraction),
            "sdf_inward_world": np.asarray(
                self.current_sdf_inward_world,
                dtype=np.float64,
            ).reshape(3).tolist(),
            "sdf_worst_inward_world": np.asarray(
                self.current_sdf_worst_inward_world,
                dtype=np.float64,
            ).reshape(3).tolist(),
            "sdf_forward_probe_distances": list(
                self.sdf_forward_probe_distances
            ),
            "sdf_forward_clearances": np.asarray(
                self.current_sdf_forward_clearances,
                dtype=np.float64,
            ).reshape(-1).tolist(),
            "sdf_outside_counter": int(self.sdf_outside_counter),
            "sdf_outside_confirm_steps": int(self.sdf_outside_confirm_steps),
            "sdf_wall_contact_steps_episode": int(
                self.sdf_wall_contact_steps_episode
            ),
            "sdf_tip_penetration_steps_episode": int(
                self.sdf_wall_contact_steps_episode
            ),
            "sdf_tip_near_wall_counter": int(
                self.sdf_tip_near_wall_counter
            ),
            "sdf_tip_near_wall_counter_max_episode": int(
                self.sdf_tip_near_wall_counter_max_episode
            ),
            "sdf_tip_near_wall_steps_episode": int(
                self.sdf_tip_near_wall_steps_episode
            ),
            "sdf_tip_near_wall_grace_steps": int(
                self.tip_near_wall_grace_steps
            ),
            "sdf_penetration_this_episode": bool(
                self.sdf_penetration_this_episode
            ),
            "sdf_penetration_depth_max_episode": float(
                self.max_sdf_penetration_depth_this_episode
            ),
            "sdf_penetration_integral_episode": float(
                self.sdf_penetration_integral_this_episode
            ),
            "centerline_graph_available": bool(
                self.centerline_graph_points is not None
            ),
            "graph_distance": float(self.current_graph_distance),
            "route_graph_distance_gap": float(
                self.current_route_graph_distance_gap
            ),
            "off_target_branch_feature": float(
                self.current_off_target_branch_feature
            ),
            "wrong_branch": bool(self.current_wrong_branch),
            "wrong_branch_this_episode": bool(self.wrong_branch_this_episode),
            "wrong_branch_counter": int(self.wrong_branch_counter),
            "wrong_branch_confirm_steps": int(self.wrong_branch_confirm_steps),
            "raw_insert": float(self.current_raw_insert),
            "effective_insert": float(self.current_effective_insert),
            "insert_action_mean_episode": float(
                self.insert_action_sum_episode / max(1, self._elapsed_steps)
            ),
            "insert_positive_fraction_episode": float(
                self.insert_positive_steps_episode / max(1, self._elapsed_steps)
            ),
            "insert_negative_fraction_episode": float(
                self.insert_negative_steps_episode / max(1, self._elapsed_steps)
            ),
            "insert_near_zero_fraction_episode": float(
                self.insert_near_zero_steps_episode / max(1, self._elapsed_steps)
            ),
            "inserted_length_final": float(
                max(0.0, self.mcr_controller_sofa._getXTipValue())
            ),
            "inserted_length_max_episode": float(self.max_inserted_length_episode),
            "no_progress_counter": int(self.no_progress_counter),
            "no_progress_failure": bool(self.no_progress_failure),
            "no_progress_net_approach": float(self.no_progress_net_approach),
            "no_progress_feature": float(self.no_progress_feature),
            "no_progress_window_steps": int(self.no_progress_window_steps),
            "no_progress_confirm_steps": int(self.no_progress_confirm_steps),
            "insert_negative_limit": float(getattr(self, "insert_negative_limit", -1.0)),
            "actor_obs_dim": int(self.actor_observation_dim),
        }
        episode_reward_totals = {
            f"episode_reward_{key}": float(value)
            for key, value in self.episode_reward_totals.items()
        }
        episode_reward_total_components = float(
            sum(float(value) for value in self.episode_reward_totals.values())
        )
        failed_episode = bool((terminated or truncated) and not self.episode_success)
        info["episode_reward_total_components"] = episode_reward_total_components
        info["positive_failure_return"] = bool(
            failed_episode and episode_reward_total_components > 1e-6
        )
        return {
            **info,
            **self.reward_info,
            **self.reward_features,
            **episode_reward_totals,
        }

    # ------------------------------------------------------------------
    # Continuous selected route / centerline / safety
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

    def _get_current_local_radius(self, centerline_progress: float):
        if (
            self.centerline_radius is None
            or self.centerline_points is None
            or self.centerline_cumlength is None
        ):
            return None
        radius = np.asarray(self.centerline_radius, dtype=np.float32).reshape(-1)
        cumulative = np.asarray(self.centerline_cumlength, dtype=np.float32).reshape(-1)
        if len(radius) == 0 or len(radius) != len(cumulative):
            return None
        local_radius = float(
            np.interp(
                float(centerline_progress),
                cumulative,
                radius,
            )
        )
        return local_radius if np.isfinite(local_radius) and local_radius > 1e-6 else None

    def _get_graph_projection_distance(self, point: np.ndarray) -> float:
        points = getattr(self, "centerline_graph_points", None)
        edges = getattr(self, "centerline_graph_edges", None)
        if points is None or edges is None:
            return np.nan
        points = np.asarray(points, dtype=np.float32)
        edges = np.asarray(edges, dtype=np.int64)
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or edges.ndim != 2
            or edges.shape[1] != 2
            or len(edges) == 0
        ):
            return np.nan
        valid = np.all((edges >= 0) & (edges < len(points)), axis=1)
        if not np.any(valid):
            return np.nan
        edges = edges[valid]
        start = points[edges[:, 0]]
        end = points[edges[:, 1]]
        vec = end - start
        length_sq = np.maximum(np.sum(vec * vec, axis=1), 1e-12)
        point = np.asarray(point, dtype=np.float32).reshape(3)
        t = np.clip(np.sum((point[None, :] - start) * vec, axis=1) / length_sq, 0.0, 1.0)
        projection = start + t[:, None] * vec
        return float(np.min(np.linalg.norm(projection - point[None, :], axis=1)))

    def _get_sdf_sample_points(self, tip_pos: np.ndarray) -> np.ndarray:
        """Return dense samples along the physically inserted catheter.

        ``xtip`` is the deployed length measured from the distal node.  Walking
        backwards through the ordered mechanical nodes and cutting the final
        segment at that length excludes the catheter tail outside the inlet.
        Every included Line primitive is then subdivided relative to the VTI
        cell size, so a wall crossing between SOFA nodes cannot be missed.
        """

        try:
            positions = np.asarray(
                self.mcr_controller_sofa.instrument.MO.position.array(),
                dtype=np.float64,
            )
            if positions.ndim == 2 and positions.shape[1] >= 3 and len(positions) > 0:
                positions = positions[:, :3]
                inserted_length = max(
                    0.0,
                    float(self.mcr_controller_sofa._getXTipValue()),
                )
                self.current_sdf_inserted_length = inserted_length

                # The distal tip is the last mechanical node.  Build a distal
                # to proximal polyline whose arclength is exactly xtip.
                reversed_nodes = positions[::-1]
                inserted_nodes = [reversed_nodes[0].copy()]
                remaining = inserted_length
                for distal, proximal in zip(
                    reversed_nodes[:-1],
                    reversed_nodes[1:],
                ):
                    if remaining <= 1e-12:
                        break
                    segment = proximal - distal
                    segment_length = float(np.linalg.norm(segment))
                    if segment_length <= 1e-12:
                        continue
                    if remaining >= segment_length:
                        inserted_nodes.append(proximal.copy())
                        remaining -= segment_length
                    else:
                        inserted_nodes.append(
                            distal + segment * (remaining / segment_length)
                        )
                        remaining = 0.0
                        break

                inserted_nodes = np.asarray(inserted_nodes, dtype=np.float64)
                if len(inserted_nodes) < 2 or self.sdf_grid is None:
                    return inserted_nodes
                sdf_cell_sim = float(
                    np.min(self.sdf_grid.spacing)
                    * float(self.asset_source_to_sim_scale)
                )
                max_sample_step = max(
                    1e-6,
                    float(self.sdf_sample_step_fraction) * sdf_cell_sim,
                )
                dense_points = [inserted_nodes[0]]
                for start, end in zip(
                    inserted_nodes[:-1],
                    inserted_nodes[1:],
                ):
                    segment_length = float(np.linalg.norm(end - start))
                    subdivisions = max(
                        1,
                        int(np.ceil(segment_length / max_sample_step)),
                    )
                    for index in range(1, subdivisions + 1):
                        dense_points.append(
                            start
                            + (end - start)
                            * (float(index) / float(subdivisions))
                        )
                return np.asarray(dense_points, dtype=np.float64)
        except Exception:
            pass
        self.current_sdf_inserted_length = 0.0
        return np.asarray(tip_pos, dtype=np.float64).reshape((1, 3))

    def _sim_points_to_asset_source(self, points_sim: np.ndarray) -> np.ndarray:
        points = np.asarray(points_sim, dtype=np.float64).reshape((-1, 3))
        transform = np.asarray(self.asset_T_env_sim, dtype=np.float64).reshape(7)
        translation = transform[:3] + np.asarray(self.asset_offset_sim, dtype=np.float64).reshape(3)
        scale = float(self.asset_source_to_sim_scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"Invalid asset_source_to_sim_scale={scale}")
        rotation = R.from_quat(transform[3:7])
        return rotation.inv().apply(points - translation[None, :]) / scale

    def _asset_vectors_to_sim(self, vectors_source: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors_source, dtype=np.float64).reshape((-1, 3))
        transform = np.asarray(self.asset_T_env_sim, dtype=np.float64).reshape(7)
        return R.from_quat(transform[3:7]).apply(vectors)

    def _get_sdf_forward_probe_features(
        self,
        tip_pos: np.ndarray,
        tip_forward_world: np.ndarray,
    ) -> np.ndarray:
        probe_count = len(self.sdf_forward_probe_distances)
        self.current_sdf_forward_clearances = np.full(
            probe_count,
            np.nan,
            dtype=np.float64,
        )
        if self.sdf_grid is None or probe_count == 0:
            return np.zeros(probe_count, dtype=np.float32)

        forward = np.asarray(tip_forward_world, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(forward))
        if not np.isfinite(norm) or norm <= 1e-12:
            return np.zeros(probe_count, dtype=np.float32)
        forward /= norm
        distances = np.asarray(
            self.sdf_forward_probe_distances,
            dtype=np.float64,
        )
        probe_points_sim = (
            np.asarray(tip_pos, dtype=np.float64).reshape(1, 3)
            + distances[:, None] * forward[None, :]
        )
        probe_points_source = self._sim_points_to_asset_source(probe_points_sim)
        signed_source = np.asarray(
            self.sdf_grid.sample(probe_points_source),
            dtype=np.float64,
        ).reshape(-1)
        signed_sim = signed_source * float(self.asset_source_to_sim_scale)
        outside_sentinel = max(
            0.05,
            2.0 * float(self.sdf_outside_center_tolerance),
        )
        signed_sim = np.where(
            np.isfinite(signed_sim),
            signed_sim,
            outside_sentinel,
        )
        clearances = -signed_sim - float(self.catheter_radius)
        self.current_sdf_forward_clearances = clearances.astype(np.float64)
        return np.clip(
            clearances
            / max(float(self.sdf_clearance_observation_scale), 1e-9),
            -2.0,
            2.0,
        ).astype(np.float32)

    def _update_sdf_safety_state(
        self,
        tip_pos: np.ndarray,
        advance_failure_counters: bool,
    ) -> None:
        tip_pos = np.asarray(tip_pos, dtype=np.float64).reshape(3)
        cached_tip = getattr(self, "_sdf_geometry_cache_tip", None)
        cache_matches = bool(
            not advance_failure_counters
            and int(getattr(self, "_sdf_geometry_cache_step", -1))
            == int(getattr(self, "_elapsed_steps", -2))
            and cached_tip is not None
            and np.array_equal(
                np.asarray(cached_tip, dtype=np.float64).reshape(3),
                tip_pos,
            )
        )
        if cache_matches:
            return

        grid = getattr(self, "sdf_grid", None)
        if grid is None:
            self.current_sdf_tip_signed_distance = np.nan
            self.current_sdf_max_signed_distance = np.nan
            self.current_sdf_body_min_surface_clearance = np.nan
            self.current_sdf_surface_clearance = np.nan
            self.current_sdf_body_warning_feature = 0.0
            self.current_sdf_body_outside_depth_feature = 0.0
            self.current_sdf_near_wall = False
            self.current_sdf_penetrating = False
            self.current_sdf_sample_count = 0
            self.current_sdf_inserted_length = 0.0
            self.current_sdf_worst_point_sim = np.zeros(3, dtype=np.float64)
            self.current_sdf_inward_world = np.zeros(3, dtype=np.float64)
            self.current_sdf_worst_inward_world = np.zeros(3, dtype=np.float64)
            self.current_sdf_worst_arc_fraction = 0.0
            self._sdf_geometry_cache_step = int(
                getattr(self, "_elapsed_steps", -1)
            )
            self._sdf_geometry_cache_tip = tip_pos.copy()
            return

        sample_points = self._get_sdf_sample_points(tip_pos)
        source_points = self._sim_points_to_asset_source(sample_points)
        signed_source = np.asarray(grid.sample(source_points), dtype=np.float64).reshape(-1)
        tip_source = self._sim_points_to_asset_source(
            np.asarray(tip_pos, dtype=np.float64).reshape((1, 3))
        )
        tip_signed_source = float(grid.sample(tip_source)[0])
        scale = float(self.asset_source_to_sim_scale)
        signed_sim = signed_source * scale
        tip_signed_sim = tip_signed_source * scale

        # The grid has a source-space margin.  Leaving its extent is therefore
        # unambiguously outside; replace +inf with a finite diagnostic sentinel.
        outside_sentinel = max(0.05, 2.0 * float(self.sdf_outside_center_tolerance))
        signed_sim = np.where(np.isfinite(signed_sim), signed_sim, outside_sentinel)
        if not np.isfinite(tip_signed_sim):
            tip_signed_sim = outside_sentinel

        worst_index = int(np.argmax(signed_sim))
        max_signed = float(signed_sim[worst_index])
        body_surface_clearance = float(
            -max_signed - float(self.catheter_radius)
        )
        tip_surface_clearance = float(
            -tip_signed_sim - float(self.catheter_radius)
        )
        self.current_sdf_tip_signed_distance = float(tip_signed_sim)
        self.current_sdf_max_signed_distance = max_signed
        self.current_sdf_body_min_surface_clearance = body_surface_clearance
        # Compatibility name now explicitly means tip-surface clearance.
        self.current_sdf_surface_clearance = tip_surface_clearance
        (
            self.current_sdf_body_warning_feature,
            self.current_sdf_body_outside_depth_feature,
        ) = body_sdf_risk_features(
            max_signed,
            outside_tolerance=self.sdf_outside_center_tolerance,
            warning_margin=self.sdf_body_warning_margin,
        )
        self.current_sdf_near_wall = bool(
            tip_surface_clearance < float(self.sdf_near_wall_margin)
        )
        self.current_sdf_penetrating = bool(tip_surface_clearance < 0.0)
        self.current_sdf_sample_count = int(len(signed_sim))
        self.current_sdf_worst_point_sim = np.asarray(
            sample_points[worst_index],
            dtype=np.float64,
        ).reshape(3)
        if len(sample_points) > 1:
            sample_arclength = np.concatenate(
                ([0.0], np.cumsum(np.linalg.norm(np.diff(sample_points, axis=0), axis=1)))
            )
            total_sample_arclength = float(sample_arclength[-1])
            self.current_sdf_worst_arc_fraction = (
                float(sample_arclength[worst_index]) / total_sample_arclength
                if total_sample_arclength > 1e-9
                else 0.0
            )
        else:
            self.current_sdf_worst_arc_fraction = 0.0

        # SDF increases from lumen interior toward the outside.  Return both
        # tip and worst-shaft inward directions; the latter is paired with its
        # shaft location in the observation, so its spatial meaning is explicit.
        worst_source = source_points[worst_index : worst_index + 1]
        gradients_source = np.asarray(
            grid.gradient(np.vstack([tip_source, worst_source])),
            dtype=np.float64,
        ).reshape(2, 3)
        gradients_sim = self._asset_vectors_to_sim(gradients_source)
        inward_vectors = []
        for gradient_sim in gradients_sim:
            gradient_norm = float(np.linalg.norm(gradient_sim))
            if np.isfinite(gradient_norm) and gradient_norm > 1e-9:
                inward_vectors.append((-gradient_sim / gradient_norm).astype(np.float64))
            else:
                inward_vectors.append(np.zeros(3, dtype=np.float64))
        self.current_sdf_inward_world = inward_vectors[0]
        self.current_sdf_worst_inward_world = inward_vectors[1]

        self._sdf_geometry_cache_step = int(
            getattr(self, "_elapsed_steps", -1)
        )
        self._sdf_geometry_cache_tip = tip_pos.copy()

        outside_candidate = bool(
            max_signed > float(self.sdf_outside_center_tolerance)
        )
        if advance_failure_counters:
            self.sdf_outside_counter = (
                int(self.sdf_outside_counter) + 1 if outside_candidate else 0
            )
            self.sdf_outside_confirmed = bool(
                self.sdf_outside_counter >= int(self.sdf_outside_confirm_steps)
            )
            self.min_sdf_surface_clearance_this_episode = min(
                float(self.min_sdf_surface_clearance_this_episode),
                tip_surface_clearance,
            )
            self.min_sdf_body_surface_clearance_this_episode = min(
                float(self.min_sdf_body_surface_clearance_this_episode),
                body_surface_clearance,
            )
            self.max_sdf_signed_distance_this_episode = max(
                float(self.max_sdf_signed_distance_this_episode),
                max_signed,
            )
            if self.current_sdf_near_wall:
                self.sdf_tip_near_wall_counter += 1
                self.sdf_tip_near_wall_steps_episode += 1
                self.sdf_tip_near_wall_counter_max_episode = max(
                    int(self.sdf_tip_near_wall_counter_max_episode),
                    int(self.sdf_tip_near_wall_counter),
                )
            else:
                self.sdf_tip_near_wall_counter = 0
            if self.current_sdf_penetrating:
                self.sdf_wall_contact_steps_episode += 1
                penetration_depth = max(0.0, -tip_surface_clearance)
                self.sdf_penetration_this_episode = True
                self.max_sdf_penetration_depth_this_episode = max(
                    float(self.max_sdf_penetration_depth_this_episode),
                    penetration_depth,
                )
                self.sdf_penetration_integral_this_episode += penetration_depth

    def _update_graph_branch_state(
        self,
        tip_pos: np.ndarray,
        selected_route_distance: float,
        advance_failure_counters: bool,
    ) -> None:
        graph_distance = self._get_graph_projection_distance(tip_pos)
        self.current_graph_distance = graph_distance
        if not np.isfinite(graph_distance):
            self.current_route_graph_distance_gap = 0.0
            self.current_off_target_branch_feature = 0.0
            self.current_wrong_branch = False
            if advance_failure_counters:
                self.wrong_branch_counter = 0
            return

        gap = max(0.0, float(selected_route_distance) - float(graph_distance))
        excess = max(0.0, gap - float(self.wrong_branch_distance_margin))
        self.current_route_graph_distance_gap = gap
        self.current_off_target_branch_feature = float(
            np.clip(
                excess / max(float(self.wrong_branch_observation_scale), 1e-9),
                0.0,
                1.0,
            )
        )
        sdf_inside = bool(
            not np.isfinite(self.current_sdf_max_signed_distance)
            or self.current_sdf_max_signed_distance
            <= float(self.sdf_outside_center_tolerance)
        )
        wrong_candidate = bool(excess > 0.0 and sdf_inside)
        if advance_failure_counters:
            self.wrong_branch_counter = (
                int(self.wrong_branch_counter) + 1 if wrong_candidate else 0
            )
            self.current_wrong_branch = bool(
                self.wrong_branch_counter >= int(self.wrong_branch_confirm_steps)
            )
            if self.current_wrong_branch:
                self.wrong_branch_this_episode = True
        else:
            self.current_wrong_branch = bool(
                self.wrong_branch_counter >= int(self.wrong_branch_confirm_steps)
            )

    def _update_vessel_safety_state(
        self,
        tip_pos: np.ndarray,
        advance_failure_counters: bool = False,
    ) -> None:
        progress, seg_idx, centerline_dist, centerline_proj, tangent = (
            self._get_tracked_centerline_projection_state(tip_pos)
        )
        local_radius = self._get_current_local_radius(progress)
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
        self._update_sdf_safety_state(
            tip_pos=tip_pos,
            advance_failure_counters=advance_failure_counters,
        )
        self._update_graph_branch_state(
            tip_pos=tip_pos,
            selected_route_distance=float(centerline_dist),
            advance_failure_counters=advance_failure_counters,
        )

        if self.sdf_grid is not None:
            self.current_out_of_vessel = bool(self.sdf_outside_confirmed)
        else:
            # Compatibility fallback for legacy real-vessel assets without VTI.
            self.current_out_of_vessel = bool(
                safety_ratio >= float(self.out_of_vessel_safety_ratio)
            )
            if not np.isfinite(safety_ratio):
                self.current_out_of_vessel = bool(
                    np.isfinite(centerline_dist)
                    and centerline_dist >= float(self.out_of_vessel_fallback_distance)
                )

    def _get_vessel_section_features(
        self,
        tip_pos: np.ndarray,
        tip_frame: np.ndarray,
        tip_forward_world: np.ndarray,
    ) -> np.ndarray:
        self._update_vessel_safety_state(tip_pos)
        if np.isfinite(self.current_sdf_surface_clearance):
            clearance_feature = float(
                np.clip(
                    self.current_sdf_surface_clearance
                    / max(float(self.sdf_clearance_observation_scale), 1e-9),
                    -2.0,
                    2.0,
                )
            )
            contact_feature = 1.0 if self.current_sdf_penetrating else 0.0
            body_clearance_feature = float(
                np.clip(
                    self.current_sdf_body_min_surface_clearance
                    / max(float(self.sdf_clearance_observation_scale), 1e-9),
                    -2.0,
                    2.0,
                )
            )
            outside_counter_feature = float(
                np.clip(
                    self.sdf_outside_counter
                    / max(1.0, float(self.sdf_outside_confirm_steps)),
                    0.0,
                    1.0,
                )
            )
            inward_local = self._world_vec_to_local(
                self.current_sdf_inward_world,
                tip_frame,
            )
            inward_norm = float(np.linalg.norm(inward_local))
            if np.isfinite(inward_norm) and inward_norm > 1e-9:
                inward_local = inward_local / inward_norm
            else:
                inward_local = np.zeros(3, dtype=np.float32)
            worst_position_local = self._world_vec_to_local(
                np.asarray(self.current_sdf_worst_point_sim, dtype=np.float32)
                - np.asarray(tip_pos, dtype=np.float32),
                tip_frame,
            ) / max(float(self.current_sdf_inserted_length), 0.001)
            worst_inward_local = self._world_vec_to_local(
                self.current_sdf_worst_inward_world,
                tip_frame,
            )
            worst_inward_norm = float(np.linalg.norm(worst_inward_local))
            if np.isfinite(worst_inward_norm) and worst_inward_norm > 1e-9:
                worst_inward_local = worst_inward_local / worst_inward_norm
            else:
                worst_inward_local = np.zeros(3, dtype=np.float32)
            forward_probe_features = self._get_sdf_forward_probe_features(
                tip_pos=tip_pos,
                tip_forward_world=tip_forward_world,
            )
        else:
            clearance_feature = float(
                np.clip(self.current_centerline_safety_margin, -2.0, 1.0)
            )
            contact_feature = 1.0 if self.current_centerline_safety_margin <= 0.0 else 0.0
            body_clearance_feature = clearance_feature
            outside_counter_feature = 0.0
            inward_local = np.zeros(3, dtype=np.float32)
            worst_position_local = np.zeros(3, dtype=np.float32)
            worst_inward_local = np.zeros(3, dtype=np.float32)
            forward_probe_features = np.zeros(
                len(self.sdf_forward_probe_distances),
                dtype=np.float32,
            )
        lookahead_tangents = self._get_centerline_lookahead_tangent_features(tip_frame)
        return np.array(
            [
                np.clip(self.current_centerline_offset_N_over_radius, -3.0, 3.0),
                np.clip(self.current_centerline_offset_B_over_radius, -3.0, 3.0),
                np.clip(self.current_centerline_local_radius_norm, 0.0, 5.0),
                clearance_feature,
                body_clearance_feature,
                outside_counter_feature,
                *np.clip(inward_local, -1.0, 1.0).tolist(),
                np.clip(self.current_sdf_worst_arc_fraction, 0.0, 1.0),
                *np.clip(worst_position_local, -1.0, 1.0).tolist(),
                *np.clip(worst_inward_local, -1.0, 1.0).tolist(),
                *np.clip(forward_probe_features, -2.0, 2.0).tolist(),
                *np.clip(lookahead_tangents, -1.0, 1.0).tolist(),
                np.clip(self.current_off_target_branch_feature, 0.0, 1.0),
                contact_feature,
            ],
            dtype=np.float32,
        )

    def _get_centerline_lookahead_tangent_features(self, tip_frame: np.ndarray) -> np.ndarray:
        """Encode selected-route tangents ahead of the current projection."""

        progress = float(getattr(self, "current_centerline_progress", np.nan))
        cumulative = getattr(self, "centerline_cumlength", None)
        if not np.isfinite(progress) or cumulative is None or len(cumulative) < 2:
            return np.zeros(3 * len(self.centerline_lookahead_distances), dtype=np.float32)
        route_end = float(np.asarray(cumulative, dtype=np.float64).reshape(-1)[-1])
        tangent_features = []
        half_window = 0.0005
        for distance in self.centerline_lookahead_distances:
            query = min(progress + float(distance), route_end)
            before = self._interpolate_centerline_point_at_progress(max(query - half_window, 0.0))
            after = self._interpolate_centerline_point_at_progress(min(query + half_window, route_end))
            tangent_world = np.asarray(after - before, dtype=np.float32)
            tangent_norm = float(np.linalg.norm(tangent_world))
            if not np.isfinite(tangent_norm) or tangent_norm <= 1e-9:
                tangent_world = np.asarray(self.current_centerline_tangent, dtype=np.float32)
                tangent_norm = float(np.linalg.norm(tangent_world))
            if np.isfinite(tangent_norm) and tangent_norm > 1e-9:
                tangent_world = tangent_world / tangent_norm
                tangent_local = self._world_vec_to_local(tangent_world, tip_frame)
            else:
                tangent_local = np.zeros(3, dtype=np.float32)
            tangent_features.extend(tangent_local.tolist())
        return np.asarray(tangent_features, dtype=np.float32)

    def _get_distance_tip_to_dest(self):
        tip = np.asarray(self.mcr_controller_sofa.get_pos_quat_catheter_tip()[0:3], dtype=np.float32)
        dist = float(np.linalg.norm(np.asarray(self.target_position, dtype=np.float32) - tip))
        if not np.isfinite(dist):
            self.non_finite_failure = True
            return float(1e3)
        return dist

    def _configure_continuous_route(self) -> None:
        """Resolve episode start/target arc lengths on the selected route."""

        if (
            self.centerline_points is None
            or self.centerline_cumlength is None
            or len(self.centerline_points) < 2
        ):
            self.current_route_start_progress = 0.0
            self.current_route_target_progress = 0.0
            return
        route_start = float(self.centerline_cumlength[0])
        route_end = float(self.centerline_cumlength[-1])
        target_progress, _, _ = self._raw_project_point_to_centerline_progress(
            self.target_position
        )
        target_progress = float(np.clip(target_progress, route_start, route_end))

        start_reference = getattr(self, "current_soft_start_position", None)
        if start_reference is None:
            start_reference = getattr(self, "current_scene_start_position", None)
        start_progress = route_start
        if start_reference is not None:
            start_np = np.asarray(start_reference, dtype=np.float32).reshape(3)
            if np.all(np.isfinite(start_np)):
                start_progress, _, _ = self._raw_project_point_to_centerline_progress(
                    start_np
                )
                start_progress = float(np.clip(start_progress, route_start, route_end))
        if target_progress <= start_progress + 1e-7:
            start_progress = route_start

        self.current_route_start_progress = float(start_progress)
        self.current_route_target_progress = float(target_progress)
        self.reward_progress_normalization = max(
            float(target_progress - start_progress),
            1e-6,
        )

    def _initialize_continuous_route_from_current_tip(self) -> None:
        """Initialize once globally, then use only local recurrent projection."""

        try:
            tip = np.asarray(
                self.mcr_controller_sofa.get_pos_quat_catheter_tip()[0:3],
                dtype=np.float32,
            )
            projection = project_to_route(
                self.centerline_points,
                self.centerline_cumlength,
                tip,
                previous_progress=None,
            )
            progress = float(
                np.clip(
                    projection.progress,
                    self.current_route_start_progress,
                    self.current_route_target_progress,
                )
            )
            self.current_route_progress = progress
            self.previous_route_progress = progress
            self.current_route_progress_delta = 0.0
            self.current_route_progress_ratio = normalized_route_progress(
                progress,
                self.current_route_start_progress,
                self.current_route_target_progress,
            )
            self.current_route_projection_segment = int(projection.segment_index)
            self.current_route_projection_distance = float(projection.distance)
            self.current_route_projection_jump_rejected = False
            self._route_projection_cache_step = int(self._elapsed_steps)
            self._route_projection_cache_tip = tip.copy()
            self._route_projection_cache_value = projection
        except Exception:
            self.current_route_progress = float(self.current_route_start_progress)
            self.previous_route_progress = float(self.current_route_start_progress)
            self.current_route_progress_delta = 0.0
            self.current_route_progress_ratio = 0.0
            self._route_projection_cache_step = -1
            self._route_projection_cache_tip = None
            self._route_projection_cache_value = None

    def _get_tracked_centerline_projection_state(self, tip_pos: np.ndarray):
        """Return a cached, local and physically gated route projection."""

        tip = np.asarray(tip_pos, dtype=np.float32).reshape(3)
        cached_tip = getattr(self, "_route_projection_cache_tip", None)
        if (
            cached_tip is not None
            and int(getattr(self, "_route_projection_cache_step", -1))
            == int(self._elapsed_steps)
            and np.allclose(tip, cached_tip, rtol=0.0, atol=1e-9)
            and getattr(self, "_route_projection_cache_value", None) is not None
        ):
            projection = self._route_projection_cache_value
            return (
                float(self.current_route_progress),
                int(projection.segment_index),
                float(projection.distance),
                np.asarray(projection.point, dtype=np.float32),
                np.asarray(projection.tangent, dtype=np.float32),
            )

        previous = float(getattr(self, "current_route_progress", np.nan))
        previous_arg = previous if np.isfinite(previous) else None
        projection = project_to_route(
            self.centerline_points,
            self.centerline_cumlength,
            tip,
            previous_progress=previous_arg,
            backward_window=self.route_projection_backward_window,
            forward_window=self.route_projection_forward_window,
            ambiguity_tolerance=self.route_projection_ambiguity_tolerance,
            max_progress_step=self.route_projection_max_progress_step,
        )
        progress = float(
            np.clip(
                projection.progress,
                self.current_route_start_progress,
                self.current_route_target_progress,
            )
        )
        self.previous_route_progress = previous if previous_arg is not None else progress
        self.current_route_progress = progress
        self.current_route_progress_delta = (
            progress - previous if previous_arg is not None else 0.0
        )
        self.current_route_progress_ratio = normalized_route_progress(
            progress,
            self.current_route_start_progress,
            self.current_route_target_progress,
        )
        self.current_route_projection_segment = int(projection.segment_index)
        self.current_route_projection_distance = float(projection.distance)
        self.current_route_projection_jump_rejected = bool(projection.jump_rejected)
        if projection.jump_rejected:
            self.route_projection_jump_rejections_episode += 1
        self._route_projection_cache_step = int(self._elapsed_steps)
        self._route_projection_cache_tip = tip.copy()
        self._route_projection_cache_value = projection
        return (
            progress,
            int(projection.segment_index),
            float(projection.distance),
            np.asarray(projection.point, dtype=np.float32),
            np.asarray(projection.tangent, dtype=np.float32),
        )

    def _continuous_route_potential(self) -> float:
        return normalized_route_progress(
            float(getattr(self, "current_route_progress", 0.0)),
            float(getattr(self, "current_route_start_progress", 0.0)),
            float(getattr(self, "current_route_target_progress", 0.0)),
        )

    # ------------------------------------------------------------------
    # Action application and scene initialization
    # ------------------------------------------------------------------

    def _apply_insert_safety_shield(self, raw_insert: float) -> float:
        """Apply only the configured action bounds.

        SOFA collision response and direct SDF safety feedback provide the
        physical constraints.  Avoid a hidden state-dependent insertion scale
        change, which makes the SAC action meaning inconsistent near the wall.
        """
        raw_insert = float(np.clip(raw_insert, -1.0, 1.0))
        self.current_raw_insert = raw_insert
        effective_insert = float(
            np.clip(
                raw_insert,
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
        self.insert_action_sum_episode += float(effective_insert)
        if effective_insert > 0.05:
            self.insert_positive_steps_episode += 1
        elif effective_insert < -0.05:
            self.insert_negative_steps_episode += 1
        else:
            self.insert_near_zero_steps_episode += 1
        self._apply_local_magnetic_action(rot_n, rot_b)
        self.mcr_controller_sofa.insertRetract(effective_insert)

    def _init_sim(self):
        super()._init_sim()
        self.mcr_controller_sofa: ControllerSofa = self.scene_creation_result["mcr_controller_sofa"]
        self.mcr_environment = self.scene_creation_result["mcr_environment"]
        self.chosen_model = self.scene_creation_result.get("chosen_model", "unknown")
        self.centerline_vtk = self.scene_creation_result.get("centerline_vtk", "unknown")
        self.task_id = self.scene_creation_result.get("task_id", str(self.chosen_model))
        self.vessel_scale_factor = float(
            self.scene_creation_result.get("vessel_scale_factor", 1.0)
        )
        self.centerline_graph_points = self.scene_creation_result.get(
            "centerline_graph_points", None
        )
        self.centerline_graph_edges = self.scene_creation_result.get(
            "centerline_graph_edges", None
        )
        self.centerline_graph_radius = self.scene_creation_result.get(
            "centerline_graph_radius", None
        )
        if self.centerline_graph_points is not None:
            self.centerline_graph_points = np.asarray(
                self.centerline_graph_points, dtype=np.float32
            )
        if self.centerline_graph_edges is not None:
            self.centerline_graph_edges = np.asarray(
                self.centerline_graph_edges, dtype=np.int64
            )
        if self.centerline_graph_radius is not None:
            self.centerline_graph_radius = np.asarray(
                self.centerline_graph_radius, dtype=np.float32
            )

        self.asset_source_to_sim_scale = float(
            self.scene_creation_result.get("asset_source_to_sim_scale", np.nan)
        )
        self.asset_T_env_sim = self.scene_creation_result.get(
            "asset_T_env_sim", None
        )
        self.asset_offset_sim = np.asarray(
            self.scene_creation_result.get("asset_offset_sim", [0.0, 0.0, 0.0]),
            dtype=np.float64,
        ).reshape(3)
        self.sdf_grid = None
        self.vessel_metadata = None
        self.collision_triangle_count = 0
        self.asset_model_family = "unknown"
        self.asset_difficulty = "unknown"
        sdf_vti = self.scene_creation_result.get("sdf_vti", None)
        metadata_json = self.scene_creation_result.get("metadata_json", None)
        is_artificial_model = bool(
            self.scene_creation_result.get("is_artificial_model", False)
        )
        if metadata_json:
            self.vessel_metadata = load_vessel_metadata(metadata_json)
            metadata_model = str(self.vessel_metadata.get("model_id", ""))
            if metadata_model and metadata_model != str(self.chosen_model):
                raise ValueError(
                    "Vessel metadata/model mismatch: "
                    f"metadata={metadata_model} chosen={self.chosen_model}"
                )
            source_units = str(self.vessel_metadata.get("source_units", ""))
            if source_units and source_units != "mm":
                raise ValueError(
                    f"Unsupported generated vessel source_units={source_units}; expected mm"
                )
            self.asset_model_family = str(
                self.vessel_metadata.get("family", "unknown")
            )
            self.asset_difficulty = str(
                self.vessel_metadata.get("difficulty", "unknown")
            )
            metadata_sofa_scale = float(
                self.vessel_metadata.get("sofa_scale", np.nan)
            )
            runtime_base_scale = float(
                self.asset_source_to_sim_scale
                / max(float(self.vessel_scale_factor), 1e-12)
            )
            if (
                np.isfinite(metadata_sofa_scale)
                and not np.isclose(
                    runtime_base_scale,
                    metadata_sofa_scale,
                    rtol=1e-6,
                    atol=1e-12,
                )
            ):
                raise ValueError(
                    "Vessel metadata/runtime scale mismatch: "
                    f"metadata={metadata_sofa_scale} runtime={runtime_base_scale}"
                )
            collision_validation = self.vessel_metadata.get(
                "collision_validation", {}
            )
            if isinstance(collision_validation, dict):
                self.collision_triangle_count = int(
                    collision_validation.get("triangle_count", 0)
                )
                degenerate_triangles = int(
                    collision_validation.get("degenerate_triangles", 0)
                )
                if self.collision_triangle_count <= 0 or degenerate_triangles != 0:
                    raise ValueError(
                        "Invalid generated collision mesh metadata: "
                        f"triangles={self.collision_triangle_count} "
                        f"degenerate={degenerate_triangles}"
                    )
            sdf_metadata = self.vessel_metadata.get("sdf", {})
            if isinstance(sdf_metadata, dict):
                convention = str(sdf_metadata.get("convention", ""))
                if convention and convention != "negative_inside_positive_outside":
                    raise ValueError(
                        f"Unsupported SDF sign convention: {convention}"
                    )
        if sdf_vti:
            if self.asset_T_env_sim is None:
                raise ValueError("SDF asset requires asset_T_env_sim from the scene.")
            self.sdf_grid = load_signed_distance_grid(sdf_vti)
            sdf_metadata = (
                self.vessel_metadata.get("sdf", {})
                if isinstance(self.vessel_metadata, dict)
                else {}
            )
            if isinstance(sdf_metadata, dict):
                metadata_dims = tuple(
                    int(value)
                    for value in sdf_metadata.get("dimensions_xyz", [])
                )
                runtime_dims = tuple(
                    int(value) for value in self.sdf_grid.values.shape[::-1]
                )
                if metadata_dims and metadata_dims != runtime_dims:
                    raise ValueError(
                        "Vessel metadata/VTI dimensions mismatch: "
                        f"metadata={metadata_dims} runtime={runtime_dims}"
                    )
                metadata_spacing = float(
                    sdf_metadata.get("spacing_mm", np.nan)
                )
                if (
                    np.isfinite(metadata_spacing)
                    and not np.allclose(
                        self.sdf_grid.spacing,
                        metadata_spacing,
                        rtol=1e-6,
                        atol=1e-9,
                    )
                ):
                    raise ValueError(
                        "Vessel metadata/VTI spacing mismatch: "
                        f"metadata={metadata_spacing} "
                        f"runtime={self.sdf_grid.spacing.tolist()}"
                    )
        if is_artificial_model and (self.sdf_grid is None or self.vessel_metadata is None):
            raise RuntimeError(
                f"Artificial model {self.chosen_model} requires VTI SDF and metadata."
            )
        if (
            is_artificial_model
            and str(self.chosen_model).upper()[0:1] in ("B", "V")
            and (
                self.centerline_graph_points is None
                or self.centerline_graph_edges is None
            )
        ):
            raise RuntimeError(
                f"Branching model {self.chosen_model} requires centerline_graph.vtk."
            )
        if self.scene_verbose:
            print(
                "[VESSEL_ASSET_BUNDLE]",
                "model=", self.chosen_model,
                "family=", self.asset_model_family,
                "difficulty=", self.asset_difficulty,
                "sdf=", bool(self.sdf_grid is not None),
                "graph=", bool(self.centerline_graph_points is not None),
                "collision_triangles=", self.collision_triangle_count,
                "source_to_sim_scale=", self.asset_source_to_sim_scale,
            )

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
        # soft single-vessel randomization. Keep that sampled start so route
        # progress begins from the actual catheter start rather than the
        # absolute centerline endpoint. Reset soft-start here to avoid stale
        # references after full scene reloads.
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

        self._apply_curriculum_target_position()
        self._configure_continuous_route()
        self._capture_soft_reset_reference_pose()

        vessel_positions = np.asarray(self.mcr_environment.get_vessel_tree_positions(), dtype=np.float32)
        if vessel_positions.ndim == 2 and vessel_positions.shape[1] == 3 and vessel_positions.shape[0] > 1 and np.all(np.isfinite(vessel_positions)):
            bbox_diag = float(np.linalg.norm(np.min(vessel_positions, axis=0) - np.max(vessel_positions, axis=0)))
            self.cartesian_scaling_factor = 1.0 / bbox_diag if np.isfinite(bbox_diag) and bbox_diag > 1e-9 else 1.0
        else:
            self.cartesian_scaling_factor = 1.0

    def _apply_curriculum_target_position(self) -> None:
        """Shorten only the training target while preserving the full route."""

        fraction = float(getattr(self, "curriculum_target_fraction", 1.0))
        self.current_full_target_position = np.asarray(
            self.target_position,
            dtype=np.float32,
        ).reshape(3).copy()
        self.current_curriculum_target_progress = np.nan
        if (
            not bool(getattr(self, "training_curriculum_enabled", False))
            or bool(getattr(self, "_explicit_force_model", ""))
            or fraction >= 1.0 - 1e-9
            or self.centerline_points is None
            or self.centerline_cumlength is None
            or len(self.centerline_points) < 2
        ):
            return

        start_reference = getattr(self, "current_scene_start_position", None)
        if start_reference is None:
            try:
                start_reference = np.asarray(
                    self.mcr_controller_sofa.get_pos_quat_catheter_tip()[0:3],
                    dtype=np.float32,
                )
            except Exception:
                start_reference = self.centerline_points[0]
        start_progress, _, _ = self._raw_project_point_to_centerline_progress(
            np.asarray(start_reference, dtype=np.float32)
        )
        full_target_progress, _, _ = self._raw_project_point_to_centerline_progress(
            self.current_full_target_position
        )
        if not (
            np.isfinite(start_progress)
            and np.isfinite(full_target_progress)
            and full_target_progress > start_progress + 1e-6
        ):
            return
        target_progress = start_progress + fraction * (
            full_target_progress - start_progress
        )
        self.target_position = self._interpolate_centerline_point_at_progress(
            target_progress
        ).astype(np.float32)
        self.current_curriculum_target_progress = float(target_progress)


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
