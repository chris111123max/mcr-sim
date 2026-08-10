from typing import Union, Tuple, Optional, Any, Dict
from pathlib import Path
from enum import Enum, unique
import gymnasium.spaces as spaces
import numpy as np
from collections import defaultdict, deque
from scipy.interpolate import splprep, splev
from scipy.spatial.transform import Rotation as R

from mcr_sim.rl_core.base import SofaEnv, RenderMode, RenderFramework
from mcr_sim.mcr_controller_sofa import ControllerSofa

HERE = Path(__file__).resolve().parent
FLAT_SCENE_DESCRIPTION_FILE_PATH = HERE / "scene_description_2d.py"
AORTIC_SCENE_DESCRIPTION_FILE_PATH = HERE.parent / "example_aortic_arch.py"
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
    """Magnetic Continuum Robot Environment.

    Clean projection-progress version for V1/0210 SAC training.

    Main choices:
    1. No gate-FSM reward/control/success logic. Centerline progress is represented by
       projection arc length ``s`` and reward is based on positive ``delta_s``.
    2. Actor observation is compact and deployable: local projection-frame geometry
       plus normalized local vessel-section safety features and a short action-response
       history.
    3. Critic uses asymmetric privileged observation: actor_obs + 34-D privileged_obs
       with task one-hot, local safety/radius state, and longer future centerline vectors.
    4. Observation frame, action frame, reward progress, lookahead, and success check
       are all based on the same continuous centerline projection state.
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
            # Clean reward surface:
            #   centerline_progress_distance: projection arc-length progress, 1 unit = 1 mm
            #   delta_tip_pos_distance_to_dest_pos: weak/terminal Euclidean refinement, 1 unit = 1 mm
            #   out_of_vessel_penalty: strong vessel-exit penalty
            #   successful_task: final success bonus
            "centerline_progress_distance": 200.0,
            "delta_tip_pos_distance_to_dest_pos": 20.0,
            # Soft wall penalty is continuous; severe out-of-vessel is reserved for
            # clearly invalid wall-crossing states.
            "wall_soft_penalty": 0.0,
            "out_of_vessel_penalty": -300.0,
            "successful_task": 4000.0,

            # Compatibility/logging keys kept zero-weighted. Gate is fully disabled
            # from reward/control/success, but old keys may still be present in logs.
            "tip_pos_distance_to_dest_pos": 0.0,
            "gate_progress": 0.0,
            "next_gate_approach": 0.0,
            "no_progress_penalty": 0.0,
            "step_penalty": -0.5,
            "action_smoothness": 0.0,
            "insert_smoothness": 0.0,
            "positive_insert_misaligned_penalty": 0.0,
            "safe_forward_insert_bonus": 0.0,
            "insert_retract_penalty": 0.0,
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

        # Required by mcr_controller_sofa.py.
        # When reducing SOFA dt, keep insertion integration stable by splitting
        # the pending insert command into several small substeps.
        create_scene_kwargs.setdefault("insert_substep_max", 1)

        # Keep initialization randomization consistent:
        # - multi-vessel training reloads the scene and lets example_aortic_arch.py
        #   sample endpoint-window starts/targets;
        # - forced single-vessel training keeps the SOFA scene and lets this env
        #   perform the same endpoint-window randomization during soft reset.
        if "soft_randomize_single_vessel" not in create_scene_kwargs:
            create_scene_kwargs["soft_randomize_single_vessel"] = bool(
                str(create_scene_kwargs.get("force_model", "") or "").strip()
            )

        self.target_distance_threshold = target_distance_threshold
        self.num_catheter_tracking_points = num_catheter_tracking_points
        self.max_episode_steps = int(max_episode_steps)
        self._elapsed_steps = 0

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

        # Actor tangent lookahead: 10 future centerline tangents sampled every 1 mm
        # over the next 10 mm. Compared with tip-relative future-point vectors,
        # tangent vectors have unit length and therefore do not collapse in tight
        # U-bends. They give the actor a stable signpost of upcoming curvature.
        self.num_actor_lookahead_points = int(create_scene_kwargs.get("num_actor_lookahead_points", 10))
        self.num_actor_lookahead_points = max(1, self.num_actor_lookahead_points)
        self.actor_lookahead_distance = float(create_scene_kwargs.get("actor_lookahead_distance", 0.010))  # 10 mm
        self.actor_lookahead_step = float(create_scene_kwargs.get("actor_lookahead_step", 0.001))          # 1 mm
        self.actor_lookahead_offsets = tuple(
            (np.arange(1, self.num_actor_lookahead_points + 1, dtype=np.float32) * self.actor_lookahead_step).tolist()
        )
        self.actor_lookahead_dim = 3 * self.num_actor_lookahead_points

        # Legacy helper dimensions. These are kept for compatibility with
        # _get_centerline_light_features, but the full long-range block is no
        # longer concatenated into actor_obs.
        self.num_centerline_lookahead_points = 12
        self.centerline_lookahead_offsets = (2, 4, 6, 8, 12, 16, 22, 30, 40, 55, 75, 100)
        self.centerline_feature_dim = 1 + 3 * self.num_centerline_lookahead_points
        self.vessel_section_feature_dim = 4

        # Explicit guidance features make the local steering / insertion safety shield
        # observable to SAC instead of forcing the policy to infer it from raw lookahead
        # vectors. The current turn-angle scalar is explicitly injected into actor_obs
        # below, because U-bend anticipation is critical for this task.
        # Layout:
        #   [lead_dir_local(3), forward_alignment, effective_insert_norm,
        #    projection_reliable, insert_gate_scale]
        self.explicit_guidance_feature_dim = 7
        self.base_observation_dim = 10 + self.vessel_section_feature_dim + self.explicit_guidance_feature_dim
        self.target_observation_scale = 0.10      # 100 mm
        self.lookahead_observation_scale = 0.05   # 50 mm
        self.magnetic_field_observation_scale = 0.10

        # Safety-aware cross-section observation.
        self.radius_observation_scale = float(create_scene_kwargs.get("radius_observation_scale", 0.005))  # 5 mm
        self.default_local_radius = float(create_scene_kwargs.get("default_local_radius", 0.005))
        self.catheter_radius = float(create_scene_kwargs.get("catheter_radius", 0.000665))

        # Conservative vessel-exit proxy.
        # safety_ratio = (centerline_distance + catheter_radius) / local_radius.
        # Values close to 1 mean the catheter is touching/crossing the vessel wall.
        # We use 0.80 by default because the centerline radius may be optimistic in 0021;
        # this prevents the policy from cutting through the wall to collect target reward.
        self.out_of_vessel_safety_ratio = float(create_scene_kwargs.get("out_of_vessel_safety_ratio", 1.00))
        self.out_of_vessel_fallback_distance = float(create_scene_kwargs.get("out_of_vessel_fallback_distance", 0.012))
        # Continuous wall penalty. It starts before actual vessel exit and grows smoothly.
        # Severe penalty is delayed to avoid a cliff exactly at safety_ratio=1.0.
        self.wall_soft_start_ratio = float(create_scene_kwargs.get("wall_soft_start_ratio", 0.80))
        self.wall_soft_hard_ratio = float(create_scene_kwargs.get("wall_soft_hard_ratio", 1.10))
        self.wall_soft_power = float(create_scene_kwargs.get("wall_soft_power", 2.0))
        self.severe_out_of_vessel_ratio = float(create_scene_kwargs.get("severe_out_of_vessel_ratio", 1.15))
        self.current_out_of_vessel = False
        self.out_of_vessel_this_episode = False
        self.out_of_vessel_failure = False

        # Prevent nearest-centerline projection jumps in highly curved / bifurcation regions.
        # The tip moves less than ~1 mm per step, so a 2 mm arc-length window is conservative.
        self.centerline_projection_max_jump = float(create_scene_kwargs.get("centerline_projection_max_jump", 0.002))

        # Reward normalization.
        # Centerline progress uses delta progress ratio, so a full path contributes
        # approximately 1.0 before multiplying by reward weight, independent of
        # absolute vessel/path length. Terminal Euclidean refinement keeps a fixed
        # metric scale because final success is judged by millimeters.
        self.euclidean_reward_scale = float(create_scene_kwargs.get("euclidean_reward_scale", 0.001))
        self.euclidean_reward_scale = max(self.euclidean_reward_scale, 1e-9)

        # Outside the terminal zone, keep a weak Euclidean progress reward as a
        # fallback when centerline projection is noisy or temporarily unreliable.
        # Inside the terminal zone, Euclidean refinement keeps full strength.
        self.outside_terminal_euclidean_weight = float(
            create_scene_kwargs.get("outside_terminal_euclidean_weight", 0.3)
        )
        self.outside_terminal_euclidean_weight = float(
            np.clip(self.outside_terminal_euclidean_weight, 0.0, 1.0)
        )

        # Centerline projection reliability / divergence protection.
        # The centerline projection is useful only when the tip remains near the
        # physical vessel path. If the tip is far away, nearest-segment projection
        # may jump to a wrong branch or an unrelated curved segment. In that case
        # centerline progress reward is disabled and a weak Euclidean progress
        # reward is used as fallback. Extremely divergent episodes are terminated
        # to avoid filling the replay buffer with long, invalid trajectories.
        self.centerline_projection_search_window = float(
            create_scene_kwargs.get("centerline_projection_search_window", 0.020)
        )  # fallback symmetric search window around previous progress
        # Progress-tracking estimator: with mostly forward insertion, use an
        # asymmetric arc-length window to avoid snapping backward or to a spatially
        # close but topologically wrong segment.
        self.centerline_projection_back_window = float(
            create_scene_kwargs.get("centerline_projection_back_window", 0.005)
        )  # 5 mm backward allowance
        self.centerline_projection_forward_window = float(
            create_scene_kwargs.get("centerline_projection_forward_window", 0.030)
        )  # 30 mm forward allowance
        self.projection_reliable_distance = float(
            create_scene_kwargs.get("projection_reliable_distance", 0.030)
        )  # 30 mm
        # Off-path guard distances are curvature-aware: stricter in curved
        # regions to prevent the policy from taking an unsafe inner shortcut.
        self.out_of_centerline_soft_distance = float(
            create_scene_kwargs.get("out_of_centerline_soft_distance", 0.020)
        )  # straight fallback: 20 mm
        self.severe_divergence_distance = float(
            create_scene_kwargs.get("severe_divergence_distance", 0.040)
        )  # straight fallback: 40 mm
        self.out_soft_distance_straight = float(create_scene_kwargs.get("out_soft_distance_straight", 0.020))
        self.out_hard_distance_straight = float(create_scene_kwargs.get("out_hard_distance_straight", 0.040))
        self.out_soft_distance_medium_turn = float(create_scene_kwargs.get("out_soft_distance_medium_turn", 0.012))
        self.out_hard_distance_medium_turn = float(create_scene_kwargs.get("out_hard_distance_medium_turn", 0.025))
        self.out_soft_distance_sharp_turn = float(create_scene_kwargs.get("out_soft_distance_sharp_turn", 0.008))
        self.out_hard_distance_sharp_turn = float(create_scene_kwargs.get("out_hard_distance_sharp_turn", 0.015))
        self.severe_divergence_patience = int(
            create_scene_kwargs.get("severe_divergence_patience", 10)
        )
        self.current_projection_reliable = True
        self.severe_centerline_divergence = False
        self.severe_centerline_divergence_counter = 0

        # 3D local-action insertion safety shield. SAC outputs [rotN, rotB, insert].
        # Bend-specific speed caps are disabled in this version. With the mechanical
        # insertion already reduced to 0.3 mm/step, slowing again in turns can cause
        # the observed 60 mm stagnation. We keep projection reliability and alignment gating.
        self.insert_cap_straight = float(create_scene_kwargs.get("insert_cap_straight", 1.0))
        self.insert_cap_medium_turn = float(create_scene_kwargs.get("insert_cap_medium_turn", 1.0))
        self.insert_cap_sharp_turn = float(create_scene_kwargs.get("insert_cap_sharp_turn", 1.0))
        self.insert_cap_terminal = float(create_scene_kwargs.get("insert_cap_terminal", 1.0))
        self.insert_cap_unreliable = float(create_scene_kwargs.get("insert_cap_unreliable", 0.0))
        self.retract_cap_default = float(create_scene_kwargs.get("retract_cap_default", 0.5))
        self.retract_cap_recovery = float(create_scene_kwargs.get("retract_cap_recovery", 0.25))
        self.local_field_action_angle = float(create_scene_kwargs.get("local_field_action_angle", 2.0 * np.pi / 180.0))
        # Kept for logging / critic curvature diagnostics only, not for speed gating.
        self.turn_angle_medium_deg = float(create_scene_kwargs.get("turn_angle_medium_deg", 10.0))
        self.turn_angle_sharp_deg = float(create_scene_kwargs.get("turn_angle_sharp_deg", 25.0))
        self.turn_angle_offsets = tuple(create_scene_kwargs.get("turn_angle_offsets", (2, 3, 5)))
        # Unified lead point for the alignment shield; no straight/medium/sharp split.
        self.lead_idx_local = int(create_scene_kwargs.get("lead_idx_local", 5))
        self.turn_path_direction_weight_medium = float(create_scene_kwargs.get("turn_path_direction_weight_medium", 0.0))
        self.turn_path_direction_weight_sharp = float(create_scene_kwargs.get("turn_path_direction_weight_sharp", 0.0))

        # Alignment-gated insertion: do not insert quickly unless the tip forward
        # direction is aligned with the lead direction. This prevents the fixed
        # insertion controller from pushing a misaligned tip into the vessel wall.
        self.alignment_stop_threshold = float(create_scene_kwargs.get("alignment_stop_threshold", 0.20))
        self.alignment_slow_threshold = float(create_scene_kwargs.get("alignment_slow_threshold", 0.50))
        self.alignment_medium_threshold = float(create_scene_kwargs.get("alignment_medium_threshold", 0.80))
        self.insert_cap_misaligned = float(create_scene_kwargs.get("insert_cap_misaligned", 0.0))
        self.insert_cap_poor_alignment = float(create_scene_kwargs.get("insert_cap_poor_alignment", 0.10))
        self.insert_cap_mid_alignment = float(create_scene_kwargs.get("insert_cap_mid_alignment", 0.30))

        self.current_turn_angle_deg = 0.0
        self.current_turn_angle_norm = 0.0
        self.current_raw_insert = 0.0
        self.current_rule_insert = 0.0
        self.current_effective_insert = 0.0
        self.current_insert_gate_scale = 1.0
        self.current_forward_alignment = 0.0
        self.current_forward_misalignment = 0.0
        self.current_goal_alignment = 0.0
        self.current_goal_misalignment = 0.0
        self.current_lead_direction_world = np.zeros(3, dtype=np.float32)
        self.current_lead_direction_local = np.zeros(3, dtype=np.float32)
        self.current_centerline_frame = np.eye(3, dtype=np.float32)
        self.current_out_of_centerline_soft = False
        self.current_out_of_vessel = False
        self.out_of_vessel_this_episode = False
        self.out_of_vessel_failure = False

        # Task list is needed both by adaptive sampling and by privileged critic one-hot encoding.
        self.training_models = [
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
        ]

        # Actor observation is intentionally compact, deployable, and projection-frame based.
        # Removed redundant centerline_distance_norm / scalar turn_angle_norm from the
        # actor input. Vessel clearance is represented by normalized local-section
        # features, which generalize better across different vessel radii.
        #
        # Actor current geometry, 45-D by default:
        #   tip_forward_local(3)
        #   magnetic_field_local(3)
        #   lead_direction_local(3)
        #   actor_tangent_lookahead_local(10 points * 3 = 30)
        #   current_dist_norm(1)
        #   centerline_progress_ratio(1)
        #   vessel_section_features(4):
        #       [offset_N_over_radius, offset_B_over_radius,
        #        local_radius_norm, safety_margin]
        # Dynamic/action-response, 7-D per step:
        #   prev_action/effective_insert(3)
        #   tip_delta_local(3)
        #   progress_delta_norm(1)
        # Default history = 4 steps, so actor_obs = 45 + 7*4 = 73-D by default.
        self.actor_current_geometry_dim = 15 + self.actor_lookahead_dim  # 45 when tangent lookahead=10*3
        self.actor_dynamic_step_dim = 7
        self.actor_history_steps = int(create_scene_kwargs.get("actor_history_steps", 4))
        self.actor_history_steps = max(1, self.actor_history_steps)
        self.actor_dynamic_history_dim = self.actor_dynamic_step_dim * self.actor_history_steps
        self.actor_observation_dim = self.actor_current_geometry_dim + self.actor_dynamic_history_dim
        self._actor_dynamic_history = deque(maxlen=self.actor_history_steps)
        self.actor_core_observation_dim = self.actor_current_geometry_dim

        # Privileged critic observation: clean but slightly multi-dimensional.
        # We keep the full 10-task one-hot for compatibility, while current sampling
        # is restricted to V1 and 0210.
        # Layout = task one-hot(10) + current safety scalars(6) + future centerline vectors(6*3=18) = 34-D.
        self.privileged_current_scalar_dim = 6
        self.privileged_lookahead_offsets = tuple(create_scene_kwargs.get(
            "privileged_lookahead_offsets", (2, 4, 6, 10, 15, 20)
        ))
        self.privileged_lookahead_dim_per_point = 3
        self.privileged_scalar_dim = (
            self.privileged_current_scalar_dim
            + len(self.privileged_lookahead_offsets) * self.privileged_lookahead_dim_per_point
        )
        self.privileged_observation_dim = len(self.training_models) + self.privileged_scalar_dim

        if self.observation_type == ObservationType.STATE:
            self.observation_space = spaces.Dict(
                {
                    "actor_obs": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(self.actor_observation_dim,),
                        dtype=np.float32,
                    ),
                    "privileged_obs": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(self.privileged_observation_dim,),
                        dtype=np.float32,
                    ),
                }
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
        self.max_progress_this_episode = 0.0
        self.previous_centerline_projection_index = -1
        self.centerline_cumlength = None
        self.centerline_points = None
        self.centerline_radius = None

        self.current_centerline_local_radius = np.nan
        self.current_centerline_safety_ratio = np.nan
        self.current_centerline_safety_margin = np.nan
        self.current_centerline_offset_N_over_radius = np.nan
        self.current_centerline_offset_B_over_radius = np.nan
        self.current_centerline_local_radius_norm = np.nan

        self.centerline_projection_raw_jump = 0.0
        self.centerline_projection_selected_jump = 0.0
        self.centerline_projection_jump_limited = False
        self.centerline_projection_limited_count = 0
        self.centerline_projection_call_count = 0
        self.centerline_projection_frozen = False
        self.centerline_projection_freeze_count = 0

        self.current_projection_reliable = True
        self.severe_centerline_divergence = False
        self.severe_centerline_divergence_counter = 0
        self.current_turn_angle_deg = 0.0
        self.current_turn_angle_norm = 0.0
        self.current_raw_insert = 0.0
        self.current_rule_insert = 0.0
        self.current_effective_insert = 0.0
        self.current_insert_gate_scale = 1.0
        self.current_forward_alignment = 0.0
        self.current_forward_misalignment = 0.0
        self.current_goal_alignment = 0.0
        self.current_goal_misalignment = 0.0
        self.current_lead_direction_world = np.zeros(3, dtype=np.float32)
        self.current_lead_direction_local = np.zeros(3, dtype=np.float32)
        self.current_out_of_vessel = False
        self.out_of_vessel_this_episode = False
        self.out_of_vessel_failure = False

        self.max_safety_ratio_this_episode = 0.0
        self.min_safety_margin_this_episode = np.inf

        self.centerline_escape_failed = False
        self.centerline_escape_counter = 0
        self.centerline_escape_patience = 10
        self.centerline_safe_ratio = 1.00
        self.centerline_fail_ratio = 1.08
        self._episode_end_printed = False

        self.episode_success = False
        self.episode_success_2mm = False
        self.min_dist_this_episode = np.inf
        self.episode_success_10mm = False
        self.episode_success_6mm = False
        self.episode_success_3mm = False
        self.action_smoothing_alpha = 0.2

        self.progress_clip = 0.003
        self.no_progress_delta_threshold = 0.00002  # kept for legacy projection diagnostics only
        # Gate-based stagnation patience. With 2 mm gates and ~0.3 mm/step insertion,
        # straight traversal needs about 7 steps per gate. Start penalizing after
        # 20 consecutive steps without passing the next ordered gate; terminate the
        # episode after a longer no-progress period to avoid filling replay with
        # repeated local oscillation.
        self.no_progress_patience = int(create_scene_kwargs.get("no_progress_patience", 20))
        self.no_progress_terminate_patience = int(create_scene_kwargs.get("no_progress_terminate_patience", 250))
        self.no_progress_counter = 0
        self.no_progress_failure = False

        # Sequential cross-section gate progress. Gates are generated from the
        # centerline every 2 mm. Each gate is a disk whose normal points from the
        # current gate center to the next gate center; the pass radius is derived
        # from the centerline Radius array with a conservative wall margin.
        self.gate_spacing = float(create_scene_kwargs.get("gate_spacing", 0.002))
        self.gate_pass_radius_scale = float(create_scene_kwargs.get("gate_pass_radius_scale", 0.80))
        self.gate_pass_margin = float(create_scene_kwargs.get("gate_pass_margin", 0.0005))
        self.gate_points = None
        self.gate_normals = None
        self.gate_radius = None
        self.gate_pass_radius = None
        self.gate_progress = None
        self.current_gate_idx = 0
        self.current_next_gate_idx = 1
        self.current_gate_progress_ratio = 0.0
        self.current_gate_passed_this_step = False
        self.current_gate_pass_count_episode = 0
        self.current_gate_pass_count_step = 0
        self.current_next_gate_signed_dist = np.nan
        self.current_next_gate_lateral_dist = np.nan
        self.current_next_gate_pass_radius = np.nan
        self.previous_next_gate_idx = None
        self.previous_next_gate_distance = None
        self.current_next_gate_approach = 0.0
        self._last_gate_tip_pos = None
        # Gate FSM guard: if the tip leaves the vessel while approaching the
        # current next gate, it cannot collect that gate by re-entering behind
        # the gate plane. It must return to the front side first.
        self.gate_invalid_bypass = False
        self.gate_bypass_reset_margin = float(create_scene_kwargs.get("gate_bypass_reset_margin", 1e-5))

        self.terminal_enter_threshold = 0.010
        self.terminal_exit_threshold = 0.012
        self.min_success_centerline_progress_ratio = float(
            create_scene_kwargs.get("min_success_centerline_progress_ratio", 0.90)
        )
        self.terminal_bad_steps_limit = 30
        self.non_finite_failure = False

        # Use task-level sampling so 0207_left and 0207_right can be learned
        # and logged independently instead of being merged into one 0207 statistic.
        self.training_models = [
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
        ]
        self.adaptive_sampling_beta = float(create_scene_kwargs.get("adaptive_sampling_beta", 2.0))
        self.adaptive_sampling_eps = float(create_scene_kwargs.get("adaptive_sampling_eps", 0.05))
        self.adaptive_sampling_enabled = bool(create_scene_kwargs.get("adaptive_task_sampling", True))
        self._explicit_force_model = str(create_scene_kwargs.get("force_model", "") or "").strip()
        if self._explicit_force_model:
            self.adaptive_sampling_enabled = False

        self._sampler_rng = np.random.default_rng()
        self.model_success_stats = {
            model: {"episodes": 0, "successes": 0}
            for model in self.training_models
        }

        self.model_sampling_probs = {model: 0.0 for model in self.training_models}
        for model in ["V1", "0210"]:
            self.model_sampling_probs[model] = 0.5

        self.current_sampling_model = self._explicit_force_model if self._explicit_force_model else None
                # Single-vessel randomized soft reset.
        # Previous implementation rebuilt the whole SOFA scene whenever start/target
        # or initial orientation randomization was enabled. That preserves randomization
        # but can leak C++/SOFA memory during long runs. In forced single-vessel mode,
        # the vessel STL/collision and centerline are fixed, so we can keep the scene
        # loaded and only randomize target_position + instrument starting pose.
        self.soft_randomize_single_vessel = bool(
            create_scene_kwargs.get("soft_randomize_single_vessel", True)
        )
        self._soft_reset_rng = np.random.default_rng()
        self._soft_reset_base_start_sim = None
        self._soft_reset_nominal_start_sim = None
        self._soft_reset_nominal_target_sim = None
        self._soft_reset_warning_printed = False
        self._soft_reset_info_printed = False

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
        angle = float(np.arccos(dot))
        return R.from_rotvec(angle * axis)

    @staticmethod
    def _sample_uniform_ball(radius: float, rng: np.random.Generator) -> np.ndarray:
        radius = max(0.0, float(radius))
        if radius <= 0.0:
            return np.zeros(3, dtype=np.float64)
        direction = rng.normal(size=3)
        n = float(np.linalg.norm(direction))
        if n < 1e-12:
            direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            direction = direction / n
        r = radius * float(rng.random() ** (1.0 / 3.0))
        return (r * direction).astype(np.float64)

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
        perp_norm = float(np.linalg.norm(perp))
        if perp_norm < 1e-12:
            ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if abs(float(np.dot(ref, base))) > 0.9:
                ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            perp = ref - float(np.dot(ref, base)) * base
            perp_norm = float(np.linalg.norm(perp))
        perp = perp / (perp_norm + 1e-12)

        cos_min = float(np.cos(max_angle_rad))
        cos_theta = float(rng.uniform(cos_min, 1.0))
        sin_theta = float(np.sqrt(max(0.0, 1.0 - cos_theta * cos_theta)))
        out = cos_theta * base + sin_theta * perp
        return out / (float(np.linalg.norm(out)) + 1e-12)

    @staticmethod
    def _assign_sofa_data_value(obj, attr_name: str, value) -> bool:
        """Best-effort SOFA Data assignment helper."""
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

    def _capture_soft_reset_reference_pose(self) -> None:
        """Store the nominal start pose of the already-created single-vessel scene."""
        if getattr(self, "_soft_reset_base_start_sim", None) is not None:
            return

        base_pose = None
        try:
            base_pose = np.asarray(self.mcr_controller_sofa.instrument.IRC.startingPos.value, dtype=np.float64).reshape(-1)
        except Exception:
            pass

        if base_pose is None or base_pose.shape[0] < 7 or (not np.all(np.isfinite(base_pose[:7]))):
            try:
                rest = np.asarray(self.mcr_controller_sofa.instrument.MO.rest_position.value, dtype=np.float64)
                if rest.ndim == 2 and rest.shape[1] >= 7 and rest.shape[0] > 0:
                    base_pose = rest[0, :7].copy()
            except Exception:
                base_pose = None

        if base_pose is None or base_pose.shape[0] < 7:
            try:
                tip_pose = np.asarray(self.mcr_controller_sofa.get_pos_quat_catheter_tip(), dtype=np.float64).reshape(-1)
                if tip_pose.shape[0] >= 7:
                    base_pose = tip_pose[:7].copy()
            except Exception:
                base_pose = None

        if base_pose is not None and len(base_pose) >= 7 and np.all(np.isfinite(base_pose[:7])):
            self._soft_reset_base_start_sim = np.asarray(base_pose[:7], dtype=np.float64).copy()

    def _apply_instrument_start_pose_sim(self, t_start_sim: np.ndarray) -> bool:
        """Move the existing SOFA instrument to a new start pose without recreating the scene.

        This updates InterventionalRadiologyController.startingPos and resets the
        rigid beam DOFs/rest positions to the same pose. It is intentionally
        best-effort because SOFA Data assignment behavior differs across versions.
        """
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
        except Exception:
            pass

        try:
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
            poses = np.tile(t_start_sim[None, :], (n, 1)).astype(np.float64)
            pose_list = poses.tolist()
            ok_any = self._assign_sofa_data_value(instrument.MO, "rest_position", pose_list) or ok_any
            ok_any = self._assign_sofa_data_value(instrument.MO, "position", pose_list) or ok_any
            ok_any = self._assign_sofa_data_value(instrument.MO, "free_position", pose_list) or ok_any
            try:
                instrument.MO.velocity.value = np.zeros((n, 6), dtype=np.float64).tolist()
                ok_any = True
            except Exception:
                pass

        return bool(ok_any)

    def _soft_randomize_single_vessel_scene(self, seed: Union[int, np.random.SeedSequence, None] = None) -> None:
        """Randomize start/target/initial orientation inside an existing single-vessel SOFA scene.

        This preserves the desired domain randomization while avoiding repeated
        unload/reload of the same vessel STL/collision scene.
        """
        if not bool(getattr(self, "soft_randomize_single_vessel", True)):
            return
        if not bool(getattr(self, "_explicit_force_model", "")):
            return

        randomize_start_target = bool(self.create_scene_kwargs.get("randomize_start_target", True))
        randomize_initial_orientation = bool(self.create_scene_kwargs.get("randomize_initial_orientation", True))
        if not (randomize_start_target or randomize_initial_orientation):
            return

        if seed is not None:
            seed_value = self._seed_to_int(seed)
            if seed_value is not None:
                self._soft_reset_rng = np.random.default_rng(seed_value)

        rng = getattr(self, "_soft_reset_rng", None)
        if rng is None:
            rng = np.random.default_rng()
            self._soft_reset_rng = rng

        points = getattr(self, "centerline_points", None)
        if points is None or len(points) < 2:
            return
        points = np.asarray(points, dtype=np.float64)

        self._capture_soft_reset_reference_pose()
        base_start_sim = getattr(self, "_soft_reset_base_start_sim", None)
        if base_start_sim is None or len(base_start_sim) < 7:
            if not bool(getattr(self, "_soft_reset_warning_printed", False)):
                print("[SOFT_RANDOM_RESET][WARN] Could not capture base start pose; target randomization only.")
                self._soft_reset_warning_printed = True
            base_start_sim = None

        nominal_start = points[0].copy()
        nominal_target = points[-1].copy()
        start_point = nominal_start.copy()
        target_point = nominal_target.copy()
        start_idx = 0
        target_idx = len(points) - 1
        start_candidates = [0]
        target_candidates = [len(points) - 1]

        if randomize_start_target:
            # Scheme-1 endpoint-window randomization, synchronized with
            # example_aortic_arch.py:
            #   start  <- one of the first K centerline points
            #   target <- one of the last K centerline points
            # No off-center ball perturbation is added.
            n_pts = int(len(points))
            start_window_points = int(self.create_scene_kwargs.get(
                "start_window_points",
                self.create_scene_kwargs.get("start_target_window_points", 5),
            ))
            target_window_points = int(self.create_scene_kwargs.get(
                "target_window_points",
                self.create_scene_kwargs.get("start_target_window_points", 5),
            ))
            start_window_points = int(np.clip(start_window_points, 1, n_pts))
            target_window_points = int(np.clip(target_window_points, 1, n_pts))
            start_candidates_arr = np.arange(0, start_window_points, dtype=np.int64)
            target_candidates_arr = np.arange(max(0, n_pts - target_window_points), n_pts, dtype=np.int64)
            start_idx = int(rng.choice(start_candidates_arr))
            target_idx = int(rng.choice(target_candidates_arr))
            if target_idx <= start_idx and n_pts > 1:
                target_idx = n_pts - 1
            start_candidates = start_candidates_arr.tolist()
            target_candidates = target_candidates_arr.tolist()
            start_point = points[start_idx].copy()
            target_point = points[target_idx].copy()

        self.target_position = np.asarray(target_point, dtype=np.float32)
        self.current_soft_start_position = np.asarray(start_point, dtype=np.float32)
        self.current_soft_target_position = np.asarray(target_point, dtype=np.float32)
        self.current_soft_start_idx = int(start_idx)
        self.current_soft_target_idx = int(target_idx)

        if base_start_sim is not None:
            step = int(self.create_scene_kwargs.get("entry_tangent_points", 5))
            step = int(np.clip(step, 1, len(points) - 1))
            next_idx = int(np.clip(start_idx + step, 0, len(points) - 1))
            if next_idx == start_idx:
                next_idx = int(np.clip(start_idx + 1, 0, len(points) - 1))
            entry_tangent = points[next_idx] - points[start_idx]
            entry_tangent = self._unit_vector(entry_tangent)
            if entry_tangent is not None:
                desired_tangent = entry_tangent
                sampled_angle_deg = 0.0
                if randomize_initial_orientation:
                    max_angle_deg = float(self.create_scene_kwargs.get("initial_orientation_max_angle_deg", 20.0))
                    sampled = self._sample_direction_within_cone(entry_tangent, np.deg2rad(max_angle_deg), rng)
                    if sampled is not None:
                        desired_tangent = sampled
                        sampled_angle_deg = float(np.degrees(np.arccos(np.clip(np.dot(entry_tangent, desired_tangent), -1.0, 1.0))))

                base_rot = R.from_quat(np.asarray(base_start_sim[3:7], dtype=np.float64))
                base_forward = base_rot.apply(np.array([1.0, 0.0, 0.0], dtype=np.float64))
                delta_rot = self._rotation_between_vectors(base_forward, desired_tangent)
                new_rot = delta_rot * base_rot

                t_start_sim = np.asarray(base_start_sim, dtype=np.float64).copy()
                t_start_sim[0:3] = np.asarray(start_point, dtype=np.float64)
                t_start_sim[3:7] = new_rot.as_quat()

                ok = self._apply_instrument_start_pose_sim(t_start_sim)
                if not bool(getattr(self, "_soft_reset_info_printed", False)):
                    print(
                        "[SOFT_ENDPOINT_WINDOW_RESET] enabled "
                        f"force_model={self._explicit_force_model} "
                        f"start_target={randomize_start_target} "
                        f"orientation={randomize_initial_orientation} "
                        f"pose_apply_ok={ok} "
                        f"start_candidates={start_candidates} "
                        f"target_candidates={target_candidates} "
                        f"sampled_start_idx={start_idx} "
                        f"sampled_target_idx={target_idx} "
                        f"example_angle_deg={sampled_angle_deg:.2f}"
                    )
                    self._soft_reset_info_printed = True

        # Target-dependent terminal geometry must follow the sampled target.
        self._update_exit_plane_normal_for_current_target()

    def _update_exit_plane_normal_for_current_target(self) -> None:
        self.exit_plane_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if self.centerline_points is None or self.target_position is None or len(self.centerline_points) == 0:
            return
        try:
            distances_to_target = np.linalg.norm(self.centerline_points - self.target_position, axis=1)
            idx_1mm = int(np.argmin(np.abs(distances_to_target - 0.001)))
            point_1mm = self.centerline_points[idx_1mm]
            direction_vec = np.asarray(self.target_position, dtype=np.float32) - np.asarray(point_1mm, dtype=np.float32)
            norm = float(np.linalg.norm(direction_vec))
            if norm > 1e-9:
                self.exit_plane_normal = (direction_vec / norm).astype(np.float32)
        except Exception:
            pass

    def _record_episode_for_sampler(self) -> None:
        if not self.adaptive_sampling_enabled:
            return

        model_key = str(getattr(self, "chosen_model", "") or "")
        if model_key not in self.model_success_stats:
            return

        stats = self.model_success_stats[model_key]
        stats["episodes"] += 1
        if bool(getattr(self, "episode_success", False)):
            stats["successes"] += 1

    def _sample_next_training_model(self, seed: Union[int, np.random.SeedSequence, None] = None) -> None:
        if self._explicit_force_model:
            self.create_scene_kwargs["force_model"] = self._explicit_force_model
            self.current_sampling_model = self._explicit_force_model
            return

        # ============================================================
        # Current active training subset: V1 + 0210.
        # Keep self.training_models unchanged for privileged critic one-hot dimension.
        # Only change the actual sampling pool here.
        # ============================================================
        active_sampling_models = [
            "0207_left",
            "0207_right",
            "0210",
            "V1",
        ]

        # Safety check: only keep valid models that exist in the original full list.
        active_sampling_models = [
            m for m in active_sampling_models if m in self.training_models
        ]
        if len(active_sampling_models) == 0:
            active_sampling_models = list(self.training_models)

        if not self.adaptive_sampling_enabled:
            chosen = self._sampler_rng.choice(active_sampling_models)
            self.create_scene_kwargs["force_model"] = str(chosen)
            self.current_sampling_model = str(chosen)

            # Log probabilities: selected subset has uniform probability, others are zero.
            self.model_sampling_probs = {model: 0.0 for model in self.training_models}
            for model in active_sampling_models:
                self.model_sampling_probs[model] = 1.0 / float(len(active_sampling_models))
            return

        if seed is not None:
            try:
                seed_value = int(seed.entropy) if isinstance(seed, np.random.SeedSequence) else int(seed)
                self._sampler_rng = np.random.default_rng(seed_value)
            except Exception:
                pass

        weights = []
        for model in active_sampling_models:
            stats = self.model_success_stats[model]
            episodes = int(stats["episodes"])
            successes = int(stats["successes"])
            success_rate = float(successes / episodes) if episodes > 0 else 0.0
            difficulty = max(0.0, 1.0 - success_rate)
            weight = float(np.power(difficulty + self.adaptive_sampling_eps, self.adaptive_sampling_beta))
            weights.append(weight)

        weights = np.asarray(weights, dtype=np.float64)
        if (not np.all(np.isfinite(weights))) or float(np.sum(weights)) <= 0.0:
            probs = np.ones(len(active_sampling_models), dtype=np.float64) / float(len(active_sampling_models))
        else:
            probs = weights / float(np.sum(weights))

        chosen = str(self._sampler_rng.choice(active_sampling_models, p=probs))
        self.create_scene_kwargs["force_model"] = chosen
        self.current_sampling_model = chosen

        # Keep full probability dict for logging, but assign zero to unused vessels.
        self.model_sampling_probs = {model: 0.0 for model in self.training_models}
        self.model_sampling_probs.update({
            model: float(prob)
            for model, prob in zip(active_sampling_models, probs)
        })

    def reset(self, seed: Union[int, np.random.SeedSequence, None] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Union[np.ndarray, None], Dict]:
        if self._initialized:
            self._record_episode_for_sampler()

        self._sample_next_training_model(seed=seed)

        # Reset strategy:
        # - Multi-vessel mode still reloads when the selected vessel changes.
        # - Forced single-vessel mode keeps the SOFA scene loaded. If start/target
        #   or initial-orientation randomization is requested, it is applied inside
        #   the existing scene by _soft_randomize_single_vessel_scene(). This avoids
        #   repeated STL/collision scene creation and prevents long-run OOM.
        single_vessel_mode = bool(getattr(self, "_explicit_force_model", ""))
        randomization_requested = bool(
            self.create_scene_kwargs.get("randomize_start_target", True)
            or self.create_scene_kwargs.get("randomize_initial_orientation", True)
        )
        soft_randomization_available = bool(
            single_vessel_mode and getattr(self, "soft_randomize_single_vessel", True)
        )
        randomized_scene_each_episode = bool(randomization_requested and not soft_randomization_available)
        need_full_reload = bool(self._initialized)

        if single_vessel_mode and self._initialized and not randomized_scene_each_episode:
            need_full_reload = False
            reset_mode = "soft_reset_randomized" if randomization_requested else "soft_reset"
        elif not self._initialized:
            need_full_reload = True
            reset_mode = "first_full_reload"
        else:
            need_full_reload = True
            reset_mode = "full_reload_randomized" if randomized_scene_each_episode else "full_reload"

        if getattr(self, "_reset_mode_last", "") != reset_mode:
            if single_vessel_mode:
                print(
                    f"[RESET_MODE] force_model={self._explicit_force_model} "
                    f"mode={reset_mode} reload_every={bool(randomized_scene_each_episode)}"
                )
            else:
                print(
                    f"[RESET_MODE] force_model={getattr(self, '_explicit_force_model', '')} "
                    f"mode={reset_mode}"
                )
            self._reset_mode_last = reset_mode

        if self._initialized and need_full_reload:
            if hasattr(self, "sofa_simulation") and self._sofa_root_node is not None:
                self.sofa_simulation.unload(self._sofa_root_node)
            self._initialized = False

        # Build a SOFA scene only when needed. During single-vessel soft reset, the
        # existing scene is reused and only controller / episode state is reset below.
        if not self._initialized:
            super().reset(seed)

        self._elapsed_steps = 0
        self.episode_success = False
        self.episode_success_2mm = False
        self.min_dist_this_episode = np.inf
        self.episode_success_10mm = False
        self.episode_success_6mm = False
        self.episode_success_3mm = False

        self.previous_centerline_progress = None
        self.max_progress_this_episode = 0.0
        self.previous_centerline_projection_index = -1
        self.current_centerline_progress = 0.0
        self.current_centerline_delta_progress = 0.0
        self.current_centerline_distance = 0.0
        self.current_centerline_progress_ratio = 0.0
        self.current_centerline_delta_progress_ratio = 0.0
        self.using_centerline_reward = False
        self.using_euclidean_terminal_reward = False
        self.non_finite_failure = False

        self.entered_terminal_zone_once = False
        self.terminal_bad_steps = 0
        self.was_in_terminal_zone_prev_step = False
        self.terminal_recovered = False
        self.terminal_escape_failed = False
        self.terminal_failed_due_to_timeout = False

        self.centerline_escape_failed = False
        self.centerline_escape_counter = 0
        self.current_centerline_local_radius = np.nan
        self.current_centerline_safety_ratio = np.nan
        self.current_centerline_safety_margin = np.nan
        self.current_centerline_offset_N_over_radius = np.nan
        self.current_centerline_offset_B_over_radius = np.nan
        self.current_centerline_local_radius_norm = np.nan

        self.centerline_projection_raw_jump = 0.0
        self.centerline_projection_selected_jump = 0.0
        self.centerline_projection_jump_limited = False
        self.centerline_projection_limited_count = 0
        self.centerline_projection_call_count = 0
        self.centerline_projection_frozen = False
        self.centerline_projection_freeze_count = 0

        self.current_projection_reliable = True
        self.severe_centerline_divergence = False
        self.severe_centerline_divergence_counter = 0
        self.current_turn_angle_deg = 0.0
        self.current_turn_angle_norm = 0.0
        self.current_raw_insert = 0.0
        self.current_rule_insert = 0.0
        self.current_effective_insert = 0.0
        self.current_insert_gate_scale = 1.0
        self.current_forward_alignment = 0.0
        self.current_forward_misalignment = 0.0
        self.current_goal_alignment = 0.0
        self.current_goal_misalignment = 0.0
        self.current_lead_direction_world = np.zeros(3, dtype=np.float32)
        self.current_lead_direction_local = np.zeros(3, dtype=np.float32)
        self.current_out_of_vessel = False
        self.out_of_vessel_this_episode = False
        self.out_of_vessel_failure = False

        self.max_safety_ratio_this_episode = 0.0
        self.min_safety_margin_this_episode = np.inf
        self.no_progress_counter = 0
        self.no_progress_failure = False
        self.current_gate_idx = 0
        self.current_next_gate_idx = 1
        self.current_gate_progress_ratio = 0.0
        self.current_gate_passed_this_step = False
        self.current_gate_pass_count_episode = 0
        self.current_gate_pass_count_step = 0
        self.current_next_gate_signed_dist = np.nan
        self.current_next_gate_lateral_dist = np.nan
        self.current_next_gate_pass_radius = np.nan
        self.previous_next_gate_idx = None
        self.previous_next_gate_distance = None
        self.current_next_gate_approach = 0.0
        self._last_gate_tip_pos = None
        self.gate_invalid_bypass = False

        self._episode_end_printed = False

        self.reached_10mm_once = False
        self.reached_6mm_once = False
        self.reached_3mm_once = False

        self.mcr_controller_sofa.reset()

        # Apply start/target/orientation randomization without recreating the
        # SOFA scene in forced single-vessel mode. For the first reset this also
        # overrides the nominal scene pose; for later resets it prevents memory
        # accumulation from repeated unload/reload cycles.
        if bool(single_vessel_mode and getattr(self, "soft_randomize_single_vessel", True)):
            self._soft_randomize_single_vessel_scene(seed=seed)

        self.reward_info = {}
        self.reward_features = {}
        self.reward_features["tip_pos_distance_to_dest_pos"] = self._get_distance_tip_to_dest()

        self._last_smoothed_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self._prev_smoothed_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self._reset_actor_history()
        self._last_actor_tip_pos = None
        self._last_actor_dist = None
        self._last_actor_progress = None

        # Reset settle: use one explicit 0.05 s SOFA integration step.
        # Do not use 0.5 s here; that is too coarse and can disturb the initial gate state.
        self.sofa_simulation.animate(self._sofa_root_node, 0.05)

        # Gate progress is intentionally disabled in this projection-progress version.
        return self._get_observation(image_observation=self._maybe_update_rgb_buffer()), {}

    def step(self, action: Any) -> Tuple[Union[np.ndarray, dict], float, bool, bool, dict]:
        action_np = np.array(action, dtype=np.float32)
        action_np = np.nan_to_num(action_np, nan=0.0, posinf=1.0, neginf=-1.0)
        action_np = np.clip(action_np, -1.0, 1.0).astype(np.float32)

        # Do not smooth or otherwise alter the SAC action.
        # Keep only NaN handling and action-space clipping above.
        previous_action = getattr(self, "_last_smoothed_action", np.zeros_like(action_np))
        executed_action = action_np.copy()
        self._prev_smoothed_action = previous_action.copy()
        self._last_smoothed_action = executed_action.copy()

        image_observation = super().step(executed_action)
        self._elapsed_steps += 1
        observation = self._get_observation(image_observation)
        reward = self._get_reward()

        terminated = self._get_done()
        non_finite_failure = False

        if self.observation_type == ObservationType.STATE:
            obs_finite = True
            if isinstance(observation, dict):
                for _obs_value in observation.values():
                    if not np.all(np.isfinite(_obs_value)):
                        obs_finite = False
                        break
            else:
                obs_finite = bool(np.all(np.isfinite(observation)))
            if not obs_finite:
                non_finite_failure = True
                if isinstance(observation, dict):
                    observation = {
                        k: np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                        for k, v in observation.items()
                    }
                else:
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
            or getattr(self, "out_of_vessel_failure", False)
            or getattr(self, "no_progress_failure", False)
            # Centerline escape / severe divergence are no longer termination conditions.
            # They are logged or penalized, so the agent can learn recovery.
            or getattr(self, "non_finite_failure", False)
        ):
            terminated = True

        truncated = (self._elapsed_steps >= self.max_episode_steps) and (not terminated)

        info = self._get_info(terminated=terminated, truncated=truncated)
        if truncated:
            info["TimeLimit.truncated"] = True

        return observation, reward, terminated, truncated, info

    def _reset_actor_history(self) -> None:
        """Reset the compact one-step actor action-response history."""
        self._actor_dynamic_history = deque(maxlen=int(getattr(self, "actor_history_steps", 4)))

    def _push_actor_dynamic_history(self, dynamic_step_obs: np.ndarray) -> None:
        dynamic_step_obs = np.asarray(dynamic_step_obs, dtype=np.float32).reshape(-1)
        expected = int(getattr(self, "actor_dynamic_step_dim", 7))
        if dynamic_step_obs.shape[0] != expected:
            fixed = np.zeros(expected, dtype=np.float32)
            n = min(expected, dynamic_step_obs.shape[0])
            fixed[:n] = dynamic_step_obs[:n]
            dynamic_step_obs = fixed
        self._actor_dynamic_history.append(
            np.nan_to_num(dynamic_step_obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        )

    # Backward-compatible alias for older internal calls.
    def _push_actor_history(self, core_obs: np.ndarray) -> None:
        self._push_actor_dynamic_history(core_obs)

    def _get_actor_dynamic_history_observation(self) -> np.ndarray:
        steps = int(getattr(self, "actor_history_steps", 4))
        dim = int(getattr(self, "actor_dynamic_step_dim", 7))
        history = list(getattr(self, "_actor_dynamic_history", []))
        missing = max(0, steps - len(history))
        chunks = [np.zeros(dim, dtype=np.float32) for _ in range(missing)]
        chunks.extend(history)
        out = np.concatenate(chunks[-steps:]).astype(np.float32)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def _get_actor_observation(self, current_geometry_obs: np.ndarray) -> np.ndarray:
        current_geometry_obs = np.asarray(current_geometry_obs, dtype=np.float32).reshape(-1)
        expected = int(getattr(self, "actor_current_geometry_dim", current_geometry_obs.shape[0]))
        if current_geometry_obs.shape[0] != expected:
            fixed = np.zeros(expected, dtype=np.float32)
            n = min(expected, current_geometry_obs.shape[0])
            fixed[:n] = current_geometry_obs[:n]
            current_geometry_obs = fixed
        obs = np.concatenate(
            [
                np.nan_to_num(current_geometry_obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
                self._get_actor_dynamic_history_observation(),
            ]
        ).astype(np.float32)
        expected_total = int(getattr(self, "actor_observation_dim", obs.shape[0]))
        if obs.shape[0] != expected_total:
            fixed = np.zeros(expected_total, dtype=np.float32)
            n = min(expected_total, obs.shape[0])
            fixed[:n] = obs[:n]
            obs = fixed
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # Backward-compatible alias. It returns only the dynamic-history part.
    def _get_actor_history_observation(self) -> np.ndarray:
        return self._get_actor_dynamic_history_observation()

    def _build_actor_current_geometry_observation(
        self,
        tip_forward_local: np.ndarray,
        magnetic_field_norm: np.ndarray,
        lead_direction_local: np.ndarray,
        actor_lookahead_local: np.ndarray,
        current_dist_norm: np.ndarray,
        centerline_progress_ratio: float,
        vessel_section_features: np.ndarray,
    ) -> np.ndarray:
        """Build the current actor geometry block in the projection local frame.

        The actor receives local vessel-section features instead of a raw
        centerline-distance scalar. This tells the deployable policy whether a
        given offset is safe in a thick vessel or dangerous in a thin vessel.
        """
        vessel_section_features = np.asarray(vessel_section_features, dtype=np.float32).reshape(-1)
        if vessel_section_features.shape[0] != self.vessel_section_feature_dim:
            fixed = np.zeros(self.vessel_section_feature_dim, dtype=np.float32)
            n = min(self.vessel_section_feature_dim, vessel_section_features.shape[0])
            fixed[:n] = vessel_section_features[:n]
            vessel_section_features = fixed

        obs = np.concatenate(
            [
                np.asarray(tip_forward_local, dtype=np.float32).reshape(3),
                np.asarray(magnetic_field_norm, dtype=np.float32).reshape(3),
                np.asarray(lead_direction_local, dtype=np.float32).reshape(3),
                np.asarray(actor_lookahead_local, dtype=np.float32).reshape(-1),
                np.asarray(current_dist_norm, dtype=np.float32).reshape(1),
                np.array([np.clip(float(centerline_progress_ratio), 0.0, 1.0)], dtype=np.float32),
                vessel_section_features.astype(np.float32),
            ]
        ).astype(np.float32)
        expected = int(getattr(self, "actor_current_geometry_dim", obs.shape[0]))
        if obs.shape[0] != expected:
            fixed = np.zeros(expected, dtype=np.float32)
            n = min(expected, obs.shape[0])
            fixed[:n] = obs[:n]
            obs = fixed
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def _build_actor_dynamic_step_observation(
        self,
        prev_action: np.ndarray,
        tip_delta_local: np.ndarray,
        progress_delta_norm: np.ndarray,
    ) -> np.ndarray:
        """Compact one-step action-response block, 7-D."""
        obs = np.concatenate(
            [
                np.asarray(prev_action, dtype=np.float32).reshape(3),
                np.asarray(tip_delta_local, dtype=np.float32).reshape(3),
                np.asarray(progress_delta_norm, dtype=np.float32).reshape(1),
            ]
        ).astype(np.float32)
        expected = int(getattr(self, "actor_dynamic_step_dim", 7))
        if obs.shape[0] != expected:
            fixed = np.zeros(expected, dtype=np.float32)
            n = min(expected, obs.shape[0])
            fixed[:n] = obs[:n]
            obs = fixed
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def _build_actor_core_observation(
        self,
        tip_forward_local: np.ndarray,
        magnetic_field_norm: np.ndarray,
        current_dist_norm: np.ndarray,
        tip_delta_local: np.ndarray,
        distance_delta_norm: np.ndarray,
        prev_action: np.ndarray,
        lead_direction_local: np.ndarray,
        actor_lookahead_local: np.ndarray,
        forward_alignment: float,
        projection_reliable_feature: float,
        insert_gate_scale: float,
    ) -> np.ndarray:
        """Backward-compatible wrapper for older internal call signatures."""
        return np.concatenate(
            [
                self._build_actor_current_geometry_observation(
                    tip_forward_local=tip_forward_local,
                    magnetic_field_norm=magnetic_field_norm,
                    lead_direction_local=lead_direction_local,
                    actor_lookahead_local=actor_lookahead_local,
                    current_dist_norm=current_dist_norm,
                    centerline_progress_ratio=float(getattr(self, "current_centerline_progress_ratio", 0.0)),
                    vessel_section_features=np.array(
                        [
                            float(getattr(self, "current_centerline_offset_N_over_radius", 0.0)),
                            float(getattr(self, "current_centerline_offset_B_over_radius", 0.0)),
                            float(getattr(self, "current_centerline_local_radius_norm", 1.0)),
                            float(getattr(self, "current_centerline_safety_margin", 1.0)),
                        ],
                        dtype=np.float32,
                    ),
                ),
                self._build_actor_dynamic_step_observation(
                    prev_action=prev_action,
                    tip_delta_local=tip_delta_local,
                    progress_delta_norm=np.array([float(getattr(self, "current_centerline_delta_progress", 0.0)) / 0.001], dtype=np.float32),
                ),
            ]
        ).astype(np.float32)

    def _get_privileged_lookahead_geometry(self, tip_pos: np.ndarray, progress: float, frame: np.ndarray) -> np.ndarray:
        """Critic-only future centerline vectors, no radius/curvature redundancy."""
        offsets = tuple(getattr(self, "privileged_lookahead_offsets", (2, 4, 6, 10, 15, 20)))
        per_point_dim = int(getattr(self, "privileged_lookahead_dim_per_point", 3))
        out = np.zeros(len(offsets) * per_point_dim, dtype=np.float32)

        if self.centerline_points is None or len(self.centerline_points) < 2:
            return out
        if getattr(self, "centerline_cumlength", None) is None:
            return out

        tip_pos = np.asarray(tip_pos, dtype=np.float32).reshape(3)
        frame = np.asarray(frame, dtype=np.float32)
        vec_scale = float(getattr(self, "lookahead_observation_scale", 0.05))
        progress = float(progress) if np.isfinite(float(progress)) else 0.0

        values = []
        for off_mm in offsets:
            future_point = self._interpolate_centerline_point_at_progress(progress + float(off_mm) * 0.001)
            v_world = future_point - tip_pos
            v_local = self._world_vec_to_local(v_world, frame) / max(vec_scale, 1e-9)
            values.append(np.clip(v_local, -5.0, 5.0))

        if len(values) > 0:
            flat = np.asarray(values, dtype=np.float32).reshape(-1)
            out[:min(out.shape[0], flat.shape[0])] = flat[:out.shape[0]]
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def _get_privileged_observation(self) -> np.ndarray:
        """Critic-only privileged observation for asymmetric SAC.

        Clean layout, 34-D by default:
            task one-hot over the full training model list(10)
            + current safety scalars(6)
            + longer future centerline vectors(6 * 3 = 18)
        """
        model_names = list(getattr(self, "training_models", []))
        one_hot = np.zeros(len(model_names), dtype=np.float32)
        key = str(getattr(self, "task_id", getattr(self, "chosen_model", getattr(self, "current_sampling_model", ""))) or "")
        if key in model_names:
            one_hot[model_names.index(key)] = 1.0

        def sf(name, default=0.0):
            try:
                value = float(getattr(self, name, default))
            except Exception:
                return float(default)
            return value if np.isfinite(value) else float(default)

        current_scalars = np.array(
            [
                sf("current_centerline_local_radius") / 0.02,
                sf("current_centerline_safety_ratio"),
                sf("current_centerline_safety_margin"),
                sf("current_centerline_offset_N_over_radius"),
                sf("current_centerline_offset_B_over_radius"),
                1.0 if bool(getattr(self, "current_out_of_vessel", False)) else 0.0,
            ],
            dtype=np.float32,
        )

        try:
            tip_pose = np.asarray(self.mcr_controller_sofa.get_pos_quat_catheter_tip(), dtype=np.float32)
            tip_pos = tip_pose[0:3]
            progress, seg_idx, _, _, tangent = self._get_centerline_projection_state(tip_pos)
            frame = self._build_local_centerline_frame(tangent)
            lookahead_geometry = self._get_privileged_lookahead_geometry(tip_pos, progress, frame)
        except Exception:
            lookahead_geometry = np.zeros(
                len(getattr(self, "privileged_lookahead_offsets", (2, 4, 6, 10, 15, 20)))
                * int(getattr(self, "privileged_lookahead_dim_per_point", 3)),
                dtype=np.float32,
            )

        out = np.concatenate([one_hot, current_scalars, lookahead_geometry]).astype(np.float32)
        expected = int(getattr(self, "privileged_observation_dim", out.shape[0]))
        if out.shape[0] != expected:
            fixed = np.zeros(expected, dtype=np.float32)
            n = min(expected, out.shape[0])
            fixed[:n] = out[:n]
            out = fixed
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def _get_observation(self, image_observation: Union[np.ndarray, None]) -> Union[np.ndarray, dict]:
        if self.observation_type == ObservationType.RGB:
            return image_observation
        elif self.observation_type == ObservationType.STATE:
            tip_pose = np.asarray(self.mcr_controller_sofa.get_pos_quat_catheter_tip(), dtype=np.float32)
            tip_pos = tip_pose[0:3]
            tip_quat = tip_pose[3:7]

            progress, seg_idx, centerline_dist, centerline_proj, tangent = self._get_centerline_projection_state(tip_pos)
            # Expose upcoming curvature directly to the actor. This value was
            # previously computed only for logging / steering diagnostics.
            turn_angle_deg = float(self._estimate_centerline_turn_angle_deg(seg_idx)) if seg_idx >= 0 else 0.0
            self.current_turn_angle_deg = turn_angle_deg
            self.current_turn_angle_norm = float(np.clip(turn_angle_deg / 90.0, 0.0, 1.0))
            frame = self._build_local_centerline_frame(tangent)
            self.current_centerline_frame = frame.astype(np.float32)
            total_len = float(self.centerline_cumlength[-1]) if getattr(self, "centerline_cumlength", None) is not None else 0.0
            progress_ratio = float(progress / (total_len + 1e-9)) if total_len > 1e-9 else 0.0

            tip_forward_world = self._quat_rotate_vector(tip_quat, np.array([1.0, 0.0, 0.0], dtype=np.float32))
            tip_forward_local = self._world_vec_to_local(tip_forward_world, frame)
            tip_forward_norm = float(np.linalg.norm(tip_forward_local))
            if tip_forward_norm > 1e-9:
                tip_forward_local = tip_forward_local / tip_forward_norm

            current_dist = float(self._get_distance_tip_to_dest())
            current_dist_norm = np.array([current_dist / float(self.target_observation_scale)], dtype=np.float32)

            last_tip_pos = getattr(self, "_last_actor_tip_pos", None)
            if last_tip_pos is None:
                tip_delta_world = np.zeros(3, dtype=np.float32)
            else:
                tip_delta_world = np.asarray(tip_pos, dtype=np.float32) - np.asarray(last_tip_pos, dtype=np.float32)
            tip_delta_local = np.clip(self._world_vec_to_local(tip_delta_world, frame) / 0.001, -5.0, 5.0).astype(np.float32)

            last_progress = getattr(self, "_last_actor_progress", None)
            if last_progress is None or not np.isfinite(float(last_progress)):
                progress_delta_norm = np.array([0.0], dtype=np.float32)
            else:
                progress_delta_norm = np.array([np.clip((float(progress) - float(last_progress)) / 0.001, -5.0, 5.0)], dtype=np.float32)

            mag_field = np.asarray(self.mcr_controller_sofa.get_mag_field_des(), dtype=np.float32)
            mag_field_local = self._world_vec_to_local(mag_field, frame)
            magnetic_field_norm = np.clip(mag_field_local / float(self.magnetic_field_observation_scale), -2.0, 2.0).astype(np.float32)

            prev_control = getattr(self, "_last_smoothed_action", np.zeros(3, dtype=np.float32))
            prev_control = np.asarray(prev_control, dtype=np.float32).reshape(-1)
            prev_action = np.zeros(3, dtype=np.float32)
            if prev_control.shape[0] > 0:
                prev_action[0] = float(prev_control[0])
            if prev_control.shape[0] > 1:
                prev_action[1] = float(prev_control[1])
            prev_action[2] = float(getattr(self, "current_effective_insert", getattr(self, "current_rule_insert", 0.0)))

            lead_distance = float(getattr(self, "actor_lookahead_distance", 0.006))
            lead_point = self._interpolate_centerline_point_at_progress(float(progress) + lead_distance)
            lead_dir_world = np.asarray(lead_point, dtype=np.float32) - np.asarray(tip_pos, dtype=np.float32)
            lead_norm = float(np.linalg.norm(lead_dir_world))
            if lead_norm > 1e-9:
                lead_dir_world = lead_dir_world / lead_norm
            else:
                lead_dir_world = np.asarray(tangent, dtype=np.float32)
            lead_direction_local = self._world_vec_to_local(lead_dir_world, frame)
            lead_local_norm = float(np.linalg.norm(lead_direction_local))
            if lead_local_norm > 1e-9:
                lead_direction_local = lead_direction_local / lead_local_norm
            self.current_lead_direction_world = lead_dir_world.astype(np.float32)
            self.current_lead_direction_local = lead_direction_local.astype(np.float32)

            actor_lookahead_local = self._get_actor_lookahead_relative_vectors(tip_pos=tip_pos, progress=progress, frame=frame)
            vessel_section_features = self._get_vessel_section_features(tip_pos)

            actor_current_geometry = self._build_actor_current_geometry_observation(
                tip_forward_local=tip_forward_local,
                magnetic_field_norm=magnetic_field_norm,
                lead_direction_local=lead_direction_local,
                actor_lookahead_local=actor_lookahead_local,
                current_dist_norm=current_dist_norm,
                centerline_progress_ratio=progress_ratio,
                vessel_section_features=vessel_section_features,
            )
            actor_dynamic_step = self._build_actor_dynamic_step_observation(
                prev_action=prev_action,
                tip_delta_local=tip_delta_local,
                progress_delta_norm=progress_delta_norm,
            )
            self._push_actor_dynamic_history(actor_dynamic_step)
            actor_obs = self._get_actor_observation(actor_current_geometry)

            self._last_actor_tip_pos = np.asarray(tip_pos, dtype=np.float32).copy()
            self._last_actor_dist = float(current_dist)
            self._last_actor_progress = float(progress)

            return {
                "actor_obs": actor_obs.astype(np.float32),
                "privileged_obs": self._get_privileged_observation().astype(np.float32),
            }
        else:
            return {}

    def _get_reward_features(self, previous_reward_features: dict) -> dict:
        """Compute the clean projection-progress reward features."""
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

        try:
            tip_pose = np.asarray(self.mcr_controller_sofa.get_pos_quat_catheter_tip(), dtype=np.float32)
            tip_pos = tip_pose[0:3]
        except Exception:
            tip_pos = np.zeros(3, dtype=np.float32)

        centerline_progress, centerline_seg_idx, centerline_dist, centerline_proj, centerline_tangent = (
            self._get_centerline_projection_state(tip_pos)
        )
        previous_centerline_progress = getattr(self, "previous_centerline_progress", None)
        if previous_centerline_progress is None or centerline_seg_idx < 0:
            delta_centerline_progress = 0.0
        else:
            delta_centerline_progress = float(centerline_progress - previous_centerline_progress)
        delta_centerline_progress = float(np.clip(delta_centerline_progress, -self.progress_clip, self.progress_clip))

        self.previous_centerline_progress = float(centerline_progress)
        self.previous_centerline_projection_index = int(centerline_seg_idx)
        self.current_centerline_distance = float(centerline_dist)
        self.current_centerline_progress = float(centerline_progress)
        self.current_centerline_delta_progress = float(delta_centerline_progress)
        total_len = float(self.centerline_cumlength[-1]) if getattr(self, "centerline_cumlength", None) is not None else 0.0
        self.current_centerline_progress_ratio = float(centerline_progress / (total_len + 1e-9)) if total_len > 1e-9 else 0.0
        if previous_centerline_progress is None or total_len <= 1e-9 or centerline_seg_idx < 0:
            delta_centerline_progress_ratio = 0.0
        else:
            delta_centerline_progress_ratio = float(delta_centerline_progress / total_len)
        self.current_centerline_delta_progress_ratio = float(np.clip(delta_centerline_progress_ratio, -0.03, 0.03))

        terminal_zone_active = bool(current_dist <= self.terminal_enter_threshold or previous_dist <= self.terminal_enter_threshold)
        projection_reliable = bool(
            centerline_seg_idx >= 0
            and np.isfinite(centerline_dist)
            and centerline_dist <= float(getattr(self, "projection_reliable_distance", 0.030))
        )
        self.current_projection_reliable = projection_reliable
        self.using_centerline_reward = bool((not terminal_zone_active) and projection_reliable)
        self.using_euclidean_terminal_reward = bool(terminal_zone_active)

        # Centerline radius/safety diagnostics and out-of-vessel proxy.
        local_radius = self._get_current_local_radius(centerline_seg_idx)
        catheter_radius = float(getattr(self, "catheter_radius", 0.000665))
        self.current_centerline_local_radius = float(local_radius) if local_radius is not None else np.nan
        self.current_centerline_safety_ratio = np.nan
        self.current_out_of_vessel = False

        if (
            local_radius is not None
            and np.isfinite(centerline_dist)
            and np.isfinite(local_radius)
            and local_radius > catheter_radius + 1e-6
        ):
            safety_ratio = float((centerline_dist + catheter_radius) / local_radius)
            self.current_centerline_safety_ratio = safety_ratio
            self.current_centerline_safety_margin = float(1.0 - safety_ratio)
            self.max_safety_ratio_this_episode = max(float(getattr(self, "max_safety_ratio_this_episode", 0.0)), safety_ratio)
            self.min_safety_margin_this_episode = min(
                float(getattr(self, "min_safety_margin_this_episode", np.inf)),
                float(1.0 - safety_ratio),
            )
            out_ratio = float(getattr(self, "out_of_vessel_safety_ratio", 1.00))
            self.current_out_of_vessel = bool(safety_ratio >= out_ratio)
            fail_ratio = float(getattr(self, "centerline_fail_ratio", 1.08))
            if safety_ratio > fail_ratio:
                self.centerline_escape_counter += 1
            else:
                self.centerline_escape_counter = 0
        else:
            self.centerline_escape_counter = 0
            fallback_dist = float(getattr(self, "out_of_vessel_fallback_distance", 0.012))
            self.current_out_of_vessel = bool(np.isfinite(centerline_dist) and centerline_dist >= fallback_dist)

        # Store cross-section offsets for critic/logging.
        try:
            frame = self._build_local_centerline_frame(centerline_tangent)
            offset_world = np.asarray(tip_pos, dtype=np.float32) - np.asarray(centerline_proj, dtype=np.float32)
            offset_local = self._world_vec_to_local(offset_world, frame)
            lr = float(local_radius) if local_radius is not None and np.isfinite(local_radius) and local_radius > 1e-9 else float(getattr(self, "default_local_radius", 0.005))
            self.current_centerline_offset_N_over_radius = float(offset_local[1] / lr)
            self.current_centerline_offset_B_over_radius = float(offset_local[2] / lr)
            self.current_centerline_local_radius_norm = float(lr / max(float(getattr(self, "radius_observation_scale", 0.005)), 1e-9))
            radial_offset = float(np.sqrt(offset_local[1] ** 2 + offset_local[2] ** 2))
            self.current_centerline_safety_margin = float(1.0 - (radial_offset + catheter_radius) / lr)
        except Exception:
            pass

        out_soft_distance = float(getattr(self, "out_of_centerline_soft_distance", 0.020))
        self.current_out_of_centerline_soft = bool(np.isfinite(centerline_dist) and centerline_dist > out_soft_distance)
        hard_dist = float(getattr(self, "severe_divergence_distance", 0.040))
        if np.isfinite(centerline_dist) and centerline_dist > hard_dist:
            self.severe_centerline_divergence_counter = int(getattr(self, "severe_centerline_divergence_counter", 0)) + 1
        else:
            self.severe_centerline_divergence_counter = 0
        if self.severe_centerline_divergence_counter >= max(1, int(getattr(self, "severe_divergence_patience", 10))):
            self.severe_centerline_divergence = True

        reward_features["tip_pos_distance_to_dest_pos"] = current_dist
        # High-watermark progress reward: only reward new maximum centerline progress.
        # This prevents forward/backward dancing from collecting the same progress reward repeatedly.
        if self.using_centerline_reward:
            prev_max_progress = float(getattr(self, "max_progress_this_episode", 0.0))
            delta_highwater = max(float(centerline_progress) - prev_max_progress, 0.0)
            delta_highwater = float(np.clip(delta_highwater, 0.0, self.progress_clip))
            reward_features["centerline_progress_distance"] = delta_highwater / 0.001
            if float(centerline_progress) > prev_max_progress:
                self.max_progress_this_episode = float(centerline_progress)
        else:
            reward_features["centerline_progress_distance"] = 0.0

        # Small time pressure to prefer faster completion once progress reward is capped by the high-watermark.
        reward_features["step_penalty"] = 1.0

        if terminal_zone_active:
            reward_features["delta_tip_pos_distance_to_dest_pos"] = clipped_delta_dist / self.euclidean_reward_scale
        else:
            reward_features["delta_tip_pos_distance_to_dest_pos"] = (
                self.outside_terminal_euclidean_weight * clipped_delta_dist / self.euclidean_reward_scale
            )

        # Wall soft penalty is fully disabled.
        # Out-of-vessel is the hard safety signal: once the vessel-exit proxy is
        # triggered, give the configured out_of_vessel_penalty and terminate the
        # current episode. The penalty magnitude remains controlled only by
        # reward_amount_dict["out_of_vessel_penalty"].
        reward_features["wall_soft_penalty"] = 0.0
        out_of_vessel_now = bool(getattr(self, "current_out_of_vessel", False))
        reward_features["out_of_vessel_penalty"] = 1.0 if out_of_vessel_now else 0.0

        if out_of_vessel_now:
            self.out_of_vessel_this_episode = True
            self.out_of_vessel_failure = True

        # No-progress penalty / termination is fully disabled.
        # Keep the counter only as a diagnostic metric; it no longer affects reward
        # or episode termination.
        if self.using_centerline_reward and max(delta_centerline_progress, 0.0) < float(getattr(self, "no_progress_delta_threshold", 0.00002)):
            self.no_progress_counter = int(getattr(self, "no_progress_counter", 0)) + 1
        else:
            self.no_progress_counter = 0
        self.no_progress_failure = False
        reward_features["no_progress_penalty"] = 0.0

        # Gate is disabled; keep diagnostic values neutral.
        self.current_gate_idx = 0
        self.current_next_gate_idx = 0
        self.current_gate_progress_ratio = 0.0
        self.current_gate_passed_this_step = False
        self.current_gate_pass_count_step = 0
        self.current_next_gate_approach = 0.0
        reward_features["gate_progress"] = 0.0
        reward_features["next_gate_approach"] = 0.0

        # Success: close to target, inside vessel proxy, and sufficiently advanced
        # along the accepted high-watermark centerline progress. Do NOT use the
        # instantaneous projection ratio here, because a nearest-centerline jump in
        # a U-bend could otherwise satisfy success without real topological progress.
        min_success_progress_ratio = float(getattr(self, "min_success_centerline_progress_ratio", 0.90))
        valid_inside_vessel = not bool(getattr(self, "current_out_of_vessel", False))
        if total_len > 1e-9:
            accepted_centerline_progress_ratio = float(
                np.clip(float(getattr(self, "max_progress_this_episode", 0.0)) / total_len, 0.0, 1.0)
            )
        else:
            accepted_centerline_progress_ratio = 0.0
        self.accepted_centerline_progress_ratio = accepted_centerline_progress_ratio
        valid_centerline_progress = accepted_centerline_progress_ratio >= min_success_progress_ratio
        reward_features["successful_task"] = 0.0
        if current_dist <= self.target_distance_threshold and valid_inside_vessel and valid_centerline_progress:
            reward_features["successful_task"] = 1.0
            self.episode_success = True
            self.episode_success_2mm = bool(current_dist <= 0.002 + 1e-12)
            self.is_out_of_bounds = True

        keep_reward_keys = (
            "tip_pos_distance_to_dest_pos",
            "centerline_progress_distance",
            "delta_tip_pos_distance_to_dest_pos",
            "successful_task",
            "wall_soft_penalty",
            "out_of_vessel_penalty",
            "no_progress_penalty",
            "step_penalty",
            "gate_progress",
            "next_gate_approach",
        )
        reward_features = {k: float(reward_features.get(k, 0.0)) for k in keep_reward_keys}
        return reward_features

    def _get_reward(self) -> float:
        reward = 0.0
        self.reward_info = {}
        reward_features = self._get_reward_features(previous_reward_features=self.reward_features)
        self.reward_features = reward_features.copy()

        for key, value in reward_features.items():
            value = self.reward_amount_dict[key] * value
            # Distance/progress features are already normalized in _get_reward_features.
            # Do not multiply by vessel bbox_diag; it makes progress rewards inconsistent
            # across long/short or curved/straight vessels.
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
        done_by_out_of_vessel = bool(getattr(self, "out_of_vessel_failure", False))
        done_by_centerline_escape = bool(getattr(self, "centerline_escape_failed", False))
        done_by_severe_divergence = bool(getattr(self, "severe_centerline_divergence", False))
        done_by_no_progress = bool(getattr(self, "no_progress_failure", False))
        done_by_non_finite = bool(getattr(self, "non_finite_failure", False))

        if done_by_target:
            terminal_reason = "target"
        elif done_by_timeout:
            terminal_reason = "timeout"
        elif done_by_terminal_escape:
            terminal_reason = "terminal_escape"
        elif done_by_out_of_vessel:
            terminal_reason = "out_of_vessel"
        elif terminated and done_by_severe_divergence:
            terminal_reason = "severe_divergence"
        elif done_by_centerline_escape:
            terminal_reason = "centerline_escape"
        elif done_by_no_progress:
            terminal_reason = "no_progress"
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

        self.info["min_dist_to_goal"] = min_dist
        self.info["current_dist_to_goal"] = current_dist
        self.info["final_dist_to_goal"] = current_dist if (terminated or truncated) else np.nan
        self.info["target_distance_threshold"] = float(self.target_distance_threshold)

        self.info["task_id"] = task_id
        self.info["chosen_model"] = chosen_model
        self.info["centerline_vtk"] = centerline_vtk
        self.info["sampling_model"] = str(getattr(self, "current_sampling_model", chosen_model))

        model_stats = self.model_success_stats.get(str(chosen_model), None)
        if model_stats is not None:
            episodes = int(model_stats["episodes"])
            successes = int(model_stats["successes"])
            self.info["model_episodes"] = float(episodes)
            self.info["model_success_rate"] = float(successes / episodes) if episodes > 0 else 0.0

        model_prob = self.model_sampling_probs.get(str(chosen_model), np.nan)
        self.info["model_sampling_prob"] = float(model_prob) if np.isfinite(model_prob) else np.nan

        self.info["terminal_reason"] = terminal_reason
        self.info["done_by_target"] = done_by_target
        self.info["done_by_timeout"] = done_by_timeout
        self.info["done_by_terminal_escape"] = done_by_terminal_escape
        self.info["done_by_out_of_vessel"] = done_by_out_of_vessel
        self.info["done_by_severe_divergence"] = done_by_severe_divergence
        self.info["done_by_no_progress"] = done_by_no_progress
        self.info["done_by_non_finite"] = done_by_non_finite

        self.info["done_by_centerline_escape"] = done_by_centerline_escape
        self.info["centerline_escape_counter"] = int(getattr(self, "centerline_escape_counter", 0))
        self.info["centerline_escape_patience"] = int(getattr(self, "centerline_escape_patience", 5))
        self.info["centerline_safe_ratio"] = float(getattr(self, "centerline_safe_ratio", 1.00))
        self.info["centerline_fail_ratio"] = float(getattr(self, "centerline_fail_ratio", 1.08))
        self.info["centerline_local_radius"] = float(getattr(self, "current_centerline_local_radius", np.nan))
        self.info["centerline_safety_ratio"] = float(getattr(self, "current_centerline_safety_ratio", np.nan))
        self.info["centerline_safety_margin"] = float(getattr(self, "current_centerline_safety_margin", np.nan))
        self.info["centerline_offset_N_over_radius"] = float(getattr(self, "current_centerline_offset_N_over_radius", np.nan))
        self.info["centerline_offset_B_over_radius"] = float(getattr(self, "current_centerline_offset_B_over_radius", np.nan))
        self.info["centerline_local_radius_norm"] = float(getattr(self, "current_centerline_local_radius_norm", np.nan))
        self.info["centerline_safety_ratio_max_episode"] = float(getattr(self, "max_safety_ratio_this_episode", np.nan))
        self.info["centerline_safety_margin_min_episode"] = float(getattr(self, "min_safety_margin_this_episode", np.nan))
        self.info["no_progress_counter"] = int(getattr(self, "no_progress_counter", 0))
        self.info["no_progress_patience"] = int(getattr(self, "no_progress_patience", 20))
        self.info["no_progress_terminate_patience"] = int(getattr(self, "no_progress_terminate_patience", 250))

        self.info["centerline_projection_raw_jump"] = float(getattr(self, "centerline_projection_raw_jump", np.nan))
        self.info["centerline_projection_selected_jump"] = float(getattr(self, "centerline_projection_selected_jump", np.nan))
        self.info["centerline_projection_jump_limited"] = bool(getattr(self, "centerline_projection_jump_limited", False))
        self.info["centerline_projection_limited_count"] = int(getattr(self, "centerline_projection_limited_count", 0))
        self.info["centerline_projection_call_count"] = int(getattr(self, "centerline_projection_call_count", 0))
        call_count = max(1, int(getattr(self, "centerline_projection_call_count", 0)))
        self.info["centerline_projection_limited_rate"] = float(
            int(getattr(self, "centerline_projection_limited_count", 0)) / call_count
        )
        self.info["centerline_projection_frozen"] = bool(getattr(self, "centerline_projection_frozen", False))
        self.info["centerline_projection_freeze_count"] = int(getattr(self, "centerline_projection_freeze_count", 0))
        self.info["centerline_projection_freeze_rate"] = float(
            int(getattr(self, "centerline_projection_freeze_count", 0)) / call_count
        )

        self.info["centerline_progress"] = float(getattr(self, "current_centerline_progress", np.nan))
        self.info["centerline_delta_progress"] = float(getattr(self, "current_centerline_delta_progress", np.nan))
        self.info["centerline_delta_progress_ratio"] = float(getattr(self, "current_centerline_delta_progress_ratio", np.nan))
        self.info["centerline_distance"] = float(getattr(self, "current_centerline_distance", np.nan))
        self.info["centerline_progress_ratio"] = float(getattr(self, "current_centerline_progress_ratio", np.nan))
        self.info["accepted_centerline_progress_ratio"] = float(getattr(self, "accepted_centerline_progress_ratio", np.nan))
        self.info["max_centerline_progress_this_episode"] = float(getattr(self, "max_progress_this_episode", np.nan))
        self.info["gate_idx"] = int(getattr(self, "current_gate_idx", 0))
        self.info["gate_next_idx"] = int(getattr(self, "current_next_gate_idx", 1))
        self.info["gate_num"] = int(len(getattr(self, "gate_points", []))) if getattr(self, "gate_points", None) is not None else 0
        self.info["gate_progress_ratio"] = float(getattr(self, "current_gate_progress_ratio", 0.0))
        self.info["gate_passed_this_step"] = bool(getattr(self, "current_gate_passed_this_step", False))
        self.info["gate_pass_count_episode"] = int(getattr(self, "current_gate_pass_count_episode", 0))
        self.info["gate_pass_count_step"] = int(getattr(self, "current_gate_pass_count_step", 0))
        self.info["gate_invalid_bypass"] = bool(getattr(self, "gate_invalid_bypass", False))
        self.info["next_gate_signed_dist"] = float(getattr(self, "current_next_gate_signed_dist", np.nan))
        self.info["next_gate_lateral_dist"] = float(getattr(self, "current_next_gate_lateral_dist", np.nan))
        self.info["next_gate_pass_radius"] = float(getattr(self, "current_next_gate_pass_radius", np.nan))
        self.info["next_gate_approach"] = float(getattr(self, "current_next_gate_approach", np.nan))
        self.info["centerline_reversed_for_progress"] = bool(getattr(self, "centerline_reversed_for_progress", False))
        self.info["using_centerline_reward"] = bool(getattr(self, "using_centerline_reward", False))
        self.info["using_euclidean_terminal_reward"] = bool(getattr(self, "using_euclidean_terminal_reward", False))
        self.info["projection_reliable"] = bool(getattr(self, "current_projection_reliable", True))
        self.info["projection_reliable_distance"] = float(getattr(self, "projection_reliable_distance", np.nan))
        self.info["severe_divergence_counter"] = int(getattr(self, "severe_centerline_divergence_counter", 0))
        self.info["severe_divergence_distance"] = float(getattr(self, "severe_divergence_distance", np.nan))
        self.info["out_of_centerline_soft_distance"] = float(getattr(self, "out_of_centerline_soft_distance", np.nan))
        self.info["out_of_centerline_soft"] = bool(getattr(self, "current_out_of_centerline_soft", False))
        self.info["out_of_vessel"] = bool(getattr(self, "current_out_of_vessel", False))
        self.info["out_of_vessel_this_episode"] = bool(getattr(self, "out_of_vessel_this_episode", False))
        self.info["out_of_vessel_safety_ratio"] = float(getattr(self, "out_of_vessel_safety_ratio", np.nan))
        self.info["min_success_centerline_progress_ratio"] = float(getattr(self, "min_success_centerline_progress_ratio", 0.90))
        self.info["rule_insert"] = float(getattr(self, "current_rule_insert", np.nan))
        self.info["raw_insert"] = float(getattr(self, "current_raw_insert", np.nan))
        self.info["effective_insert"] = float(getattr(self, "current_effective_insert", np.nan))
        self.info["insert_gate_scale"] = float(getattr(self, "current_insert_gate_scale", np.nan))
        self.info["turn_angle_deg"] = float(getattr(self, "current_turn_angle_deg", np.nan))
        self.info["forward_alignment"] = float(getattr(self, "current_forward_alignment", np.nan))
        self.info["forward_misalignment"] = float(getattr(self, "current_forward_misalignment", np.nan))
        self.info["turn_angle_norm"] = float(getattr(self, "current_turn_angle_norm", np.nan))
        self.info["goal_alignment"] = float(getattr(self, "current_goal_alignment", np.nan))
        self.info["goal_misalignment"] = float(getattr(self, "current_goal_misalignment", np.nan))
        self.info["non_finite_failure"] = bool(getattr(self, "non_finite_failure", False))


        self.info["privileged_obs"] = self._get_privileged_observation()
        self.info["privileged_obs_dim"] = int(getattr(self, "privileged_observation_dim", 0))
        self.info["actor_obs_dim"] = int(getattr(self, "actor_observation_dim", 0))

        return {**self.info, **self.reward_info, **self.episode_info, **self.reward_features}

    def _get_vessel_section_features(self, tip_pos: np.ndarray) -> np.ndarray:
        """Return 4-D safety-aware features in the local centerline cross-section.

        Layout:
            [offset_N_over_radius, offset_B_over_radius, local_radius_norm, safety_margin]

        This does NOT force the policy to stay on the centerline. It gives the policy
        enough information to know where the tip is inside the local vessel section and
        how much wall-clearance remains.
        """

        features = np.zeros(self.vessel_section_feature_dim, dtype=np.float32)

        if self.centerline_points is None or len(self.centerline_points) < 2:
            return features
        if getattr(self, "centerline_cumlength", None) is None:
            return features

        progress, seg_idx, centerline_dist, proj, tangent = self._get_centerline_projection_state(tip_pos)
        if seg_idx < 0:
            return features

        local_radius = self._get_current_local_radius(seg_idx)
        if local_radius is None or (not np.isfinite(local_radius)) or local_radius <= 1e-9:
            local_radius = float(getattr(self, "default_local_radius", 0.005))

        frame = self._build_local_centerline_frame(tangent)
        offset_world = np.asarray(tip_pos, dtype=np.float32) - np.asarray(proj, dtype=np.float32)
        offset_local = self._world_vec_to_local(offset_world, frame)

        local_radius = max(float(local_radius), 1e-9)
        radius_scale = max(float(getattr(self, "radius_observation_scale", 0.005)), 1e-9)
        catheter_radius = float(getattr(self, "catheter_radius", 0.000665))

        offset_N_over_radius = float(offset_local[1] / local_radius)
        offset_B_over_radius = float(offset_local[2] / local_radius)
        radial_offset = float(np.sqrt(offset_local[1] ** 2 + offset_local[2] ** 2))

        safety_ratio = float((radial_offset + catheter_radius) / local_radius)
        safety_margin = float(1.0 - safety_ratio)
        local_radius_norm = float(local_radius / radius_scale)

        # Store for info/logging. These are observation-side values.
        self.current_centerline_offset_N_over_radius = float(offset_N_over_radius)
        self.current_centerline_offset_B_over_radius = float(offset_B_over_radius)
        self.current_centerline_local_radius_norm = float(local_radius_norm)
        self.current_centerline_safety_margin = float(safety_margin)

        features[0] = float(np.clip(offset_N_over_radius, -3.0, 3.0))
        features[1] = float(np.clip(offset_B_over_radius, -3.0, 3.0))
        features[2] = float(np.clip(local_radius_norm, 0.0, 5.0))
        features[3] = float(np.clip(safety_margin, -3.0, 1.0))

        return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def _get_centerline_light_features(self, tip_pos: np.ndarray) -> np.ndarray:
        """Return local, normalized centerline features.

        Layout:
            [progress_ratio, lookahead_1_local_norm(3), ..., lookahead_12_local_norm(3)]

        Each lookahead vector is expressed in the current local centerline frame:
            [forward, lateral_1, lateral_2] / 0.05
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
        """Project tip onto centerline with optional arc-length continuity limiting.

        A pure nearest-segment projection can jump to a far-away centerline segment in
        curved or bifurcated vessels. That corrupts progress, lookahead, radius, and
        safety-ratio signals. We therefore keep the selected projection within
        `centerline_projection_max_jump` meters of the previous accepted progress
        whenever a previous progress exists.
        """
        if self.centerline_points is None or len(self.centerline_points) < 2:
            return 0.0, -1, 0.0, np.asarray(tip_pos, dtype=np.float32), np.zeros(3, dtype=np.float32)
        if getattr(self, "centerline_cumlength", None) is None:
            return 0.0, -1, 0.0, np.asarray(tip_pos, dtype=np.float32), np.zeros(3, dtype=np.float32)

        points = np.asarray(self.centerline_points, dtype=np.float32)
        seg_start = points[:-1]
        seg_end = points[1:]
        seg_vec = seg_end - seg_start

        tip_pos = np.asarray(tip_pos, dtype=np.float32)
        tip_vec = tip_pos[None, :] - seg_start

        seg_len_sq = np.sum(seg_vec * seg_vec, axis=1)
        seg_len_sq = np.maximum(seg_len_sq, 1e-12)

        t = np.sum(tip_vec * seg_vec, axis=1) / seg_len_sq
        t = np.clip(t, 0.0, 1.0)

        proj = seg_start + t[:, None] * seg_vec
        dists = np.linalg.norm(proj - tip_pos[None, :], axis=1)

        seg_lengths = np.linalg.norm(seg_vec, axis=1)
        progress_candidates = self.centerline_cumlength[:-1] + t * seg_lengths

        raw_best_idx = int(np.argmin(dists))
        raw_progress = float(progress_candidates[raw_best_idx])

        prev_progress = getattr(self, "previous_centerline_progress", None)

        selected_idx = raw_best_idx
        limited = False
        raw_jump = 0.0
        selected_jump = 0.0

        # Prefer a local arc-length search window around the previous accepted
        # progress instead of unconstrained global nearest projection. This avoids
        # snapping to a different spatially-close but topologically-wrong segment
        # in curved or branched vessels.
        if prev_progress is not None and np.isfinite(prev_progress):
            prev_progress = float(prev_progress)
            progress_diff = np.abs(progress_candidates - prev_progress)
            raw_jump = float(abs(raw_progress - prev_progress))

            back_window = float(getattr(self, "centerline_projection_back_window", 0.005))
            forward_window = float(getattr(self, "centerline_projection_forward_window", 0.030))
            if back_window > 0.0 or forward_window > 0.0:
                progress_delta = progress_candidates - prev_progress
                valid_window = np.where((progress_delta >= -abs(back_window)) & (progress_delta <= abs(forward_window)))[0]
                if valid_window.size > 0:
                    selected_idx = int(valid_window[np.argmin(dists[valid_window])])
                else:
                    search_window = float(getattr(self, "centerline_projection_search_window", 0.020))
                    if search_window > 0.0:
                        valid_window = np.where(progress_diff <= search_window)[0]
                        if valid_window.size > 0:
                            selected_idx = int(valid_window[np.argmin(dists[valid_window])])

            # Optional small-step limiter kept as an additional stabilizer. It is
            # applied after local-window selection and only changes the selected
            # index when the accepted progress would still jump too far.
            max_jump = float(getattr(self, "centerline_projection_max_jump", 0.002))
            selected_jump_candidate = float(abs(float(progress_candidates[selected_idx]) - prev_progress))
            if max_jump > 0.0 and selected_jump_candidate > max_jump:
                valid_jump = np.where(progress_diff <= max_jump)[0]
                if valid_jump.size > 0:
                    selected_idx = int(valid_jump[np.argmin(dists[valid_jump])])
                    limited = True
                else:
                    # Hard continuity constraint: never accept an illegal projection
                    # jump. If no candidate lies within max_jump, freeze the
                    # projection at the previous accepted arc-length progress.
                    # This prevents the progress-credit-card exploit and keeps
                    # actor lookahead from flickering across U-bend branches.
                    freeze_progress = float(np.clip(prev_progress, float(self.centerline_cumlength[0]), float(self.centerline_cumlength[-1])))
                    freeze_proj = self._interpolate_centerline_point_at_progress(freeze_progress)
                    freeze_tangent = self._interpolate_centerline_tangent_at_progress(freeze_progress)
                    freeze_idx = int(np.clip(np.searchsorted(self.centerline_cumlength, freeze_progress, side="right") - 1, 0, len(points) - 2))
                    freeze_dist = float(np.linalg.norm(np.asarray(freeze_proj, dtype=np.float32) - tip_pos))

                    self.centerline_projection_raw_jump = float(raw_jump)
                    self.centerline_projection_selected_jump = 0.0
                    self.centerline_projection_jump_limited = True
                    self.centerline_projection_frozen = True
                    self.centerline_projection_call_count = int(getattr(self, "centerline_projection_call_count", 0)) + 1
                    self.centerline_projection_limited_count = int(getattr(self, "centerline_projection_limited_count", 0)) + 1
                    self.centerline_projection_freeze_count = int(getattr(self, "centerline_projection_freeze_count", 0)) + 1
                    return freeze_progress, freeze_idx, freeze_dist, np.asarray(freeze_proj, dtype=np.float32), np.asarray(freeze_tangent, dtype=np.float32)

        best_seg_idx = int(selected_idx)
        best_t = float(t[best_seg_idx])
        closest_dist = float(dists[best_seg_idx])
        best_proj = proj[best_seg_idx]

        seg = seg_vec[best_seg_idx]
        seg_length = float(np.linalg.norm(seg))
        tangent = seg / (seg_length + 1e-9)
        progress = float(self.centerline_cumlength[best_seg_idx] + best_t * seg_length)

        if prev_progress is not None and np.isfinite(prev_progress):
            selected_jump = float(abs(progress - float(prev_progress)))
        else:
            selected_jump = 0.0

        self.centerline_projection_raw_jump = float(raw_jump)
        self.centerline_projection_selected_jump = float(selected_jump)
        self.centerline_projection_jump_limited = bool(limited)
        self.centerline_projection_frozen = False
        self.centerline_projection_call_count = int(getattr(self, "centerline_projection_call_count", 0)) + 1
        if limited:
            self.centerline_projection_limited_count = int(
                getattr(self, "centerline_projection_limited_count", 0)
            ) + 1

        return progress, best_seg_idx, closest_dist, best_proj.astype(np.float32), tangent.astype(np.float32)

    def _build_local_centerline_frame(self, tangent: np.ndarray) -> np.ndarray:
        T = np.asarray(tangent, dtype=np.float32)
        T_norm = float(np.linalg.norm(T))
        if T_norm < 1e-9:
            return np.eye(3, dtype=np.float32)
        T = T / T_norm

        ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if abs(float(np.dot(T, ref))) > 0.90:
            ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        N = np.cross(ref, T)
        N_norm = float(np.linalg.norm(N))
        if N_norm < 1e-9:
            return np.eye(3, dtype=np.float32)
        N = N / N_norm

        B = np.cross(T, N)
        B_norm = float(np.linalg.norm(B))
        if B_norm < 1e-9:
            return np.eye(3, dtype=np.float32)
        B = B / B_norm

        return np.stack([T, N, B], axis=1).astype(np.float32)

    def _world_vec_to_local(self, v_world: np.ndarray, frame: np.ndarray) -> np.ndarray:
        v_world = np.asarray(v_world, dtype=np.float32)
        frame = np.asarray(frame, dtype=np.float32)
        return np.array(
            [
                np.dot(v_world, frame[:, 0]),
                np.dot(v_world, frame[:, 1]),
                np.dot(v_world, frame[:, 2]),
            ],
            dtype=np.float32,
        )

    def _quat_rotate_vector(self, quat_xyzw: np.ndarray, vec: np.ndarray) -> np.ndarray:
        q = np.asarray(quat_xyzw, dtype=np.float32)
        v = np.asarray(vec, dtype=np.float32)

        if q.shape[0] != 4:
            return v.astype(np.float32)

        q_norm = float(np.linalg.norm(q))
        if q_norm < 1e-9:
            return v.astype(np.float32)

        q = q / q_norm
        q_xyz = q[:3]
        q_w = float(q[3])

        t = 2.0 * np.cross(q_xyz, v)
        rotated = v + q_w * t + np.cross(q_xyz, t)
        return rotated.astype(np.float32)

    def _get_centerline_arc_progress(self, tip_pos: np.ndarray) -> Tuple[float, int, float]:
        progress, best_seg_idx, closest_dist, _, _ = self._get_centerline_projection_state(tip_pos)
        return progress, best_seg_idx, closest_dist

    def _get_closest_centerline_index_and_distance(self, tip_pos: np.ndarray) -> Tuple[int, float]:
        if self.centerline_points is None or len(self.centerline_points) == 0:
            return -1, 0.0
        distances = np.linalg.norm(self.centerline_points - tip_pos, axis=1)
        closest_idx = int(np.argmin(distances))
        return closest_idx, float(distances[closest_idx])

    def _interpolate_centerline_point_at_progress(self, query_progress: float) -> np.ndarray:
        """Interpolate a centerline point by arc-length progress in meters."""
        points = getattr(self, "centerline_points", None)
        cum = getattr(self, "centerline_cumlength", None)
        if points is None or cum is None or len(points) == 0 or len(cum) == 0:
            return np.zeros(3, dtype=np.float32)

        points = np.asarray(points, dtype=np.float32)
        cum = np.asarray(cum, dtype=np.float32)
        q = float(np.clip(float(query_progress), float(cum[0]), float(cum[-1])))
        x = float(np.interp(q, cum, points[:, 0]))
        y = float(np.interp(q, cum, points[:, 1]))
        z = float(np.interp(q, cum, points[:, 2]))
        return np.array([x, y, z], dtype=np.float32)

    def _interpolate_centerline_tangent_at_progress(self, query_progress: float) -> np.ndarray:
        """Return a unit centerline tangent at the queried arc-length progress."""
        points = getattr(self, "centerline_points", None)
        cum = getattr(self, "centerline_cumlength", None)
        if points is None or cum is None or len(points) < 2 or len(cum) < 2:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)

        points = np.asarray(points, dtype=np.float32)
        cum = np.asarray(cum, dtype=np.float32)
        q = float(np.clip(float(query_progress), float(cum[0]), float(cum[-1])))

        # Use a small central difference on the arc-length parameter. This makes
        # the tangent smoother than selecting a single raw segment, while keeping
        # the returned vector normalized and non-collapsing in U-bends.
        half_window = 0.0005  # 0.5 mm
        q0 = float(np.clip(q - half_window, float(cum[0]), float(cum[-1])))
        q1 = float(np.clip(q + half_window, float(cum[0]), float(cum[-1])))
        if abs(q1 - q0) < 1e-9:
            idx = int(np.clip(np.searchsorted(cum, q, side="right") - 1, 0, len(points) - 2))
            tangent = points[idx + 1] - points[idx]
        else:
            p0 = self._interpolate_centerline_point_at_progress(q0)
            p1 = self._interpolate_centerline_point_at_progress(q1)
            tangent = p1 - p0

        n = float(np.linalg.norm(tangent))
        if n < 1e-9:
            idx = int(np.clip(np.searchsorted(cum, q, side="right") - 1, 0, len(points) - 2))
            tangent = points[idx + 1] - points[idx]
            n = float(np.linalg.norm(tangent))
        if n < 1e-9:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return (tangent / n).astype(np.float32)

    def _interpolate_centerline_radius_at_progress(self, query_progress: float) -> float:
        """Interpolate the centerline Radius array by arc-length progress."""
        radius = getattr(self, "centerline_radius", None)
        cum = getattr(self, "centerline_cumlength", None)
        if radius is None or cum is None or len(radius) == 0 or len(cum) == 0:
            return float(getattr(self, "default_local_radius", 0.005))
        radius = np.asarray(radius, dtype=np.float32).reshape(-1)
        cum = np.asarray(cum, dtype=np.float32).reshape(-1)
        if len(radius) != len(cum):
            idx = int(np.clip(np.searchsorted(cum, float(query_progress)), 0, len(radius) - 1))
            value = float(radius[idx])
        else:
            q = float(np.clip(float(query_progress), float(cum[0]), float(cum[-1])))
            value = float(np.interp(q, cum, radius))
        if not np.isfinite(value) or value <= 1e-9:
            value = float(getattr(self, "default_local_radius", 0.005))
        return value

    def _build_gate_progress_map(self) -> None:
        """Create ordered 2 mm cross-section gates from the centerline.

        Gate i is a circular disk:
          center  = gate_points[i]
          normal  = normalize(gate_points[i + 1] - gate_points[i])
          radius  = Radius at that centerline progress

        The pass radius is slightly smaller than the anatomical radius to avoid
        rewarding wall-hugging traversal. Progress is monotonic: only the next
        gate is checked and current_gate_idx only increases.
        """
        self.gate_points = None
        self.gate_normals = None
        self.gate_radius = None
        self.gate_pass_radius = None
        self.gate_progress = None

        points = getattr(self, "centerline_points", None)
        cum = getattr(self, "centerline_cumlength", None)
        if points is None or cum is None or len(points) < 2 or len(cum) < 2:
            return

        total_len = float(cum[-1])
        if not np.isfinite(total_len) or total_len <= 1e-9:
            return

        spacing = max(float(getattr(self, "gate_spacing", 0.002)), 1e-6)
        gate_progress = np.arange(0.0, total_len + 0.5 * spacing, spacing, dtype=np.float32)
        if gate_progress.size == 0 or float(gate_progress[-1]) < total_len:
            gate_progress = np.append(gate_progress, np.float32(total_len))
        gate_progress[-1] = np.float32(total_len)
        if gate_progress.size < 2:
            return

        gate_points = np.asarray(
            [self._interpolate_centerline_point_at_progress(float(q)) for q in gate_progress],
            dtype=np.float32,
        )
        gate_radius = np.asarray(
            [self._interpolate_centerline_radius_at_progress(float(q)) for q in gate_progress],
            dtype=np.float32,
        )

        normals = np.zeros_like(gate_points, dtype=np.float32)
        for i in range(len(gate_points)):
            if i < len(gate_points) - 1:
                v = gate_points[i + 1] - gate_points[i]
            else:
                v = gate_points[i] - gate_points[i - 1]
            n = float(np.linalg.norm(v))
            if n <= 1e-9:
                normals[i] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            else:
                normals[i] = (v / n).astype(np.float32)

        catheter_radius = float(getattr(self, "catheter_radius", 0.000665))
        margin = float(getattr(self, "gate_pass_margin", 0.0005))
        scale = float(getattr(self, "gate_pass_radius_scale", 0.80))
        pass_radius = []
        for r in gate_radius:
            rr = float(r) if np.isfinite(float(r)) and float(r) > 1e-9 else float(getattr(self, "default_local_radius", 0.005))
            conservative = scale * rr
            clearance = rr - catheter_radius - margin
            if clearance > 1e-9:
                pr = min(conservative, clearance)
            else:
                pr = conservative
            pass_radius.append(max(1e-6, float(pr)))

        self.gate_points = gate_points.astype(np.float32)
        self.gate_normals = normals.astype(np.float32)
        self.gate_radius = gate_radius.astype(np.float32)
        self.gate_pass_radius = np.asarray(pass_radius, dtype=np.float32)
        self.gate_progress = np.asarray(gate_progress, dtype=np.float32)

    def _get_active_gate_direction_world(self) -> np.ndarray:
        """Return the active gate-based forward direction in world coordinates.

        This replaces projection-derived tangents for actor/magnetic local frames.
        The direction follows the ordered gate sequence:
            current_gate -> next_gate.
        """
        points = getattr(self, "gate_points", None)
        normals = getattr(self, "gate_normals", None)

        if points is None or len(points) < 2:
            tangent = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            try:
                if getattr(self, "centerline_points", None) is not None and len(self.centerline_points) >= 2:
                    tangent = np.asarray(self.centerline_points[1] - self.centerline_points[0], dtype=np.float32)
            except Exception:
                pass
            n = float(np.linalg.norm(tangent))
            return (tangent / (n + 1e-9)).astype(np.float32)

        current_idx = int(np.clip(getattr(self, "current_gate_idx", 0), 0, len(points) - 1))
        next_idx = int(np.clip(getattr(self, "current_next_gate_idx", current_idx + 1), 0, len(points) - 1))

        if next_idx > current_idx:
            direction = np.asarray(points[next_idx] - points[current_idx], dtype=np.float32)
        elif normals is not None and len(normals) > 0:
            direction = np.asarray(normals[int(np.clip(current_idx, 0, len(normals) - 1))], dtype=np.float32)
        else:
            direction = np.asarray(points[-1] - points[-2], dtype=np.float32)

        n = float(np.linalg.norm(direction))
        if n < 1e-9:
            if normals is not None and len(normals) > 0:
                direction = np.asarray(normals[int(np.clip(current_idx, 0, len(normals) - 1))], dtype=np.float32)
            else:
                direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            n = float(np.linalg.norm(direction))
        return (direction / (n + 1e-9)).astype(np.float32)

    def _get_active_gate_frame(self) -> np.ndarray:
        """Build the local T/N/B frame from active gate direction, not projection."""
        return self._build_local_centerline_frame(self._get_active_gate_direction_world())

    def _get_active_gate_progress(self) -> float:
        """Return arc-length progress of current gate for lookahead sampling."""
        gate_progress = getattr(self, "gate_progress", None)
        if gate_progress is not None and len(gate_progress) > 0:
            idx = int(np.clip(getattr(self, "current_gate_idx", 0), 0, len(gate_progress) - 1))
            return float(gate_progress[idx])
        if getattr(self, "centerline_cumlength", None) is not None and len(self.centerline_cumlength) > 0:
            return float(self.centerline_cumlength[0])
        return 0.0

    def _get_active_gate_lead_direction(self, tip_pos: np.ndarray) -> np.ndarray:
        """Return direction from tip to next gate center, with gate-normal fallback."""
        points = getattr(self, "gate_points", None)
        if points is None or len(points) < 2:
            return self._get_active_gate_direction_world()

        next_idx = int(np.clip(getattr(self, "current_next_gate_idx", 1), 0, len(points) - 1))
        tip_pos = np.asarray(tip_pos, dtype=np.float32).reshape(3)
        vec = np.asarray(points[next_idx], dtype=np.float32) - tip_pos
        n = float(np.linalg.norm(vec))
        if n > 1e-9:
            return (vec / n).astype(np.float32)
        return self._get_active_gate_direction_world()

    def _compute_gate_metrics(self, tip_pos: np.ndarray, gate_idx: Optional[int] = None) -> dict:
        points = getattr(self, "gate_points", None)
        normals = getattr(self, "gate_normals", None)
        pass_radius = getattr(self, "gate_pass_radius", None)
        if points is None or normals is None or pass_radius is None or len(points) == 0:
            return {"signed": np.nan, "lateral": np.nan, "pass_radius": np.nan, "distance": np.nan}
        if gate_idx is None:
            gate_idx = int(getattr(self, "current_next_gate_idx", 1))
        gate_idx = int(np.clip(gate_idx, 0, len(points) - 1))
        x = np.asarray(tip_pos, dtype=np.float32).reshape(3)
        p = np.asarray(points[gate_idx], dtype=np.float32)
        t = np.asarray(normals[gate_idx], dtype=np.float32)
        t = t / (float(np.linalg.norm(t)) + 1e-9)
        v = x - p
        signed = float(np.dot(v, t))
        lateral_vec = v - signed * t
        lateral = float(np.linalg.norm(lateral_vec))
        dist = float(np.linalg.norm(v))
        pr = float(pass_radius[gate_idx])
        return {"signed": signed, "lateral": lateral, "pass_radius": pr, "distance": dist}

    def _gate_crossed_between(self, x_prev: np.ndarray, x_curr: np.ndarray, gate_idx: int) -> bool:
        points = getattr(self, "gate_points", None)
        normals = getattr(self, "gate_normals", None)
        pass_radius = getattr(self, "gate_pass_radius", None)
        if points is None or normals is None or pass_radius is None:
            return False
        if gate_idx < 0 or gate_idx >= len(points):
            return False

        p = np.asarray(points[gate_idx], dtype=np.float32)
        t = np.asarray(normals[gate_idx], dtype=np.float32)
        t = t / (float(np.linalg.norm(t)) + 1e-9)
        x_prev = np.asarray(x_prev, dtype=np.float32).reshape(3)
        x_curr = np.asarray(x_curr, dtype=np.float32).reshape(3)

        h_prev = float(np.dot(x_prev - p, t))
        h_curr = float(np.dot(x_curr - p, t))
        if not (h_prev < 0.0 and h_curr >= 0.0):
            return False

        denom = h_curr - h_prev
        if abs(denom) < 1e-12:
            return False
        alpha = float(np.clip(-h_prev / denom, 0.0, 1.0))
        x_cross = x_prev + alpha * (x_curr - x_prev)
        v_cross = x_cross - p
        signed_cross = float(np.dot(v_cross, t))
        lateral_vec = v_cross - signed_cross * t
        lateral_cross = float(np.linalg.norm(lateral_vec))
        return bool(lateral_cross <= float(pass_radius[gate_idx]))

    def _initialize_gate_progress_from_current_tip(self) -> None:
        """Set the starting gate index from the actual reset tip pose.

        This handles randomized starts without rebuilding the scene. A gate is
        considered already passed at reset only if the tip is on/after its plane
        and inside that gate disk. Episode pass count still starts from zero.
        """
        try:
            tip = np.asarray(self.mcr_controller_sofa.get_pos_quat_catheter_tip()[0:3], dtype=np.float32)
        except Exception:
            tip = np.zeros(3, dtype=np.float32)

        points = getattr(self, "gate_points", None)
        if points is None or len(points) < 2:
            self.current_gate_idx = 0
            self.current_next_gate_idx = 1
            self.current_gate_progress_ratio = 0.0
            self._last_gate_tip_pos = tip.copy()
            return

        passed_idx = 0
        for i in range(len(points) - 1):
            m = self._compute_gate_metrics(tip, gate_idx=i)
            if np.isfinite(m["signed"]) and m["signed"] >= 0.0 and m["lateral"] <= m["pass_radius"]:
                passed_idx = i
            else:
                # Since gates are ordered from entry to target and starts are near
                # the entry, stop at the first unpassed gate to avoid loop artifacts.
                if i > 0:
                    break

        self.current_gate_idx = int(np.clip(passed_idx, 0, len(points) - 1))
        self.current_next_gate_idx = int(min(self.current_gate_idx + 1, len(points) - 1))
        denom = max(1, len(points) - 1)
        self.current_gate_progress_ratio = float(self.current_gate_idx / denom)
        self.current_gate_passed_this_step = False
        self.current_gate_pass_count_episode = 0
        self.current_gate_pass_count_step = 0
        nm = self._compute_gate_metrics(tip, self.current_next_gate_idx)
        self.current_next_gate_signed_dist = float(nm["signed"])
        self.current_next_gate_lateral_dist = float(nm["lateral"])
        self.current_next_gate_pass_radius = float(nm["pass_radius"])
        self._last_gate_tip_pos = tip.copy()

    def _update_gate_progress(self, tip_pos: np.ndarray, current_out_of_vessel: Optional[bool] = None) -> int:
        """Finite-state update for ordered gate progress.

        Invariants:
        1. Only the current next gate can be passed. Old gates cannot be rewarded
           again, and future gates cannot be skipped.
        2. At most one gate can be passed per environment step.
        3. A gate can be passed either by a valid front-to-back crossing inside the
           disk, or by being already behind the plane and inside the disk. The
           second branch prevents deadlock when the first crossing is slightly
           outside the disk and the tip later corrects laterally.
        4. If the tip is outside the vessel proxy, gate state cannot advance.
        5. If the tip has gone outside while approaching this gate, it cannot collect
           the gate by re-entering behind the plane. It must first return to the
           front side of the same next gate, then cross from inside.
        """
        points = getattr(self, "gate_points", None)
        if points is None or len(points) < 2:
            self.current_gate_passed_this_step = False
            self.current_gate_pass_count_step = 0
            return 0

        curr = np.asarray(tip_pos, dtype=np.float32).reshape(3)
        prev = getattr(self, "_last_gate_tip_pos", None)
        if prev is None:
            self._initialize_gate_progress_from_current_tip()
            self._last_gate_tip_pos = curr.copy()
            return 0
        prev = np.asarray(prev, dtype=np.float32).reshape(3)

        next_idx = int(np.clip(getattr(self, "current_next_gate_idx", 1), 0, len(points) - 1))
        current_out = bool(getattr(self, "current_out_of_vessel", False)) if current_out_of_vessel is None else bool(current_out_of_vessel)

        def _refresh_metrics_and_return_zero() -> int:
            self.current_gate_pass_count_step = 0
            self.current_gate_passed_this_step = False
            denom = max(1, len(points) - 1)
            self.current_gate_progress_ratio = float(
                np.clip(float(getattr(self, "current_gate_idx", 0)) / float(denom), 0.0, 1.0)
            )
            nm = self._compute_gate_metrics(curr, int(getattr(self, "current_next_gate_idx", next_idx)))
            self.current_next_gate_signed_dist = float(nm["signed"])
            self.current_next_gate_lateral_dist = float(nm["lateral"])
            self.current_next_gate_pass_radius = float(nm["pass_radius"])
            self._last_gate_tip_pos = curr.copy()
            return 0

        # Outside-vessel states cannot advance the gate FSM. Mark the current next
        # gate as contaminated by a possible shortcut until the tip returns to the
        # front side of this same gate.
        if current_out:
            self.gate_invalid_bypass = True
            return _refresh_metrics_and_return_zero()

        # If a shortcut/outside segment happened before, require the tip to return
        # to the front side of the same next gate before any pass condition is
        # accepted. This prevents re-entering behind the gate and receiving reward.
        if bool(getattr(self, "gate_invalid_bypass", False)):
            m = self._compute_gate_metrics(curr, next_idx)
            signed = float(m.get("signed", np.nan))
            reset_margin = float(getattr(self, "gate_bypass_reset_margin", 1e-5))
            if np.isfinite(signed) and signed < -abs(reset_margin):
                self.gate_invalid_bypass = False
            else:
                return _refresh_metrics_and_return_zero()

        crossed_inside = bool(self._gate_crossed_between(prev, curr, next_idx))
        m_curr = self._compute_gate_metrics(curr, next_idx)
        current_inside_after = bool(
            np.isfinite(m_curr.get("signed", np.nan))
            and np.isfinite(m_curr.get("lateral", np.nan))
            and np.isfinite(m_curr.get("pass_radius", np.nan))
            and float(m_curr["signed"]) >= 0.0
            and float(m_curr["lateral"]) <= float(m_curr["pass_radius"])
        )

        passed = int(crossed_inside or current_inside_after)
        if passed > 0:
            self.current_gate_idx = int(next_idx)
            self.current_next_gate_idx = int(min(next_idx + 1, len(points) - 1))
            self.current_gate_pass_count_step = 1
            self.current_gate_passed_this_step = True
            self.current_gate_pass_count_episode = int(getattr(self, "current_gate_pass_count_episode", 0)) + 1
        else:
            self.current_gate_pass_count_step = 0
            self.current_gate_passed_this_step = False

        denom = max(1, len(points) - 1)
        self.current_gate_progress_ratio = float(np.clip(float(self.current_gate_idx) / float(denom), 0.0, 1.0))
        nm = self._compute_gate_metrics(curr, int(getattr(self, "current_next_gate_idx", 1)))
        self.current_next_gate_signed_dist = float(nm["signed"])
        self.current_next_gate_lateral_dist = float(nm["lateral"])
        self.current_next_gate_pass_radius = float(nm["pass_radius"])
        self._last_gate_tip_pos = curr.copy()
        return int(passed)

    def _get_actor_lookahead_relative_vectors(
        self,
        tip_pos: np.ndarray,
        progress: Optional[float] = None,
        frame: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return future centerline tangent directions in the current local frame.

        Unlike future-point relative vectors p(s+ds)-p_tip, these are unit
        tangent vectors T(s+ds). Their magnitude does not collapse in tight
        U-bends, so the actor sees a smooth upcoming rotation instead of a
        near-zero ambiguous vector.
        """
        n_points = int(getattr(self, "num_actor_lookahead_points", 10))
        dim = 3 * n_points
        out = np.zeros(dim, dtype=np.float32)

        if self.centerline_points is None or len(self.centerline_points) < 2:
            return out
        if getattr(self, "centerline_cumlength", None) is None:
            return out

        tip_pos = np.asarray(tip_pos, dtype=np.float32).reshape(3)
        if progress is None:
            progress, _, _, _, tangent = self._get_centerline_projection_state(tip_pos)
        if frame is None:
            _, _, _, _, tangent = self._get_centerline_projection_state(tip_pos)
            frame = self._build_local_centerline_frame(tangent)

        frame = np.asarray(frame, dtype=np.float32)
        offsets = tuple(getattr(
            self,
            "actor_lookahead_offsets",
            tuple((np.arange(1, n_points + 1, dtype=np.float32) * 0.001).tolist()),
        ))

        values = []
        for off in offsets[:n_points]:
            future_tangent_world = self._interpolate_centerline_tangent_at_progress(float(progress) + float(off))
            future_tangent_local = self._world_vec_to_local(future_tangent_world, frame)
            tn = float(np.linalg.norm(future_tangent_local))
            if tn > 1e-9:
                future_tangent_local = future_tangent_local / tn
            values.append(np.clip(future_tangent_local, -1.0, 1.0))

        if len(values) > 0:
            flat = np.asarray(values, dtype=np.float32).reshape(-1)
            out[:min(dim, flat.shape[0])] = flat[:dim]
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def _get_lookahead_relative_vectors(self, tip_pos: np.ndarray) -> np.ndarray:
        dim = 3 * self.num_centerline_lookahead_points
        if self.centerline_points is None or len(self.centerline_points) == 0:
            return np.zeros(dim, dtype=np.float32)
        if getattr(self, "centerline_cumlength", None) is None:
            return np.zeros(dim, dtype=np.float32)

        tip_pos = np.asarray(tip_pos, dtype=np.float32)
        progress, seg_idx, closest_dist, best_proj, tangent = self._get_centerline_projection_state(tip_pos)
        if seg_idx < 0:
            return np.zeros(dim, dtype=np.float32)

        frame = self._build_local_centerline_frame(tangent)
        points = np.asarray(self.centerline_points, dtype=np.float32)
        last_idx = len(points) - 1
        current_idx = int(np.clip(seg_idx, 0, last_idx))

        lookahead_vectors = []
        scale = float(getattr(self, "lookahead_observation_scale", 0.05))
        for offset in self.centerline_lookahead_offsets:
            idx = min(current_idx + int(offset), last_idx)
            v_world = points[idx] - tip_pos
            v_local = self._world_vec_to_local(v_world, frame)
            lookahead_vectors.append(v_local / scale)

        out = np.asarray(lookahead_vectors, dtype=np.float32).reshape(-1)
        if out.shape[0] != dim:
            out = np.pad(out, (0, max(0, dim - out.shape[0])))[:dim]
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

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
            return points

    def _resample_centerline_radius_1mm(self, raw_points: np.ndarray, raw_radius, resampled_points: np.ndarray):
        if raw_points is None or raw_radius is None or resampled_points is None:
            return None

        raw_points = np.asarray(raw_points, dtype=np.float32)
        raw_radius = np.asarray(raw_radius, dtype=np.float32).reshape(-1)
        resampled_points = np.asarray(resampled_points, dtype=np.float32)

        if raw_points.ndim != 2 or raw_points.shape[1] != 3:
            return None
        if resampled_points.ndim != 2 or resampled_points.shape[1] != 3:
            return None
        if len(raw_points) < 2 or len(raw_radius) != len(raw_points) or len(resampled_points) < 2:
            return None
        if not np.all(np.isfinite(raw_radius)):
            return None

        raw_seg = np.linalg.norm(np.diff(raw_points, axis=0), axis=1)
        raw_cum = np.concatenate(([0.0], np.cumsum(raw_seg))).astype(np.float32)
        raw_total = float(raw_cum[-1])

        res_seg = np.linalg.norm(np.diff(resampled_points, axis=0), axis=1)
        res_cum = np.concatenate(([0.0], np.cumsum(res_seg))).astype(np.float32)
        res_total = float(res_cum[-1])

        if raw_total < 1e-9 or res_total < 1e-9:
            return None

        target_raw_dist = res_cum / (res_total + 1e-12) * raw_total
        resampled_radius = np.interp(target_raw_dist, raw_cum, raw_radius).astype(np.float32)

        if len(resampled_radius) != len(resampled_points):
            return None

        return np.nan_to_num(resampled_radius, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def _get_current_local_radius(self, centerline_seg_idx: int):
        radius = getattr(self, "centerline_radius", None)
        points = getattr(self, "centerline_points", None)

        if radius is None or points is None:
            return None

        radius = np.asarray(radius, dtype=np.float32).reshape(-1)
        if len(radius) == 0 or len(points) == 0:
            return None

        idx = int(np.clip(centerline_seg_idx, 0, len(radius) - 1))
        local_radius = float(radius[idx])

        if not np.isfinite(local_radius) or local_radius <= 1e-6:
            return None

        return local_radius

    def _get_distance_tip_to_dest(self):
        tip = np.asarray(self.mcr_controller_sofa.get_pos_quat_catheter_tip()[0:3], dtype=np.float32)
        dist = np.linalg.norm(self.target_position - tip)
        if not np.isfinite(dist):
            self.non_finite_failure = True
            return float(1e3)
        return float(dist)

    def _estimate_centerline_turn_angle_deg(self, centerline_seg_idx: int) -> float:
        """Estimate upcoming centerline turn angle using multi-scale lookahead.

        The centerline is resampled to roughly 1 mm spacing in _init_sim, so integer
        offsets are approximately millimeters. We use the maximum angle among several
        future offsets as a conservative curvature indicator.
        """
        points = getattr(self, "centerline_points", None)
        if points is None or len(points) < 3 or centerline_seg_idx < 0:
            return 0.0

        idx = int(np.clip(centerline_seg_idx, 0, len(points) - 2))
        base_next = min(idx + 1, len(points) - 1)
        v0 = np.asarray(points[base_next] - points[idx], dtype=np.float32)
        v0_norm = float(np.linalg.norm(v0))
        if v0_norm < 1e-9:
            return 0.0
        v0 = v0 / v0_norm

        max_angle = 0.0
        for off in getattr(self, "turn_angle_offsets", (5, 10, 20, 30)):
            j0 = min(idx + int(off), len(points) - 2)
            j1 = min(j0 + 1, len(points) - 1)
            v1 = np.asarray(points[j1] - points[j0], dtype=np.float32)
            v1_norm = float(np.linalg.norm(v1))
            if v1_norm < 1e-9:
                continue
            v1 = v1 / v1_norm
            dot = float(np.clip(np.dot(v0, v1), -1.0, 1.0))
            angle = float(np.degrees(np.arccos(dot)))
            max_angle = max(max_angle, angle)
        return float(max_angle)

    def _get_centerline_lead_direction(
        self,
        tip_pos: np.ndarray,
        centerline_seg_idx: int,
        turn_angle_deg: float,
        fallback_tangent: np.ndarray,
    ):
        """Return a unified local lead direction for steering/alignment.

        There is no straight / medium / sharp bend branch here. The controller
        always uses a short lead point around 5 mm ahead of the current
        centerline projection. If the tip is off-center, the direction from tip
        to the lead point naturally pulls the tip back toward the centerline.
        """
        points = getattr(self, "centerline_points", None)
        if points is None or len(points) < 2 or centerline_seg_idx < 0:
            tangent = np.asarray(fallback_tangent, dtype=np.float32)
            n = float(np.linalg.norm(tangent))
            return tangent / n if n > 1e-9 else None

        idx0 = int(np.clip(centerline_seg_idx, 0, len(points) - 1))
        offset = int(max(1, getattr(self, "lead_idx_local", 5)))
        idx1 = int(np.clip(idx0 + offset, 0, len(points) - 1))

        p0 = np.asarray(points[idx0], dtype=np.float32)
        p1 = np.asarray(points[idx1], dtype=np.float32)
        tip_pos = np.asarray(tip_pos, dtype=np.float32)

        path_vec = p1 - p0
        recover_vec = p1 - tip_pos

        def _unit(v):
            v = np.asarray(v, dtype=np.float32)
            n = float(np.linalg.norm(v))
            return v / n if n > 1e-9 else None

        path_dir = _unit(path_vec)
        recover_dir = _unit(recover_vec)

        # By default use recover_dir, because it is tip-relative and naturally
        # includes both forward tracking and lateral recovery. A small path_dir
        # fallback keeps behavior stable when the tip is already close to p1.
        w_recover = float(np.clip(getattr(self, "lead_recovery_weight", 1.0), 0.0, 1.0))
        if path_dir is not None and recover_dir is not None:
            lead_dir = w_recover * recover_dir + (1.0 - w_recover) * path_dir
            lead_norm = float(np.linalg.norm(lead_dir))
            if lead_norm > 1e-9:
                return (lead_dir / lead_norm).astype(np.float32)
        if recover_dir is not None:
            return recover_dir.astype(np.float32)
        if path_dir is not None:
            return path_dir.astype(np.float32)

        tangent = np.asarray(fallback_tangent, dtype=np.float32)
        n = float(np.linalg.norm(tangent))
        return tangent / n if n > 1e-9 else None

    def _compute_current_steering_context(self):
        """Projection-frame steering context used for logging and optional insert gating."""
        try:
            tip_pose = np.asarray(self.mcr_controller_sofa.get_pos_quat_catheter_tip(), dtype=np.float32)
            tip_pos = tip_pose[0:3]
            tip_quat = tip_pose[3:7]
        except Exception:
            return {
                "projection_reliable": False,
                "turn_angle_deg": 0.0,
                "turn_angle_norm": 0.0,
                "lead_direction_world": np.zeros(3, dtype=np.float32),
                "forward_alignment": 0.0,
            }

        progress, seg_idx, centerline_dist, _, tangent = self._get_centerline_projection_state(tip_pos)
        frame = self._build_local_centerline_frame(tangent)
        lead_point = self._interpolate_centerline_point_at_progress(progress + float(getattr(self, "actor_lookahead_distance", 0.006)))
        lead_dir = np.asarray(lead_point, dtype=np.float32) - np.asarray(tip_pos, dtype=np.float32)
        n = float(np.linalg.norm(lead_dir))
        if n > 1e-9:
            lead_dir = lead_dir / n
        else:
            lead_dir = np.asarray(tangent, dtype=np.float32)

        tip_forward_world = self._quat_rotate_vector(tip_quat, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        fn = float(np.linalg.norm(tip_forward_world))
        if fn > 1e-9:
            tip_forward_world = tip_forward_world / fn
        forward_alignment = float(np.clip(np.dot(tip_forward_world, lead_dir), -1.0, 1.0))

        turn_angle_deg = float(self._estimate_centerline_turn_angle_deg(seg_idx)) if seg_idx >= 0 else 0.0
        turn_angle_norm = float(np.clip(turn_angle_deg / 90.0, 0.0, 1.0))
        projection_reliable = bool(
            seg_idx >= 0
            and np.isfinite(centerline_dist)
            and centerline_dist <= float(getattr(self, "projection_reliable_distance", 0.030))
        )
        self.current_forward_alignment = forward_alignment
        self.current_turn_angle_deg = turn_angle_deg
        self.current_turn_angle_norm = turn_angle_norm
        self.current_lead_direction_world = lead_dir.astype(np.float32)
        self.current_centerline_frame = frame.astype(np.float32)
        return {
            "projection_reliable": projection_reliable,
            "turn_angle_deg": turn_angle_deg,
            "turn_angle_norm": turn_angle_norm,
            "lead_direction_world": lead_dir.astype(np.float32),
            "forward_alignment": forward_alignment,
        }

    def _apply_insert_safety_shield(self, raw_insert: float, ctx: Optional[dict] = None) -> float:
        """Pass-through insert action.

        No alignment/projection safety cap is applied here. The SAC action is
        executed directly after the standard action-space clipping in step().
        """
        effective_insert = float(np.clip(raw_insert, -1.0, 1.0))
        self.current_raw_insert = effective_insert
        self.current_rule_insert = effective_insert
        self.current_effective_insert = effective_insert
        self.current_insert_gate_scale = 1.0
        return effective_insert

    def _apply_local_magnetic_action(self, rot_n: float, rot_b: float, tip_pos: np.ndarray) -> None:
        """Apply projection-frame steering action to the world-frame B-field."""
        try:
            _, _, _, _, tangent = self._get_centerline_projection_state(tip_pos)
            frame = self._build_local_centerline_frame(tangent)
            self.current_centerline_frame = frame.astype(np.float32)

            axis_n = np.asarray(frame[:, 1], dtype=np.float32)
            axis_b = np.asarray(frame[:, 2], dtype=np.float32)
            axis_n = axis_n / (float(np.linalg.norm(axis_n)) + 1e-9)
            axis_b = axis_b / (float(np.linalg.norm(axis_b)) + 1e-9)

            field = np.asarray(self.mcr_controller_sofa.get_mag_field_des(), dtype=np.float32)
            angle_scale = float(getattr(self, "local_field_action_angle", 2.0 * np.pi / 180.0))
            if abs(float(rot_n)) > 1e-8:
                field = R.from_rotvec(float(rot_n) * angle_scale * axis_n).apply(field).astype(np.float32)
            if abs(float(rot_b)) > 1e-8:
                field = R.from_rotvec(float(rot_b) * angle_scale * axis_b).apply(field).astype(np.float32)

            old_mag = float(np.linalg.norm(self.mcr_controller_sofa.get_mag_field_des()))
            new_mag = float(np.linalg.norm(field))
            if old_mag > 1e-9 and new_mag > 1e-9:
                field = field / new_mag * old_mag
            self.mcr_controller_sofa.mag_controller.field_des = field
        except Exception:
            self.mcr_controller_sofa.rotateZ(float(rot_n))
            self.mcr_controller_sofa.rotateX(float(rot_b))

    def _do_action(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        rot_n = float(action[0]) if action.shape[0] > 0 else 0.0
        rot_b = float(action[1]) if action.shape[0] > 1 else 0.0
        raw_insert = float(action[2]) if action.shape[0] > 2 else 0.0

        try:
            tip_pose = np.asarray(self.mcr_controller_sofa.get_pos_quat_catheter_tip(), dtype=np.float32)
            tip_pos = tip_pose[0:3]
        except Exception:
            tip_pos = np.zeros(3, dtype=np.float32)

        # Keep steering-context computation only for logging (forward_alignment, turn_angle, etc.).
        # It must not modify or cap the SAC action.
        self._compute_current_steering_context()
        effective_insert = self._apply_insert_safety_shield(raw_insert, ctx=None)

        self._apply_local_magnetic_action(rot_n, rot_b, tip_pos)
        self.mcr_controller_sofa.insertRetract(effective_insert)

    def _init_sim(self):
        super()._init_sim()
        self.mcr_controller_sofa: ControllerSofa = self.scene_creation_result["mcr_controller_sofa"]
        self.mcr_environment = self.scene_creation_result["mcr_environment"]
        self.chosen_model = self.scene_creation_result.get("chosen_model", "unknown")
        self.centerline_vtk = self.scene_creation_result.get("centerline_vtk", "unknown")
        self.task_id = self.scene_creation_result.get("task_id", str(self.chosen_model))

        if not self._explicit_force_model and str(self.chosen_model) in self.model_sampling_probs:
            print(
                "[SAMPLER] chosen_model=", self.chosen_model,
                "sampling_prob=", self.model_sampling_probs[str(self.chosen_model)],
            )

        raw_centerline = self.scene_creation_result.get("centerline_points", None)
        raw_centerline_radius = self.scene_creation_result.get("centerline_radius", None)
        if raw_centerline is not None:
            raw_centerline_arr = np.array(raw_centerline, dtype=np.float32)
            self.centerline_points = self._resample_centerline_points_1mm(raw_centerline_arr)
            radius_reason = None
            raw_radius_arr = None
            if raw_centerline_radius is None:
                radius_reason = "raw_centerline_radius is None"
            else:
                try:
                    raw_radius_arr = np.asarray(raw_centerline_radius, dtype=np.float32).reshape(-1)
                except Exception:
                    radius_reason = "raw_centerline_radius conversion failed"
                else:
                    if len(raw_radius_arr) != len(raw_centerline_arr):
                        radius_reason = (
                            f"len(raw_centerline_radius)={len(raw_radius_arr)} "
                            f"!= len(raw_centerline)={len(raw_centerline_arr)}"
                        )
                    elif not np.all(np.isfinite(raw_radius_arr)):
                        radius_reason = "raw_centerline_radius has non-finite values"

            if radius_reason is None:
                self.centerline_radius = self._resample_centerline_radius_1mm(
                    raw_points=raw_centerline_arr,
                    raw_radius=raw_radius_arr,
                    resampled_points=self.centerline_points,
                )
                if self.centerline_radius is None:
                    radius_reason = "resampling failed"
            else:
                self.centerline_radius = None

            if self.centerline_radius is not None:
                print(f"[CENTERLINE_RADIUS] task_id={self.task_id}")
                print("[CENTERLINE_RADIUS] raw count =", len(raw_centerline_arr))
                print("[CENTERLINE_RADIUS] resampled count =", len(self.centerline_radius))
                print(
                    "[CENTERLINE_RADIUS] min/mean/max mm =",
                    float(np.min(self.centerline_radius) * 1000.0),
                    float(np.mean(self.centerline_radius) * 1000.0),
                    float(np.max(self.centerline_radius) * 1000.0),
                )
            else:
                print("[CENTERLINE_RADIUS] No valid Radius array found. Local-radius safety disabled.")
                if radius_reason:
                    print("[CENTERLINE_RADIUS] reason =", radius_reason)
        else:
            self.centerline_points = None
            self.centerline_radius = None

        scene_target = self.scene_creation_result.get("target_position", None)
        if scene_target is not None:
            self.target_position = np.asarray(scene_target, dtype=np.float32)

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
                d_first = float(np.linalg.norm(self.centerline_points[0] - self.target_position))
                d_last = float(np.linalg.norm(self.centerline_points[-1] - self.target_position))
                self.centerline_reversed_for_progress = bool(d_first < d_last)

            if self.centerline_reversed_for_progress:
                self.centerline_points = self.centerline_points[::-1].copy()
                if self.centerline_radius is not None:
                    self.centerline_radius = self.centerline_radius[::-1].copy()

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

        # Gate FSM is disabled in this projection-progress version.
        self.gate_points = None
        self.gate_normals = None
        self.gate_radius = None
        self.gate_pass_radius = None
        self.gate_progress = None

        # Store nominal references for randomized soft reset. These are fixed for
        # a forced single-vessel run and are reused episode after episode.
        if self.centerline_points is not None and len(self.centerline_points) >= 2:
            self._soft_reset_nominal_start_sim = np.asarray(self.centerline_points[0], dtype=np.float32).copy()
            self._soft_reset_nominal_target_sim = np.asarray(self.centerline_points[-1], dtype=np.float32).copy()
        self._capture_soft_reset_reference_pose()

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

        vessel_positions = np.asarray(self.mcr_environment.get_vessel_tree_positions(), dtype=np.float32)
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