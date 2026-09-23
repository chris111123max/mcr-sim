"""Scale-aware defaults for non-ROS artificial-vessel RL training.

The ten generated training vessels define the design envelope below.  Keep
these values synchronized with ``tools/generate_artificial_vessels.py`` when
the generator geometry changes.
"""

import math
import numpy as np


# Geometry envelope (B01..B05 and C01..C05, source assets scaled to metres).
CATHETER_OUTER_DIAMETER_M = 0.00133
CATHETER_RADIUS_M = CATHETER_OUTER_DIAMETER_M / 2.0
TRAIN_VESSEL_MIN_RADIUS_M = 0.0028
TRAIN_VESSEL_MAX_RADIUS_M = 0.0052
TRAIN_ROUTE_MAX_LENGTH_M = 0.495
CONTROLLER_MAX_INSERTION_M = 0.510
# One policy step may advance the catheter by up to 0.8 mm.  The episode
# horizon remains 2048; this doubles the reachable distance per episode without
# changing termination semantics or the learned observation/reward definition.
MAX_INSERTION_PER_ACTION_M = 0.0008
# Symmetric, zero-preserving authority: raw -1/0/+1 -> -0.8/0/+0.8 mm.
INSERT_ACTION_NEGATIVE_LIMIT = -1.0

# Ordered-point navigation starts from a forward reference rather than the
# first few samples immediately after the entry.  Those samples are often
# behind the settled catheter pose and would create an artificial reverse-turn
# objective.  The distance is converted to an index on each route, so tight
# bends with 2 mm spacing may skip more samples than a straight segment.
DISCRETE_INITIAL_SKIP_DISTANCE_M = 0.008

# Environment/task defaults.
# V15.2-C: one 10 ms policy action drives two complete 5 ms SOFA steps.
RL_CONTROL_PERIOD_S = 0.01
PHYSICS_SUBSTEPS = 2
SOFA_TIME_STEP_S = RL_CONTROL_PERIOD_S / PHYSICS_SUBSTEPS
FRAME_SKIP = 1  # Legacy action-repeat setting; do not use for physics substeps.
SETTLE_STEPS = 8
TARGET_THRESHOLD_M = 0.003
MAX_EPISODE_STEPS = 2048
RADIUS_OBSERVATION_SCALE_M = 0.005
# A flexible magnetic catheter has delayed, hysteretic response.  One previous
# sample cannot distinguish "command has not taken effect yet" from "command is
# ineffective" and encourages repeated over-correction.  Keep 320 ms of local
# action/motion/progress history at the default 10 ms control interval.
ACTOR_HISTORY_STEPS = 32
ACTOR_SHAFT_LOOKBACK_DISTANCES_M = (0.010, 0.030, 0.060)
# Active point plus future points.  Since the active point is normally about
# 4 mm ahead on gentle segments, these offsets yield approximately the agreed
# 4/12/24/40/60 mm tip-relative preview while keeping the rewarded point
# explicitly observable.
DISCRETE_PREVIEW_DISTANCES_M = (0.000, 0.008, 0.020, 0.036, 0.056)
VESSEL_SECTION_FEATURE_DIM = 26
ACTOR_STATIC_ROUTE_FEATURE_DIM = 12
ACTOR_CURRENT_GEOMETRY_DIM = 31  # field3 + multiscale points15 + remaining1 + shaft9 + time/insertion/radius3
ACTOR_DYNAMIC_STEP_DIM = 7
ACTOR_OBSERVATION_DIM = (
    ACTOR_CURRENT_GEOMETRY_DIM + ACTOR_HISTORY_STEPS * ACTOR_DYNAMIC_STEP_DIM
)

# Continuous selected-route tracking. Initial localization may inspect the
# complete route; recurrent tracking is local and physically gated so nearby
# arms of a U-turn cannot create artificial progress. Two moving guidance
# points replace discrete waypoint spheres.
# Branch steering must start before the tip reaches the junction.  These remain
# tip-local vectors (not vessel IDs or global coordinates), so the additional
# preview is available equally to MLP, recurrent, and future transformer agents.
ROUTE_GUIDANCE_LOOKAHEAD_DISTANCES_M = (0.020, 0.060)
ROUTE_GUIDANCE_OBSERVATION_SCALE_M = 0.040
# Translation/rotation-invariant horizon feature used by the actor instead of
# absolute route completion.  The longest generated training route is the
# natural normalization scale; held-out routes are clipped rather than exposing
# a vessel-specific global coordinate or percentage.
ROUTE_REMAINING_DISTANCE_SCALE_M = TRAIN_ROUTE_MAX_LENGTH_M
ROUTE_PROJECTION_BACKWARD_WINDOW_M = 0.020
ROUTE_PROJECTION_FORWARD_WINDOW_M = 0.040
ROUTE_PROJECTION_AMBIGUITY_TOLERANCE_M = 0.00075
ROUTE_PROJECTION_MAX_PROGRESS_STEP_M = 0.002
ROUTE_SUCCESS_PROGRESS_MARGIN_M = 0.005

LOCAL_FIELD_ACTION_ANGLE_RAD = 3.0 * math.pi / 180.0
MAX_ACTION_DELTA = 0.30
# The ratio is retained only for legacy vessels that do not provide a VTI SDF.
OUT_OF_VESSEL_SAFETY_RATIO = 1.00
OUT_OF_VESSEL_FALLBACK_DISTANCE_M = 0.012

# Multi-model vessel safety.  The VTI stores centre-to-wall signed distance.
# A genuine outside termination requires any sampled catheter centre to remain
# at least 0.5 mm outside for three consecutive environment steps. Reward V13
# exposes the already-computed whole-body margin and starts a bounded warning
# ramp 0.5 mm before the catheter surface reaches the wall; shaft contact
# remains legal but is explicitly visible to the policy and reward.
SDF_CLEARANCE_OBSERVATION_SCALE_M = 0.002
SDF_NEAR_WALL_MARGIN_M = 0.001
SDF_OUTSIDE_CENTER_TOLERANCE_M = 0.0005
SDF_BODY_WARNING_MARGIN_M = 0.0005
SDF_OUTSIDE_CONFIRM_STEPS = 3
SDF_SAMPLE_STEP_FRACTION = 0.5
CENTERLINE_LOOKAHEAD_DISTANCES_M = (0.010, 0.030, 0.060)
TIP_NEAR_WALL_GRACE_STEPS = 5
TIP_NEAR_WALL_RAMP_STEPS = 20

# On branching vessels, compare distance to the selected target route with
# distance to the complete centerline graph.  A 2 mm preference for another
# graph branch, sustained for five steps, activates a recoverable dense penalty.
WRONG_BRANCH_DISTANCE_MARGIN_M = 0.002
WRONG_BRANCH_OBSERVATION_SCALE_M = 0.005
WRONG_BRANCH_CONFIRM_STEPS = 5

# Domain randomization only shrinks the generated vessels.  Keeping the upper
# bound at 1.0 preserves the nominal geometry and avoids making C05 longer than
# the controller's 510 mm insertion limit.
VESSEL_SCALE_MIN = 0.90
VESSEL_SCALE_MAX = 1.00
START_WINDOW_DISTANCE_M = 0.010
TARGET_WINDOW_DISTANCE_M = 0.010
INITIAL_ORIENTATION_MAX_ANGLE_DEG = 10.0
ENTRY_TANGENT_POINTS = 5

# Reward profile V15.2-A. PPO receives signed distance progress to the active
# ordered navigation point.
# Retracting cancels forward credit; switching points never creates a bonus.
REWARD_PROFILE_VERSION = "15.2A-discrete-forward-relaxed-waypoints"
REWARD_PROGRESS_NORMALIZATION_M = TRAIN_ROUTE_MAX_LENGTH_M  # metadata/fallback only
REWARD_PROGRESS_SCALE = 1000.0
# Compatibility alias for older reporting code. In V13 this is per unit route
# completion, not per physical metre.
REWARD_PROGRESS_PER_M = REWARD_PROGRESS_SCALE
REWARD_ROUTE_PROGRESS = REWARD_PROGRESS_SCALE
REWARD_PROGRESS_BUDGET = REWARD_PROGRESS_SCALE * TRAIN_ROUTE_MAX_LENGTH_M
# Common learner discount; V13 progress itself is an undiscounted difference.
REWARD_DISCOUNT_GAMMA = 0.9995
# The diagnostic audit showed body escape while the tip was still inside. The
# earlier coefficient was typically 8-15x smaller than immediate progress near
# the failure bend. This remains a single bounded safety term, but now provides
# useful pre-contact credit assignment.
REWARD_WALL_PROXIMITY = 0.0
REWARD_OFF_TARGET_BRANCH = 0.0
REWARD_SUCCESS = 300.0
REWARD_OUT_OF_VESSEL = -30.0
REWARD_NON_FINITE = -30.0
REWARD_TIMEOUT = -10.0
REWARD_STEP = -0.0005
# Stagnation never terminates an episode.  This bounded per-step cost provides
# local credit assignment after a grace period; the terminal timeout alone is
# too delayed to discourage retract-and-wait under gamma=0.9995.
REWARD_STAGNATION = 0.0

# A short, rollout-visible window distinguishes deliberate steering pauses from
# a policy that has collapsed to zero insertion/retraction.  It never terminates
# the episode; it only adds one bounded cost after the initial grace period.
NO_PROGRESS_WINDOW_STEPS = 32
NO_PROGRESS_GRACE_STEPS = 64
NO_PROGRESS_CONFIRM_STEPS = 512
NO_PROGRESS_MIN_NET_APPROACH_M = 0.001

# SAC has no gradient clipping in upstream SB3.  The shared distributed SAC
# implementation applies this bound after cross-rank averaging and before the
# optimizer step.  PPO retains its own PPO_MAX_GRAD_NORM below.
SAC_MAX_GRAD_NORM = 10.0
SAC_MIN_ENT_COEF = 0.02


def reward_profile(discount_gamma: float = REWARD_DISCOUNT_GAMMA) -> dict:
    """Return the exact shared reward settings for run metadata."""

    return {
        "version": REWARD_PROFILE_VERSION,
        "progress_normalization": "none_distance_in_metres",
        "progress_normalization_m": REWARD_PROGRESS_NORMALIZATION_M,
        "progress_scale": REWARD_PROGRESS_SCALE,
        "discount_gamma": float(discount_gamma),
        "progress_formula": "1000*(distance_before-distance_after)_same_active_point_metres",
        "discrete_navigation": {"gentle_spacing_m": .004, "tight_spacing_m": .002, "gentle_radius_m": .0015, "tight_radius_m": .0012, "final_radius_m": .003, "preview_points": 5, "initial_skip_distance_m": DISCRETE_INITIAL_SKIP_DISTANCE_M},
        "discrete_bend_rule": "offline_4mm_window_direction_change_ge_10deg_or_graph_degree_ge_3_within_2mm",
        "progress_budget": REWARD_PROGRESS_BUDGET,
        "route_progress": REWARD_ROUTE_PROGRESS,
        "wall_proximity": REWARD_WALL_PROXIMITY,
        "off_target_branch": REWARD_OFF_TARGET_BRANCH,
        "success": REWARD_SUCCESS,
        "out_of_vessel": REWARD_OUT_OF_VESSEL,
        "non_finite": REWARD_NON_FINITE,
        "timeout": REWARD_TIMEOUT,
        "step": REWARD_STEP,
        "stagnation": REWARD_STAGNATION,
        "stagnation_formula": "disabled_no_stagnation_termination_or_cost",
        "body_sdf_warning_margin_m": SDF_BODY_WARNING_MARGIN_M,
        "body_sdf_warning_formula": "linear_surface_clearance_0.5mm_to_contact",
        "insert_action_mapping": "symmetric_-1_0_1_max_0.8mm",
        "max_insertion_per_action_m": MAX_INSERTION_PER_ACTION_M,
        "no_progress_window_steps": NO_PROGRESS_WINDOW_STEPS,
        "no_progress_grace_steps": NO_PROGRESS_GRACE_STEPS,
        "no_progress_confirm_steps": NO_PROGRESS_CONFIRM_STEPS,
        "no_progress_min_net_approach_m": NO_PROGRESS_MIN_NET_APPROACH_M,
    }


def body_sdf_risk_features(
    body_surface_clearance: float,
    max_signed_distance: float,
    outside_tolerance: float = SDF_OUTSIDE_CENTER_TOLERANCE_M,
    warning_margin: float = SDF_BODY_WARNING_MARGIN_M,
):
    """Return bounded whole-body warning and outside-depth features.

    ``body_surface_clearance`` is wall-to-catheter-surface clearance: positive
    is safe, zero is contact, and negative is body penetration. The warning
    ramps over the final ``warning_margin`` before surface contact.

    ``max_signed_distance`` remains the worst catheter-centre SDF sample:
    negative values are inside the lumen and positive values are outside. It is
    used only for the terminal outside-depth feature. Keeping these two
    geometries separate avoids starting the warning after the catheter body has
    already penetrated the wall by roughly one catheter radius.
    """

    clearance = float(body_surface_clearance)
    signed = float(max_signed_distance)
    tolerance = max(float(outside_tolerance), 1e-12)
    margin = max(float(warning_margin), 0.0)
    if not math.isfinite(clearance) or not math.isfinite(signed):
        return 0.0, 0.0
    warning = min(max((margin - clearance) / max(margin, 1e-12), 0.0), 1.0)
    outside_depth = min(max(signed / tolerance, 0.0), 1.0)
    return warning, outside_depth


def map_insert_action(
    raw_insert: float,
    negative_limit: float = INSERT_ACTION_NEGATIVE_LIMIT,
) -> float:
    """Map policy insertion to actuator command without moving the zero point."""

    raw = min(max(float(raw_insert), -1.0), 1.0)
    if raw >= 0.0:
        return raw
    return abs(min(float(negative_limit), 0.0)) * raw


# Training-only five-stage target-route curriculum for B01/B02.  Stages add
# geometrically deeper branches without removing mastered routes.  The final
# policy is therefore trained jointly on all twelve B01/B02 target tasks.
TRAINING_CURRICULUM_ENABLED = True
TRAINING_CURRICULUM_ALL_MODELS = (
    "B01", "B02", "B03", "B04", "B05",
    "C01", "C02", "C03", "C04", "C05",
)
TRAINING_CURRICULUM_BRANCH_MODELS = ("B01", "B02")
TRAINING_CURRICULUM_MODELS = (
    TRAINING_CURRICULUM_BRANCH_MODELS,
    TRAINING_CURRICULUM_BRANCH_MODELS,
    TRAINING_CURRICULUM_BRANCH_MODELS,
    TRAINING_CURRICULUM_BRANCH_MODELS,
    TRAINING_CURRICULUM_BRANCH_MODELS,
)
TRAINING_CURRICULUM_ROUTES = (
    {"B01": (1, 4), "B02": (1, 4)},
    {"B01": (1, 4, 5, 6), "B02": (1, 4)},
    {"B01": (1, 4, 5, 6), "B02": (1, 4, 5, 6)},
    {"B01": (1, 2, 3, 4, 5, 6), "B02": (1, 4, 5, 6)},
    {"B01": (1, 2, 3, 4, 5, 6), "B02": (1, 2, 3, 4, 5, 6)},
)
TRAINING_CURRICULUM_STAGE_NAMES = (
    "shallow_routes",
    "b01_medium_routes",
    "both_medium_routes",
    "b01_all_routes",
    "b01_b02_all_routes",
)
TRAINING_CURRICULUM_TARGET_FRACTIONS = (1.00,) * 5
TRAINING_CURRICULUM_SUCCESS_THRESHOLDS = (0.70, 0.70, 0.70, 0.70)
TRAINING_CURRICULUM_PROMOTION_MODES = ("per_route_min",) * 4
TRAINING_CURRICULUM_CONSECUTIVE_SUCCESS_EPISODES = 3
TRAINING_CURRICULUM_ROLLING_EPISODES_PER_VESSEL = 100
TRAINING_CURRICULUM_MIN_EPISODES_PER_VESSEL = 30
TRAINING_CURRICULUM_ROLLING_EPISODES_PER_ROUTE = 50
TRAINING_CURRICULUM_MIN_EPISODES_PER_ROUTE = 30
# Keep geometry fixed while establishing the twelve-route control baseline.
TRAINING_CURRICULUM_DR_FRACTIONS = (0.00,) * 5
# Half the sampling distribution remains uniform.  The adaptive half focuses
# on weak vessels but is capped so a single failure mode cannot erase skills
# already acquired on the rest of the active pool.
TRAINING_CURRICULUM_UNIFORM_SAMPLING_MIX = 0.50
TRAINING_CURRICULUM_DIFFICULTY_POWER = 2.0
TRAINING_CURRICULUM_MAX_SAMPLING_FACTOR = 2.0
# Use the previously stable exploration floors in every stage.  Exploration
# still anneals naturally through the learned PPO log_std / SAC entropy tuner.
PPO_ACTION_STD_FLOOR_BY_STAGE = (0.20,) * 5
SAC_ENT_COEF_FLOOR_BY_STAGE = (0.02,) * 5


def curriculum_domain_randomization_profile(current_stage: int) -> dict:
    """Return the stage-specific subset of the final DR envelope."""

    stage = min(max(int(current_stage), 0), len(TRAINING_CURRICULUM_DR_FRACTIONS) - 1)
    fraction = float(TRAINING_CURRICULUM_DR_FRACTIONS[stage])
    return {
        "fraction": fraction,
        "vessel_scale_min": 1.0 - fraction * (1.0 - VESSEL_SCALE_MIN),
        "vessel_scale_max": VESSEL_SCALE_MAX,
        "start_window_distance_m": fraction * START_WINDOW_DISTANCE_M,
        "target_window_distance_m": fraction * TARGET_WINDOW_DISTANCE_M,
        "initial_orientation_max_angle_deg": (
            fraction * INITIAL_ORIENTATION_MAX_ANGLE_DEG
        ),
    }


def curriculum_sampling_weights(
    active_models,
    per_model_success_rates=None,
    uniform_mix: float = TRAINING_CURRICULUM_UNIFORM_SAMPLING_MIX,
    difficulty_power: float = TRAINING_CURRICULUM_DIFFICULTY_POWER,
    max_sampling_factor: float = TRAINING_CURRICULUM_MAX_SAMPLING_FACTOR,
) -> dict:
    """Blend uniform sampling with failure-rate-weighted hard-vessel sampling."""

    models = tuple(str(model_id) for model_id in active_models)
    if not models:
        return {}
    rates = per_model_success_rates or {}
    difficulties = []
    for model_id in models:
        rate = rates.get(model_id)
        if rate is None or not math.isfinite(float(rate)):
            rate = 0.0
        rate = min(max(float(rate), 0.0), 1.0)
        difficulties.append(max(1.0 - rate, 0.05) ** float(difficulty_power))
    difficulty_total = sum(difficulties)
    uniform_probability = 1.0 / len(models)
    mix = min(max(float(uniform_mix), 0.0), 1.0)
    raw_weights = {
        model_id: (
            mix * uniform_probability
            + (1.0 - mix) * difficulty / difficulty_total
        )
        for model_id, difficulty in zip(models, difficulties)
    }
    cap = min(max(float(max_sampling_factor) * uniform_probability, uniform_probability), 1.0)
    weights = dict(raw_weights)
    fixed = set()
    while True:
        newly_fixed = {
            model_id for model_id, probability in weights.items()
            if model_id not in fixed and probability > cap + 1e-12
        }
        if not newly_fixed:
            break
        fixed.update(newly_fixed)
        for model_id in newly_fixed:
            weights[model_id] = cap
        remaining = [model_id for model_id in models if model_id not in fixed]
        remaining_mass = max(1.0 - cap * len(fixed), 0.0)
        raw_remaining = sum(raw_weights[model_id] for model_id in remaining)
        if not remaining or raw_remaining <= 0.0:
            break
        for model_id in remaining:
            weights[model_id] = remaining_mass * raw_weights[model_id] / raw_remaining
    total = sum(weights.values())
    return {model_id: probability / total for model_id, probability in weights.items()}


def curriculum_route_sampling_weights(
    current_stage: int,
    per_route_success_rates=None,
    uniform_mix: float = 0.30,
    max_sampling_factor: float = 2.0,
) -> dict:
    """Return bounded adaptive probabilities for each active vessel/route.

    At least 30% of the distribution remains uniform, so mastered routes are
    rehearsed while the remaining mass focuses on routes with low success.
    """

    stage = min(max(int(current_stage), 0), len(TRAINING_CURRICULUM_ROUTES) - 1)
    rates = per_route_success_rates or {}
    result = {}
    for model_id, route_indices in TRAINING_CURRICULUM_ROUTES[stage].items():
        keys = [f"{model_id}/target_{index:02d}" for index in route_indices]
        difficulties = []
        for key in keys:
            rate = rates.get(key)
            rate = 0.0 if rate is None or not math.isfinite(float(rate)) else float(rate)
            difficulties.append(max(1.0 - min(max(rate, 0.0), 1.0), 0.05) ** 2)
        uniform = 1.0 / len(keys)
        total_difficulty = max(sum(difficulties), 1e-12)
        raw = [
            float(uniform_mix) * uniform
            + (1.0 - float(uniform_mix)) * difficulty / total_difficulty
            for difficulty in difficulties
        ]
        cap = min(float(max_sampling_factor) * uniform, 1.0)
        clipped = np.minimum(np.asarray(raw, dtype=np.float64), cap)
        # Iteratively redistribute unused probability without exceeding cap.
        for _ in range(len(keys) + 1):
            missing = 1.0 - float(clipped.sum())
            if missing <= 1e-12:
                break
            eligible = clipped < cap - 1e-12
            if not np.any(eligible):
                break
            add = missing * np.asarray(raw)[eligible] / max(float(np.asarray(raw)[eligible].sum()), 1e-12)
            clipped[eligible] = np.minimum(clipped[eligible] + add, cap)
        clipped /= max(float(clipped.sum()), 1e-12)
        result[model_id] = {
            int(index): float(probability)
            for index, probability in zip(route_indices, clipped)
        }
    return result


def curriculum_exploration_profile(current_stage: int) -> dict:
    """Return shared algorithm-specific exploration floors for one stage."""

    ppo_stage = min(max(int(current_stage), 0), len(PPO_ACTION_STD_FLOOR_BY_STAGE) - 1)
    sac_stage = min(max(int(current_stage), 0), len(SAC_ENT_COEF_FLOOR_BY_STAGE) - 1)
    return {
        "ppo_min_action_std": float(PPO_ACTION_STD_FLOOR_BY_STAGE[ppo_stage]),
        "sac_min_ent_coef": float(SAC_ENT_COEF_FLOOR_BY_STAGE[sac_stage]),
    }


def curriculum_protocol_profile() -> dict:
    """Return the shared curriculum/observation protocol for run metadata."""

    return {
        "stage_names": list(TRAINING_CURRICULUM_STAGE_NAMES),
        "models_by_stage": [list(models) for models in TRAINING_CURRICULUM_MODELS],
        "routes_by_stage": [
            {model_id: list(indices) for model_id, indices in routes.items()}
            for routes in TRAINING_CURRICULUM_ROUTES
        ],
        "target_fraction_by_stage": list(TRAINING_CURRICULUM_TARGET_FRACTIONS),
        "success_thresholds": list(TRAINING_CURRICULUM_SUCCESS_THRESHOLDS),
        "promotion_modes": list(TRAINING_CURRICULUM_PROMOTION_MODES),
        "consecutive_success_episodes": (
            TRAINING_CURRICULUM_CONSECUTIVE_SUCCESS_EPISODES
        ),
        "rolling_episodes_per_vessel": TRAINING_CURRICULUM_ROLLING_EPISODES_PER_VESSEL,
        "minimum_episodes_per_vessel": TRAINING_CURRICULUM_MIN_EPISODES_PER_VESSEL,
        "domain_randomization_fractions": list(TRAINING_CURRICULUM_DR_FRACTIONS),
        "uniform_sampling_mix": TRAINING_CURRICULUM_UNIFORM_SAMPLING_MIX,
        "difficulty_power": TRAINING_CURRICULUM_DIFFICULTY_POWER,
        "max_sampling_factor": TRAINING_CURRICULUM_MAX_SAMPLING_FACTOR,
        "ppo_action_std_floor_by_stage": list(PPO_ACTION_STD_FLOOR_BY_STAGE),
        "ppo_initial_action_std": PPO_INITIAL_ACTION_STD,
        "ppo_max_action_std": PPO_MAX_ACTION_STD,
        "sac_ent_coef_floor_by_stage": list(SAC_ENT_COEF_FLOOR_BY_STAGE),
        "centerline_lookahead_distances_m": list(CENTERLINE_LOOKAHEAD_DISTANCES_M),
        "actor_future_tangent_features": "five_multiscale_route_point_vectors_no_tangent_features",
        "discrete_preview_offsets_from_active_point_m": list(DISCRETE_PREVIEW_DISTANCES_M),
        "actor_shaft_lookback_distances_m": list(ACTOR_SHAFT_LOOKBACK_DISTANCES_M),
        "actor_time_feature": "fraction_of_episode_remaining",
        "actor_inserted_length_feature": "controller_insertion_fraction",
        "actor_history_steps": ACTOR_HISTORY_STEPS,
        "max_insertion_per_action_m": MAX_INSERTION_PER_ACTION_M,
        "local_field_action_angle_deg": math.degrees(LOCAL_FIELD_ACTION_ANGLE_RAD),
        "navigation": "ordered_discrete_points",
        "route_guidance_lookahead_distances_m": list(
            ROUTE_GUIDANCE_LOOKAHEAD_DISTANCES_M
        ),
        "route_projection_backward_window_m": ROUTE_PROJECTION_BACKWARD_WINDOW_M,
        "route_projection_forward_window_m": ROUTE_PROJECTION_FORWARD_WINDOW_M,
        "route_projection_ambiguity_tolerance_m": (
            ROUTE_PROJECTION_AMBIGUITY_TOLERANCE_M
        ),
        "route_projection_max_progress_step_m": (
            ROUTE_PROJECTION_MAX_PROGRESS_STEP_M
        ),
        "route_success_progress_margin_m": ROUTE_SUCCESS_PROGRESS_MARGIN_M,
        "route_remaining_distance_scale_m": ROUTE_REMAINING_DISTANCE_SCALE_M,
        "actor_coordinate_frame": "catheter_tip_local",
        "actor_route_horizon_feature": "five_tip_relative_multiscale_points_and_remaining_route_distance",
        "actor_wall_features": False,
        "actor_dynamic_features": "previous_effective_action3_tip_motion3_actual_insertion_delta1",
        "branch_target_sampling": "stage_filtered_failure_adaptive_with_uniform_rehearsal",
        "branch_target_routes": [f"target_{index:02d}" for index in range(1, 7)],
        "branch_mastery_metric": "minimum_per_route_success",
        "observation_dim": ACTOR_OBSERVATION_DIM,
    }


def update_curriculum_progress(
    current_stage: int,
    consecutive_success_episodes: int,
    train_success_rate: float,
    per_model_success_rates=None,
    per_model_episode_counts=None,
):
    """Update the stable-success streak and advance at most one stage.

    Stage 0 uses the weighted aggregate rolling success of B01/B02.  Later
    stages use the weakest active vessel.  All stages still require the minimum
    rolling sample count for every active vessel.
    """

    stage = min(max(int(current_stage), 0), len(TRAINING_CURRICULUM_MODELS) - 1)
    streak = max(int(consecutive_success_episodes), 0)
    if stage >= len(TRAINING_CURRICULUM_SUCCESS_THRESHOLDS):
        return stage, 0
    mastery_success_rate = float(train_success_rate)
    enough_samples = True
    if per_model_success_rates is not None:
        active_rates = []
        for model_id in TRAINING_CURRICULUM_MODELS[stage]:
            rate = per_model_success_rates.get(model_id)
            if rate is None or not math.isfinite(float(rate)):
                active_rates = []
                mastery_success_rate = -math.inf
                break
            if per_model_episode_counts is not None:
                count = int(per_model_episode_counts.get(model_id, 0))
                if count < TRAINING_CURRICULUM_MIN_EPISODES_PER_VESSEL:
                    enough_samples = False
            active_rates.append(float(rate))
        if active_rates:
            mode = TRAINING_CURRICULUM_PROMOTION_MODES[stage]
            if mode == "aggregate":
                if per_model_episode_counts is None:
                    mastery_success_rate = sum(active_rates) / len(active_rates)
                else:
                    active_counts = [
                        max(int(per_model_episode_counts.get(model_id, 0)), 0)
                        for model_id in TRAINING_CURRICULUM_MODELS[stage]
                    ]
                    total_count = sum(active_counts)
                    mastery_success_rate = (
                        sum(rate * count for rate, count in zip(active_rates, active_counts))
                        / total_count
                        if total_count > 0
                        else -math.inf
                    )
            else:
                mastery_success_rate = min(active_rates)
    if (
        enough_samples
        and mastery_success_rate >= TRAINING_CURRICULUM_SUCCESS_THRESHOLDS[stage]
        and streak >= TRAINING_CURRICULUM_CONSECUTIVE_SUCCESS_EPISODES
    ):
        stage += 1
        streak = 0
    elif (
        not enough_samples
        or mastery_success_rate < TRAINING_CURRICULUM_SUCCESS_THRESHOLDS[stage]
    ):
        streak = 0
    return stage, streak


def update_validation_unlocked(
    previously_unlocked: bool,
    train_success_rate: float,
    minimum_train_success_rate: float,
    full_task_ready: bool = True,
) -> bool:
    """Latch validation only after success on a complete-route training task."""

    return bool(
        previously_unlocked
        or (
            bool(full_task_ready)
            and (
                float(minimum_train_success_rate) <= 0.0
                or float(train_success_rate) >= float(minimum_train_success_rate)
            )
        )
    )

# V15.2-B Physics Wall.  Mechanical beam resolution remains 30 body + 3 tip;
# only the mapped collision representation is sampled more densely.  The SDF
# term is a SOFA mechanical force and is never part of reward, observation, or
# action processing.
CATHETER_COLLISION_BODY_EDGES = 80
CATHETER_COLLISION_TIP_EDGES = 12
SDF_PHYSICS_WALL_ENABLED = True
SDF_WALL_ACTIVATION_CLEARANCE_M = 0.0003
SDF_WALL_STIFFNESS_N_PER_M = 10.0
SDF_WALL_MAX_FORCE_N = 0.010

# Collision/contact defaults.  Catheter Line/Point primitives represent their
# physical radius through proximity.  Vessel collision remains triangle-only.
VESSEL_TRIANGLE_PROXIMITY_M = 0.0002
CATHETER_COLLISION_PROXIMITY_M = CATHETER_RADIUS_M
LMD_CONTACT_DISTANCE_M = 0.0002
LMD_ALARM_DISTANCE_M = 0.0010
LMD_ANGLE_CONE = 0.02
FRICTION_COEFFICIENT = 0.01
CONSTRAINT_TOLERANCE = 1e-6
CONSTRAINT_MAX_ITERATIONS = 20000

# Shared formal experiment protocol.  Training episode counts are global
# across every distributed rank, not per-rank budgets.
NUM_EPOCHS = 100
TRAIN_EPISODES_PER_EPOCH = 100
CHECKPOINT_INTERVAL = 1
VALID_INTERVAL = 2
VALID_MIN_TRAIN_SUCCESS_RATE = 0.20
VALID_VESSELS = 5
VALID_EPISODES_PER_VESSEL = 2
VALID_EPISODES_TOTAL = VALID_VESSELS * VALID_EPISODES_PER_VESSEL

# SAC defaults for the 4-NPU target. Batch size and environment count are
# global values; train_sac.py divides them evenly between ranks.
# Comparison runs use an episode-based budget.  A vectorized step can finish
# several environments at once, so the callback stops at the first vector
# step reaching the boundary (the reported count is therefore >= 100).
SAC_EPOCHS = NUM_EPOCHS
SAC_EPISODES_PER_EPOCH = TRAIN_EPISODES_PER_EPOCH
SAC_STEPS_PER_EPOCH = MAX_EPISODE_STEPS * SAC_EPISODES_PER_EPOCH
SAC_TOTAL_TIMESTEPS = SAC_EPOCHS * SAC_STEPS_PER_EPOCH
SAC_N_ENVS = 64
SAC_LEARNING_RATE = 3e-4
SAC_BATCH_SIZE = 1024
SAC_BUFFER_SIZE = 500_000
SAC_LEARNING_STARTS = 50_000
SAC_TRAIN_FREQ = 1
SAC_GRADIENT_STEPS = 1
SAC_TAU = 0.005
SAC_GAMMA = REWARD_DISCOUNT_GAMMA

# PPO baseline defaults.  ``n_steps`` is per environment; batch size is global
# and is divided evenly between synchronized ranks, just like SAC.
PPO_EPOCHS = NUM_EPOCHS
PPO_EPISODES_PER_EPOCH = TRAIN_EPISODES_PER_EPOCH
# One 910B3 + the allocated CPU quota is balanced at 32 independent SOFA envs.
# Keep this explicit instead of inheriting the SAC throughput-oriented default.
PPO_N_ENVS = 32
PPO_LEARNING_RATE = 1e-4
PPO_N_STEPS = 256
PPO_BATCH_SIZE = 1024
PPO_N_EPOCHS = 5
PPO_GAMMA = SAC_GAMMA
PPO_GAE_LAMBDA = 0.98
PPO_CLIP_RANGE = 0.10
PPO_ENT_COEF = 0.002
PPO_VF_COEF = 0.5
PPO_MAX_GRAD_NORM = 0.3
PPO_INITIAL_ACTION_STD = 0.50
PPO_MIN_ACTION_STD = 0.20
PPO_MAX_ACTION_STD = 0.60


def validate_training_defaults() -> None:
    """Fail early if a future edit makes the default task unreachable."""
    if not (0.0 < VESSEL_SCALE_MIN <= VESSEL_SCALE_MAX):
        raise ValueError("Invalid vessel scale range.")
    longest_scaled_route = TRAIN_ROUTE_MAX_LENGTH_M * VESSEL_SCALE_MAX
    if longest_scaled_route >= CONTROLLER_MAX_INSERTION_M:
        raise ValueError(
            "Largest scaled vessel route exceeds the controller insertion limit: "
            f"{longest_scaled_route:.6f} >= {CONTROLLER_MAX_INSERTION_M:.6f} m"
        )
    available_motion = MAX_EPISODE_STEPS * MAX_INSERTION_PER_ACTION_M
    if available_motion < 1.5 * longest_scaled_route:
        raise ValueError(
            "Episode action budget is too short for the largest scaled vessel."
        )
    if not (0.0 <= LMD_CONTACT_DISTANCE_M < LMD_ALARM_DISTANCE_M):
        raise ValueError("LocalMinDistance requires contactDistance < alarmDistance.")
    if CATHETER_COLLISION_BODY_EDGES < 1 or CATHETER_COLLISION_TIP_EDGES < 1:
        raise ValueError("Catheter collision edge counts must be positive.")
    if not (
        SDF_WALL_ACTIVATION_CLEARANCE_M > 0.0
        and SDF_WALL_STIFFNESS_N_PER_M > 0.0
        and SDF_WALL_MAX_FORCE_N > 0.0
    ):
        raise ValueError("SDF physics-wall parameters must be positive.")
    if not (0.0 < OUT_OF_VESSEL_SAFETY_RATIO <= 1.0):
        raise ValueError("Out-of-vessel ratio must be in (0, 1] for physical containment.")
    if START_WINDOW_DISTANCE_M < 0.0 or TARGET_WINDOW_DISTANCE_M < 0.0:
        raise ValueError("Endpoint randomization distances must be non-negative.")
    if REWARD_PROGRESS_NORMALIZATION_M <= 0.0:
        raise ValueError("Reward progress normalization must be positive.")
    if len(ROUTE_GUIDANCE_LOOKAHEAD_DISTANCES_M) != 2 or any(
        distance <= 0.0 for distance in ROUTE_GUIDANCE_LOOKAHEAD_DISTANCES_M
    ):
        raise ValueError("Exactly two positive continuous route guidance distances are required.")
    if not (
        ROUTE_PROJECTION_BACKWARD_WINDOW_M > ROUTE_PROJECTION_MAX_PROGRESS_STEP_M
        and ROUTE_PROJECTION_FORWARD_WINDOW_M > ROUTE_PROJECTION_MAX_PROGRESS_STEP_M
        and ROUTE_PROJECTION_AMBIGUITY_TOLERANCE_M >= 0.0
        and ROUTE_SUCCESS_PROGRESS_MARGIN_M > 0.0
    ):
        raise ValueError("Continuous route tracking settings are invalid.")
    if not (
        REWARD_ROUTE_PROGRESS > 0.0
        and REWARD_WALL_PROXIMITY == 0.0
        and REWARD_OFF_TARGET_BRANCH == 0.0
        and REWARD_SUCCESS > 0.0
        and REWARD_OUT_OF_VESSEL < 0.0
        and REWARD_NON_FINITE < 0.0
        and REWARD_TIMEOUT < 0.0
        and REWARD_STEP < 0.0
    ):
        raise ValueError("Reward V13 signs are invalid.")
    maximum_navigation_credit = REWARD_PROGRESS_BUDGET
    # The discrete-point baseline intentionally gives meaningful positive
    # credit for safe advancement before a complete episode is available.
    # Terminal failure remains a clear negative event, while a sufficiently
    # long forward trajectory must be preferable to waiting for timeout.
    if not (
        REWARD_PROGRESS_SCALE * 0.12 > abs(REWARD_OUT_OF_VESSEL)
        and REWARD_PROGRESS_SCALE > 0.0
        and REWARD_SUCCESS > abs(REWARD_OUT_OF_VESSEL)
    ):
        raise ValueError("Discrete progress reward must dominate short forward motion and failure cost.")
    if REWARD_STAGNATION != 0.0:
        raise ValueError("Reward V13 stagnation must be a non-terminal penalty.")
    if not (
        NO_PROGRESS_WINDOW_STEPS > 0
        and NO_PROGRESS_GRACE_STEPS >= NO_PROGRESS_WINDOW_STEPS
        and NO_PROGRESS_CONFIRM_STEPS > 0
        and NO_PROGRESS_MIN_NET_APPROACH_M > 0.0
    ):
        raise ValueError("Invalid no-progress detection settings.")
    if SAC_MAX_GRAD_NORM <= 0.0:
        raise ValueError("SAC_MAX_GRAD_NORM must be positive.")
    if SAC_MIN_ENT_COEF <= 0.0:
        raise ValueError("SAC_MIN_ENT_COEF must be positive.")
    if len(TRAINING_CURRICULUM_MODELS) != len(TRAINING_CURRICULUM_SUCCESS_THRESHOLDS) + 1:
        raise ValueError("Curriculum stages and thresholds are inconsistent.")
    if len(TRAINING_CURRICULUM_DR_FRACTIONS) != len(TRAINING_CURRICULUM_MODELS):
        raise ValueError("Curriculum DR stages are inconsistent.")
    if len(TRAINING_CURRICULUM_TARGET_FRACTIONS) != len(TRAINING_CURRICULUM_MODELS):
        raise ValueError("Curriculum target-distance stages are inconsistent.")
    if any(not (0.0 < fraction <= 1.0) for fraction in TRAINING_CURRICULUM_TARGET_FRACTIONS):
        raise ValueError("Curriculum target fractions must be in (0, 1].")
    if any(not (0.0 <= fraction <= 1.0) for fraction in TRAINING_CURRICULUM_DR_FRACTIONS):
        raise ValueError("Curriculum DR fractions must be in [0, 1].")
    if len(TRAINING_CURRICULUM_STAGE_NAMES) != len(TRAINING_CURRICULUM_MODELS):
        raise ValueError("Curriculum stage names are inconsistent.")
    if any(tuple(models) != TRAINING_CURRICULUM_BRANCH_MODELS for models in TRAINING_CURRICULUM_MODELS):
        raise ValueError("Every target-route curriculum stage must train B01/B02.")
    if len(TRAINING_CURRICULUM_ROUTES) != len(TRAINING_CURRICULUM_MODELS):
        raise ValueError("Target-route curriculum stages are inconsistent.")
    for routes in TRAINING_CURRICULUM_ROUTES:
        if set(routes) != set(TRAINING_CURRICULUM_BRANCH_MODELS):
            raise ValueError("Every route stage must configure B01 and B02.")
        if any(not indices or any(index not in range(1, 7) for index in indices) for indices in routes.values()):
            raise ValueError("Curriculum target indices must lie in [1, 6].")
    if any(fraction != 1.0 for fraction in TRAINING_CURRICULUM_TARGET_FRACTIONS):
        raise ValueError("Every curriculum stage must use the complete route.")
    if TRAINING_CURRICULUM_DR_FRACTIONS != (0.0,) * 5:
        raise ValueError("The five-stage curriculum DR schedule is inconsistent.")
    if TRAINING_CURRICULUM_PROMOTION_MODES != ("per_route_min",) * 4:
        raise ValueError("The curriculum promotion modes are inconsistent.")
    if not (0.0 <= TRAINING_CURRICULUM_UNIFORM_SAMPLING_MIX <= 1.0):
        raise ValueError("Curriculum uniform sampling mix must be in [0, 1].")
    if TRAINING_CURRICULUM_DIFFICULTY_POWER <= 0.0:
        raise ValueError("Curriculum difficulty power must be positive.")
    if TRAINING_CURRICULUM_MAX_SAMPLING_FACTOR < 1.0:
        raise ValueError("Curriculum maximum sampling factor must be at least 1.")
    if not (
        len(PPO_ACTION_STD_FLOOR_BY_STAGE)
        == len(SAC_ENT_COEF_FLOOR_BY_STAGE)
        == len(TRAINING_CURRICULUM_MODELS)
    ):
        raise ValueError("Curriculum exploration schedules are inconsistent.")
    if TRAINING_CURRICULUM_CONSECUTIVE_SUCCESS_EPISODES < 1:
        raise ValueError("Curriculum consecutive-success episode count must be positive.")
    if TRAINING_CURRICULUM_ROLLING_EPISODES_PER_VESSEL < 1:
        raise ValueError("Curriculum rolling window must be positive.")
    if not (
        1
        <= TRAINING_CURRICULUM_MIN_EPISODES_PER_VESSEL
        <= TRAINING_CURRICULUM_ROLLING_EPISODES_PER_VESSEL
    ):
        raise ValueError("Invalid curriculum minimum per-vessel sample count.")
    if not (
        1 <= TRAINING_CURRICULUM_MIN_EPISODES_PER_ROUTE
        <= TRAINING_CURRICULUM_ROLLING_EPISODES_PER_ROUTE
    ):
        raise ValueError("Invalid curriculum minimum per-route sample count.")
    if any(
        not (0.0 < threshold <= 1.0)
        for threshold in TRAINING_CURRICULUM_SUCCESS_THRESHOLDS
    ):
        raise ValueError("Curriculum success thresholds must be in (0, 1].")
    if not (0.0 < PPO_MIN_ACTION_STD <= PPO_MAX_ACTION_STD):
        raise ValueError("Invalid PPO action standard-deviation bounds.")
    if not (PPO_MIN_ACTION_STD <= PPO_INITIAL_ACTION_STD <= PPO_MAX_ACTION_STD):
        raise ValueError("Invalid PPO initial action standard deviation.")
    if any(
        not (PPO_MIN_ACTION_STD <= floor <= PPO_MAX_ACTION_STD)
        for floor in PPO_ACTION_STD_FLOOR_BY_STAGE
    ):
        raise ValueError("Invalid PPO curriculum exploration floor.")
    if any(floor < SAC_MIN_ENT_COEF for floor in SAC_ENT_COEF_FLOOR_BY_STAGE):
        raise ValueError("Invalid SAC curriculum entropy floor.")
    if not (SDF_NEAR_WALL_MARGIN_M > 0.0 and SDF_CLEARANCE_OBSERVATION_SCALE_M > 0.0):
        raise ValueError("SDF clearance scales must be positive.")
    if SDF_OUTSIDE_CENTER_TOLERANCE_M < 0.0 or SDF_OUTSIDE_CONFIRM_STEPS < 1:
        raise ValueError("Invalid SDF outside confirmation settings.")
    if SDF_BODY_WARNING_MARGIN_M <= 0.0:
        raise ValueError("SDF body warning margin must be positive.")
    if not (0.0 < SDF_SAMPLE_STEP_FRACTION <= 1.0):
        raise ValueError("SDF sample step fraction must be in (0, 1].")
    if (
        len(ACTOR_SHAFT_LOOKBACK_DISTANCES_M) != 3
        or any(distance <= 0.0 for distance in ACTOR_SHAFT_LOOKBACK_DISTANCES_M)
        or tuple(sorted(ACTOR_SHAFT_LOOKBACK_DISTANCES_M))
        != tuple(ACTOR_SHAFT_LOOKBACK_DISTANCES_M)
    ):
        raise ValueError("Actor shaft lookback distances must contain three ordered values.")
    if (
        len(CENTERLINE_LOOKAHEAD_DISTANCES_M) != 3
        or any(distance <= 0.0 for distance in CENTERLINE_LOOKAHEAD_DISTANCES_M)
        or tuple(sorted(CENTERLINE_LOOKAHEAD_DISTANCES_M))
        != tuple(CENTERLINE_LOOKAHEAD_DISTANCES_M)
    ):
        raise ValueError("Three ordered positive centerline lookahead distances are required.")
    if TIP_NEAR_WALL_GRACE_STEPS < 0 or TIP_NEAR_WALL_RAMP_STEPS < 1:
        raise ValueError("Invalid tip near-wall persistence settings.")
    if WRONG_BRANCH_DISTANCE_MARGIN_M <= 0.0 or WRONG_BRANCH_CONFIRM_STEPS < 1:
        raise ValueError("Invalid wrong-branch confirmation settings.")


validate_training_defaults()
