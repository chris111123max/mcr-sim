"""Scale-aware defaults for non-ROS artificial-vessel RL training.

The ten generated training vessels define the design envelope below.  Keep
these values synchronized with ``tools/generate_artificial_vessels.py`` when
the generator geometry changes.
"""

import math


# Geometry envelope (B01..B05 and C01..C05, source assets scaled to metres).
CATHETER_OUTER_DIAMETER_M = 0.00133
CATHETER_RADIUS_M = CATHETER_OUTER_DIAMETER_M / 2.0
TRAIN_VESSEL_MIN_RADIUS_M = 0.0028
TRAIN_VESSEL_MAX_RADIUS_M = 0.0052
TRAIN_ROUTE_MAX_LENGTH_M = 0.495
CONTROLLER_MAX_INSERTION_M = 0.510
MAX_INSERTION_PER_ACTION_M = 0.0004

# Environment/task defaults.
SOFA_TIME_STEP_S = 0.01
FRAME_SKIP = 1
SETTLE_STEPS = 8
TARGET_THRESHOLD_M = 0.003
MAX_EPISODE_STEPS = 2048
RADIUS_OBSERVATION_SCALE_M = 0.005
ACTOR_HISTORY_STEPS = 1
ACTOR_SHAFT_LOOKBACK_DISTANCES_M = (0.010, 0.030, 0.060)
VESSEL_SECTION_FEATURE_DIM = 33
ACTOR_STATIC_ROUTE_FEATURE_DIM = 12
ACTOR_CURRENT_GEOMETRY_DIM = ACTOR_STATIC_ROUTE_FEATURE_DIM + VESSEL_SECTION_FEATURE_DIM
ACTOR_DYNAMIC_STEP_DIM = 7
ACTOR_OBSERVATION_DIM = (
    ACTOR_CURRENT_GEOMETRY_DIM + ACTOR_HISTORY_STEPS * ACTOR_DYNAMIC_STEP_DIM
)

# Continuous selected-route tracking. Initial localization may inspect the
# complete route; recurrent tracking is local and physically gated so nearby
# arms of a U-turn cannot create artificial progress. Two moving guidance
# points replace discrete waypoint spheres.
ROUTE_GUIDANCE_LOOKAHEAD_DISTANCES_M = (0.010, 0.020)
ROUTE_GUIDANCE_OBSERVATION_SCALE_M = 0.020
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
# at least 0.5 mm outside for three consecutive environment steps.  Reward V10
# exposes the already-computed whole-body margin and starts a bounded warning
# ramp 0.5 mm before the centre reaches the wall; shaft contact remains legal.
SDF_CLEARANCE_OBSERVATION_SCALE_M = 0.002
SDF_NEAR_WALL_MARGIN_M = 0.001
SDF_OUTSIDE_CENTER_TOLERANCE_M = 0.0005
SDF_BODY_WARNING_MARGIN_M = 0.0005
SDF_OUTSIDE_CONFIRM_STEPS = 3
SDF_SAMPLE_STEP_FRACTION = 0.5
CENTERLINE_LOOKAHEAD_DISTANCES_M = (0.005, 0.010, 0.020)
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

# Reward profile v10.  The dense term is a potential over physical selected-route
# metres, not a route-length-normalized percentage.  Discount-matched potential
# shaping redistributes feedback without changing the policy ordering defined by
# the base terminal/safety/time objective.  Retraction is naturally negative;
# curve anticipation is learned from state instead of action-dependent shaping.
REWARD_PROFILE_VERSION = 10
REWARD_PROGRESS_NORMALIZATION_M = TRAIN_ROUTE_MAX_LENGTH_M  # fallback before route setup
REWARD_PROGRESS_PER_M = 400.0
REWARD_ROUTE_PROGRESS = REWARD_PROGRESS_PER_M
REWARD_PROGRESS_BUDGET = REWARD_PROGRESS_PER_M * TRAIN_ROUTE_MAX_LENGTH_M
# Potential-based route shaping must use the same discount as every learner:
# F(s,s') = gamma * Phi(s') - Phi(s).  Terminal Phi is exactly zero.
REWARD_DISCOUNT_GAMMA = 0.9995
REWARD_WALL_PROXIMITY = -0.10
REWARD_OFF_TARGET_BRANCH = -0.10
REWARD_SUCCESS = 500.0
REWARD_OUT_OF_VESSEL = -500.0
REWARD_NON_FINITE = -500.0
REWARD_TIMEOUT = -500.0
REWARD_STEP = -0.01

# Net continuous route progress is measured over a long window for diagnostics.
# It neither changes reward nor terminates an episode in Reward V10.
NO_PROGRESS_WINDOW_STEPS = 256
NO_PROGRESS_GRACE_STEPS = 256
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
        "progress_normalization": "physical_selected_route_potential_m",
        "progress_normalization_m": REWARD_PROGRESS_NORMALIZATION_M,
        "progress_per_m": REWARD_PROGRESS_PER_M,
        "discount_gamma": float(discount_gamma),
        "progress_formula": "gamma*route_progress_m_next-route_progress_m_previous",
        "progress_budget": REWARD_PROGRESS_BUDGET,
        "route_progress": REWARD_ROUTE_PROGRESS,
        "wall_proximity": REWARD_WALL_PROXIMITY,
        "off_target_branch": REWARD_OFF_TARGET_BRANCH,
        "success": REWARD_SUCCESS,
        "out_of_vessel": REWARD_OUT_OF_VESSEL,
        "non_finite": REWARD_NON_FINITE,
        "timeout": REWARD_TIMEOUT,
        "step": REWARD_STEP,
        "body_sdf_warning_margin_m": SDF_BODY_WARNING_MARGIN_M,
        "no_progress_window_steps": NO_PROGRESS_WINDOW_STEPS,
        "no_progress_grace_steps": NO_PROGRESS_GRACE_STEPS,
        "no_progress_confirm_steps": NO_PROGRESS_CONFIRM_STEPS,
        "no_progress_min_net_approach_m": NO_PROGRESS_MIN_NET_APPROACH_M,
    }


def body_sdf_risk_features(
    max_signed_distance: float,
    outside_tolerance: float = SDF_OUTSIDE_CENTER_TOLERANCE_M,
    warning_margin: float = SDF_BODY_WARNING_MARGIN_M,
):
    """Return bounded whole-body warning and outside-depth features.

    ``max_signed_distance`` is the worst catheter-centre SDF sample: negative
    values are inside the lumen and positive values are outside.  Contact is
    not itself a failure.  The warning ramps from ``-warning_margin`` to the
    configured outside threshold, while penetration starts only after the
    sampled centre has crossed the wall.
    """

    signed = float(max_signed_distance)
    tolerance = max(float(outside_tolerance), 1e-12)
    margin = max(float(warning_margin), 0.0)
    if not math.isfinite(signed):
        return 0.0, 0.0
    warning = min(
        max((signed + margin) / max(tolerance + margin, 1e-12), 0.0),
        1.0,
    )
    outside_depth = min(max(signed / tolerance, 0.0), 1.0)
    return warning, outside_depth


# Training-only five-stage vessel/robustness curriculum.  Every stage uses the
# complete route.  Branching B01/B02 are learned first; their aggregate rolling
# success must stably reach 90% before the continuously curved C01/C02 are
# isolated for bend-control learning.  The four-vessel pool is then trained with
# full DR, followed by all vessels on fixed geometry and finally full DR.
# Validation remains locked until the final all-vessel/full-DR stage is active.
TRAINING_CURRICULUM_ENABLED = True
TRAINING_CURRICULUM_ALL_MODELS = (
    "B01", "B02", "B03", "B04", "B05",
    "C01", "C02", "C03", "C04", "C05",
)
TRAINING_CURRICULUM_BRANCH_MODELS = ("B01", "B02")
TRAINING_CURRICULUM_CURVED_MODELS = ("C01", "C02")
TRAINING_CURRICULUM_SIMPLE_MODELS = (
    *TRAINING_CURRICULUM_BRANCH_MODELS,
    *TRAINING_CURRICULUM_CURVED_MODELS,
)
TRAINING_CURRICULUM_MODELS = (
    TRAINING_CURRICULUM_BRANCH_MODELS,
    TRAINING_CURRICULUM_CURVED_MODELS,
    TRAINING_CURRICULUM_SIMPLE_MODELS,
    TRAINING_CURRICULUM_ALL_MODELS,
    TRAINING_CURRICULUM_ALL_MODELS,
)
TRAINING_CURRICULUM_STAGE_NAMES = (
    "branch_fixed",
    "curved_fixed",
    "simple_full_dr",
    "all_fixed",
    "all_full_dr",
)
TRAINING_CURRICULUM_TARGET_FRACTIONS = (1.00,) * 5
TRAINING_CURRICULUM_SUCCESS_THRESHOLDS = (0.90, 0.50, 0.50, 0.50)
# Stage 0 follows the requested aggregate B01/B02 criterion.  Later stages use
# the weakest active vessel so one geometry cannot hide another vessel's failure.
TRAINING_CURRICULUM_PROMOTION_MODES = (
    "aggregate",
    "per_vessel_min",
    "per_vessel_min",
    "per_vessel_min",
)
TRAINING_CURRICULUM_CONSECUTIVE_SUCCESS_EPISODES = 3
TRAINING_CURRICULUM_ROLLING_EPISODES_PER_VESSEL = 100
TRAINING_CURRICULUM_MIN_EPISODES_PER_VESSEL = 100
# DR is deliberately disabled again when difficult vessels are first added.
TRAINING_CURRICULUM_DR_FRACTIONS = (0.00, 0.00, 1.00, 0.00, 1.00)
# Half the sampling distribution remains uniform.  The adaptive half focuses
# on weak vessels but is capped so a single failure mode cannot erase skills
# already acquired on the rest of the active pool.
TRAINING_CURRICULUM_UNIFORM_SAMPLING_MIX = 0.50
TRAINING_CURRICULUM_DIFFICULTY_POWER = 2.0
TRAINING_CURRICULUM_MAX_SAMPLING_FACTOR = 2.0
# Use the previously stable exploration floors in every stage.  Exploration
# still anneals naturally through the learned PPO log_std / SAC entropy tuner.
PPO_ACTION_STD_FLOOR_BY_STAGE = (0.25,) * 5
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
        "sac_ent_coef_floor_by_stage": list(SAC_ENT_COEF_FLOOR_BY_STAGE),
        "centerline_lookahead_distances_m": list(CENTERLINE_LOOKAHEAD_DISTANCES_M),
        "actor_future_tangent_features": "tip_local_tangents_5_10_20mm",
        "actor_shaft_lookback_distances_m": list(ACTOR_SHAFT_LOOKBACK_DISTANCES_M),
        "actor_time_feature": "fraction_of_episode_remaining",
        "actor_inserted_length_feature": "controller_insertion_fraction",
        "actor_history_steps": ACTOR_HISTORY_STEPS,
        "local_field_action_angle_deg": math.degrees(LOCAL_FIELD_ACTION_ANGLE_RAD),
        "navigation": "continuous_selected_route",
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
        "actor_route_horizon_feature": "remaining_route_distance",
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
PPO_N_ENVS = SAC_N_ENVS
PPO_LEARNING_RATE = 3e-4
PPO_N_STEPS = 256
PPO_BATCH_SIZE = 1024
PPO_N_EPOCHS = 10
PPO_GAMMA = SAC_GAMMA
PPO_GAE_LAMBDA = 0.98
PPO_CLIP_RANGE = 0.2
PPO_ENT_COEF = 0.001
PPO_VF_COEF = 0.5
PPO_MAX_GRAD_NORM = 0.5
PPO_MIN_ACTION_STD = 0.25
PPO_MAX_ACTION_STD = 1.0


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
        and REWARD_WALL_PROXIMITY < 0.0
        and REWARD_OFF_TARGET_BRANCH < 0.0
        and REWARD_SUCCESS > 0.0
        and REWARD_OUT_OF_VESSEL < 0.0
        and REWARD_NON_FINITE < 0.0
        and REWARD_TIMEOUT < 0.0
        and REWARD_STEP < 0.0
    ):
        raise ValueError("Reward V10 signs are invalid.")
    maximum_navigation_credit = REWARD_PROGRESS_BUDGET
    if not (
        REWARD_OUT_OF_VESSEL < -maximum_navigation_credit
        and REWARD_NON_FINITE < -maximum_navigation_credit
        and REWARD_TIMEOUT <= -maximum_navigation_credit
    ):
        raise ValueError("Every terminal failure must remain negative after maximum navigation credit.")
    longest_route_safe_forward_reward = (
        REWARD_ROUTE_PROGRESS
        * (
            REWARD_DISCOUNT_GAMMA * TRAIN_ROUTE_MAX_LENGTH_M
            - (TRAIN_ROUTE_MAX_LENGTH_M - MAX_INSERTION_PER_ACTION_M)
        )
        + REWARD_STEP
    )
    if longest_route_safe_forward_reward <= 0.0:
        raise ValueError("Safe full insertion must remain positive on the longest route.")
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
    if TRAINING_CURRICULUM_MODELS != (
        TRAINING_CURRICULUM_BRANCH_MODELS,
        TRAINING_CURRICULUM_CURVED_MODELS,
        TRAINING_CURRICULUM_SIMPLE_MODELS,
        TRAINING_CURRICULUM_ALL_MODELS,
        TRAINING_CURRICULUM_ALL_MODELS,
    ):
        raise ValueError("The five-stage vessel curriculum is inconsistent.")
    if not set(TRAINING_CURRICULUM_SIMPLE_MODELS).issubset(
        set(TRAINING_CURRICULUM_ALL_MODELS)
    ):
        raise ValueError("Simple curriculum vessels must belong to the training pool.")
    if any(fraction != 1.0 for fraction in TRAINING_CURRICULUM_TARGET_FRACTIONS):
        raise ValueError("Every curriculum stage must use the complete route.")
    if TRAINING_CURRICULUM_DR_FRACTIONS != (0.0, 0.0, 1.0, 0.0, 1.0):
        raise ValueError("The five-stage curriculum DR schedule is inconsistent.")
    if TRAINING_CURRICULUM_PROMOTION_MODES != (
        "aggregate",
        "per_vessel_min",
        "per_vessel_min",
        "per_vessel_min",
    ):
        raise ValueError("The curriculum promotion modes are inconsistent.")
    if TRAINING_CURRICULUM_DR_FRACTIONS[-1] != 1.0:
        raise ValueError("The final curriculum stage must use full domain randomization.")
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
    if any(
        not (0.0 < threshold <= 1.0)
        for threshold in TRAINING_CURRICULUM_SUCCESS_THRESHOLDS
    ):
        raise ValueError("Curriculum success thresholds must be in (0, 1].")
    if not (0.0 < PPO_MIN_ACTION_STD <= PPO_MAX_ACTION_STD):
        raise ValueError("Invalid PPO action standard-deviation bounds.")
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
