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
MAX_INSERTION_PER_ACTION_M = 0.0002

# Environment/task defaults.
SOFA_TIME_STEP_S = 0.01
FRAME_SKIP = 1
SETTLE_STEPS = 8
TARGET_THRESHOLD_M = 0.003
MAX_EPISODE_STEPS = 4096
RADIUS_OBSERVATION_SCALE_M = 0.005
ACTOR_HISTORY_STEPS = 4

WAYPOINT_SPACING_M = 0.005
WAYPOINT_REACH_THRESHOLD_M = 0.002
PRE_TARGET_WAYPOINT_OFFSET_M = 0.001
WAYPOINT_OBSERVATION_SCALE_M = 0.010
WAYPOINT_PROGRESS_CLIP_M = 0.001
WAYPOINT_HANDOFF_MARGIN_M = 0.0005
WAYPOINT_HANDOFF_CONFIRM_STEPS = 2

LOCAL_FIELD_ACTION_ANGLE_RAD = 2.0 * math.pi / 180.0
MAX_ACTION_DELTA = 0.30
# The ratio is retained only for legacy vessels that do not provide a VTI SDF.
OUT_OF_VESSEL_SAFETY_RATIO = 1.00
OUT_OF_VESSEL_FALLBACK_DISTANCE_M = 0.012

# Multi-model vessel safety.  The VTI stores centre-to-wall signed distance.
# Tip clearance drives dense safety features/rewards; whole-body samples are
# containment-only so the flexible shaft can contact and slide along the wall.
# A genuine outside termination requires any sampled catheter centre to remain
# at least 0.5 mm outside for three consecutive environment steps.
SDF_CLEARANCE_OBSERVATION_SCALE_M = 0.002
SDF_NEAR_WALL_MARGIN_M = 0.001
SDF_OUTSIDE_CENTER_TOLERANCE_M = 0.0005
SDF_OUTSIDE_CONFIRM_STEPS = 3
SDF_SAMPLE_STEP_FRACTION = 0.5
SDF_FORWARD_PROBE_DISTANCES_M = (0.001, 0.002, 0.004)
TIP_NEAR_WALL_GRACE_STEPS = 5
TIP_NEAR_WALL_RAMP_STEPS = 20

# On branching vessels, compare distance to the selected target route with
# distance to the complete centerline graph.  A 2 mm preference for another
# graph branch, sustained for five steps, is treated as a wrong-branch failure.
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

# Reward profile v4.  Dense navigation credit is the difference of an ordered
# route potential in [0, 1].  It therefore telescopes over a trajectory:
# forward/backward oscillation cannot farm reward, while progress credit becomes
# available again after a necessary correction in a tight bend.  Ordered
# waypoint hits remain one-shot by construction.  Every failed trajectory still
# remains negative and all magnitudes stay at O(1)..O(100).
REWARD_PROFILE_VERSION = 4
REWARD_PROGRESS_NORMALIZATION_M = TRAIN_ROUTE_MAX_LENGTH_M  # fallback before route setup
REWARD_PROGRESS_BUDGET = 100.0
REWARD_WAYPOINT_BUDGET = 20.0
REWARD_WAYPOINT_APPROACH = REWARD_PROGRESS_BUDGET
REWARD_WAYPOINT_REACHED = REWARD_WAYPOINT_BUDGET
REWARD_TARGET_APPROACH = REWARD_PROGRESS_BUDGET
REWARD_WALL_PROXIMITY = -0.02
REWARD_WALL_PENETRATION = -0.50
REWARD_OFF_TARGET_BRANCH = -0.10
REWARD_RETRACTION = -0.03
REWARD_NO_PROGRESS = -0.02
REWARD_WRONG_BRANCH = -150.0
REWARD_SUCCESS = 150.0
REWARD_OUT_OF_VESSEL = -150.0
REWARD_NON_FINITE = -150.0
REWARD_TIMEOUT = -120.0
REWARD_NO_PROGRESS_TERMINAL = -120.0
REWARD_STEP = -0.002

# A policy that stays at the insertion lower bound must not fill the replay
# buffer with 4096-step timeout episodes.  Net ordered-waypoint approach is
# measured over a long window so normal magnetic steering pauses are allowed.
# Only sustained stagnation after the window is full becomes terminal.
NO_PROGRESS_WINDOW_STEPS = 256
NO_PROGRESS_GRACE_STEPS = 256
NO_PROGRESS_CONFIRM_STEPS = 512
NO_PROGRESS_MIN_NET_APPROACH_M = 0.001

# SAC has no gradient clipping in upstream SB3.  The shared distributed SAC
# implementation applies this bound after cross-rank averaging and before the
# optimizer step.  PPO retains its own PPO_MAX_GRAD_NORM below.
SAC_MAX_GRAD_NORM = 10.0
SAC_MIN_ENT_COEF = 0.02


def reward_profile() -> dict:
    """Return the exact shared reward/no-progress settings for run metadata."""

    return {
        "version": REWARD_PROFILE_VERSION,
        "progress_normalization": "ordered_route_potential_difference",
        "progress_normalization_m": REWARD_PROGRESS_NORMALIZATION_M,
        "progress_budget": REWARD_PROGRESS_BUDGET,
        "waypoint_budget": REWARD_WAYPOINT_BUDGET,
        "waypoint_approach": REWARD_WAYPOINT_APPROACH,
        "waypoint_reached": REWARD_WAYPOINT_REACHED,
        "target_approach": REWARD_TARGET_APPROACH,
        "wall_proximity": REWARD_WALL_PROXIMITY,
        "wall_penetration": REWARD_WALL_PENETRATION,
        "off_target_branch": REWARD_OFF_TARGET_BRANCH,
        "retraction": REWARD_RETRACTION,
        "no_progress": REWARD_NO_PROGRESS,
        "wrong_branch": REWARD_WRONG_BRANCH,
        "success": REWARD_SUCCESS,
        "out_of_vessel": REWARD_OUT_OF_VESSEL,
        "non_finite": REWARD_NON_FINITE,
        "timeout": REWARD_TIMEOUT,
        "no_progress_terminal": REWARD_NO_PROGRESS_TERMINAL,
        "step": REWARD_STEP,
        "no_progress_window_steps": NO_PROGRESS_WINDOW_STEPS,
        "no_progress_grace_steps": NO_PROGRESS_GRACE_STEPS,
        "no_progress_confirm_steps": NO_PROGRESS_CONFIRM_STEPS,
        "no_progress_min_net_approach_m": NO_PROGRESS_MIN_NET_APPROACH_M,
    }


def ordered_route_potential(
    start_progress: float,
    target_progress: float,
    waypoint_progress,
    active_waypoint_index: int,
    active_waypoint_distance: float,
) -> float:
    """Return bounded progress through the strictly ordered waypoint task.

    The completed prefix comes from the active waypoint index.  Progress inside
    the active segment comes only from Euclidean approach to that waypoint, so
    the reward never exposes a privileged global centerline projection to the
    policy and cannot jump to an unrelated branch.
    """

    start = float(start_progress)
    target = float(target_progress)
    route_length = target - start
    points = [float(value) for value in waypoint_progress]
    if not points or not math.isfinite(route_length) or route_length <= 1e-9:
        return 0.0

    index = min(max(int(active_waypoint_index), 0), len(points) - 1)
    active = min(max(points[index], start), target)
    previous = start if index == 0 else min(max(points[index - 1], start), target)
    segment_length = max(active - previous, 0.0)
    distance = float(active_waypoint_distance)
    if not math.isfinite(distance):
        completion = 0.0
    elif segment_length <= 1e-9:
        completion = 0.0
    else:
        completion = min(max(1.0 - distance / segment_length, 0.0), 1.0)
    travelled = max(previous - start, 0.0) + segment_length * completion
    return min(max(travelled / route_length, 0.0), 1.0)


# Training-only vessel curriculum.  Domain randomization remains enabled in
# every stage; only the geometry pool expands.  Validation always uses V01..V05
# directly and is never simplified by this curriculum.
TRAINING_CURRICULUM_ENABLED = True
TRAINING_CURRICULUM_MODELS = (
    ("B01", "B02"),
    ("B01", "B02", "B03", "B04", "B05"),
    ("B01", "B02", "B03", "B04", "B05", "C01", "C02"),
    ("B01", "B02", "B03", "B04", "B05", "C01", "C02", "C03", "C04", "C05"),
)
TRAINING_CURRICULUM_SUCCESS_THRESHOLDS = (0.10, 0.10, 0.10)
TRAINING_CURRICULUM_CONSECUTIVE_EPOCHS = 3


def update_curriculum_progress(
    current_stage: int,
    consecutive_success_epochs: int,
    train_success_rate: float,
):
    """Update the stable-success streak and advance at most one stage."""

    stage = min(max(int(current_stage), 0), len(TRAINING_CURRICULUM_MODELS) - 1)
    streak = max(int(consecutive_success_epochs), 0)
    if stage >= len(TRAINING_CURRICULUM_SUCCESS_THRESHOLDS):
        return stage, 0
    if float(train_success_rate) >= TRAINING_CURRICULUM_SUCCESS_THRESHOLDS[stage]:
        streak += 1
        if streak >= TRAINING_CURRICULUM_CONSECUTIVE_EPOCHS:
            stage += 1
            streak = 0
    else:
        streak = 0
    return stage, streak


def update_validation_unlocked(
    previously_unlocked: bool,
    train_success_rate: float,
    minimum_train_success_rate: float,
) -> bool:
    """Latch validation on once the shared training-success gate is reached."""

    return bool(
        previously_unlocked
        or float(minimum_train_success_rate) <= 0.0
        or float(train_success_rate) >= float(minimum_train_success_rate)
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
SAC_BATCH_SIZE = 2048
SAC_BUFFER_SIZE = 500_000
SAC_LEARNING_STARTS = 50_000
SAC_TRAIN_FREQ = 1
SAC_GRADIENT_STEPS = 4
SAC_TAU = 0.005
SAC_GAMMA = 0.995

# PPO baseline defaults.  ``n_steps`` is per environment; batch size is global
# and is divided evenly between synchronized ranks, just like SAC.
PPO_EPOCHS = NUM_EPOCHS
PPO_EPISODES_PER_EPOCH = TRAIN_EPISODES_PER_EPOCH
PPO_N_ENVS = SAC_N_ENVS
PPO_LEARNING_RATE = 3e-4
PPO_N_STEPS = 512
PPO_BATCH_SIZE = 512
PPO_N_EPOCHS = 10
PPO_GAMMA = SAC_GAMMA
PPO_GAE_LAMBDA = 0.95
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
    minimum_scaled_lumen_radius = TRAIN_VESSEL_MIN_RADIUS_M * VESSEL_SCALE_MIN
    if WAYPOINT_REACH_THRESHOLD_M >= minimum_scaled_lumen_radius:
        raise ValueError("Waypoint threshold must be smaller than the minimum lumen radius.")
    if not (0.0 <= LMD_CONTACT_DISTANCE_M < LMD_ALARM_DISTANCE_M):
        raise ValueError("LocalMinDistance requires contactDistance < alarmDistance.")
    if not (0.0 < OUT_OF_VESSEL_SAFETY_RATIO <= 1.0):
        raise ValueError("Out-of-vessel ratio must be in (0, 1] for physical containment.")
    if START_WINDOW_DISTANCE_M < 0.0 or TARGET_WINDOW_DISTANCE_M < 0.0:
        raise ValueError("Endpoint randomization distances must be non-negative.")
    if REWARD_PROGRESS_NORMALIZATION_M <= 0.0:
        raise ValueError("Reward progress normalization must be positive.")
    if not (
        REWARD_SUCCESS > 0.0
        and REWARD_OUT_OF_VESSEL < 0.0
        and REWARD_WRONG_BRANCH < 0.0
        and REWARD_NON_FINITE < 0.0
        and REWARD_TIMEOUT < 0.0
        and REWARD_NO_PROGRESS_TERMINAL < 0.0
    ):
        raise ValueError("Terminal reward signs are invalid.")
    maximum_navigation_credit = REWARD_PROGRESS_BUDGET + REWARD_WAYPOINT_BUDGET
    if not (
        REWARD_OUT_OF_VESSEL < -maximum_navigation_credit
        and REWARD_WRONG_BRANCH < -maximum_navigation_credit
        and REWARD_NON_FINITE < -maximum_navigation_credit
        and REWARD_TIMEOUT <= -maximum_navigation_credit
        and REWARD_NO_PROGRESS_TERMINAL <= -maximum_navigation_credit
    ):
        raise ValueError("Every failed trajectory must remain negative after maximum navigation credit.")
    longest_route_forward_reward = (
        REWARD_PROGRESS_BUDGET
        * MAX_INSERTION_PER_ACTION_M
        / TRAIN_ROUTE_MAX_LENGTH_M
        + REWARD_WALL_PROXIMITY
        + REWARD_STEP
    )
    if longest_route_forward_reward <= 0.0:
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
    if TRAINING_CURRICULUM_CONSECUTIVE_EPOCHS < 1:
        raise ValueError("Curriculum consecutive epoch count must be positive.")
    if not (0.0 < PPO_MIN_ACTION_STD <= PPO_MAX_ACTION_STD):
        raise ValueError("Invalid PPO action standard-deviation bounds.")
    if not (SDF_NEAR_WALL_MARGIN_M > 0.0 and SDF_CLEARANCE_OBSERVATION_SCALE_M > 0.0):
        raise ValueError("SDF clearance scales must be positive.")
    if SDF_OUTSIDE_CENTER_TOLERANCE_M < 0.0 or SDF_OUTSIDE_CONFIRM_STEPS < 1:
        raise ValueError("Invalid SDF outside confirmation settings.")
    if not (0.0 < SDF_SAMPLE_STEP_FRACTION <= 1.0):
        raise ValueError("SDF sample step fraction must be in (0, 1].")
    if (
        len(SDF_FORWARD_PROBE_DISTANCES_M) == 0
        or any(distance <= 0.0 for distance in SDF_FORWARD_PROBE_DISTANCES_M)
        or tuple(sorted(SDF_FORWARD_PROBE_DISTANCES_M))
        != tuple(SDF_FORWARD_PROBE_DISTANCES_M)
    ):
        raise ValueError("SDF forward probe distances must be positive and ordered.")
    if TIP_NEAR_WALL_GRACE_STEPS < 0 or TIP_NEAR_WALL_RAMP_STEPS < 1:
        raise ValueError("Invalid tip near-wall persistence settings.")
    if WRONG_BRANCH_DISTANCE_MARGIN_M <= 0.0 or WRONG_BRANCH_CONFIRM_STEPS < 1:
        raise ValueError("Invalid wrong-branch confirmation settings.")


validate_training_defaults()
