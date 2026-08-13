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

# Reward shaping.  Dense progress is normalized by 1 mm.  Its unit weight and
# the small one-shot waypoint bonus keep total route shaping below terminal
# success/failure magnitudes even for the longest 495 mm route.  VTI clearance
# supplies smooth wall-risk/penetration terms; the complete graph supplies a
# continuous off-route term plus a separately confirmed wrong-branch terminal.
REWARD_PROGRESS_NORMALIZATION_M = 0.001
REWARD_WAYPOINT_APPROACH = 1.0
REWARD_WAYPOINT_REACHED = 2.0
REWARD_TARGET_APPROACH = 1.0
REWARD_WALL_PROXIMITY = -0.5
REWARD_WALL_PENETRATION = -5.0
REWARD_OFF_TARGET_BRANCH = -2.0
REWARD_WRONG_BRANCH = -1000.0
REWARD_SUCCESS = 1500.0
REWARD_OUT_OF_VESSEL = -1500.0
REWARD_TIMEOUT = -1000.0
REWARD_STEP = -0.01

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
NUM_EPOCHS = 50
TRAIN_EPISODES_PER_EPOCH = 100
CHECKPOINT_INTERVAL = 1
VALID_INTERVAL = 2
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
SAC_GRADIENT_STEPS = 2
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
PPO_ENT_COEF = 0.0
PPO_VF_COEF = 0.5
PPO_MAX_GRAD_NORM = 0.5


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
        and REWARD_TIMEOUT < 0.0
    ):
        raise ValueError("Terminal reward signs are invalid.")
    dense_progress_upper = (
        longest_scaled_route
        / REWARD_PROGRESS_NORMALIZATION_M
        * max(REWARD_WAYPOINT_APPROACH, REWARD_TARGET_APPROACH)
    )
    waypoint_bonus_upper = (
        math.ceil(longest_scaled_route / WAYPOINT_SPACING_M)
        * REWARD_WAYPOINT_REACHED
    )
    shaping_upper = dense_progress_upper + waypoint_bonus_upper
    terminal_magnitude_floor = min(
        REWARD_SUCCESS,
        abs(REWARD_OUT_OF_VESSEL),
        abs(REWARD_WRONG_BRANCH),
        abs(REWARD_TIMEOUT),
    )
    if terminal_magnitude_floor <= shaping_upper:
        raise ValueError(
            "Terminal rewards must dominate the longest-route shaping upper bound."
        )
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
