"""Scale-aware defaults for non-ROS artificial-vessel SAC training.

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
OUT_OF_VESSEL_SAFETY_RATIO = 1.00
OUT_OF_VESSEL_FALLBACK_DISTANCE_M = 0.012

# Domain randomization only shrinks the generated vessels.  Keeping the upper
# bound at 1.0 preserves the nominal geometry and avoids making C05 longer than
# the controller's 510 mm insertion limit.
VESSEL_SCALE_MIN = 0.90
VESSEL_SCALE_MAX = 1.00
START_WINDOW_DISTANCE_M = 0.010
TARGET_WINDOW_DISTANCE_M = 0.010
INITIAL_ORIENTATION_MAX_ANGLE_DEG = 10.0
ENTRY_TANGENT_POINTS = 5

# Reward shaping.  Dense progress is normalized by 1 mm and therefore scales
# with actual motion rather than giving the same reward to a micron and a full
# insertion step.  Terminal outcomes remain larger than accumulated waypoint
# bonuses so a near-complete failure is distinct from a success.
REWARD_PROGRESS_NORMALIZATION_M = 0.001
REWARD_WAYPOINT_APPROACH = 5.0
REWARD_WAYPOINT_REACHED = 5.0
REWARD_TARGET_APPROACH = 5.0
REWARD_SUCCESS = 1000.0
REWARD_OUT_OF_VESSEL = -1000.0
REWARD_TIMEOUT = -500.0
REWARD_STEP = -0.05

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

# SAC defaults for the 4-NPU target.  Batch size and environment count are
# global values; train_sac.py divides them evenly between ranks.
SAC_EPOCHS = 50
SAC_STEPS_PER_EPOCH = 100_000
SAC_TOTAL_TIMESTEPS = SAC_EPOCHS * SAC_STEPS_PER_EPOCH
SAC_N_ENVS = 4
SAC_LEARNING_RATE = 3e-4
SAC_BATCH_SIZE = 512
SAC_BUFFER_SIZE = 500_000
SAC_LEARNING_STARTS = 50_000
SAC_TRAIN_FREQ = 1
SAC_GRADIENT_STEPS = -1
SAC_TAU = 0.005
SAC_GAMMA = 0.995


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
    if not (REWARD_SUCCESS > 0.0 and REWARD_OUT_OF_VESSEL < 0.0 and REWARD_TIMEOUT < 0.0):
        raise ValueError("Terminal reward signs are invalid.")


validate_training_defaults()
