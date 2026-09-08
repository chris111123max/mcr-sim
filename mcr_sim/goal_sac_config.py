"""Defaults for the isolated Goal-conditioned SAC experiment.

The server target is Huawei Ascend 910B3 (64 GiB HBM).  These defaults are
intentionally conservative for a twin-Q baseline: the default single-card
minibatch is 1024; distributed runs divide the same global batch by rank.
Arguments remain overrideable only for explicit ablation/smoke tests.
"""

GOAL_SAC_BATCH_SIZE = 1024
GOAL_SAC_BUFFER_SIZE = 500_000
GOAL_SAC_LEARNING_STARTS = 50_000
GOAL_SAC_LEARNING_RATE = 3e-4
GOAL_SAC_TRAIN_FREQ = 1
GOAL_SAC_TAU = 0.005
GOAL_SAC_GAMMA = 0.999
GOAL_SAC_ENT_COEF = "auto_0.001"
# 32 parallel SOFA environments is the single-910B3 default.  It preserves
# rollout diversity while avoiding the severe CPU oversubscription of 64 envs
# on the commonly allocated 22-core worker.
GOAL_SAC_N_ENVS = 32
GOAL_SAC_EPOCHS = 300
GOAL_SAC_EPISODES_PER_EPOCH = 100
GOAL_SAC_MAX_EPISODE_STEPS = 2048

# HER is deliberately opt-in.  It cannot create useful supervision before the
# policy produces safe, monotonic route progress; the previous run therefore
# requested 50% HER but actually relabelled 0% of sampled transitions.
GOAL_SAC_HER_RATIO = 0.0
GOAL_SAC_HER_SAFE_MARGIN_M = 0.0005
GOAL_SAC_GOAL_TOLERANCE = 0.01
GOAL_SAC_HER_MIN_GOAL_ADVANCE = 0.0
GOAL_SAC_STEP_COST = 0.005
GOAL_SAC_SAFETY_WEIGHT = 0.002
GOAL_SAC_ACTION_SMOOTHNESS_WEIGHT = 0.0001
GOAL_SAC_FAILURE_TERMINAL_PENALTY = 120.0
GOAL_SAC_TIMEOUT_PENALTY = 120.0
GOAL_SAC_OUT_OF_VESSEL_PENALTY = 150.0
GOAL_SAC_NON_FINITE_PENALTY = 200.0
GOAL_SAC_POTENTIAL_SCALE = 10.0
GOAL_SAC_SUCCESS_BONUS = 100.0
GOAL_SAC_MIN_ENT_COEF = 0.001
GOAL_SAC_CRITIC_ENSEMBLE_SIZE = 2
GOAL_SAC_TARGET_CRITIC_SUBSET_SIZE = 2
# UTD counts optimizer updates per vector step, not per environment transition.
# With 32 envs, UTD=2 means 2/32 updates per new transition (batch=1024).
GOAL_SAC_UTD_RATIO = 2
GOAL_SAC_UTD_WARMUP_STEPS = 50_000
GOAL_SAC_ACTOR_UPDATE_INTERVAL = 1
GOAL_SAC_CRITIC_LAYER_NORM = False
GOAL_SAC_METRIC_LOG_INTERVAL = 64
GOAL_SAC_PERFORMANCE_LOG_INTERVAL_STEPS = 512
