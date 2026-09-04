"""Defaults for the isolated Goal-conditioned SAC experiment.

The server target is Huawei Ascend 910B3 (64 GiB HBM).  These defaults are
intentionally conservative for a ten-critic ensemble: the default single-card
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
GOAL_SAC_ENT_COEF = "auto"
# 32 parallel SOFA environments is the single-910B3 default.  It preserves
# rollout diversity while avoiding the severe CPU oversubscription of 64 envs
# on the commonly allocated 22-core worker.
GOAL_SAC_N_ENVS = 32
GOAL_SAC_EPOCHS = 100
GOAL_SAC_EPISODES_PER_EPOCH = 100
GOAL_SAC_MAX_EPISODE_STEPS = 2048

GOAL_SAC_HER_RATIO = 0.50
GOAL_SAC_HER_SAFE_MARGIN_M = 0.0005
GOAL_SAC_GOAL_TOLERANCE = 0.01
GOAL_SAC_STEP_COST = 0.01
GOAL_SAC_SAFETY_WEIGHT = 0.0005
GOAL_SAC_FAILURE_TERMINAL_PENALTY = 10.0
GOAL_SAC_CRITIC_ENSEMBLE_SIZE = 10
GOAL_SAC_TARGET_CRITIC_SUBSET_SIZE = 2
# With 32 vector environments and a 1024-sample minibatch, UTD=10 repeats far
# too much optimization on the earliest replay data.  A 1 -> 2 warmup keeps
# the ten-Q ensemble useful without collapsing entropy or dominating rollout.
GOAL_SAC_UTD_RATIO = 2
GOAL_SAC_UTD_WARMUP_STEPS = 50_000
GOAL_SAC_ACTOR_UPDATE_INTERVAL = 2
GOAL_SAC_CRITIC_LAYER_NORM = True
GOAL_SAC_METRIC_LOG_INTERVAL = 64
GOAL_SAC_PERFORMANCE_LOG_INTERVAL_STEPS = 512
