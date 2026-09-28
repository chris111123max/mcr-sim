# Formal PPO Beam Safety Metrics Acceptance

## Scope

Validate the formal PPO Beam-unilateral training entry after adding read-only
Beam safety telemetry.

This task MUST NOT change reward, observation/state, action, termination,
curriculum, PPO optimization, or Beam safety mathematics.

Production training path under test:

- training/py/train_ppo_beam_unilateral.py
- mcr_sim/beam_safety_env.py
- mcr_sim/beam_sdf_unilateral_fast.py
- mcr_sim/rl_core/beam_safety_metrics.py

Legacy baseline must remain untouched:

- training/py/train_ppo.py

## Invariants

Keep exactly:

- observation/state definition: legacy PPO baseline
- reward definition: legacy PPO baseline
- action definition: legacy PPO baseline
- done/termination logic: legacy PPO baseline
- curriculum logic: legacy PPO baseline
- physics: 2 x 5 ms
- requested Beam margin: +0.100 mm
- q_free dense spacing: 10 um
- Fast-FD finite-difference convention
- Beam-level scalar unilateral rows
- GenericConstraintSolver
- audit near-wall threshold: 0.300 mm
- audit interval: 16 physics substeps

The new metrics code may only consume info["beam_safety"] and write logger /
TensorBoard values. It must not feed values back into the environment or model.

## 1. Preflight

Work from:

cd /data/home/3220251075/mcr_sim/mcr_project/mCR_simulator-master/python

Record:

git status --short
git log -1 --oneline

Use:

PYTHON=/data/home/3220251075/mcr_sim/mcr_env/miniforge3/envs/mcr_sofa/bin/python

Run:

"$PYTHON" -m py_compile \
  mcr_sim/rl_core/beam_safety_metrics.py \
  mcr_sim/beam_safety_env.py \
  training/py/train_ppo_beam_unilateral.py

Require PASS.

## 2. Read-only unit smoke

Run:

"$PYTHON" testing/beam_feasible_domain/tests/beam_safety_metrics_read_only_smoke.py

Require:

BEAM_SAFETY_METRICS_READ_ONLY=PASS

The test proves that the collector does not mutate input info and correctly
handles cumulative-counter reset across episodes.

## 3. Static isolation check

Inspect git history/diff for the metrics patch.

Require:

- training/py/train_ppo.py unchanged
- reward code unchanged
- observation/state code unchanged
- action code unchanged
- done/termination code unchanged
- curriculum code unchanged
- mcr_sim/beam_sdf_unilateral_fast.py safety math unchanged
- native unilateral constraint unchanged

Only these tracked source changes are expected for this metrics patch:

- mcr_sim/rl_core/beam_safety_metrics.py added
- mcr_sim/beam_safety_env.py read-only counters added
- training/py/train_ppo_beam_unilateral.py Beam metrics callback attached
- testing/beam_feasible_domain/tests/beam_safety_metrics_read_only_smoke.py added
- this task document

Do not modify scientific parameters to make the test pass.

## 4. Short real 32-env logging smoke

Load Ascend runtime:

source /usr/local/Ascend/driver/bin/setenv.bash || true
source /data/home/3220251075/mcr_sim/mcr_env/cann/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH

Confirm plugin exists:

mcr_sim/native/beam_unilateral/_build/lib/libMCRBeamLinearizedUnilateral.so

Run a SHORT real PPO smoke only to verify the logging callback under
SubprocVecEnv. Do not start formal long training.

Use:

- device npu
- 32 env
- epochs 1
- episodes-per-epoch 1
- n_steps 64
- batch-size 1024
- PPO n_epochs 1
- max_episode_steps 64
- 2 x 5 ms physics
- training curriculum ON
- Beam validation audit
- audit interval 16
- near threshold 0.300 mm

The reduced max_episode_steps and PPO n_epochs are smoke-only runtime controls.
They do not change Beam safety mathematics.

Require:

- NPU npu:0
- 32 workers start
- real rollout runs
- at least one optimizer update
- no NaN/Inf
- no Beam safety exception
- final checkpoint/log created

## 5. Verify TensorBoard Beam namespaces

Locate the TensorBoard event file for the short run and inspect it with
TensorBoard EventAccumulator or equivalent local Python API.

Require keys from these namespaces:

beam_safety/
beam_safety_total/
beam_episode_recent_w50/

At minimum verify values exist for:

- beam_safety/physics_substeps_seen_rollout
- beam_safety/safe_free_substeps_rollout
- beam_safety/active_substeps_rollout
- beam_safety/near_wall_substeps_rollout
- beam_safety/unilateral_rows_built_rollout
- beam_safety/q_free_penetration_substeps_rollout
- beam_safety/committed_audits_rollout
- beam_safety/committed_audit_skips_rollout
- beam_safety/committed_verification_fraction_rollout
- beam_safety/worst_free_clearance_mm_rollout
- beam_safety/worst_audited_committed_clearance_mm_rollout
- beam_safety_total/completed_episodes_observed

Also confirm ordinary PPO/task metrics still exist, such as training loss / KL
when emitted by SB3 and the existing train/ / rollout_recent/ namespaces.

## 6. Counter consistency

For a successful rollout require:

safe_free_substeps + active_substeps == physics_substeps_seen

Require:

unilateral_rows_built >= active_substeps

Require:

0 <= committed_verification_fraction_rollout <= 1

q_free penetration substeps are diagnostic free-state events. They are not
committed penetration. Do not interpret them as a safety failure.

The actual hard safety failure remains a runtime exception from
BeamSafetyMCREnv.

## 7. No behavior change claim

Do not claim bitwise action/reward equivalence solely from TensorBoard values.

Instead verify by source isolation that this patch only:

- increments read-only counters after the already-computed planner result;
- adds fields to info["beam_safety"];
- consumes those fields in a callback;
- writes logger scalars.

No new metric may be used by observation, reward, action, done, curriculum,
constraint construction, or PPO loss.

## 8. Final report

Return:

FORMAL PPO BEAM METRICS ACCEPTANCE

Build:
py_compile =
read-only unit smoke =

Isolation:
legacy train_ppo changed = NO
reward changed = NO
observation/state changed = NO
action changed = NO
done changed = NO
curriculum changed = NO
Beam safety math changed = NO

32-env logging smoke:
NPU =
workers =
rollout =
optimizer update =
NaN/Inf =
Beam safety exceptions =
checkpoint/log =

TensorBoard:
beam_safety namespace =
beam_safety_total namespace =
beam_episode_recent_w50 namespace =
ordinary PPO/task metrics still present =

Counters:
physics substeps =
safe-free =
active =
safe-free + active == physics =
rows =
q_free penetration substeps =
audits =
skips =
verification fraction =
worst free clearance mm =
worst audited committed clearance mm =

FINAL:
METRICS INTEGRATION = PASS/FAIL
FORMAL 100-EPOCH TRAINING READY = YES/NO

Only say FORMAL 100-EPOCH TRAINING READY=YES if all required checks pass.

Stop after acceptance. Do not launch the formal 100-epoch training.
