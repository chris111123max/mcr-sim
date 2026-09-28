# Production Beam Safety Audit Integration Acceptance

## Goal

Validate the newly promoted production Fast-FD Beam/SDF unilateral path before
any PPO training is started.

Production files under test:

- mcr_sim/beam_sdf_unilateral_fast.py
- mcr_sim/beam_safety_env.py
- mcr_sim/native/beam_unilateral/
- training/py/train_ppo_beam_unilateral.py

Legacy training/py/train_ppo.py must remain unchanged.

## Semantics

Physical safety is active on EVERY physics substep in all modes:

live Beam Rigid3 free state
-> one 10 um Beam/SDF profile
-> Fast local-point FD rows when q_free < +0.100 mm
-> GenericConstraintSolver
-> committed Beam state

Only the independent committed verifier changes:

- debug: committed 10 um validation every physics substep.
- audit: validate every ACTIVE substep, every near-wall substep
  (q_free <= 0.300 mm by default), every 16th far-safe physics substep,
  reset, and terminal state.
- off: independent verifier disabled; physical constraint stays active.
  Do not use off for this acceptance.

Committed penetration audit limit stays 0.001 mm.

## 1. Pull / preflight

cd /data/home/3220251075/mcr_sim/mcr_project/mCR_simulator-master/python

git status --short
git log -1 --oneline

PYTHON=/data/home/3220251075/mcr_sim/mcr_env/miniforge3/envs/mcr_sofa/bin/python

"$PYTHON" -m py_compile \
  mcr_sim/beam_sdf_unilateral_fast.py \
  mcr_sim/beam_safety_env.py \
  training/py/train_ppo_beam_unilateral.py

If a trivial new-file syntax/import issue appears, make only the minimum
correction in the new Beam-safety files. Do not change old baselines or
scientific constants.

## 2. Build production native plugin

Use the same SOFA 21.12 build environment that compiled the accepted test
plugin:

bash mcr_sim/native/beam_unilateral/build.sh

Require:

mcr_sim/native/beam_unilateral/_build/lib/libMCRBeamLinearizedUnilateral.so

If a compile-only SOFA compatibility error appears, compare against:

testing/beam_feasible_domain/beam_unilateral_lcp/native/

and make only the minimum compile fix. Do not change constraint semantics.

## 3. Ascend runtime

source /usr/local/Ascend/driver/bin/setenv.bash || true
source /data/home/3220251075/mcr_sim/mcr_env/cann/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:\${LD_LIBRARY_PATH:-}

Then:

"$PYTHON" - <<'PY'
from mcr_sim.distributed import resolve_device
d = resolve_device("npu", distributed=False, local_rank=0)
print("resolved =", d.resolved)
print("accelerator =", d.accelerator)
assert d.resolved == "npu:0"
assert d.accelerator == "npu"
print("NPU_SMOKE=PASS")
PY

No CPU fallback.

## 4. Fixed-frame production versus validated V2

Write a TEST-ONLY runner under testing/beam_feasible_domain/tests/.

Replay the protected B02 / target_04 / seed15204 step665 state and compare:

validated V2:
testing/beam_feasible_domain/beam_unilateral_lcp/beam_unilateral_fast_fd_v2.py

production:
mcr_sim/beam_sdf_unilateral_fast.py

Use the exact same q_free state and production SDF.

Require:

- dense sample count equal;
- selected dense indices equal;
- selected elements equal;
- row offsets equal;
- DOF indices equal;
- source clearances equal;
- linear Jacobian max abs error = 0 if the exact path is preserved;
- angular Jacobian max abs error = 0 if the exact path is preserved;
- row SHA equal;
- support mode exact_selected_point_local_fd;
- production q_free internal full profile count = 0;
- q_prev full dense measurement = 0;
- no CollisionDOF safety source;
- no projection/rollback/action shielding/state writes.

If the production live adapter differs from the validated V2 geometry, STOP at
the first mismatch. Do not tune thresholds.

## 5. Production DEBUG full episode

Only after fixed-frame PASS.

Use BeamSafetyMCREnv directly in debug mode:

B02 / target_04 / seed15204
epoch74 deterministic
2048 RL steps
4096 physics substeps
2 x 5 ms
GenericConstraintSolver
margin +0.100 mm
dense spacing 0.010 mm
committed penetration limit 0.001 mm

Reference action SHA:

e6854f5496eb2d43eb913d1116238d831666f7589ada051325bfdf14d294900e

Reference V2:

safe-free = 2660
active = 1436
total rows = 7000
q_free < 0 = 636
q_free < -0.001 mm = 625
worst free = -0.395220 mm
worst committed = +0.026822 mm
committed penetration = 0

DEBUG must audit every physics substep.

Require:
- action SHA match;
- no committed penetration violation;
- activeCount mismatch = 0;
- planner/build failures = 0;
- NaN/Inf = 0.

## 6. Production AUDIT fixed episode

Run the same fixed episode in audit mode with defaults:

interval = 16 physics substeps
near threshold = 0.300 mm

The physical q_free profile, Fast FD rows and Generic solve remain every
physics substep.

Require:
- same action SHA as DEBUG and V2;
- same active substeps / row behavior;
- every ACTIVE substep audited;
- every q_free <= 0.300 mm substep audited;
- far-safe periodic audits occur;
- reset audit occurs;
- terminal audit occurs when needed;
- at least one far-safe committed audit is skipped;
- no audited penetration violation;
- no planner/activeCount/nonfinite failure.

Do not claim audit alone checks every skipped committed state. DEBUG is the
full per-substep oracle.

## 7. Optional 32-env audit throughput

Only after sections 1-6 PASS, run one rollout-only test:

32 SubprocVecEnv spawn workers
B02 / target_04
epoch74 deterministic
NPU npu:0
700 vector steps
100 warm-up / 600 measured
audit mode
NO PPO optimizer update

Report:
- env-steps/s;
- vector mean / p95;
- committed audit count / skip count;
- active substeps;
- safety failures.

Reference V2 with every-substep committed validation:
171.69 env-steps/s.

Measure; do not force a target speed.

## Hard restrictions

Do not modify:
- training/py/train_ppo.py
- old PPO/SAC baselines
- +0.100 mm margin
- 10 um q_free dense spacing
- FD step sizes
- row selection
- GenericConstraintSolver formulation
- 2 x 5 ms physics

Do not:
- use CPU fallback;
- train;
- run optimizer updates;
- add action shielding;
- add rollback/projection;
- write Beam free/committed positions;
- use CollisionDOFs as safety source.

If a true scientific/safety mismatch occurs, stop. Do not auto-tune.

## Final report

Return:

PRODUCTION BEAM AUDIT ACCEPTANCE

Build:
py_compile = ...
native plugin = ...
legacy train_ppo modified = NO
NPU smoke = ...

Step665 production/V2:
selected indices identical = ...
row structure identical = ...
linear max abs error = ...
angular max abs error = ...
row SHA equal = ...
live adapter equivalent = ...
FINAL = PASS/FAIL

DEBUG 2048:
action SHA match = ...
RL steps = ...
physics substeps = ...
active substeps = ...
total rows = ...
committed audits = ...
worst free = ...
worst committed = ...
max committed penetration = ...
activeCount mismatch = ...
planning failures = ...
FINAL = PASS/FAIL

AUDIT 2048:
action SHA match = ...
physical trajectory equivalent = ...
committed audits = ...
committed audit skips = ...
all ACTIVE audited = ...
all near-wall audited = ...
periodic far-safe audits = ...
reset/terminal audit = ...
audited penetration violations = ...
FINAL = PASS/FAIL

32-env audit throughput, if run:
env-steps/s = ...
vector mean/p95 = ...
audit count/skips = ...
safety failures = ...

TRAINING READY = YES/NO

Do not say TRAINING READY=YES unless build, fixed-frame, DEBUG full episode,
and AUDIT full episode all PASS.

Then STOP. Do not start training.
