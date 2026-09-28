# Codex task: B02 2048-step fast local-point FD Beam unilateral acceptance

## Goal

Run the fixed B02 / target_04 / seed15204 2048-step engineering acceptance
using the fast local-point finite-difference Beam unilateral builder.

This is the next validation after:

```text
STEP665 FAST LOCAL-POINT FD BENCHMARK = PASS
baseline full-profile FD = 0.449908 s
fast local-point FD = 0.048582 s
speedup = 9.26x
linear Jacobian error = 0
angular Jacobian error = 0
committed clearance = +0.420600 mm
```

The scientific formulation must remain identical to the validated baseline.

## New files only

This task uses only the newly added test files:

```text
testing/beam_feasible_domain/beam_unilateral_lcp/beam_unilateral_fast_full_episode.py
testing/beam_feasible_domain/tests/b02_full_episode_beam_unilateral_fast_fd.py
```

Do not modify existing production or old baseline validation files.

## Configuration

```text
B02 / target_04
seed = 15204
deterministic epoch74 PPO checkpoint
max RL steps = 2048
physics = 2 x 5 ms
constraint solver = GenericConstraintSolver
requested margin = +0.100 mm
dense Beam spacing = 0.010 mm
committed penetration limit = 0.001 mm
builder = fast_local_point_fd
```

## Safety route

Every physics substep:

```text
real Beam committed/free state
-> 10 um dense Beam/SDF profile
-> if q_free >= +0.100 mm:
     no Beam unilateral rows
-> otherwise:
     select dense worst/near-worst samples
     map each selected sample to exact Beam element/t
     local Rigid3 +/- finite differences on only that point
     batch all perturbed points
     one production-SDF query
     build scalar Beam unilateral rows
-> GenericConstraintSolver
-> committed Beam
-> independent 10 um dense committed check
```

## Hard restrictions

Do NOT modify:

```text
mcr_sim/
training/
testing/beam_feasible_domain/beam_unilateral_lcp/beam_unilateral_full_episode.py
testing/beam_feasible_domain/tests/b02_full_episode_beam_unilateral_generic.py
```

Do NOT change:

- requested margin;
- dense spacing;
- row selection;
- translation FD step;
- rotation FD step;
- GenericConstraintSolver;
- penetration limit.

Do NOT use:

- SLSQP q_candidate;
- native candidate injection;
- CollisionDOFs as safety constraint source;
- direct Beam free_position write;
- direct committed position write;
- rollback;
- projection;
- action shielding;
- analytic gradient/Jacobian;
- LCPConstraintSolver.

Do NOT train.

## 1. Work directory / git

```bash
cd /data/home/3220251075/mcr_sim/mcr_project/mCR_simulator-master/python

git status --short
git log -1 --oneline
```

Do not reset/clean unrelated files.

## 2. Python preflight

```bash
python -m py_compile \
  testing/beam_feasible_domain/beam_unilateral_lcp/beam_unilateral_fast_fd.py \
  testing/beam_feasible_domain/beam_unilateral_lcp/beam_unilateral_fast_full_episode.py \
  testing/beam_feasible_domain/tests/b02_full_episode_beam_unilateral_fast_fd.py
```

If there is only a trivial syntax/import issue in the newly added fast files,
make the minimum test-only fix and show the diff.

Do not change scientific thresholds or existing baseline files.

## 3. Plugin

Reuse the already validated plugin:

```bash
find testing/beam_feasible_domain/_runtime \
  -name 'libMCRBeamLinearizedUnilateral.so' -print
```

If missing, rebuild the existing test-only plugin only.

## 4. Old output protection

Before running, confirm these fast-full-episode paths do not contain a previous
completed result from another attempt:

```text
testing/beam_feasible_domain/_runtime/beam_unilateral_fast_fd_full_episode/
```

Do NOT delete unrelated runtime data.

## 5. Execute exactly one episode

```bash
mkdir -p testing/beam_feasible_domain/_runtime/beam_unilateral_fast_fd_full_episode/logs
mkdir -p testing/beam_feasible_domain/_runtime/beam_unilateral_fast_fd_full_episode/results

python testing/beam_feasible_domain/tests/b02_full_episode_beam_unilateral_fast_fd.py \
  --max-rl-steps 2048 \
  --progress-every 25 \
  > testing/beam_feasible_domain/_runtime/beam_unilateral_fast_fd_full_episode/logs/run.log 2>&1
```

Run ONE copy only.

If the runner fails, stop at the FIRST failure.
Do not tune, change margin, patch physics, or rerun automatically.

## 6. Fast-builder invariants

For every row-build substep require:

```text
support_mode = exact_selected_point_local_fd
selected-point reconstruction error <= 1e-10 m
batched FD SDF query count = 1
post-selection full-Beam profile count = 0
native activeCount = intended row count
```

The runner fail-fast checks these.

## 7. Hard safety PASS requirements

PASS requires:

- 2048 RL steps completed, or normal successful completion before horizon;
- no safety-related termination;
- every committed state penetration <= 0.001 mm;
- committed violation count = 0;
- row-build/planning failures = 0;
- activeCount mismatches = 0;
- fast-builder invariant failures = 0;
- NaN/Inf = 0;
- no forbidden safety fallback/write.

The +0.100 mm requested margin remains a buffer target, not the penetration
definition. A committed state may be below +0.100 mm and still be physically
safe if clearance remains >= -0.001 mm.

## 8. Baseline comparison

Validated full-profile 2048-step baseline:

```text
RL steps = 2048
physics substeps = 4096
row-build substeps = 1436
q_free < 0 substeps = 636
q_free < -0.001 mm substeps = 625
total unilateral rows = 7000
worst free clearance = -0.395220 mm
worst committed clearance = +0.026822 mm
max committed penetration = 0 mm
committed violations = 0
row-build total runtime = 1410.52 s
row-build mean runtime = 0.9823 s
action SHA256 =
e6854f5496eb2d43eb913d1116238d831666f7589ada051325bfdf14d294900e
```

The fast runner records whether the action SHA matches.

Action SHA match is DIAGNOSTIC, not a hard safety PASS condition.

For performance report both:

```text
baseline mean row-build / fast mean row-build
baseline total row-build / fast total row-build
```

If action SHA and row-build count match baseline, the total-runtime comparison
is directly apples-to-apples.

If they differ, emphasize the per-row mean speedup as the fairer metric.

## 9. Results

Primary JSON:

```text
testing/beam_feasible_domain/_runtime/beam_unilateral_fast_fd_full_episode/results/b02_fast_fd_full_episode.json
```

Report:

```text
testing/beam_feasible_domain/_runtime/beam_unilateral_fast_fd_full_episode/results/b02_fast_fd_full_episode_report.md
```

Trace:

```text
testing/beam_feasible_domain/_runtime/beam_unilateral_fast_fd_full_episode/results/b02_fast_fd_full_episode_trace.jsonl
```

Read the JSON/report yourself. Do not paste the complete trace.

## 10. Final response format

Return:

```text
B02 2048-STEP FAST LOCAL-POINT FD BEAM UNILATERAL ACCEPTANCE

Configuration:
seed = 15204
solver = GenericConstraintSolver
builder = fast_local_point_fd
requested margin = 0.100 mm
dense spacing = 0.010 mm
committed penetration limit = 0.001 mm
physics = 2 x 5 ms
production modified = NO
training = NO

Run:
RL steps = ...
physics substeps = ...
terminated / truncated = ...
terminal reason = ...
action SHA256 = ...
baseline action SHA match = YES/NO

Constraint activity:
safe-free substeps = ...
q_free < +0.100 mm row-build substeps = ...
q_free < 0 substeps = ...
q_free < -0.001 mm substeps = ...
total rows = ...
max rows/substep = ...
row distribution = ...
support_mode counts = ...
activeCount mismatches = ...

Fast builder:
row-build runtime total = ...
row-build runtime mean = ...
row-build runtime max = ...
baseline mean = 0.9823 s
mean speedup = ... x
baseline total = 1410.52 s
total speedup = ... x
batched FD points total = ...
batched SDF query total = ...
post-selection full-Beam profiles = ...
max selected-point reconstruction error = ...
fast-builder invariant failures = ...

Clearance:
worst free = ...
worst committed = ...
max committed penetration = ...
committed violations = ...
committed states below +0.100 mm = ...

Numerics:
NaN/Inf = ...
planning/row-build failures = ...
first failure = ...

FINAL = PASS / FAIL / INCONCLUSIVE

Q1. Did all geometrically negative q_free states end as safe committed states?
Q2. Did fast local-point FD preserve the 2048-step nonpenetration acceptance?
Q3. What was the mean row-build speedup versus 0.9823 s baseline?
Q4. Did the fast trajectory/action SHA exactly match the validated baseline?
Q5. If PASS, is fast FD ready for production integration work?
Q6. Does this prove all future policies/states/vessels never penetrate?
```

Q6 must be NO.

If PASS, STOP. Do not start production integration or training automatically.
If FAIL, STOP at first failure. Do not tune automatically.
