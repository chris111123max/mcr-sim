# Codex task: Step665 fast local-point FD Beam unilateral benchmark

## Goal

Reduce the dominant runtime cost of the validated Beam-level SDF unilateral
route WITHOUT changing the physical formulation yet.

Current validated full-episode row builder:

```text
one 10 um dense Beam profile
-> select active dense samples
-> for every +/- Rigid3 finite-difference perturbation:
     rebuild the entire ~5300-point Beam
     query the entire ~5300-point SDF profile
-> assemble unilateral rows
```

New test-only fast builder:

```text
one 10 um dense Beam profile
-> select the exact same dense samples
-> map each selected sample to its Beam element and exact t
-> for every +/- Rigid3 finite-difference perturbation:
     reconstruct ONLY that selected Beam point
-> batch all perturbed points
-> ONE batched production-SDF query
-> assemble the same unilateral rows
```

This is intentionally NOT an analytic-gradient rewrite yet.

The finite-difference sizes, quaternion perturbation convention, SDF clearance
definition, dense row selection, and two-endpoint Beam support remain the same
as the validated baseline.

## New files

```text
testing/beam_feasible_domain/beam_unilateral_lcp/beam_unilateral_fast_fd.py
testing/beam_feasible_domain/tests/diagnose_step665_fast_local_fd.py
```

## Protected target

Use the known fixed protected state:

```text
B02 / target_04 / seed 15204
step 665 / substep 1
q_prev ~= +0.468945 mm
q_free ~= -0.032075 mm
requested margin = +0.100 mm
dense spacing = 0.010 mm
solver = GenericConstraintSolver
```

The prefix through step664 may use the archived strict feasible-domain route
ONLY to reproduce this known state.

At target step665/substep1:

- q_candidate SLSQP = OFF
- native candidate injection = OFF
- CollisionDOFs safety source = OFF
- direct free/committed writes = OFF

## What the target test does

On the exact same q_prev/q_free target state:

1. build the existing full-profile-FD row set for comparison only;
2. time it;
3. build the fast local-point-FD row set;
4. time it;
5. compare:
   - selected dense indices;
   - row offsets;
   - Beam DOF indices;
   - row block counts;
   - source clearances;
   - linear Jacobian;
   - angular Jacobian;
6. install ONLY the fast row set into:
   `BeamLinearizedUnilateralConstraint`;
7. let GenericConstraintSolver finish step665/substep1;
8. independently check committed Beam clearance at 10 um;
9. stop before substep2.

Do NOT run a full episode in this task.

## Hard restrictions

Do NOT modify:

```text
mcr_sim/
training/
```

Do NOT change:

- requested margin;
- dense spacing;
- FD translation step;
- FD rotation step;
- row selection;
- GenericConstraintSolver;
- committed penetration limit.

Do NOT introduce:

- analytic SDF gradient;
- analytic Beam Jacobian;
- LCP;
- projection;
- rollback;
- action shielding.

This task isolates ONLY the performance effect of replacing repeated
full-profile FD evaluation with selected-point local FD evaluation.

## 1. Pull / inspect

```bash
cd /data/home/3220251075/mcr_sim/mcr_project/mCR_simulator-master/python

git status --short
git log -1 --oneline
```

Do not clean/reset unrelated files.

## 2. Python preflight

```bash
python -m py_compile \
  testing/beam_feasible_domain/beam_unilateral_lcp/beam_unilateral_fast_fd.py \
  testing/beam_feasible_domain/tests/diagnose_step665_fast_local_fd.py
```

If py_compile exposes a trivial test-only issue, fix only under
`testing/beam_feasible_domain/` and show the diff.

## 3. Plugins

Reuse the already validated test-only plugins when present:

```bash
find testing/beam_feasible_domain/_runtime -name 'libMCRBeamLinearizedUnilateral.so' -print
find testing/beam_feasible_domain/_runtime -name 'libMCRBeamFeasibleHook.so' -print
```

If a runtime artifact is missing, rebuild the corresponding existing test-only
plugin without changing its algorithm.

## 4. Execute exactly one target benchmark

```bash
mkdir -p testing/beam_feasible_domain/_runtime/beam_unilateral_fast_fd_step665/logs
mkdir -p testing/beam_feasible_domain/_runtime/beam_unilateral_fast_fd_step665/results

python testing/beam_feasible_domain/tests/diagnose_step665_fast_local_fd.py \
  --progress-every 50 \
  > testing/beam_feasible_domain/_runtime/beam_unilateral_fast_fd_step665/logs/run.log 2>&1
```

Run one copy only.

## 5. Required checks

Protected-state gate must reproduce approximately:

```text
q_prev = +0.468945 mm
q_free = -0.032075 mm
```

Fast selected-point reconstruction must satisfy:

```text
selected_point_reconstruction_max_error <= 1e-10 m
```

The fast and baseline builders should have identical:

```text
selected_dense_indices
row_offsets
dof_indices
row_block_counts
source_clearances
```

The runner also checks numerical Jacobian equivalence using small
rtol/atol tolerances.

The target solve must have:

```text
constraint activeCount == intended fast row_count
committed penetration <= 0.001 mm
finite state
```

## Performance numbers to report

Report:

- baseline full-profile-FD runtime;
- fast local-point-FD runtime;
- speedup factor;
- row count;
- dense sample count;
- number of batched perturbed points;
- number of post-selection full-Beam profile evaluations in the fast path
  (expected 0);
- number of batched FD SDF calls (expected 1).

The benchmark is successful scientifically if the row/Jacobian result remains
equivalent and committed safety passes.

Do NOT fail the science result merely because speedup is smaller than an
arbitrary target; report the measured speedup.

## Result

Primary:

```text
testing/beam_feasible_domain/_runtime/beam_unilateral_fast_fd_step665/results/step665_fast_local_fd_result.json
```

## Final response format

Return:

```text
STEP665 FAST LOCAL-POINT FD BEAM UNILATERAL BENCHMARK

Preflight:
python compile = PASS/FAIL
plugins = PASS/FAIL
production modified = NO
training = NO

Protected state:
q_prev = ...
q_free = ...
state matched = YES/NO
action SHA256 = ...

Row equivalence:
row count = ...
selected indices equal = YES/NO
row offsets equal = YES/NO
DOF indices equal = YES/NO
row block counts equal = YES/NO
source clearances equal = YES/NO
linear max abs error = ...
linear max relative error = ...
angular max abs error = ...
angular max relative error = ...
jacobian equivalent = YES/NO
selected-point reconstruction max error = ...

Performance:
baseline full-profile FD = ... s
fast local-point FD = ... s
speedup = ... x
dense sample count = ...
batched FD points = ...
batched FD SDF query count = ...
fast post-selection full-Beam profiles = ...

Generic solve:
active rows = ...
committed clearance = ...
committed penetration = ...
finite = YES/NO

FINAL = PASS / FAIL

Q1. Did the fast builder reproduce the validated baseline rows/Jacobian?
Q2. Did GenericConstraintSolver remain physically safe at step665?
Q3. What speedup was measured?
Q4. If PASS, is the next test a 2048-step fast-builder full-episode acceptance?
```

If PASS, answer Q4 = YES, but STOP.
Do not automatically run the 2048-step fast episode yet.
Do not train.
