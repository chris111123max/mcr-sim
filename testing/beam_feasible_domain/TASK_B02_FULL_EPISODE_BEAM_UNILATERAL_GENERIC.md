# Codex task: B02 2048-step Beam-level SDF unilateral + Generic acceptance

## Goal

Run one fixed full B02 episode using the new physical safety route from the
FIRST physics substep onward:

```text
real Beam Rigid3 free state
-> production BeamAdapter 10 um dense geometry
-> production B02 SDF
-> Beam-level linearized unilateral rows
-> GenericConstraintSolver
-> committed Beam state
-> independent 10 um dense committed check
```

This run answers whether the new route prevents catheter body penetration over
the whole fixed B02/target_04/seed15204 episode / 2048-step horizon.

Do NOT use the SLSQP q_candidate route anywhere in this episode.

## Configuration

```text
B02 / target_04
seed = 15204
deterministic epoch74 PPO checkpoint
max RL steps = 2048
physics = 2 x 5 ms
solver = GenericConstraintSolver
requested unilateral margin = +0.100 mm
dense Beam spacing = 10 um
committed penetration limit = 0.001 mm
```

The existing soft SDF wall and normal collision setup remain part of the
historical B02 environment. The NEW hard safety constraint source is the real
BeamAdapter/SDF, not CollisionDOFs.

## New files

```text
testing/beam_feasible_domain/beam_unilateral_lcp/beam_unilateral_full_episode.py
testing/beam_feasible_domain/tests/b02_full_episode_beam_unilateral_generic.py
```

The native component remains:

```text
testing/beam_feasible_domain/beam_unilateral_lcp/native/BeamLinearizedUnilateralConstraint.cpp
```

## Important formulation change for the full episode

The full-episode row builder does NOT restrict the Jacobian to Beam nodes that
happened to move in the free step.

For each selected 10-um dense sample, it maps that sample to its active Beam
element and uses the exact two Rigid3 endpoint DOFs of that element as the
constraint support.

This is the correct local support for the cubic Beam sample and is also much
cheaper than finite-differencing the entire catheter.

If the derived dense sample indexing cannot be matched exactly, the test-only
code conservatively falls back to all active Beam nodes and records:

```text
support_mode = all_active_nodes_fallback
```

Do not silently change this logic.

## Hard restrictions

Do NOT modify production:

```text
mcr_sim/
training/
```

Do NOT train.

Do NOT switch to LCPConstraintSolver.

Do NOT use:

- SLSQP q_candidate solver;
- native q_candidate injection;
- direct Beam free_position writes;
- direct committed position writes;
- CollisionDOFs as the new safety constraint source;
- rollback;
- projection;
- action shielding.

Do NOT loosen the 0.001 mm committed penetration acceptance limit.

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
  testing/beam_feasible_domain/beam_unilateral_lcp/beam_linearized_unilateral.py \
  testing/beam_feasible_domain/beam_unilateral_lcp/beam_unilateral_full_episode.py \
  testing/beam_feasible_domain/tests/b02_full_episode_beam_unilateral_generic.py
```

If this exposes only a trivial test-only syntax/import issue, make the minimum
fix under `testing/beam_feasible_domain/`, show the diff, and continue.

Do not alter the scientific thresholds.

## 3. Native plugin

First look for the plugin built by the successful step665 test:

```bash
find testing/beam_feasible_domain/_runtime -name 'libMCRBeamLinearizedUnilateral.so' -print
```

If present, reuse it.

If absent, build:

```bash
BUILD=testing/beam_feasible_domain/_runtime/beam_unilateral_generic_full_episode/native_build

cmake -S testing/beam_feasible_domain/beam_unilateral_lcp/native \
      -B "$BUILD" \
      -G Ninja \
      -DCMAKE_BUILD_TYPE=Release

cmake --build "$BUILD" -j2
```

Only minimum test-only SOFA 21.12 compile compatibility fixes are permitted.

## 4. Execute exactly one full episode

Create runtime directories:

```bash
mkdir -p testing/beam_feasible_domain/_runtime/beam_unilateral_generic_full_episode/logs
mkdir -p testing/beam_feasible_domain/_runtime/beam_unilateral_generic_full_episode/results
```

Run:

```bash
python testing/beam_feasible_domain/tests/b02_full_episode_beam_unilateral_generic.py \
  --max-rl-steps 2048 \
  --progress-every 25 \
  > testing/beam_feasible_domain/_runtime/beam_unilateral_generic_full_episode/logs/run.log 2>&1
```

Run ONE copy only.

The runner is fail-fast. If the first real physical/constraint failure occurs,
stop there. Do not automatically tune parameters and do not rerun with a
larger margin.

## 5. Every physics substep must do this

At CollisionBeginEvent:

1. read real Beam committed `position`;
2. read real Beam `free_position`;
3. independently evaluate 10-um dense Beam clearance;
4. if free clearance >= +0.100 mm:
   - disable Beam unilateral rows for that solve;
5. if free clearance < +0.100 mm:
   - select dense worst/near-worst samples;
   - map each selected sample to its active Beam element;
   - finite-difference only the exact two endpoint Rigid3 DOFs;
   - write scalar unilateral rows;
   - require component `activeCount == intended row_count`;
6. let GenericConstraintSolver finish the normal SOFA solve;
7. independently measure committed Beam clearance at 10 um spacing.

## 6. Hard full-episode PASS requirements

PASS requires:

- no SLSQP q_candidate solve;
- no native candidate injection;
- no direct free-position/committed-position write;
- no rollback/projection/action shielding;
- GenericConstraintSolver is the only root constraint solver;
- every required unilateral row set is built successfully;
- intended row count equals native component activeCount;
- no NaN/Inf;
- every committed Beam state has penetration <= 0.001 mm;
- committed violation count = 0;
- no safety-related environment termination;
- episode either:
  - reaches 2048 RL steps, or
  - ends normally due to successful task completion before 2048.

The requested +0.100 mm margin is a buffer/constraint target. Report how often
the final committed state is below +0.100 mm, but do NOT classify a physically
safe committed state as penetration solely because it is below the margin.

## 7. Fail-fast data

If FAIL, return the FIRST failing frame with:

- RL step / substep;
- q_prev clearance;
- q_free clearance;
- free penetration;
- dense sample count;
- selected dense indices;
- selected Beam element indices;
- support_mode;
- support Beam nodes;
- source clearances;
- unilateral row count;
- native activeCount;
- minimum freeViolation;
- row-build runtime;
- committed clearance;
- committed penetration;
- exact failure reason.

Do not automatically change anything after the first failure.

## 8. Results

Primary:

```text
testing/beam_feasible_domain/_runtime/beam_unilateral_generic_full_episode/results/b02_beam_unilateral_generic_full_episode.json
```

Report:

```text
testing/beam_feasible_domain/_runtime/beam_unilateral_generic_full_episode/results/b02_beam_unilateral_generic_full_episode_report.md
```

Trace:

```text
testing/beam_feasible_domain/_runtime/beam_unilateral_generic_full_episode/results/b02_beam_unilateral_generic_full_episode_trace.jsonl
```

Read the JSON and report yourself. Do not paste the entire trace.

## Final response format

Return:

```text
B02 2048-STEP BEAM-LEVEL SDF UNILATERAL + GENERIC ACCEPTANCE

Configuration:
seed = 15204
solver = GenericConstraintSolver
requested margin = 0.100 mm
dense spacing = 0.010 mm
committed penetration limit = 0.001 mm
physics = 2 x 5 ms
SLSQP candidate route = OFF
native candidate injection = OFF
production modified = NO
training = NO

Run:
RL steps executed = ...
physics substeps = ...
terminated / truncated = ...
terminal reason = ...
action SHA256 = ...

Constraint activity:
safe-free substeps = ...
row-build substeps = ...
penetrating free substeps = ...
total unilateral rows = ...
max rows in one substep = ...
support_mode counts = ...
activeCount mismatches = ...
row-build runtime total / mean / max = ...

Clearance:
worst free clearance = ...
worst committed clearance = ...
max committed penetration = ...
committed penetration violations = ...
committed states below +0.100 mm margin = ...

Numerics:
NaN/Inf = ...
planning/row-build failures = ...

First failure:
...

FINAL:
PASS / FAIL / INCONCLUSIVE

Q1. Did the Beam-level unilateral layer successfully correct penetrating free states?
Q2. Did every committed Beam state stay within the 0.001 mm penetration limit?
Q3. Did the fixed episode reach 2048 steps or end normally before that?
Q4. Did any failure come from row construction / GenericConstraintSolver rather than physical penetration?
Q5. For this fixed B02 episode, can we call the catheter body nonpenetration engineering acceptance passed?
Q6. Does this prove all future policies/states/vessels can never penetrate?
```

Q6 must be NO.

If PASS, STOP. Do not start training automatically.
If FAIL, STOP at the first failure and return it. Do not tune automatically.
