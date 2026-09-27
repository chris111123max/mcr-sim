# Codex task: B02 step665 Beam-level SDF unilateral Generic/LCP A/B

## Goal

Test whether the step665 SLSQP stagnation can be bypassed by moving the safety
solve into SOFA's native constraint layer.

The target is the exact protected B02 frame:

- B02 / target_04 / seed 15204
- step 665 / substep 1
- q_prev clearance ~= +0.468945 mm
- q_free clearance ~= -0.032075 mm
- requested safety margin = +0.100 mm
- committed penetration acceptance limit = 0.001 mm
- 2 x 5 ms physics

The prefix through step664 MUST still use the current dense-aware strict
0.100 mm feasible-domain solver so the target state remains the same.

At step665/substep1, DO NOT call the SLSQP q_candidate solver.

Instead, use a test-only Beam-level linearized SDF unilateral constraint that:

1. samples the real production BeamAdapter geometry at 10 um spacing;
2. queries the production B02 SDF;
3. selects the dense worst region/local minima;
4. computes finite-difference Jacobians with respect to the actual Rigid3 Beam
   DOFs;
5. supplies g = clearance - 0.100 mm >= 0 rows directly to SOFA;
6. lets the native constraint solver compute the correction.

The constraint source is the real BeamAdapter/SDF. CollisionDOFs are NOT a
constraint source.

## Files

Native test-only component:

```text
testing/beam_feasible_domain/beam_unilateral_lcp/native/CMakeLists.txt
testing/beam_feasible_domain/beam_unilateral_lcp/native/BeamLinearizedUnilateralConstraint.cpp
```

Python row builder:

```text
testing/beam_feasible_domain/beam_unilateral_lcp/beam_linearized_unilateral.py
```

A/B runner:

```text
testing/beam_feasible_domain/tests/diagnose_step665_beam_unilateral_lcp.py
```

## A/B branches

The runner replays the protected prefix once and forks after step664.

Both branches inherit the exact same in-memory SOFA state and exact same
step665 policy action.

Branch G:

```text
GenericConstraintSolver + BeamLinearizedUnilateralConstraint
```

Branch L:

```text
LCPConstraintSolver + BeamLinearizedUnilateralConstraint
```

The LCP branch switches solver only after the protected prefix and verifies
that the fork-state fingerprint is unchanged by the solver replacement.

This directly tests whether the older LCP solver can consume/use this
Beam-level unilateral formulation.

## Hard restrictions

Everything remains test-only.

Do NOT modify:

```text
mcr_sim/
training/
cpp/SDFUnilateralConstraint/
```

Do NOT train.

Do NOT change:

- requested margin 0.100 mm;
- 10 um dense Beam check;
- committed penetration limit 0.001 mm;
- seed/checkpoint;
- 2 x 5 ms substeps.

Do NOT use:

- CollisionDOFs as the safety constraint source;
- q_candidate SLSQP at step665;
- rollback;
- post-step projection;
- action shielding;
- committed-position writes;
- Beam free-position writes at step665.

The prefix may continue using the already-validated native candidate injection,
because that is required only to reproduce the protected state through step664.

## 1. Pull / inspect

Work from:

```bash
cd /data/home/3220251075/mcr_sim/mcr_project/mCR_simulator-master/python
```

Run:

```bash
git status --short
git log -1 --oneline
```

Do not clean/reset unrelated server files.

## 2. Python syntax preflight

```bash
python -m py_compile \
  testing/beam_feasible_domain/beam_unilateral_lcp/beam_linearized_unilateral.py \
  testing/beam_feasible_domain/tests/diagnose_step665_beam_unilateral_lcp.py
```

If this finds a trivial test-only syntax/import issue, fix only under
`testing/beam_feasible_domain/beam_unilateral_lcp/` or the new runner.

## 3. Build native Beam constraint plugin

Use the same SOFA 21.12 compiler/environment that successfully built the
existing test-only MCRBeamFeasibleHook.

```bash
BUILD=testing/beam_feasible_domain/_runtime/beam_unilateral_lcp_step665/native_build

cmake -S testing/beam_feasible_domain/beam_unilateral_lcp/native \
      -B "$BUILD" \
      -G Ninja \
      -DCMAKE_BUILD_TYPE=Release

cmake --build "$BUILD" -j2
```

Expected library name:

```text
libMCRBeamLinearizedUnilateral.so
```

If SOFA 21.12 exposes a small API naming/type incompatibility in the new
test-only C++ file, you may make the MINIMUM compile-only correction under:

```text
testing/beam_feasible_domain/beam_unilateral_lcp/native/
```

Do not change the scientific formulation:

- Rigid3 Beam Constraint;
- arbitrary supplied linear/angular Jacobian blocks;
- one unilateral lambda >= 0 resolution per row;
- freeViolation = dense clearance - 0.100 mm.

Do not move code into production.

After any compile fix, show exactly what changed.

## 4. Existing prefix hook

The runner also needs the previously validated:

```text
libMCRBeamFeasibleHook.so
```

normally under:

```text
testing/beam_feasible_domain/_runtime/native_build/
```

If it is missing only because runtime artifacts were not preserved, rebuild
the existing test-only hook from:

```text
testing/beam_feasible_domain/native_precommit_hook/
```

Do not modify its algorithm.

## 5. Execute one A/B run

```bash
mkdir -p testing/beam_feasible_domain/_runtime/beam_unilateral_lcp_step665/logs
mkdir -p testing/beam_feasible_domain/_runtime/beam_unilateral_lcp_step665/results

python testing/beam_feasible_domain/tests/diagnose_step665_beam_unilateral_lcp.py \
  --progress-every 50 \
  > testing/beam_feasible_domain/_runtime/beam_unilateral_lcp_step665/logs/run.log 2>&1
```

Run one copy only.

The runner must stop after step665/substep1. It must never execute target
substep2.

## 6. Replay gate

Both branches must originate from the same fork fingerprint and same target
action SHA.

At CollisionBeginEvent the target state must match approximately:

```text
q_prev = +0.468945 mm
q_free = -0.032075 mm
```

with the runner's small tolerance.

If not, report:

```text
STEP665_PROTECTED_STATE_MISMATCH
```

and stop.

## 7. What to inspect

Primary combined JSON:

```text
testing/beam_feasible_domain/_runtime/beam_unilateral_lcp_step665/results/step665_beam_unilateral_ab.json
```

Branch JSONs:

```text
testing/beam_feasible_domain/_runtime/beam_unilateral_lcp_step665/results/step665_generic_beam_unilateral.json
testing/beam_feasible_domain/_runtime/beam_unilateral_lcp_step665/results/step665_lcp_beam_unilateral.json
```

For each branch report:

- solver class;
- state matched;
- dense sample count;
- selected dense indices;
- selected source clearances;
- moved Beam nodes;
- number of unilateral rows supplied;
- component activeCount;
- minimum freeViolation;
- target runtime;
- committed dense clearance;
- committed penetration;
- finite/nonfinite.

Also inspect whether the LCP branch actually consumes the rows:

- activeCount > 0 only proves rows were built;
- the decisive evidence is whether committed Beam motion changes in the
  expected safe direction and whether committed dense clearance is safe.

## Interpretation

Possible outcomes:

### A. Generic PASS, LCP PASS

Strong evidence that a Beam-level SDF unilateral constraint can bypass the
step665 q_candidate/SLSQP stagnation, and that the older LCP solver can consume
the formulation.

### B. Generic PASS, LCP FAIL

The Beam-level unilateral formulation is viable, but the old LCP solver is not
compatible/effective for these custom 1-row unilateral ConstraintResolution
rows. Continue with GenericConstraintSolver rather than forcing LCP.

### C. Generic FAIL, LCP FAIL

The issue is not merely the outer SLSQP. Inspect whether:

- rows were built/active;
- the linearized Jacobian points in the wrong direction;
- the correction is too small;
- the local linearization is inadequate at step665.

Do not automatically change margin or add more experiments.

### D. LCP build/solver compatibility error

Report the exact compatibility error. Do not reinterpret it as physical
failure.

## Diagnostic PASS criterion

For either branch, diagnostic PASS means:

- protected target state matched;
- no target q_candidate SLSQP was used;
- Beam-level rows were built from real BeamAdapter/SDF;
- activeCount > 0;
- no CollisionDOF safety constraints were used;
- no direct free/committed position write at target;
- target substep1 completed;
- committed dense penetration <= 0.001 mm;
- no NaN/Inf.

Also report the committed clearance relative to the requested +0.100 mm target.
It is acceptable for this diagnostic to distinguish:

```text
physically safe committed state
```

from

```text
full +0.100 mm linearized target achieved
```

Do not conflate those two.

## Final response format

Return:

```text
B02 STEP665 BEAM-LEVEL SDF UNILATERAL GENERIC/LCP A/B

Build:
Beam plugin = PASS/FAIL
compile-only fixes = ...
production modified = NO
training = NO

Protected prefix:
steps completed = 664
physics substeps = ...
strict solver calls = ...
fork fingerprint = ...
target action SHA256 = ...

Target replay:
q_prev clearance = ...
q_free clearance = ...
state matched = YES/NO

Beam unilateral rows:
dense sample count = ...
selected dense indices = ...
selected clearances = ...
moved Beam nodes = ...
row count = ...
minimum freeViolation = ...

Generic branch:
solver = GenericConstraintSolver
active rows = ...
committed clearance = ...
committed penetration = ...
decision = ...

LCP branch:
solver = LCPConstraintSolver
solver switch state unchanged = ...
active rows = ...
committed clearance = ...
committed penetration = ...
decision = ...

Comparison:
same fork state = YES/NO
same target action = YES/NO
did Generic use the rows effectively = YES/NO/INCONCLUSIVE
did LCP use the rows effectively = YES/NO/INCONCLUSIVE

FINAL:
...

Q1. Can Beam-level SDF unilateral rows bypass the step665 SLSQP stagnation?
Q2. Does GenericConstraintSolver make the target committed state safe?
Q3. Does the older LCPConstraintSolver make the same target committed state safe?
Q4. Which solver, if either, is compatible with this Beam-level formulation?
Q5. Should this route proceed to a second-substep/full-episode test?
```

Then STOP.

Do not run a full episode.
Do not train.
Do not modify production.
