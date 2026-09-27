# Codex autonomous task: B02 post-constraint failure diagnosis, max 4 iterations

## Mission

Autonomously diagnose the full-episode safety failure observed at:

- B02
- target_04
- seed 15204
- protected trajectory
- RL step 1293 / physics substep 1

Observed key evidence:

```text
free clearance       = -0.054565 mm
accepted candidate   = +0.002014 mm
native hook           = PASS
mapping propagation   = PASS
velocity correction   = PASS
committed clearance   = -0.023640 mm
next substep committed = -0.089670 mm
```

Across the full protected replay before failure:

```text
unsafe free states          = 162
solver calls/success/fail   = 162 / 162 / 0
native injections/failures  = 162 / 0
mapping failures            = 0
velocity failures           = 0
NaN/Inf                     = 0
forbidden fallback          = 0
```

The current leading hypothesis is:

> The feasible solver currently targets a candidate too close to the geometric boundary. The later native collision/constraint correction can move that candidate back outside, so pre-constraint feasibility does not imply post-constraint feasibility.

Your job is to test this hypothesis and, if it is wrong or incomplete, iteratively design the next smallest diagnostic.

Do not ask the user what to test next.

You own the diagnostic loop for at most FOUR iterations.

If a clear causal explanation is established earlier, STOP EARLY.

---

## Hard scope boundary

ALL new or modified code must stay under:

```text
testing/
```

Prefer:

```text
testing/beam_feasible_domain/
```

Runtime outputs must stay under:

```text
testing/beam_feasible_domain/_runtime/
```

Do NOT modify:

```text
mcr_sim/
training/
training_runs/
```

Do NOT train.

Do NOT change production reward / observation / done / policy / physics-substep count.

Do NOT migrate SOFA.

Do NOT add production plugins.

Do NOT use rollback, projection, action shielding, post-step repair, direct committed-position write, or direct CollisionDOF write.

CollisionDOFs remain diagnostics only.

---

## Existing validated components to reuse

Use the existing test-only stack and preserve all server-local compatibility work:

```text
testing/beam_feasible_domain/tests/b02_full_episode_safety_acceptance.py
testing/beam_feasible_domain/tests/b02_online_feasible_solver_bridge.py
testing/beam_feasible_domain/tests/b02_step776_feasible_solve.py
testing/beam_feasible_domain/tests/b02_beam_adapter.py
testing/beam_feasible_domain/native_precommit_hook/NativePrecommitMappingHook.cpp
```

Previous result files:

```text
testing/beam_feasible_domain/_runtime/results/b02_full_episode_safety_acceptance.json
testing/beam_feasible_domain/_runtime/results/b02_full_episode_safety_acceptance_report.md
testing/beam_feasible_domain/_runtime/results/b02_full_episode_safety_acceptance_trace.jsonl
```

The online solver for all new diagnostics MUST explicitly consume the current frame's:

```text
q_prev
q_free
```

Do not reuse a saved step776 candidate.

Do not use a zero-argument monolithic main() as if it were an online per-frame solver.

---

# Autonomous iteration protocol

You may perform up to FOUR complete diagnostic iterations.

Each iteration is:

1. inspect all evidence from the previous iteration;
2. state one concrete hypothesis;
3. write the SMALLEST test-only code needed to discriminate that hypothesis;
4. run the test;
5. read the result files and relevant log tail yourself;
6. decide:
   - ROOT CAUSE ESTABLISHED -> stop early;
   - hypothesis rejected / incomplete -> design next iteration;
   - infrastructure blocked -> fix only a minimal test-only compatibility issue and continue within the same iteration when reasonable.

Do not count a syntax/import typo fix as a scientific iteration.

Do not run broad experiments just because time is available.

Do not run training.

At the end, report the actual number of scientific iterations used: 1 to 4.

---

# Iteration 1 — robust candidate-margin test

This is the required first iteration.

## Question

At the actual first failure frame, is the failure explained by insufficient pre-constraint clearance margin?

## Target

Protected trajectory:

```text
B02 / target_04 / seed 15204
RL step 1293
physics substep 1
```

Use the same deterministic checkpoint and the safety layer from step 1 so the trajectory remains the protected trajectory.

The test must preserve:

- q_prev from the real target frame;
- q_free from the real target frame;
- same BeamAdapter geometry;
- same B02 SDF;
- same native hook;
- same production mapping;
- same collision/constraint solve;
- same 5 ms physics substep.

Only the solver's required accepted geometric safety margin may change.

## Candidate margins

Start with a small informative set. Suggested targets:

```text
0.002 mm   # approximately current behavior / control
0.025 mm
0.050 mm
0.100 mm
0.200 mm
```

But do not blindly perform five complete 1293-step replays if a cheaper faithful same-state method is available.

First inspect whether a faithful test-only branch/snapshot of the target physical state can be produced.

A faithful snapshot must restore all mechanical/controller state needed for the same native substep. Do not call an incomplete q-only restore a faithful physical snapshot.

If no faithful snapshot is available, use the minimum number of deterministic prefix replays needed to discriminate the hypothesis.

Prefer adaptive probing, e.g.:
- current margin / control;
- one clearly larger margin around 0.05 mm;
- increase only if necessary.

## Important solver rule

Do not fake a margin by merely changing the PASS threshold after the solver returns.

The solver must actually generate a candidate satisfying the requested geometric minimum clearance target.

Implement this only in test-only code.

Do not permanently retune production.

## Required measurements per tested margin

Record:

```text
q_prev dense clearance
q_free dense clearance
requested margin
accepted candidate dense clearance
candidate vs q_free translation/rotation
native fire delta
parent write error
mapped child free-state change
native status
committed dense clearance
candidate -> committed clearance loss
NaN/Inf
```

The key quantity is:

```text
post_constraint_clearance_loss
    = candidate_clearance - committed_clearance
```

## Iteration-1 early-stop condition

You may declare the root cause established and STOP if the evidence clearly shows:

1. current near-zero-margin candidate commits outside;
2. a modest positive margin candidate, through the SAME native pipeline, commits safely;
3. no other native/mapping failure appears;
4. the result is repeatable enough in the fixed deterministic target frame to rule out a harness artifact.

Then report:

```text
ROOT CAUSE:
PRECOMMIT_FEASIBILITY_MARGIN_TOO_SMALL_FOR_POST-CONSTRAINT_MOTION
```

Also report the smallest tested margin that produced a safe committed state.

Do NOT run a full 2048-step replay merely to confirm this diagnosis.

---

# If Iteration 1 does NOT establish the cause

Choose Iteration 2 based on the data.

## Branch A — even large margin is consumed by native solve

If candidates with substantial positive margin still commit outside, test:

> Is the native collision/constraint correction itself producing a systematic unsafe displacement that cannot be covered by a fixed local margin?

Write a same-frame post-constraint displacement diagnostic.

Capture, at the failure frame:

- q_prev;
- q_free;
- q_acc;
- mapped CollisionDOF free state before/after native propagation;
- committed q;
- worst Beam sample location before/after;
- per-node translation and rotation q_acc -> q_committed;
- dense SDF clearance profile along the Beam, not only its scalar minimum;
- location/index of worst-clearance sample before and after;
- native contact/constraint row counts if already accessible from existing APIs;
- any existing contact diagnostic that can be read without writing production code.

Determine whether:
- one local region is pushed outside;
- worst-clearance location jumps to another part of the body;
- correction magnitude grows with margin;
- constraint response appears to fight the feasible correction.

If this clearly establishes a systematic post-constraint response mechanism, STOP.

## Branch B — margin response is irregular/non-monotonic

Test:

> Is the feasible candidate changing which collision/contact constraints activate, causing a discontinuous post-constraint response?

Design the smallest same-frame sweep needed to compare:
- accepted clearance;
- mapped collision free geometry;
- contact/constraint activation;
- committed clearance.

Do not add new physics.

If a contact-mode switch clearly explains safe/unsafe outcomes, STOP.

## Branch C — solver cannot actually realize larger requested margin

Test:

> Is the current real-Beam feasible solver only solving boundary feasibility and incapable of interior-margin feasibility at this frame?

Diagnose the solver output itself:
- requested margin;
- achieved dense margin;
- convergence/status;
- active constraints;
- candidate displacement magnitude.

If the solver cannot create an interior candidate even when one exists or appears locally reachable, identify that as the limitation and STOP if evidence is clear.

---

# Iterations 3 and 4

Only use them if the previous iteration leaves two or more plausible causes.

Each additional iteration must be narrower than the previous one.

Good examples:

- compare two candidate margins around the observed transition;
- compare one contact-active vs one contact-inactive candidate;
- isolate one Beam segment / worst-clearance region;
- compare q_acc -> q_committed motion with and without a specific existing contact mode, only if this can be done in a test-only scene without changing production;
- test whether a candidate satisfying margin delta at the real Beam also provides equivalent margin on the mapped collision representation.

Bad examples:

- another full 2048-step replay;
- random parameter sweeps;
- changing reward;
- increasing substeps;
- replacing the solver;
- writing a new collision engine;
- adding a new production plugin.

Iteration 4 is the hard stop.

At the end of Iteration 4, return the best-supported causal conclusion even if not fully proven.

---

# File organization

Create a dedicated diagnostic subdirectory, for example:

```text
testing/beam_feasible_domain/postconstraint_diagnosis/
```

Suggested structure:

```text
postconstraint_diagnosis/
  iter1_*.py
  iter2_*.py
  iter3_*.py
  iter4_*.py
  README.md
```

Runtime:

```text
testing/beam_feasible_domain/_runtime/postconstraint_diagnosis/
  logs/
  results/
  captures/
```

You may choose different test-only names if clearer.

Do not scatter new files outside testing/.

---

# Efficiency rules

The previous full episode consumed 162 online solves and approximately 364 seconds of solver time.

Do not casually repeat that cost.

Use existing result/trace files first.

If the target physical state cannot be restored faithfully, deterministic prefix replay to step1293 is allowed, but:
- stop immediately after the target diagnostic;
- do not continue the episode;
- reuse captured evidence when scientifically valid;
- minimize the number of full prefixes.

Do not use a stale/off-trajectory state just to save time.

---

# What counts as a clear root cause

Examples that are sufficient for early stop:

### Cause 1 — margin too small

```text
candidate +0.002 mm -> committed -0.024 mm
candidate +0.050 mm -> committed +0.02 mm
native/mapping healthy in both
```

Conclusion:
precommit candidate needs a robust positive margin because native solve consumes finite clearance.

### Cause 2 — post-constraint response defeats fixed margin

```text
increasing candidate margin does not reliably increase committed clearance
and native/contact correction pushes a specific region outward systematically
```

Conclusion:
same-frame geometry-only feasibility is insufficient; the constraint response must be modeled/anticipated.

### Cause 3 — mapping safety mismatch

```text
real Beam candidate has positive margin
but mapped collision geometry presented to native collision remains unsafe/inconsistent
despite nominal propagation success
```

Conclusion:
real-Beam safety margin is not preserved by the mapped collision representation.

### Cause 4 — solver interior-feasibility limitation

```text
requested positive margin cannot be achieved even though the target state appears locally reachable
```

Conclusion:
current solver formulation is boundary-feasibility only / insufficient for robust interior feasibility.

---

# Required final response

Return a concise report:

```text
B02 POST-CONSTRAINT FAILURE AUTONOMOUS DIAGNOSIS

Scientific iterations used: N / 4
Stopped early: YES/NO

ITERATION 1
Hypothesis:
Test written:
Run:
Key measurements:
Conclusion:

ITERATION 2
...

ROOT CAUSE:
...

Evidence:
1. ...
2. ...
3. ...

Ruled out:
- solver outright failure: YES/NO
- native hook failure: YES/NO
- mapping propagation failure: YES/NO
- NaN/Inf: YES/NO
- insufficient geometric margin: YES/NO/UNRESOLVED
- post-constraint response not captured by solver: YES/NO/UNRESOLVED
- mapping representation mismatch: YES/NO/UNRESOLVED

Recommended next engineering change:
...

Confidence:
HIGH / MEDIUM / LOW

Production files modified:
NO

Training launched:
NO
```

If the cause is not fully resolved after four scientific iterations, say so plainly and provide the strongest remaining hypothesis plus the exact evidence gap.

Do not launch training.

Do not make production changes.

Stop after diagnosis.
