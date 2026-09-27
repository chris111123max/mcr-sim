# Luna XHigh task: B02 full-episode end-to-end safety acceptance

## Goal

Run one final engineering acceptance test for the current catheter-body penetration problem.

This replaces further step776/777/780 micro-tests.

Use the fixed difficult setup:

- B02
- target_04
- seed 15204
- epoch-74 PPO checkpoint already used by the previous deterministic replay
- deterministic policy
- 2 physics substeps per RL step
- 5 ms per physics substep
- safety layer active from the FIRST physics substep
- run until normal episode termination/truncation or 2048 RL steps

The question is:

> Can the validated real-Beam feasible-domain + native precommit mechanism keep every committed real-Beam state within the accepted feasible domain for the complete fixed episode, with zero solver/native failures and no forbidden fallback?

This is an engineering acceptance test for this fixed worst-case episode.

It is NOT a mathematical proof for every reachable state.

## Files

New runner:

~~~text
testing/beam_feasible_domain/tests/b02_full_episode_safety_acceptance.py
~~~

Existing validated pieces to reuse:

~~~text
testing/beam_feasible_domain/tests/b02_online_feasible_solver_bridge.py
testing/beam_feasible_domain/tests/b02_step776_feasible_solve.py
testing/beam_feasible_domain/tests/b02_beam_adapter.py
testing/beam_feasible_domain/native_precommit_hook/NativePrecommitMappingHook.cpp
~~~

The server-local solver/adapter compatibility work that produced the previous two-substep PASS must be preserved.

Do not rewrite or retune the solver for this acceptance run.

## Important change from the step776 deterministic replay

Do NOT require the historical step776 action-prefix SHA after the safety layer becomes active.

Reason:

The safety layer is active from physics substep 1. If it corrects any earlier unsafe state, the subsequent observation trajectory changes, so later deterministic PPO actions are allowed to differ from the unprotected baseline replay.

Deterministic identity for this test is instead anchored by:

- exact checkpoint path;
- checkpoint SHA256 recorded by the runner;
- seed 15204;
- deterministic model.predict;
- B02 / target_04 environment from the validated test harness;
- exactly 2 x 5 ms physics substeps.

The runner records the resulting protected action-stream SHA256 for reproducibility.

Do not attempt to force the old step776 action sequence after an earlier safety correction.

## Hard safety criterion

Dense real-Beam committed-state limit:

~~~text
max committed penetration <= 1e-6 m
                           <= 0.001 mm
~~~

At every physics substep:

~~~text
real committed Beam
    -> real free_position
    -> production BeamAdapter + B02 SDF dense check
    -> if safe: no intervention
    -> if unsafe: validated feasible solver
    -> independently validate accepted candidate
    -> native Beam freePosition/freeVelocity write
    -> native mechanical mapping propagation
    -> native collision/constraint solve
    -> committed real-Beam dense check
~~~

A later correction cannot erase an earlier committed-state violation.

If any committed substep exceeds the 0.001 mm penetration limit, the acceptance test is FAIL.

## PASS requirements

PASS requires all of the following over the entire episode/horizon:

1. reset committed Beam state is finite and within the accepted domain;
2. configured physics_substeps == 2;
3. every env RL step executes exactly two physics substeps;
4. every physics substep receives the CollisionBeginEvent planner stage;
5. every real Beam free state is independently dense-checked;
6. every unsafe free state invokes the validated real-Beam feasible solver;
7. every returned candidate is finite and dense-safe;
8. every unsafe correction arms the native hook exactly once;
9. native hook reports PASS_NATIVE_WRITE_AND_PROPAGATE;
10. coherent freeVelocity correction executes;
11. native mechanical mapping propagation executes;
12. mapped CollisionDOF free state changes when a correction is injected;
13. CollisionDOFs remain diagnostic only;
14. every committed real-Beam state is finite;
15. max committed penetration <= 0.001 mm;
16. solver_failure_count == 0;
17. native_failure_count == 0;
18. committed_violation_count == 0;
19. non_finite_count == 0;
20. forbidden_fallback_count == 0;
21. at least one real unsafe free state challenges the solver during the episode;
22. no production files are modified.

If the complete 2048-step horizon is reached without normal environment termination, that is still sufficient for the safety acceptance question.

Route success is not required for this safety test.

## Forbidden methods

Absolutely do not use:

- rollback;
- position projection;
- post-step repair;
- action shielding;
- clamping action/motion to zero;
- direct write to committed Beam position;
- direct write to CollisionDOFs;
- CollisionDOFs as feasible-domain constraints;
- native post-contact state as feasible-solver input;
- updateVisual;
- applyRestPosition;
- reward changes;
- observation changes;
- done/terminal changes;
- extra physics substeps;
- training.

Do not modify production mcr_sim/.

## Preserve current test-only compatibility state

Before running:

~~~bash
cd /data/home/3220251075/mcr_sim/mcr_project/mCR_simulator-master/python

git status --short
~~~

Do not run:

~~~text
git clean
git reset --hard
~~~

The current testing/beam_feasible_domain tree may contain server-local validated solver/adapter/CMake compatibility files from previous PASS gates.

Preserve them.

## Preflight 1 — confirm previous gate

Read:

~~~text
testing/beam_feasible_domain/_runtime/results/b02_step776_two_substep_invariance.json
~~~

Confirm the previous result was PASS and specifically:

~~~text
substep1:
free ~ -0.284 mm
candidate ~ +0.00175 mm
committed ~ +0.4823 mm

substep2:
free ~ +0.2986 mm
no solver
committed ~ +0.2986 mm
~~~

Do not rerun that test unless the result file is missing.

## Preflight 2 — native plugin

Use the existing test-only native build if current and loadable:

~~~text
testing/beam_feasible_domain/_runtime/native_build/
~~~

Required runtime Data:

~~~text
fireCount
~~~

Required behavior:

~~~text
re-arm across repeated unsafe physics substeps
~~~

If rebuild is necessary, preserve the already working SOFA 21.12 local test-only CMake compatibility changes.

Do not reinstall SOFA.

Do not touch production.

## Preflight 3 — solver bridge

The exact real-B02 solver used for the previous PASS is the solver to reuse now.

The prior two-substep run reported the existing PASS solver source around:

~~~text
b02_step776_feasible_solve
~~~

For this full-episode acceptance, the online callable MUST explicitly accept the current-frame state, e.g.:

~~~python
solve_feasible_state(q_prev, q_free, adapter, context)
~~~

The runner enables strict bridge mode and rejects a zero-argument/monolithic main() entry point for online multi-frame use. If the previous PASS implementation is monolithic, add only a thin test-only wrapper that explicitly consumes the supplied q_prev and q_free and calls the same validated solver mathematics.

Do not return a saved step776 candidate on later frames. Do not retune or redesign the solver.

Use the current server-local bridge/wrapper compatibility that made the prior PASS possible.

Do not silently fall back to:

~~~text
beam_feasible_poc.py
~~~

because that is only the simplified plane PoC.

Do not redesign the algorithm.

## Static syntax check

Before the long run:

~~~bash
python -m py_compile \
  testing/beam_feasible_domain/tests/b02_full_episode_safety_acceptance.py
~~~

If this exposes only a test-runner compatibility issue, fix only inside:

~~~text
testing/beam_feasible_domain/
~~~

Do not change solver mathematics to make the test pass.

## Execute exactly one acceptance run

Create runtime directories:

~~~bash
mkdir -p \
  testing/beam_feasible_domain/_runtime/logs \
  testing/beam_feasible_domain/_runtime/results
~~~

Run:

~~~bash
python testing/beam_feasible_domain/tests/b02_full_episode_safety_acceptance.py \
  --max-rl-steps 2048 \
  --progress-every 25 \
  > testing/beam_feasible_domain/_runtime/logs/b02_full_episode_safety_acceptance.log 2>&1
~~~

Do not launch a second copy in parallel.

Do not run a baseline OFF episode.

The historical failure behavior is already established; this acceptance run is only for the protected mechanism.

If the runner terminates early because it found a genuine hard safety violation or solver/native failure, do not continue the episode merely to collect more failures.

If execution stops because of a concrete test-harness/API compatibility bug, fix that minimal testing-only bug and rerun once.

## Runtime outputs

Summary JSON:

~~~text
testing/beam_feasible_domain/_runtime/results/b02_full_episode_safety_acceptance.json
~~~

Human report:

~~~text
testing/beam_feasible_domain/_runtime/results/b02_full_episode_safety_acceptance_report.md
~~~

Incremental per-substep trace:

~~~text
testing/beam_feasible_domain/_runtime/results/b02_full_episode_safety_acceptance_trace.jsonl
~~~

The JSONL file is intentionally incremental so a crash/failure does not erase prior substep evidence.

## What to report

Do not paste thousands of substeps.

Report aggregate metrics plus:

- first unsafe free state;
- worst free penetration state;
- worst committed state;
- first committed violation if any;
- first solver/native failure if any;
- terminal reason;
- whether episode ended naturally or reached 2048.

If needed for diagnosis, extract only the relevant trace lines.

## Final decision meanings

### PASS

Use PASS only if:

~~~text
max committed penetration <= 0.001 mm
solver failures = 0
native failures = 0
committed violations = 0
NaN/Inf = 0
forbidden fallback = 0
solver was genuinely challenged at least once
episode ended normally or reached 2048 RL steps
~~~

Interpretation:

The current feasible-domain safety layer passes the fixed worst-case B02 full-episode engineering acceptance.

This is enough to stop asking "can it solve the current penetration failure?" for this trajectory.

It still does not prove universal mathematical nonpenetration.

### FAIL

Use FAIL for any genuine physical/safety failure, including:

- committed penetration > 0.001 mm;
- solver fails on an unsafe free state;
- accepted candidate is unsafe;
- native injection/mapping fails;
- non-finite state;
- forbidden fallback used;
- environment reports a safety/out-of-bounds terminal condition.

### INCONCLUSIVE

Use only for infrastructure/test-harness problems, for example:

- validated solver implementation unavailable;
- native plugin cannot load for non-physical reasons;
- required event not exposed;
- episode unexpectedly aborts because of a runner/API error.

Do not call a physical failure INCONCLUSIVE.

## Required final response

Return:

~~~text
B02 FULL-EPISODE FEASIBLE-DOMAIN SAFETY ACCEPTANCE

Environment:
PASS / FAIL / INCONCLUSIVE
checkpoint = ...
checkpoint SHA256 = ...
seed = 15204
physics dt = 5 ms
physics substeps / RL step = 2
production files modified = YES/NO

Run:
RL steps executed = ...
physics substeps observed = ...
episode terminated = YES/NO
episode truncated = YES/NO
terminal reason = ...
action-stream SHA256 = ...

Safety challenge:
unsafe free states = ...
safe free states = ...
first unsafe free = step ... / substep ... / ... mm
worst free clearance = ... mm
max free penetration = ... mm

Feasible solver:
solver calls = ...
safe candidates = ...
solver failures = ...
total solver runtime = ... s
mean solver runtime = ... s
max solver runtime = ... s
worst accepted candidate clearance = ... mm

Native integration:
native injections = ...
native failures = ...
fireCount final = ...
mapping propagation failures = ...
velocity correction failures = ...

Committed real-Beam safety:
worst committed clearance = ... mm
max committed penetration = ... mm
acceptance limit = 0.001 mm
committed violations = ...

Forbidden fallback audit:
CollisionDOFs used as solver constraints = YES/NO
native post-contact state used as solver input = YES/NO
rollback = YES/NO
projection = YES/NO
action shielding = YES/NO
direct committed-position repair = YES/NO

NaN/Inf:
count = ...

FINAL:
PASS / FAIL / INCONCLUSIVE

Reason:
...

Q1. Did every unsafe free state either receive a safe feasible candidate and native correction, or fail explicitly?

Q2. Did every committed real-Beam state stay within the 0.001 mm penetration acceptance limit?

Q3. Were solver failures, native failures, NaN/Inf and forbidden fallback all zero?

Q4. Did the mechanism survive the complete episode / 2048-step horizon rather than only step776?

Q5. Does this establish that the current penetration failure is solved for this fixed worst-case engineering acceptance trajectory?

Q6. Does this mathematically prove nonpenetration for every future policy/state/vessel?

Q6 must be NO even on PASS.
~~~

## Mandatory stop after this test

After obtaining the final acceptance result:

STOP.

Do not start training.
Do not optimize the solver yet.
Do not add another penetration-validation replay.

Return the result first. The next decision will be made from this single acceptance test.
