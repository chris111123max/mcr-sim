# Luna XHigh task: B02 step776 two-substep feasibility invariance

## Goal

The previous native pre-commit gate PASSED at B02 / target_04 / seed 15204 / RL step 776 / physics substep 1.

It proved:

- CollisionBeginEvent is the correct pre-collision stage.
- The accepted real-Beam feasible state can be written to Beam freePosition.
- freeVelocity can be corrected coherently with Rigid3 tangent-space difference / dt.
- SOFA native mechanical propagation moves the production mapped collision free state.
- The same physical substep can finish as a safe committed real-Beam state.

The next question is only:

Does the feasibility mechanism remain valid across the full configured RL step, exactly two consecutive 5 ms physics substeps, when substep2 starts from the real committed state produced by substep1?

This is not a long replay and not training.

## Scope

Run exactly:

- B02
- target_04
- seed 15204
- RL step 776
- physics substep 1
- physics substep 2
- then STOP

Configured execution remains:

~~~text
1 RL step = 10 ms
2 physics substeps = 5 ms + 5 ms
~~~

Do not add more substeps.

## New / changed files

Runner:

~~~text
testing/beam_feasible_domain/tests/b02_step776_two_substep_invariance.py
~~~

Bridge to the already validated server-local real-Beam solver:

~~~text
testing/beam_feasible_domain/tests/b02_online_feasible_solver_bridge.py
~~~

Native hook was updated to support re-arming and now exposes fireCount:

~~~text
testing/beam_feasible_domain/native_precommit_hook/NativePrecommitMappingHook.cpp
~~~

The native hook remains test-only.

## Critical conceptual rule

Do not reuse the saved step776/substep1 accepted candidate for substep2.

At each target CollisionBeginEvent:

1. read the real current committed Beam state as q_prev;
2. read the real current Beam free_position as q_free;
3. independently measure q_free with production BeamAdapter geometry + B02 SDF;
4. if q_free is already safe within numerical tolerance, do not intervene;
5. if q_free is unsafe, run the same already-PASSed real-Beam feasible solver from that frame's q_prev and q_free;
6. independently certify the new accepted candidate;
7. arm the native hook;
8. allow production native mapping + collision/constraint completion;
9. independently measure the resulting committed Beam state.

Substep2 therefore starts from the actual committed state produced by substep1.

## Hard prohibitions

Do not:

- train;
- run step777;
- run 10/20/100-step windows;
- modify mcr_sim/;
- change reward / observation / done;
- increase substeps;
- use rollback;
- use position projection as a safety repair;
- use action shielding;
- clamp motion to zero;
- write committed Beam position at precommit;
- write CollisionDOFs;
- use CollisionDOFs as a feasible-domain constraint source;
- use native post-contact state as feasible-solver input;
- use updateVisual;
- use applyRestPosition;
- redesign the feasible solver;
- widen numerical tolerance to get PASS.

CollisionDOFs are mapping diagnostics only.

## Step 1 — inspect current server-local test state

Work from:

~~~bash
cd /data/home/3220251075/mcr_sim/mcr_project/mCR_simulator-master/python
~~~

Confirm previous PASS outputs and local solver/adapter artifacts still exist:

~~~bash
find testing/beam_feasible_domain/tests -maxdepth 1 -type f \
  \( -name '*feasible*solve*.py' -o -name 'b02_beam_adapter.py' \) -print

ls -lh testing/beam_feasible_domain/_runtime/results/ | tail -30
~~~

Do not delete previous runtime results.

## Step 2 — rebuild the native test plugin

The native source changed because fireCount and re-arming were added.

Reuse the already working test-only SOFA 21.12 CMake compatibility fix from the previous native PASS.

Do not revert a working test-only CMake adjustment merely to match GitHub exactly.

Build only under:

~~~text
testing/beam_feasible_domain/_runtime/native_build/
~~~

Use the same compiler/CMake/Ninja setup that passed the previous gate.

At minimum:

~~~bash
cmake --build testing/beam_feasible_domain/_runtime/native_build -j2
~~~

If regeneration is required, reuse the previous working configure command / test-only direct-library link adaptation.

No production SOFA reinstall.

After build, confirm the plugin contains the updated class and SofaPython exposes fireCount.

If the native plugin cannot be rebuilt only because the old test-only CMake compatibility edit was lost, recreate that same minimal compatibility edit under:

~~~text
testing/beam_feasible_domain/native_precommit_hook/
~~~

Nothing outside testing.

## Step 3 — bridge the validated solver, not a new solver

The exact real-Beam offline feasible solver that previously PASSED was created/used on this server but is not fully committed upstream.

The new bridge:

~~~text
testing/beam_feasible_domain/tests/b02_online_feasible_solver_bridge.py
~~~

first tries to discover a compatible callable automatically.

Run a static import/discovery check before the expensive replay.

If the existing PASS solver is found but its current script has only a monolithic main() or an incompatible function signature, make the smallest possible test-only wrapper so the bridge can call:

~~~python
solve_feasible_state(q_prev, q_free, adapter, context)
~~~

The wrapper must call the existing validated solver logic.

Do not rewrite or retune the solver.

The returned result must expose an accepted Rigid3 state shaped (N, 7) and must not use:

- native post-contact state;
- CollisionDOFs as constraint source;
- rollback;
- projection.

If the previous validated solver implementation is genuinely no longer present on the server and cannot be recovered from existing test files/runtime artifacts, STOP:

~~~text
INCONCLUSIVE: VALIDATED_SOLVER_IMPLEMENTATION_NOT_AVAILABLE
~~~

Do not silently substitute beam_feasible_poc.py. That PoC is a simple plane model and is not the validated B02 real-Beam solver.

## Step 4 — static audit of event ordering

The runner adds objects in this order inside InstrumentCombined:

1. Python OnlineFeasibleController
2. native BeamFeasibleNativePrecommitHook

This ordering is deliberate.

At each target CollisionBeginEvent, the Python controller must:

- inspect q_prev / q_free;
- if needed, solve and set native.candidateFreePosition;
- set native.armed = True;

then the native hook must receive the same event and perform:

- parent free-state write;
- coherent free-velocity update;
- native mechanical mapping propagation.

Do not move planning to AnimateEndEvent.

Do not solve after native collision completion.

## Step 5 — execute exactly once

Create log directory if necessary:

~~~bash
mkdir -p testing/beam_feasible_domain/_runtime/logs
~~~

Run:

~~~bash
python testing/beam_feasible_domain/tests/b02_step776_two_substep_invariance.py \
  > testing/beam_feasible_domain/_runtime/logs/b02_step776_two_substep_invariance.log 2>&1
~~~

Do not start a second replay unless fixing a concrete test-harness/API compatibility bug.

Avoid frequent polling.

## Deterministic gate

Required action prefix SHA256:

~~~text
4304fdb75b63859597716b6016a489e68cdb1202e078cca9d844954e40f26ad2
~~~

Mismatch = FAIL and STOP.

## Required substep behavior

Exactly two target animate calls must occur:

~~~text
target_animate_count == 2
configured_physics_substeps == 2
~~~

No third target animate.

### Case A — free state unsafe

If free clearance < -1e-6 m, all must happen:

1. validated solver called;
2. accepted real-Beam candidate independently safe;
3. native hook armed;
4. fireCount increments exactly by 1;
5. native hook status = PASS_NATIVE_WRITE_AND_PROPAGATE;
6. coherent velocity correction = YES;
7. native mapping propagation = YES;
8. parent candidate write error negligible;
9. mapped CollisionDOF free-state change finite and nonzero;
10. same-substep committed real-Beam state safe.

### Case B — free state already safe

If free clearance >= -1e-6 m:

- solver is not required;
- native hook should not fire for that substep;
- fireCount delta must be 0;
- native completion must still leave committed real-Beam state safe.

Do not intentionally perturb a safe state just to force the solver to run.

## Numerical classification

Keep raw values.

Classification tolerance:

~~~text
1e-6 m
~~~

Meaningful post-commit penetration:

~~~text
> 0.05 mm
~~~

Rules:

- committed clearance >= -1e-6 m: PASS region;
- residual between 0.001 mm and 0.05 mm: PARTIAL;
- residual penetration > 0.05 mm: FAIL.

Do not widen tolerance.

## PASS gate

Overall PASS only if:

1. action SHA matches;
2. exactly two target substeps execute;
3. substep1 begins from its true real committed state and true free state;
4. substep1 obeys unsafe/safe branch rules above;
5. substep1 committed dense real-Beam state is safe;
6. substep2 begins from the actual committed state produced by substep1;
7. substep2 uses its own newly generated free state;
8. if substep2 free is unsafe, a new same-frame solve runs from substep2 q_prev/q_free;
9. if substep2 free is safe, no unnecessary intervention occurs;
10. substep2 committed dense real-Beam state is safe;
11. no forbidden fallback is used;
12. CollisionDOFs remain diagnostics only;
13. all relevant states are finite;
14. production files remain untouched.

## Required outputs

Read first:

~~~text
testing/beam_feasible_domain/_runtime/results/b02_step776_two_substep_invariance.json
testing/beam_feasible_domain/_runtime/results/b02_step776_two_substep_invariance_report.md
~~~

Only inspect the last 100 log lines if needed.

## Mandatory stop

After step776 substep2:

STOP.

Do not run step777.
Do not run a short multi-RL-step window.
Do not train.

## Final response format

Return:

~~~text
B02 STEP776 TWO-SUBSTEP FEASIBILITY INVARIANCE

Build / native plugin:
PASS / FAIL / INCONCLUSIVE
plugin = ...
fireCount exposed = YES/NO

Action prefix:
PASS / FAIL
SHA256 = ...

Target:
RL step = 776
physics substeps executed = 2

SUBSTEP 1
previous committed clearance = ... mm
free clearance = ... mm
solver required = YES/NO
solver source = ...
solver runtime = ... s
accepted dense clearance = ... mm / N/A
native fire delta = ...
native mapping change = ... mm / N/A
committed dense clearance = ... mm
result = PASS / PARTIAL / FAIL / INCONCLUSIVE

SUBSTEP 2
previous committed clearance = ... mm
free clearance = ... mm
solver required = YES/NO
solver source = ...
solver runtime = ... s
accepted dense clearance = ... mm / N/A
native fire delta = ...
native mapping change = ... mm / N/A
committed dense clearance = ... mm
result = PASS / PARTIAL / FAIL / INCONCLUSIVE

Continuity check:
substep2 starts from actual substep1 committed state = YES/NO
substep2 candidate reused from substep1 = YES/NO

Forbidden fallback audit:
CollisionDOFs as solver constraints = YES/NO
native post-contact solver input = YES/NO
rollback = YES/NO
projection = YES/NO
production files modified = YES/NO

NaN/Inf:
YES/NO

FINAL:
PASS / PARTIAL / FAIL / INCONCLUSIVE

Reason:
...
~~~

Then answer explicitly:

Q1. Did substep1 finish with a safe committed real-Beam state?

Q2. Did substep2 start from that real committed state rather than a replayed/saved candidate?

Q3. Was substep2's own free state independently evaluated?

Q4. If substep2 was unsafe, did the validated solver generate a new same-frame candidate and did native mapping/commit succeed?

Q5. Did safety hold across the complete 10 ms RL step, both 5 ms substeps?

Q6. Does this prove long-horizon nonpenetration?

For Q6 always answer NO.

A PASS only unlocks the next diagnostic stage. It does not unlock training automatically.
