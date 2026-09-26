# Luna XHigh execution task: B02 step776 pre-commit integration

## Goal

Execute the already-written test:

```text
testing/beam_feasible_domain/tests/b02_step776_precommit_integration.py
```

This is an execution and result-analysis task. Do not redesign the architecture.

The previous offline real-beam feasible solve already PASSed. This task asks only:

> Can the accepted safe Rigid3d Beam state be inserted at the actual SOFA pre-collision stage and survive the remainder of the same physical substep as a safe committed state?

Only B02 / target04 / seed15204 / RL step776 / physics substep1 is tested.

## Hard limits

- No training.
- No B02 10/20/100/1000-step windows.
- No production-code edits.
- No C-IPC.
- No SOFA26 CCD route.
- No SDF penalty-wall redesign.
- No CollisionDOF constraint source.
- No position projection.
- No rollback.
- No action shielding.
- No updateVisual state injection.
- No direct manual overwrite of CollisionDOFs as a substitute for production mapping.

All edits, if ordinary compatibility fixes are needed, stay under:

```text
testing/beam_feasible_domain/
```

## Before running

Work from:

```bash
cd /data/home/3220251075/mcr_sim/mcr_project/mCR_simulator-master/python
```

The previous single-frame offline test outputs should already exist under:

```text
testing/beam_feasible_domain/_runtime/results/
```

The previous test-only real-beam adapter should exist under:

```text
testing/beam_feasible_domain/tests/
```

Do not delete or regenerate valid previous results unnecessarily.

## First: static compatibility audit only

Read:

```text
testing/beam_feasible_domain/tests/b02_step776_precommit_integration.py
testing/beam_feasible_domain/tests/b02_beam_adapter.py
testing/beam_feasible_domain/_runtime/results/b02_step776_final_report.md
```

If the prior adapter filename/class/method names differ slightly from what the new test expects, make the smallest compatibility-only change inside `testing/beam_feasible_domain/`.

Allowed examples:

- adapt constructor signature;
- adapt `sample_state` argument name;
- adapt returned dictionary key names;
- point the candidate loader at the actual previously generated accepted-state NPZ/JSON.

Not allowed:

- change safety architecture;
- use CollisionDOFs as the constraint source;
- use native post-contact position as accepted candidate;
- loosen tolerances;
- add rollback/projection.

## Accepted-state artifact gate

The new test intentionally reuses the accepted Rigid3d state from the already-passed offline solve.

If that state was not saved in the existing runtime outputs, do not invent it.

In that case, minimally update/re-run the existing **test-only offline step776 script** only to export these arrays:

```text
q_prev
q_free
q_accepted
```

to an NPZ under:

```text
testing/beam_feasible_domain/_runtime/results/
```

Do not change its solver logic.

Then rerun the pre-commit test.

## Run

Use the existing mcr_sofa environment.

Run:

```bash
python testing/beam_feasible_domain/tests/b02_step776_precommit_integration.py \
  > testing/beam_feasible_domain/_runtime/logs/b02_step776_precommit_integration.log 2>&1
```

Do not run it repeatedly unless fixing a concrete compatibility/test-harness error.

## Token / waiting discipline

This replay takes time.

Do not:

- poll every few seconds;
- repeatedly run ps;
- repeatedly tail the log;
- cat the whole log;
- narrate each waiting interval.

Launch the test and let it finish.

After completion, read in this order:

1. `testing/beam_feasible_domain/_runtime/results/b02_step776_precommit_integration.json`
2. `testing/beam_feasible_domain/_runtime/results/b02_step776_precommit_integration_report.md`
3. only if needed, the last 60-100 lines of the log.

Do not load large JSONL or giant logs into context.

## What the test must prove

PASS is allowed only if:

1. action SHA256 is exactly:
   `4304fdb75b63859597716b6016a489e68cdb1202e078cca9d844954e40f26ad2`;
2. the target stage is the real `CollisionBeginEvent`;
3. the live `beam_dofs.free_position` is the input state;
4. the accepted state comes from the previous real-beam offline feasible solve;
5. committed `beam_dofs.position` is not directly overwritten at pre-commit;
6. CollisionDOFs are not the feasible-domain constraint source;
7. native post-contact state is not used as solver input;
8. `beam_dofs.free_position` is changed to the accepted state;
9. `beam_dofs.free_velocity` is updated coherently;
10. production `MultiAdaptiveBeamMapping` propagates the changed parent free state before collision detection/constraint solve;
11. the same SOFA substep then completes normally;
12. final committed real-beam clearance is within/above numerical tolerance;
13. independent dense real-beam/SDF clearance is also within/above tolerance;
14. no NaN/Inf occurs.

## Mapping gate is critical

If SofaPython exposes `CollisionBeginEvent` but cannot refresh production `MultiAdaptiveBeamMapping` for the modified parent free state at that stage, the required result is:

```text
INCONCLUSIVE
PRODUCTION_MAPPING_REFRESH_NOT_AVAILABLE_AT_PRECOMMIT_STAGE
```

Do not work around this by:

- `updateVisual`;
- writing CollisionDOFs directly;
- applying a post-step correction;
- modifying production files.

That outcome means the next engineering layer is a true animation-loop / mapping-level hook, likely requiring a native/test-only component.

## If CollisionBeginEvent is unavailable

If the event is not exposed in this SofaPython build, report:

```text
INCONCLUSIVE
COLLISION_BEGIN_EVENT_NOT_EXPOSED_TO_SOFAPYTHON
```

Do not substitute `onAnimateEndEvent`.

## Ordinary errors

You may autonomously fix ordinary test-only errors such as:

- import path;
- prior adapter constructor mismatch;
- result key naming mismatch;
- accepted NPZ key naming mismatch;
- simple SofaPython callback signature mismatch.

Every such fix must remain under `testing/beam_feasible_domain/`.

Do not ask the user for confirmation for these ordinary fixes.

## Stop rule

After the single target substep result is obtained:

STOP.

No next substep.
No 10-step replay.
No training.

## Final analysis

Return one compact report containing:

```text
B02 STEP776 PRE-COMMIT INTEGRATION

Action prefix:
PASS/FAIL
SHA256 = ...

Pre-commit stage:
CollisionBeginEvent seen = YES/NO
injection executed = YES/NO

Accepted candidate source:
...

Free state before injection:
clearance = ... mm

Accepted offline candidate:
clearance = ... mm

Production mapping:
refresh method = ...
mapped CollisionDOF free-state change = ... mm
NOTE: CollisionDOFs diagnostic only

Final committed Beam state:
clearance = ... mm

Independent dense check:
clearance = ... mm
penetration >0.01 mm count = ...
penetration >0.05 mm count = ...
penetration >0.1 mm count = ...

Final vs accepted candidate:
max translation difference = ... mm
max rotation difference = ... deg

NaN/Inf:
...

FINAL:
PASS / PARTIAL / FAIL / INCONCLUSIVE

Reason:
...
```

Then answer these three questions explicitly:

Q1. Did the hook really run between free motion and native collision/constraint completion?

Q2. Did the safe accepted Beam state actually propagate through production BeamAdapter mapping and become a safe committed state after the same SOFA substep?

Q3. Does this prove long-horizon nonpenetration?

For Q3, even if PASS, the answer is NO. A PASS only validates one real pre-commit physical substep. The next stage would be a very short feasibility-invariance replay starting from the last feasible committed state before the dangerous region.
