# Luna XHigh task: B02 step776 native pre-commit mapping hook

## Goal

The previous Python pre-commit integration test ended:

```text
INCONCLUSIVE
PRODUCTION_MAPPING_REFRESH_NOT_AVAILABLE_AT_PRECOMMIT_STAGE
```

That result already established:

- deterministic action prefix PASS;
- `CollisionBeginEvent` is exposed at the correct stage;
- live unsafe Beam `free_position` is readable there;
- the accepted offline feasible Beam state exists;
- Python cannot refresh production `MultiAdaptiveBeamMapping` at that stage.

Do **not** redesign the feasible solver.

This task tests exactly one new layer:

> Can a test-only native SOFA component, at the real `CollisionBeginEvent`, replace the Beam Rigid3 free state with the already-PASSed accepted candidate, coherently correct `freeVelocity`, propagate `freePosition/freeVelocity` through SOFA's native mechanical mapping visitor, and let the same physical substep finish as a safe committed state?

Only test:

- B02
- target_04
- seed 15204
- RL step 776
- physics substep 1

No multi-step replay. No training.

## New files

Native test-only component:

```text
testing/beam_feasible_domain/native_precommit_hook/CMakeLists.txt
testing/beam_feasible_domain/native_precommit_hook/NativePrecommitMappingHook.cpp
```

Runner:

```text
testing/beam_feasible_domain/tests/b02_step776_native_precommit_integration.py
```

All build/runtime files stay under:

```text
testing/beam_feasible_domain/_runtime/
```

## Hard architecture rules

The native hook may:

1. read live parent Beam `freePosition/freeVelocity`;
2. write the accepted candidate to parent Beam `freePosition`;
3. coherently update parent Beam `freeVelocity`;
4. propagate the **free vector IDs** through SOFA's native mechanical mapping visitor;
5. read CollisionDOF `freePosition` before/after only to verify mapping movement.

It may not:

- write committed Beam `position`;
- write CollisionDOFs;
- use CollisionDOFs as a feasible-domain constraint source;
- use native post-contact state as solver input;
- call `updateVisual`;
- call `applyRestPosition`;
- rollback/project/clamp the accepted state;
- change production files;
- change the offline feasible solver;
- run substep 2.

The runner has an explicit guard that raises **before the second target animate call**. The target result is invalid unless:

```text
target_completed_substeps == 1
target_substep2_executed == false
```

## Step 1 — toolchain/build audit

Work from:

```bash
cd /data/home/3220251075/mcr_sim/mcr_project/mCR_simulator-master/python
```

Check once:

```bash
command -v cmake || true
command -v c++ || command -v g++ || command -v clang++ || true
echo "CONDA_PREFIX=$CONDA_PREFIX"
echo "SOFA_ROOT=$SOFA_ROOT"
find "$CONDA_PREFIX" "${SOFA_ROOT:-/nonexistent}" \
  \( -name 'Sofa.CoreConfig.cmake' -o -name 'SofaConfig.cmake' \) \
  2>/dev/null | head -20
```

If there is no usable C++ compiler or no CMake, stop with:

```text
INCONCLUSIVE: NATIVE_BUILD_TOOLCHAIN_NOT_AVAILABLE
```

Do not install system packages.

If compiler/CMake exist but package names/include paths differ in this installed SOFA version, inspect the active environment and make the smallest compatibility-only changes under:

```text
testing/beam_feasible_domain/native_precommit_hook/
```

Allowed examples:

- old/new `CollisionBeginEvent` include path;
- old/new `MechanicalPropagateOnlyPositionAndVelocityVisitor` include path;
- CMake package/target spelling;
- `RegisterObject` syntax;
- missing test-only link target.

If there is no native equivalent for propagating `freePosition/freeVelocity` through mechanical mappings, stop:

```text
INCONCLUSIVE: NATIVE_FREE_VECTOR_PROPAGATION_API_NOT_AVAILABLE
```

Do not substitute a visual mapping update.

## Step 2 — build only the test plugin

```bash
mkdir -p testing/beam_feasible_domain/_runtime/native_build

cmake \
  -S testing/beam_feasible_domain/native_precommit_hook \
  -B testing/beam_feasible_domain/_runtime/native_build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${SOFA_ROOT:-$CONDA_PREFIX};$CONDA_PREFIX"

cmake --build testing/beam_feasible_domain/_runtime/native_build -j2
```

Ordinary version-compatibility errors may be fixed autonomously, but only in the test-only native directory.

## Step 3 — static audit before replay

Verify all of the following in the actual built source:

1. `CollisionBeginEvent` is the only injection trigger.
2. `candidateFreePosition` is the prior accepted Rigid3 state.
3. Only Beam `freePosition/freeVelocity` are written.
4. Rotation/velocity correction uses `Rigid3Types::coordDifference(candidate, live_free) / dt`, not Euclidean quaternion subtraction.
5. `MechanicalPropagateOnlyPositionAndVelocityVisitor` is called with the free-position/free-velocity vector IDs.
6. CollisionDOFs are read only for before/after mapping diagnostics.
7. No committed `position` write exists.
8. No direct CollisionDOF write exists.
9. The hook is one-shot through `armed`.
10. Production files remain unchanged.

If one of these is false, fix only the test-only implementation before replay.

## Step 4 — execute once

The previous PASSed offline adapter and accepted-state artifact must already exist on the server.

Run:

```bash
mkdir -p testing/beam_feasible_domain/_runtime/logs

python testing/beam_feasible_domain/tests/b02_step776_native_precommit_integration.py \
  > testing/beam_feasible_domain/_runtime/logs/b02_step776_native_precommit_integration.log 2>&1
```

Do not rerun unless fixing a concrete build/test-harness compatibility error.

Avoid frequent polling.

## Deterministic gate

Required action SHA256:

```text
4304fdb75b63859597716b6016a489e68cdb1202e078cca9d844954e40f26ad2
```

Mismatch = FAIL and stop.

## PASS gate

PASS only if all are true:

1. action SHA256 matches;
2. exactly one target physics substep completes;
3. substep2 never executes;
4. native hook fires on target `CollisionBeginEvent`;
5. captured pre-injection real-Beam free state is unsafe;
6. accepted state is the prior offline feasible candidate;
7. coherent `freeVelocity` correction executes;
8. native free-vector mapping propagation executes;
9. parent candidate write error is numerically negligible;
10. mapped CollisionDOF free-state change is finite and nonzero;
11. CollisionDOFs remain diagnostic only;
12. the same native collision/constraint substep completes normally;
13. final committed real-Beam clearance is within/above `1e-6 m` tolerance;
14. independent dense real-Beam/SDF clearance is also within/above tolerance;
15. no NaN/Inf;
16. no production-file edits or fallback workaround.

Residual penetration > 0.05 mm = FAIL.

Residual violation <= 0.05 mm but outside numerical tolerance may be PARTIAL.

Important: the previous ordinary native-contact baseline was already around +0.488 mm. Therefore a positive final clearance alone is **not** a PASS. The native hook/mapping evidence must also pass.

## Required outputs

Read first:

```text
testing/beam_feasible_domain/_runtime/results/b02_step776_native_precommit_integration.json
testing/beam_feasible_domain/_runtime/results/b02_step776_native_precommit_integration_report.md
```

Only inspect the last 80 lines of the log if required.

## Stop rule

After this single target physical substep:

**STOP.**

- no substep2;
- no 10/20-step replay;
- no training.

## Final response format

Return:

```text
B02 STEP776 NATIVE PRE-COMMIT HOOK

Build:
PASS / FAIL / INCONCLUSIVE
compiler = ...
cmake = ...
SOFA config/target = ...
plugin = ...

Action prefix:
PASS / FAIL
SHA256 = ...

Target isolation:
completed substeps = ...
substep2 executed = YES/NO

Native event:
CollisionBeginEvent fired = YES/NO
native status = ...

Unsafe free state:
clearance = ... mm

Accepted candidate:
dense clearance before replay = ... mm

Velocity:
coherent correction executed = YES/NO
method = Rigid3 coordDifference / dt

Native mapping propagation:
ran = YES/NO
parent candidate write error = ... mm
mapped CollisionDOF free-state change = ... mm
NOTE: diagnostic only, never constraint source

Final committed Beam:
clearance = ... mm

Independent dense check:
clearance = ... mm
penetration >0.01 mm count = ...
penetration >0.05 mm count = ...
penetration >0.1 mm count = ...

Final vs accepted:
max translation difference = ... mm
max rotation difference = ... deg

NaN/Inf:
YES/NO

FINAL:
PASS / PARTIAL / FAIL / INCONCLUSIVE

Reason:
...
```

Then answer explicitly:

Q1. Did the native hook replace the real Beam free state at `CollisionBeginEvent`?

Q2. Did SOFA's native mechanical mapping propagation move the mapped collision free state before collision solving?

Q3. Did that candidate survive the rest of the same substep as a safe committed real-Beam state?

Q4. Does this prove long-horizon nonpenetration?

For Q4 the answer is always **NO**. A PASS only unlocks the next test: one very short feasibility-invariance test; it does not unlock training.
