# Luna XHigh task: SOFA 26.06 official CCD migration-precheck

## Purpose

Before changing the mCR production environment, validate whether official SOFA 26.06 collision/mapping components can plausibly provide the catheter-body nonpenetration guarantee we need.

This task is intentionally isolated from the current SOFA 21.12 mCR environment.

Do not migrate the project.
Do not touch B02 yet.
Do not train.

Run only the three tests under:

~~~text
testing/sofa26_official_ccd_validation/
~~~

The three gates are:

1. official component/runtime audit;
2. minimal continuous-collision tunneling test;
3. BeamAdapter + MultiAdaptiveBeamMapping collision-coverage audit.

## Important upstream facts to verify, not assume away

The official SOFA v26.06.00 CCDTightInclusionIntersection source natively registers:

~~~text
Cube/Cube
Line/Line
Triangle/Point
~~~

and ignores, among others:

~~~text
Triangle/Line
Line/Point
Triangle/Triangle
~~~

The current mCR catheter collision representation is conceptually:

~~~text
vessel = TriangleCollisionModel
catheter = mapped LineCollisionModel + PointCollisionModel
~~~

Therefore the critical question is not merely whether TightInclusion exists.

The critical question is:

Can official Triangle/Point CCD, without Triangle/Line CCD, prevent a catheter LINE INTERIOR from tunneling through a vessel triangle when its sampled endpoints do not hit that triangle?

The tests deliberately target this case.

## Directory

Everything for this experiment stays under:

~~~text
testing/sofa26_official_ccd_validation/
~~~

Runtime-only data stays under:

~~~text
testing/sofa26_official_ccd_validation/_runtime/
~~~

Do not write generated files elsewhere unless an external SOFA installation already exists and is only being read.

## Absolutely protect the current mCR environment

Current production/research environment:

~~~text
mcr_sofa
SOFA 21.12
~~~

Do not install, upgrade, remove, or overwrite any package in mcr_sofa.

In particular, do not run package installation commands while CONDA_PREFIX points at the existing mcr_sofa environment.

Do not modify:

~~~text
mcr_sim/
training/
training_runs/
~~~

No production source edits.

## Step 0 — locate or create an isolated SOFA 26.06 runtime

First inspect, read-only:

~~~bash
which python
python - <<'PY'
import sys
print(sys.executable)
try:
    import Sofa
    print("Sofa module:", Sofa)
except Exception as e:
    print("No Sofa:", repr(e))
PY

find /data/home/3220251075 -maxdepth 5 \
  \( -iname '*sofa*26*' -o -iname '*v26.06*' \) \
  2>/dev/null | head -100
~~~

Also inspect available environment managers without changing anything:

~~~bash
command -v conda || true
command -v micromamba || true
command -v mamba || true
command -v pixi || true
command -v cmake || true
command -v ninja || true
command -v c++ || command -v g++ || command -v clang++ || true
~~~

### If an existing SOFA 26.06 + SofaPython3 + BeamAdapter runtime exists

Use it.

Record:
- Python executable
- SOFA version
- SofaPython3 path/version
- BeamAdapter path/version
- LD_LIBRARY_PATH / plugin paths required

### If no SOFA 26.06 runtime exists

You may create a completely isolated user-space runtime only under:

~~~text
testing/sofa26_official_ccd_validation/_runtime/
~~~

Preferred prefixes:

~~~text
testing/sofa26_official_ccd_validation/_runtime/env/
testing/sofa26_official_ccd_validation/_runtime/src/
testing/sofa26_official_ccd_validation/_runtime/build/
testing/sofa26_official_ccd_validation/_runtime/install/
~~~

Rules:

- no sudo;
- no apt;
- no system package changes;
- no package changes to mcr_sofa;
- do not replace the existing SOFA 21.12 installation;
- use SOFA v26.06.00, not master/development HEAD;
- use a BeamAdapter version/tag compatible with SOFA 26.06;
- enable SofaPython3 because tests are Python;
- use existing compiler/CMake/Ninja only.

If the server lacks dependencies that cannot be satisfied entirely inside the isolated user-space runtime, STOP with:

~~~text
INCONCLUSIVE: ISOLATED_SOFA26_RUNTIME_NOT_AVAILABLE
~~~

Do not damage the working mCR environment just to make the test run.

## Test files

Run:

~~~text
testing/sofa26_official_ccd_validation/tests/test_01_component_audit.py
testing/sofa26_official_ccd_validation/tests/test_02_minimal_tunneling.py
testing/sofa26_official_ccd_validation/tests/test_03_beamadapter_mapping_ccd.py
~~~

Use the SAME isolated SOFA 26.06 Python executable for all three tests.

For example, conceptually:

~~~bash
SOFA26_PYTHON=/path/to/isolated/sofa26/python

"$SOFA26_PYTHON" \
  testing/sofa26_official_ccd_validation/tests/test_01_component_audit.py

"$SOFA26_PYTHON" \
  testing/sofa26_official_ccd_validation/tests/test_02_minimal_tunneling.py

"$SOFA26_PYTHON" \
  testing/sofa26_official_ccd_validation/tests/test_03_beamadapter_mapping_ccd.py
~~~

If environment setup requires LD_LIBRARY_PATH/SOFA_PLUGIN_PATH/PYTHONPATH, set them only in the shell used for this isolated test.

## Test 1 — official component audit

Script:

~~~text
test_01_component_audit.py
~~~

Verify the actual runtime is SOFA 26.06 and that these official capabilities are available:

- FreeMotionAnimationLoop
- BlockGaussSeidelConstraintSolver
- CollisionPipeline
- CompositeCollisionPipeline
- SubCollisionPipeline
- CCDTightInclusionIntersection
- BruteForceBroadPhase
- BVHNarrowPhase
- CollisionResponse
- BeamAdapter plugin
- MultiAdaptiveBeamMapping factory component

Also record the official v26.06 primitive-pair support matrix.

Expected important warning:

~~~text
Triangle/Point = supported
Triangle/Line  = not natively supported
~~~

Do not classify this warning alone as a runtime failure.

Test 1 may return:

~~~text
PASS_WITH_PRIMITIVE_SUPPORT_WARNING
~~~

That means the official components exist, but body coverage still needs Test 2.

Output:

~~~text
testing/sofa26_official_ccd_validation/_runtime/results/test_01_component_audit.json
~~~

## Test 2 — minimal tunneling experiment

Script:

~~~text
test_02_minimal_tunneling.py
~~~

It runs four one-step 5 ms cases:

### Case A
Point crosses a static triangle with ordinary discrete detection.

This is a negative control.

### Case B
Same point crossing using:

~~~text
CCDTightInclusionIntersection
continuousCollisionType = FreeMotion
~~~

The point starts +2 mm above the triangle and has enough velocity to end about -2 mm below it in one 5 ms step if unconstrained.

Independent geometry says its swept trajectory intersects the triangle.

PASS expectation:

The official CCD/contact pipeline prevents complete point tunneling.

### Case C
Long line segment crosses the triangle using discrete collision.

Both line endpoints cross the z=0 plane OUTSIDE the triangle footprint.

The line interior passes through the triangle.

This is another negative control.

### Case D — CRITICAL
Same line-interior crossing with official TightInclusion FreeMotion CCD.

Representation deliberately includes:

~~~text
LineCollisionModel
PointCollisionModel
~~~

but both PointCollisionModel endpoints pass outside the triangle, so Triangle/Point CCD cannot detect the interior crossing.

Independent geometry verifies the line interior sweeps through the triangle.

This case directly targets the missing Triangle/Line support.

### Critical interpretation

If Case B passes but Case D tunnels, classify:

~~~text
FAIL_BODY_GUARANTEE
~~~

This is not a test-harness failure.

It means:

Official TightInclusion v26.06 can prevent supported point/triangle tunneling, but the current Triangle + Line/Point catheter representation still has an uncovered line-interior continuous-collision mode.

That result is directly relevant to mCR.

Output:

~~~text
testing/sofa26_official_ccd_validation/_runtime/results/test_02_minimal_tunneling.json
~~~

Do not weaken the geometry so endpoints start hitting the triangle.

Do not add extra point samples to make the critical case pass.

The purpose is to test primitive coverage, not tune around it.

## Test 3 — BeamAdapter mapping + CCD coverage relevance

Script:

~~~text
test_03_beamadapter_mapping_ccd.py
~~~

Build a small isolated BeamAdapter scene based on the official BeamAdapter deployment example.

Required real components include:

- RodStraightSection
- WireRestShape
- WireBeamInterpolation
- AdaptiveBeamForceFieldAndMass
- InterventionalRadiologyController
- MultiAdaptiveBeamMapping
- mapped CollisionDOFs
- LineCollisionModel
- PointCollisionModel

Do not import the mCR production scene.

The test must verify:

1. BeamAdapter scene initializes;
2. Rigid3 beam state is finite;
3. MultiAdaptiveBeamMapping creates finite mapped collision DOFs;
4. LineCollisionModel exists;
5. PointCollisionModel exists;
6. non-zero mapped line interiors exist between collision points.

Then combine that runtime evidence with Test 2.

### Migration relevance gate

If all are true:

- BeamAdapter mapping works;
- mapped collision representation has non-zero line interiors;
- official Triangle/Line TightInclusion CCD is not available;
- Test 2 Case D actually tunneled;

then classify:

~~~text
FAIL_BODY_GUARANTEE
migration_gate = BLOCKED_FOR_NONPENETRATION_GUARANTEE
~~~

This is the most important result.

It means the official 26.06 components, used with our current kind of BeamAdapter Line/Point collision representation, are insufficient by themselves to establish the body nonpenetration guarantee.

Do NOT proceed to B02 in that case.

Output:

~~~text
testing/sofa26_official_ccd_validation/_runtime/results/test_03_beamadapter_mapping_ccd.json
~~~

## Allowed compatibility fixes

The tests were written against official SOFA 26.06 / BeamAdapter naming.

If the actual v26.06 binary package has minor Python API differences, Luna may make minimal fixes only inside:

~~~text
testing/sofa26_official_ccd_validation/
~~~

Examples:
- Data value access syntax;
- exact plugin loading path;
- exact SOFA version getter;
- collision solver name if the distributed binary uses the same official v26.06 component under an alias;
- BeamAdapter example parameter syntax;
- topology Data formatting.

Not allowed:
- custom CCD implementation;
- custom Triangle/Line intersector;
- custom collision plugin;
- mCR production edits;
- replacing TightInclusion with our feasible-domain solver;
- adding dense point samples specifically to hide the Triangle/Line gap;
- changing test geometry so the line endpoints hit the triangle;
- using rollback/projection.

If an official plugin bundled with SOFA 26.06 independently registers Triangle/Line support at runtime, preserve it and report the evidence. Do not assume the source matrix is the final runtime truth.

## Stop conditions

After the three tests:

STOP.

Do not:
- migrate mCR;
- modify the SOFA 21.12 environment;
- test B02;
- replay checkpoint step776;
- train.

The next decision will be made from these three results.

## Required final report

Return exactly the following information:

~~~text
SOFA 26.06 OFFICIAL CCD MIGRATION PRECHECK

Isolated environment:
PASS / INCONCLUSIVE
Python = ...
SOFA = ...
SofaPython3 = ...
BeamAdapter = ...
Location = ...
Existing mcr_sofa modified = YES/NO

TEST 1 — COMPONENT AUDIT
result = ...
CompositeCollisionPipeline available = YES/NO
CCDTightInclusionIntersection available = YES/NO
BeamAdapter available = YES/NO
MultiAdaptiveBeamMapping available = YES/NO
Triangle/Point CCD = SUPPORTED / NOT SUPPORTED
Triangle/Line CCD = SUPPORTED / NOT SUPPORTED

TEST 2 — MINIMAL TUNNELING
Point discrete:
start z = ... mm
unconstrained end z = ... mm
final z = ... mm
tunneled = YES/NO

Point TightInclusion CCD:
start z = ... mm
unconstrained end z = ... mm
final z = ... mm
tunneled = YES/NO
result = ...

Line-interior discrete:
endpoint paths hit triangle = YES/NO
line interior swept through triangle = YES/NO
final side = ...
tunneled = YES/NO

Line-interior TightInclusion CCD:
endpoint paths hit triangle = YES/NO
line interior swept through triangle = YES/NO
final side = ...
tunneled = YES/NO
result = ...

TEST 3 — BEAMADAPTER MAPPING
scene initialized = YES/NO
parent Rigid3 state finite = YES/NO
mapped CollisionDOFs finite = YES/NO
mapped points = ...
mapped line edges = ...
max mapped edge length = ... mm
LineCollisionModel present = YES/NO
PointCollisionModel present = YES/NO
nonzero line interiors = YES/NO
result = ...

OFFICIAL-ONLY NONPENETRATION GATE:
PASS / FAIL / INCONCLUSIVE

MIGRATION GATE:
PROCEED_TO_B02_OFFICIAL_ONLY_TEST
or
BLOCKED_FOR_NONPENETRATION_GUARANTEE
or
INCONCLUSIVE_ENVIRONMENT

Reason:
...

Q1. Does official TightInclusion prevent the supported Triangle/Point tunneling case?

Q2. Does it also protect a Triangle/Line interior crossing when endpoints miss the triangle?

Q3. Does BeamAdapter's real mapped collision representation contain line interiors for which point samples alone are not a mathematical continuous-body certificate?

Q4. Can SOFA 26.06 official components, without our custom feasible-domain correction, currently guarantee mCR catheter-body nonpenetration?

Q5. Should we proceed to B02 migration testing?

Do not call Q4 PASS merely because the point case passes.
~~~

## Decision policy

Only set:

~~~text
MIGRATION GATE = PROCEED_TO_B02_OFFICIAL_ONLY_TEST
~~~

if the official-only tests provide actual evidence that BOTH sampled points AND line interiors are continuously protected for the BeamAdapter representation.

If Triangle/Line remains unsupported and the line-interior minimal case tunnels, the correct conclusion is:

~~~text
OFFICIAL-ONLY NONPENETRATION GATE = FAIL
MIGRATION GATE = BLOCKED_FOR_NONPENETRATION_GUARANTEE
~~~

That is a useful result, not a failed experiment.
