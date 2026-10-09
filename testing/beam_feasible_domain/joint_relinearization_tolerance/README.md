# Joint native A/B: 3 relinearizations × solver tolerance — TEST ONLY

This folder contains runnable **numerical acceptance, native-trace consistency, and
test-only native-harness launch code**. No production implementation is changed.

**Question:** On the exact B02/target_04 diagnostic step 2009, substep 1,
does 3 same-substep native SOFA solves + tolerance `1e-9` achieve
independent 10-um Beam/SDF committed clearance >= **+0.100 mm**?
The control is 3 solves + the original tolerance `1e-6`.

The historical **-0.766410143 mm** formal PPO penetration has not been
reproduced. Success here is **not** proof that the historical failure is fixed.

## Files

- `verify_joint_ab.py` — read-only, strict two-arm native snapshot/tracing gate
- `run_joint_ab.py` — optionally launches a **pre-existing test-only** native
  replay harness twice (once for each tolerance) and runs the gate
- `test_joint_ab.py` — synthetic regressions of the gate's arithmetic and
  safety/metadata checks (never calls SOFA)

## First: pull & smoke-test the new code

On the server, from the real repository `python/` working directory:

```bash
cd /data/home/3220251075/mcr_sim/mcr_project/mCR_simulator-master/python
git pull origin master
PYTHON=/data/home/3220251075/mcr_sim/mcr_env/miniforge3/envs/mcr_sofa/bin/python
DIR=testing/beam_feasible_domain/joint_relinearization_tolerance
"$PYTHON" -m py_compile "$DIR/"*.py
"$PYTHON" -m unittest discover -s "$DIR" -p "test_*.py" -v
```

These synthetic tests validate ONLY the code's consistency checks, never
the true native SOFA operation.

## Actual native experiment: strict same-frame fork

Use the **existing server-side test-only native fork harness** that already
produced the reported genuine stages:

| Native stage | Committed minimum | Typical rows |
|---|---:|---:|
| original pass1 | +0.082681 mm | 6 |
| original pass2 | +0.095125 mm | 5 |
| original pass3 | +0.096711 mm | 5 |

Known action SHA256:
`6f534ad515bfa635666f9d3237783681e65dc378a3d5608706db30bb00e181c`.

Make only two native forks **of exactly the same q_prev, q_free, action prefix,
live topology, vessel/SDF and RL step/substep**:

1. `nominal`: three native constraint solves; tolerance **1e-6** at ALL passes
2. `tight`: three native constraint solves; tolerance **1e-9** at ALL passes

Target `step2009/substep1` and **NO** second substep; no extra `animate`,
projection, rollback, action shielding, position/free_position overwrite or
training. Exactly 3 native GenericConstraintSolver invocations and 2
mapping refreshes per branch.

For each pass, preserve the *original* q_prev/q_free arrays in the canonical
NPZ, PLUS pass-specific CSR/Jacobians, independently measured dense
q_committed clearance, and the actual q_committed. On passes 2/3, store
`relinearized_about_committed_state_sha256`, computed as SHA256 over previous
native committed Rigid3 `np.asarray(q,dtype='<f8').tobytes(order='C')`.
The row builder must actually use that state; do not merely copy the SHA.

**Important:** The reported value of `active_count` must be compared with
`rebuilt_row_count` **on that same pass**, not the initial 6. If the
relinearized solver built 5 and activated 5, that's correct.

Capture actual native solver invocation/iteration counts, solver error,
tolerance, min linear row gap, wall time, row sets, event order and a trace
proving all branches use the same single physics substep. Solver error greater
than configured tolerance must be disclosed; do NOT report convergence just
because the margin passes.

### Why a server-side harness is required

The existing validated **fork-at-CollisionBeginEvent plus native ressolve and
mapping-refresh code** was run on the server but its exact callable/CLI
implementation is not present in this GitHub tree. It cannot be truthfully
reconstructed by a post-hoc NPZ script.

`run_joint_ab.py` can call that real existing test-only harness when an
explicit CLI adapter is provided. Do not fabricate its result, and do not
substitute NumPy optimization for native constraint solves. If no supported
native harness interface is available, report
`INCONCLUSIVE_NATIVE_HARNESS_INTERFACE_UNAVAILABLE`; do not train or alter
production to work around it.

## Native branch output contract

Each harness invocation receives `arm`, `tolerance`, `passes=3`,
`output_dir`, and `action_sha256` through its OWN supported CLI. It must
write `{output_dir}/arm_manifest.json` and actual `.npz` snapshots.

The JSON schema (illustration; numeric fields must come from measured data):

```json
{
  "requested_tolerance": 1e-6,
  "physics_substeps_observed_in_fork": 1,
  "target_animate_calls": 1,
  "native_solver_invocations": 3,
  "mapping_refresh_count": 2,
  "solver_tolerance_applied_on_all_stages": true,
  "native_solver": "GenericConstraintSolver",
  "direct_position_write": false,
  "projection_used": false,
  "rollback_used": false,
  "action_shielding_used": false,
  "modified_production": false,
  "stages": [
    {
      "pass_index": 1,
      "npz": "pass1.npz",
      "solver_tolerance": 1e-6,
      "native_solver": "GenericConstraintSolver",
      "native_solver_invocations_cumulative": 1,
      "native_solver_iterations": 5,
      "solver_error": 0.00003337,
      "rebuilt_row_count": 6,
      "active_count": 6,
      "native_linear_row_gap_min_mm": -0.002672,
      "native_solver_wall_ms": 0.0,
      "direct_position_write": false,
      "projection_used": false,
      "rollback_used": false,
      "action_shielding_used": false,
      "modified_production": false
    }
  ]
}
```

The displayed values are **an example of format**, not authorization to
synthesize measurements. Stage 2 and 3 additionally need
`rows_rebuilt_from_native_state=true` and the *actual previous native
committed state* SHA256. Their row counts can differ from stage 1.
Omit optional timing fields if unavailable: the report will say timing is
unknown instead of inventing it.

The NPZ must use the existing
`three_factor_validation/validate_three_factors.py` canonical schema:
`q_prev`, `q_free`, `q_committed`, `row_offsets`, `dof_indices`,
`linear_jacobian`, `angular_jacobian`, `free_violations`,
`selected_dense_indices`, `q_free_dense_clearance`,
`q_committed_dense_clearance`. All geometry in SI units.

### Running actual native branches (only if harness already supports CLI)

The following invocation shows the runner interface. Adapt the arguments
to the **actual test-only harness CLI**; `<NATIVE_HARNESS.py>` is NOT a
literal existing repository file.

```bash
"$PYTHON" "$DIR/run_joint_ab.py" \
  --native-runner testing/beam_feasible_domain/<NATIVE_HARNESS.py> \
  --python "$PYTHON" \
  --runner-args '--arm {arm} --solver-tolerance {tolerance} --native-passes {passes} --output-dir {output_dir} --action-prefix-sha256 {action_sha256}' \
  --output-dir "$DIR/results/joint_ab"
```

The launcher runs the two native branches **sequentially** and then the gate.
The harness must write each arm's real `arm_manifest.json` and stage NPZs.
A nonzero result from the runner/gate must be investigated; margin failure
returns code `1`, invalid evidence code `2`.

Alternatively, execute both arms with the server's existing fork driver and
construct a `joint_ab_manifest.json` with its recorded real measurements,
then run read-only verification:

```bash
"$PYTHON" "$DIR/verify_joint_ab.py" \
  --manifest "$DIR/results/joint_ab/joint_ab_manifest.json" \
  --output "$DIR/results/joint_ab/joint_ab_report.json"
```

The verifier demands `nominal` and `tight`, identical starting q_prev/q_free
(translation <=1e-10 m, quaternion geodesic <=1e-10 rad), pass counters 1/2/3,
exactly 3 native solver calls, 2 mapping refreshes, one physical substep,
no position overwrite/projection/rollback, correct parent-state SHA,
same-pass rebuilt row counts and activeCount, correct tolerance at each stage.
Nominal pass1 must match +0.082681 mm and nominal pass3 +0.096711 mm
(within 0.00005 mm); otherwise **do not attribute any A/B difference**.

**Caveat:** JSON flags and numeric snapshots cannot independently prove a
native SOFA C++ solve. Codex MUST audit real event logs/native counters.
A numeric `PASS_MARGIN` means the reported final clearance reaches the
target, conditional on genuine native provenance. It does not prove historical
deep-penetration safety or general invariance.

On passes 2/3, the original q_free is intentionally preserved for action
prefix comparison while rows are relinearized about the prior committed
state. Consequently the verifier does **not** fabricate a per-pass nonlinear
Jacobian prediction error from those stale original q_free arrays.
Capture a genuine stage-local linearization state + Jacobian and verify it
separately to attribute the 2nd/3rd stage nonlinear error.

## Codex acceptance report

```text
JOINT_NATIVE_SAME_FRAME_AB
RUNNER_PATH =
PRODUCTION_SHA_BEFORE_AFTER_MATCH =
SYNTHETIC_TESTS =
ACTION_SHA =
Q_PREV_Q_FREE_MAX_DELTA =
PHYSICS_SUBSTEPS_EXECUTED = 1
NATIVE_SOLVES_NOMINAL/TIGHT = 3 / 3
MAPPING_REFRESH_NOMINAL/TIGHT = 2 / 2
PASS_ROW_COUNT / ACTIVE_COUNT (each arm, each pass) =
NOMINAL_FINAL_CLEARANCE_MM =
TIGHT_FINAL_CLEARANCE_MM =
TIGHT_MARGIN_REACHED = YES/NO
NOMINAL_FINAL_MIN_LINEAR_ROW_GAP_MM =
TIGHT_FINAL_MIN_LINEAR_ROW_GAP_MM =
REPORTED_NATIVE_SOLVER_ERROR/TOLERANCE =
NATIVE_SOLVER_WALL_TIME_RATIO =
NO_DIRECT_POSITION_WRITE / NO_EXTRA_ANIMATE = verified with native traces
ROOT_CAUSE_HISTORICAL_-0.766410MM = UNPROVEN
RESULT = PASS_MARGIN / FAIL_MARGIN / INCONCLUSIVE
```

No rollout, no PPO/optimizer, no production patch.
