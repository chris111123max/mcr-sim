# Beam same-frame nonlinear relinearization diagnostics (TEST ONLY)

These tests are **not** a production fix. The original -0.766410143 mm
training failure remains unreproduced.

Reference saved frame: B02 / target_04, diagnostic worker 23, RL step 2009,
substep 1. Original native committed clearance +0.082681 mm. This frame is
SAFE but misses the intended +0.100 mm margin.

The source snapshot has a known episode-metadata inconsistency: case episode 3,
top-level metadata episode 1. Preserve the inconsistency, never silently
overwrite the data.

## Pull + synthetic smoke tests

Run from the repository python/ directory on the SOFA server:

    cd /data/home/3220251075/mcr_sim/mcr_project/mCR_simulator-master/python
    git pull origin master
    PYTHON=/data/home/3220251075/mcr_sim/mcr_env/miniforge3/envs/mcr_sofa/bin/python
    DIR=testing/beam_feasible_domain/nonlinear_relinearization

    "$PYTHON" -m py_compile \
        "$DIR/offline_directional_probe.py" \
        "$DIR/verify_native_trace.py" \
        "$DIR/test_nonlinear_relinearization.py"

    "$PYTHON" -m unittest discover -s "$DIR" -p 'test_*.py' -v
    "$PYTHON" "$DIR/offline_directional_probe.py" --self-test

Synthetic tests prove only script arithmetic, never SOFA correctness.

## Part A: Read-only real-SDF directional nonlinearization

Identify the actual normalized .npz snapshot from previous
testing/beam_feasible_domain/three_factor_validation work. It needs
q_free, q_committed, q_free_dense_clearance, q_committed_dense_clearance.
DO NOT INVENT MISSING FIELDS.

A second JSON is required with EXACT frame-specific captured geometry:

    {
      "vessel": "B02",
      "target": "target_04",
      "rl_step": 2009,
      "substep": 1,
      "sdf_vti": "/REAL/ABSOLUTE/PATH/vessel_sdf.vti",
      "asset_T_env_sim": [TX, TY, TZ, QX, QY, QZ, QW],
      "asset_offset_sim": [OX, OY, OZ],
      "asset_source_to_sim_scale": REAL_POSITIVE_SCALE,
      "catheter_radius_m": REAL_RADIUS_M,
      "active_elements": [
        {"nodes": [REAL_NODE_0, REAL_NODE_1], "rest_length_m": REAL_LENGTH_M}
      ]
    }

The symbolic fields above are a format illustration, NOT known measurements.
The real geometry must be extracted from this same B02 episode's live
WireBeamInterpolation edgeList/lengthList and mesh topology, SDF VTI,
asset transform, offset, scale and catheter radius including DR.

The executable enforces B02/target_04 step2009/substep1, exactly the 0.100mm
margin, and the saved baseline +0.082681mm (within 0.00001mm). Missing
provenance is INCONCLUSIVE, not a successful numeric test.

If the real active topology or asset transform was not saved, do not guess,
borrow geometry from initial/reset state or another episode. Report
INCONCLUSIVE: GEOMETRY_NOT_RECOVERABLE. Reuse a previously validated target
frame replay only if actually required and its q_prev/q_free matches within
1e-10 m; do not launch a new episode sweep.

Run:

    "$PYTHON" "$DIR/offline_directional_probe.py" \
        --snapshot /REAL/NORMALIZED/SAFE_FRAME.npz \
        --geometry /REAL/RECORDED/FRAME_GEOMETRY.json \
        --max-rounds 3 \
        --output "$DIR/results/directional_report.json"

The probe:
1. Recomputes full 10-um dense Beam/SDF profiles at saved q_free and
   q_committed using production-equivalent geometry and real SDF.
2. Refuses analysis if sample counts differ or either profile disagrees with
   capture by more than 1e-8 m.
3. Checks alpha=1 reconstructs the exact recorded q_committed Rigid3 state.
4. Starting at alpha=1, evaluates a bounded scalar directional Newton probe
   along the already recorded q_free -> native q_committed correction.
5. At most THREE iterations, reports clearance and worst dense index each time.

It NEVER calls SOFA, PPO, env.step, or GenericConstraintSolver. Hypothetical
q(alpha) is in-memory NumPy only and is NEVER committed to any scene. The
directional Newton result is NOT a native constraint re-solve. Even if the
directional curve reaches +0.100mm, this cannot prove that SOFA is capable
of native same-frame relinearization. A negative result does not rule out
a full multi-DOF nonlinear relinearization.

## Part B: Real native same-substep relinearization acceptance

A second real GenericConstraintSolver solve must execute WITHIN the original
step2009/substep1, after the first native committed state is measured and
fresh rows are built from that actual state. Running another 5ms animate,
manually editing position/free_position/velocity, or resuming from a reset is
NOT a same-substep re-solve.

The current production Python interfaces do NOT establish that SOFA 21.12
can legally do a second native solve at that point. If no safe supported API
is available, return INCONCLUSIVE: NATIVE_SECOND_SOLVE_API_NOT_AVAILABLE.

Use the existing test-only fork-at-CollisionBeginEvent harness *only if* it
has real native instrumentation and supports the additional same-substep solve.
Do not claim capabilities by assertion. No production modification.

If genuine native re-solve is supported, save one .npz PER PASS with:
 q_prev, q_free, q_committed, q_committed_dense_clearance.

Additionally save an instrumented JSON trace, with exact schema:

    {
      "capture_kind": "REAL_SOFA_NATIVE_GENERIC_CONSTRAINT_SOLVER",
      "physics_substeps_per_action": 2,
      "physics_dt_s": 0.005,
      "target_rl_step": 2009,
      "target_substep": 1,
      "physics_substeps_observed_in_fork": 1,
      "passes": [
        {
          "npz": "pass0.npz",
          "native_solver": "GenericConstraintSolver",
          "native_solver_invocations_cumulative": 1,
          "rows_rebuilt_from_native_state": false,
          "direct_position_write": false,
          "projection_used": false,
          "rollback_used": false,
          "action_shielding_used": false
        },
        {
          "npz": "pass1.npz",
          "native_solver": "GenericConstraintSolver",
          "native_solver_invocations_cumulative": 2,
          "rows_rebuilt_from_native_state": true,
          "relinearized_about_committed_state_sha256": "REAL_PASS0_COMMITTED_SHA256",
          "direct_position_write": false,
          "projection_used": false,
          "rollback_used": false,
          "action_shielding_used": false
        }
      ]
    }

This schema is a verification contract, not authorization to synthesize data.
All native solver invocation counts must come from real native instrumentation,
not predicted or intended counts. The SHA is computed from actual float64
little-endian SOFA Rigid3 committed (N,7). Rows must really have been rebuilt
about that state. q_prev/q_free must match the original snapshot within
1e-10m/1e-10 quaternion-component absolute differences. Do not run substep2.

Only after a genuine test-only native trace exists, run:

    "$PYTHON" "$DIR/verify_native_trace.py" \
        --trace /ACTUAL/INSTRUMENTED/NATIVE_TRACE.json \
        --baseline /ACTUAL/NORMALIZED/SAFE_FRAME.npz \
        --output "$DIR/results/native_trace_gate.json"

This verifies internal consistency and baseline +0.082681 mm. It cannot
independently observe native SOFA from JSON; truthfulness of instrumentation
must ALSO be verified by reviewing actual native solver counters/logs.

## Stop conditions

- REAL_SDF_BASELINE_MATCH: PASS only if actual profile matches captured profile.
- DIRECTIONAL_MATH_RESULT: numeric counterfactual, NEVER native success.
- NATIVE_SAME_FRAME: PASS/FAIL only after a genuinely instrumented second native
  constraint solve (possibly third) without extra dt/state writes.
- NATIVE_SAME_FRAME: INCONCLUSIVE when unsupported, geometry absent, capture
  differs, or trace provenance is not proven.
- The original deep penetration root cause is still UNPROVEN.

DO NOT alter production, PPO, reward, observation, curriculum, physics substeps,
margin, row count, solver parameters, or training. No expanded rollout.

Return a compact report:

    BASELINE_RECOMPUTED_MM =
    BASELINE_CAPTURE_MAX_ERROR_M =
    DIRECTIONAL_ITERATIONS =
    DIRECTIONAL_CLEARANCE_MM =
    DIRECTIONAL_REACHES_0.100MM = YES/NO
    REAL_NATIVE_SECOND_SOLVE_EXECUTED = YES/NO
    NATIVE_CLEARANCE_MM_PER_SOLVE =
    NATIVE_MARGIN_REACHED = YES/NO/INCONCLUSIVE
    HISTORICAL_-0.766410143MM_ROOT_CAUSE = UNPROVEN
    PRODUCTION_FILES_CHANGED = NO
