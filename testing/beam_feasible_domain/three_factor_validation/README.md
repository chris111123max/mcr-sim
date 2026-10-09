# Three-factor Beam unilateral same-frame validation (TEST ONLY)

This directory contains **offline, read-only** diagnostics. It tests numerical symptoms on a
saved Beam frame; **it cannot identify the original -0.766410143 mm training failure**
without reproducing that failure frame.

Reference B02/target_04 safe active frame: diagnostic worker23, episode3, RL step2009,
substep1. Native same-frame measurements previously reported:
- baseline: +0.082681 mm
- only rotational Jacobian coordinate conversion: +0.080878 mm
- only tighter solver tolerance: +0.085538 mm

All three are *non-penetrating*, but under the intended +0.100 mm safety margin.

## Files

- validate_three_factors.py: independently analyzes (1) rotational Jacobian frame,
  (2) linear CSR row residual and (3) nonlinear committed SDF mismatch. Optionally
  checks full-snapshot native A/B results against exactly the same q_prev/q_free.
- normalize_capture.py: inspects captured NPZ fields and maps them to required keys;
  never creates missing measurements or edits the source capture.

## Commands (from repo python/ directory)

    cd /data/home/3220251075/mcr_sim/mcr_project/mCR_simulator-master/python
    PYTHON=/data/home/3220251075/mcr_sim/mcr_env/miniforge3/envs/mcr_sofa/bin/python
    DIR=testing/beam_feasible_domain/three_factor_validation

    "$PYTHON" "$DIR/validate_three_factors.py" --self-test

    "$PYTHON" "$DIR/normalize_capture.py" \
      --source /path/to/existing_snapshot.npz --inspect

Create a local JSON mapping actual source keys to canonical keys (see below).
If original NPZ already uses the canonical names, a mapping of {} suffices.

    "$PYTHON" "$DIR/normalize_capture.py" \
      --source /path/to/existing_snapshot.npz \
      --mapping /path/to/mapping.json \
      --output "$DIR/results/baseline.npz"

    "$PYTHON" "$DIR/validate_three_factors.py" \
      --snapshot "$DIR/results/baseline.npz" --case jacobian

    "$PYTHON" "$DIR/validate_three_factors.py" \
      --snapshot "$DIR/results/baseline.npz" --case solver

    "$PYTHON" "$DIR/validate_three_factors.py" \
      --snapshot "$DIR/results/baseline.npz" --case nonlinear

    "$PYTHON" "$DIR/validate_three_factors.py" \
      --snapshot "$DIR/results/baseline.npz" --case all \
      --output "$DIR/results/three_factor_report.json"

If full native A/B captures are available (rather than only the clearance summary),
normalize them using the same method, then supply optional paths:

    "$PYTHON" "$DIR/validate_three_factors.py" \
      --snapshot "$DIR/results/baseline.npz" \
      --jacobian-fixed "$DIR/results/jacobian_fixed.npz" \
      --solver-tight "$DIR/results/solver_tight.npz" \
      --case all --output "$DIR/results/native_ab_report.json"

## Required normalized snapshot schema

Every item is an array in one NPZ, SI units/metres, SOFA Rigid3 quaternion XYZW.

Canonical key                     Shape      Meaning
q_free                            (N,7)      Free Beam position at CollisionBeginEvent
q_committed                       (N,7)      Beam position after SAME substep
row_offsets                       (R+1,)     CSR row offset, first 0 last K
dof_indices                       (K,)       Beam DOF index of each Jacobian block
linear_jacobian                   (K,3)      Native row linear gradients
angular_jacobian                  (K,3)      Native row angular gradients (raw)
free_violations                   (R,)       Native input g_free = clearance - 0.100mm
selected_dense_indices            (R,)       Selected dense sample for each row
q_free_dense_clearance            (M,)       Independent radius-adjusted free SDF
q_committed_dense_clearance       (M,)       Independent radius-adjusted committed SDF

Optional:
- q_prev (N,7): REQUIRED for native A/B comparison of identical initial states
- angular_world_oracle (K,3): independently measured world-axis rotational finite
  difference on the EXACT same selected sample and Beam DOFs; do NOT fill with a
  rotation of recorded local Jacobian (circular proof)
- sibling JSON metadata: vessel/target/seed, worker, episode, step/substep,
  interpolation topology signature, SDF identifier, action prefix SHA256

If the capture does not contain the full q_free/committed SDF profiles, CSR values
or exact row selection, do not invent them. Return INCONCLUSIVE for that check.
If independent world-axis FD oracle is not available, the Jacobian-frame case
explicitly returns INCONCLUSIVE, even if raw and rotated blocks differ.

## Normalizer mapping JSON example

    {
      "q_free": "beam_free_state",
      "q_committed": "beam_committed_state",
      "row_offsets": "csr_row_offsets",
      "dof_indices": "csr_dof_indices",
      "linear_jacobian": "csr_linear_blocks",
      "angular_jacobian": "csr_angular_blocks",
      "free_violations": "csr_free_g",
      "selected_dense_indices": "selected_dense_indices",
      "q_free_dense_clearance": "free_sdf_10um",
      "q_committed_dense_clearance": "committed_sdf_10um",
      "q_prev": "beam_prev_state",
      "angular_world_oracle": "independent_world_axis_fd"
    }

Replace example source keys with actual captured NPZ keys from --inspect.
Optional keys may be omitted.

## Verification and safety rules for Codex

1. Run synthetic self-test. A synthetic PASS validates only script arithmetic.
2. Normalize the real existing B02/target_04 snapshot without overwriting raw data.
3. Verify the baseline dense minimum is approximately +0.082681 mm and that the
   saved q_free and committed profiles use exactly the same dense interpolation
   topology/sample ordering. Otherwise stop with CAPTURE_MISMATCH.
4. Run three individual cases. Record separately:
   - Jacobian: original vs independent world FD error, rotated vs independent
     world FD error. If missing independent oracle, INCONCLUSIVE.
   - Solver: smallest reconstructed original and transformed linear gap,
     violated row count. Do not mislabel reconstructed gaps as solver lambdas.
   - Nonlinear: selected actual gap minus linear prediction, global committed
     min clearance, worst-index migration and +0.100mm margin criterion.
5. If full native A/B snapshots exist, verify q_prev and q_free max translation
   difference <= 1e-10 m and matching quaternion state; baseline Jacobian
   and solver-tight must have identical original rows. Reject confounded A/B.
6. Do NOT modify production, restart PPO, extend episode replay, increase physics
   substeps beyond 2x5ms, change margins/row counts or add rollback/projection.
   If a missing measurement needs native integration, the only permitted extra
   work is the ALREADY RECORDED target substep in the existing test-only fork.
7. Stop after the three results and provide a numerical report. State clearly
   that the original historical deep-penetration cause remains unproven.

An independent oracle is not part of production's current snapshot contract.
The numerical pattern described previously suggests a frame mismatch; its native
causal effect on this frame was negative (baseline +0.082681 mm vs conversion
+0.080878 mm). The script intentionally does not claim this correction fixes safety.
