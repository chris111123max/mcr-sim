# Joint three-native-solves × tight solver tolerance, TEST ONLY

This experiment isolates whether **three genuine native same-substep relinearizations** coupled to `GenericConstraintSolver` tolerance `1e-9` can recover the required independent full-Beam SDF +0.100 mm margin.

**New executable:** `verify_joint_tolerance.py` — validates *captured native results* and compares them against two reference arms. This is not a SOFA simulation runner; the server's already-successful test-only fork-at-CollisionBeginEvent harness must execute the new native branch, because that actual harness and its in-memory SOFA state are not in this GitHub directory.

## Reference frame

- B02 / target_04, step 2009/substep 1.
- Action SHA256 `6f534ad5155bfa635666f9d3237783681e65dc378a3d5608706db30bb00e181c`.
- Expected exact q_prev/q_free with <=1e-10 m and <=1e-10 rad quaternion geodesic discrepancy.
- Baseline original 1 native solve: +0.082681 mm.
- Three native relinearizations, original 1e-6 tolerance: +0.096711 mm.
- Target: >= +0.100 mm, using independent 10 um committed full-Beam SDF, *not* a row-linearized estimate.
- Historical -0.766410143 mm training crash **not reproduced**.

## Codex server execution

1. Inspect actual test-only native solver harness and existing reference captures in `testing/beam_feasible_domain/nonlinear_relinearization_test/` and related existing results. Do not guess paths. Never overwrite raw snapshots.
2. Check available branch measurements and reproduce baseline and 3-native original-tolerance results. Fix only stale test-level `expected=6 / active=5` assertion to use each **rebuilt** CSR row count; do not change production row logic.
3. Create **one additional test-only same-frame native branch**: 3 native solves, 2 mapping refreshes, tight `solver tolerance = 1e-9`, otherwise identical to reference 3-solve branch. Do **not** silently modify tolerance during other arms. If 1e-9 is rejected or ignored by current SOFA API, report INCONCLUSIVE.
4. Only target RL step2009, substep1. No substep2 or further `animate`; no PPO or optimizer, no new rollout. No manual `position/free_position` overwrite, projection, rollback or action clipping. Each native solve must truly run and relinearize at an updated state. Reuse the existing legal native mechanism.
5. Capture independent committed 10um Beam/SDF profile, rebuilt row CSR/activeCount, linear residual, nonlinear prediction discrepancy, worst dense point/element, per-solve iteration/error, actual tolerance, runtime, exact q_prev/q_free and SHA256. If the native solver residual remains above nominal tolerance, **do not** claim convergence just from a configured parameter.
6. Normalize saved NPZ arrays from actual recorded fields (test-only adapter allowed; no invented data): `q_prev`, `q_free`, `q_committed_dense_clearance`, `selected_dense_indices`, `rebuild_row_count` or `row_count`, `activeCount` or `active_count`; optional `linear_row_gaps_committed`. Store branch metadata JSON separately containing `native_solve_count=3`, `mapping_refresh_count=2`, `physics_substeps_executed=1`, `extra_animate_count=0`, `position_overwrite_count=0`, `solver_tolerance=1e-9`, `action_prefix_sha256`, `target_rl_step=2009`, `target_substep=1`. These must be factual readouts, not assumed defaults.
7. Run the verifier after preparing the three *real* captures:

```bash
cd /data/home/3220251075/mcr_sim/mcr_project/mCR_simulator-master/python
PYTHON=/data/home/3220251075/mcr_sim/mcr_env/miniforge3/envs/mcr_sofa/bin/python
DIR=testing/beam_feasible_domain/nonlinear_relinearization_test
"$PYTHON" "$DIR/verify_joint_tolerance.py" \
  --baseline /actual/saved/baseline.npz \
  --triple /actual/saved/three_native_original_tolerance.npz \
  --joint /actual/saved/three_native_tight_tolerance.npz \
  --joint-meta /actual/saved/three_native_tight_tolerance.json \
  --output "$DIR/results/joint_tolerance_report.json"
```

8. Audit raw native logs to verify counts, solver calls and mapping refreshes. The verifier only labels `PASS/FAIL` for the **numerical + self-reported evidence**; do not label native physics provenance PASS until logs corroborate. `INCONCLUSIVE` is correct if the new native branch is unavailable.
9. Compare the three native arms, optionally the existing one-solve tight-tolerance reference +0.085538 mm, and explicitly separate: clearance margin success, remaining nonlinear error, row residual, iteration counts, solver convergence and computation time.

**Stop after one new native branch and its validation.** Do not extend to 4/5 relinearizations, change physics substeps/margin, rerun PPO or alter production `mcr_sim/` and `training/`.

## Expected report

```text
FRAME_ID / ACTION_SHA =
IDENTICAL q_prev/q_free = PASS / FAIL
NATIVE BRANCH EVIDENCE = PASS / INCONCLUSIVE
BASELINE 1x ORIGINAL = +0.082681 mm
3x ORIGINAL TOLERANCE = +0.096711 mm
3x 1e-9 TOLERANCE =
REBUILT ROWS / activeCount =
MIN LINEAR ROW GAP =
MAX NONLINEAR SDF ERROR =
SOLVER ITERATIONS / ERROR AT EACH PASS =
SOLVER ACTUALLY CONVERGED = YES / NO / UNKNOWN
TIMING COST =
MARGIN >= +0.100 mm = PASS / FAIL / INCONCLUSIVE
HISTORICAL -0.766410143 mm CAUSE = UNKNOWN
PRODUCTION MODIFIED = NO
```
