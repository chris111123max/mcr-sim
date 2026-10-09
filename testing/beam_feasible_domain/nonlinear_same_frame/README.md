# Nonlinear same-frame Beam constraint verification — TEST ONLY

This directory validates the numerical nonlinear-SDF shortfall at the saved B02/target_04 **safe** active frame. It does **not** reproduce the historical training failure at -0.766410143 mm.

## Entry point

From the Git repository's `python/` directory:

```bash
PYTHON=/data/home/3220251075/mcr_sim/mcr_env/miniforge3/envs/mcr_sofa/bin/python
DIR=testing/beam_feasible_domain/nonlinear_same_frame
"$PYTHON" "$DIR/verify_nonlinear_same_frame.py" \
  --baseline testing/beam_feasible_domain/three_factor_validation/results/baseline.npz \
  --require-baseline-match \
  --output "$DIR/results/offline_nonlinearity.json"
```

The baseline NPZ should be the **real normalized** capture for B02/target_04 step 2009, substep 1. The repository may not contain the capture itself. First use `testing/beam_feasible_domain/three_factor_validation/normalize_capture.py` or a test-only schema adapter to derive that path from saved artifacts, with no invented values.

The verifier requires `q_free`, `q_committed`, full independent 10 um `q_free_dense_clearance`/`q_committed_dense_clearance`, full CSR row arrays, `selected_dense_indices`, and `free_violations`. Optional `q_prev` is mandatory for native branch comparisons. Units are **metres** in NPZ, output millimetres. The baseline +0.082681 mm comparison tolerance is 0.00005 mm.

## What the script can establish

- A frozen-frame reconstruction of row-linearized predictions and actual independent 10 um SDF at the same sample indices.
- Maximum and mean nonlinear prediction discrepancies, whether the worst point moved, whether it was selected, whether linear rows satisfy their first-order inequalities, and the global +0.100 mm margin deficit.
- Cross-capture identical q_prev/q_free translational and quaternion state, before accepting any candidate same-frame native A/B branch.

It uses world-axis quaternion difference for the predicted first-order row increments. The production raw angular-Jacobian frame mismatch is separately validated by `three_factor_validation`. Accordingly, the row-gap reported here is a diagnostic **not** a claim about measured native solver lambdas or exact SOFA constraint residual.

## Native relinearization: evidence required, NOT silently emulated

A numerical second evaluation of SDF is **not** a second SOFA solve. This script deliberately does **not** rewrite position/free_position, advance extra substeps, or call a non-existent SOFA solve API.

Codex should inspect the existing test-only fork-at-CollisionBeginEvent native replay implementation and SOFA solver binding/API for a **valid same-5ms-substep, repeated-native-constraint-solve** mechanism:

1. Replay the recorded raw action prefix to the captured CollisionBeginEvent; verify q_prev and q_free match frozen snapshot within `1e-10 m` and quaternion geodesic angle `1e-10 rad` (or stricter exact equality if feasible). Stop if not.
2. Baseline must reproduce +0.082681 mm from the **single target substep**. No second physics substep.
3. Attempt test-only branch with row rebuild from the actual first native solve correction, then execute **another native constraint solve on the same physical state** without manual state projection, q_committed/free_position overwrite, rollback, action shielding, or altered training/safety config.
4. Record true native solver logs and run a 2-linearization and optional 3-linearization branch. Capture each round's row count, selection, linear row gap, SDF global minimum, worst index/element and whether the margin was attained. Each relinearization must build rows from a legitimately updated linearization point. Preserve unchanged free-state and physical context in the capture.
5. If SOFA current API cannot re-enter native constraint solve safely before commit, mark `INCONCLUSIVE_NATIVE_RELINEARIZATION_UNAVAILABLE`. Do not fake this using a scipy optimizer or a manually modified DOF state and call it physically validated.
6. Any native branch NPZ fed to the verifier must contain all snapshot arrays and provenance fields `native_branch_kind="SOFA_NATIVE"`, `native_solve_count`, `native_substeps_executed=1`, `native_position_overwrite_count=0`, `native_solver_used=True`, `native_same_frame=True`, `native_relinearization_count`; metadata flags are **self-reported**, not independently proven. Matching native logs must be manually verified to consider a physical result. The verifier reports `EVIDENCE_REQUIRES_LOG_AUDIT` rather than PASS.

Optional, after real native branch snapshots:

```bash
"$PYTHON" "$DIR/verify_nonlinear_same_frame.py" \
  --baseline testing/beam_feasible_domain/three_factor_validation/results/baseline.npz \
  --native-round2 "$DIR/results/native_round2.npz" \
  --native-round3 "$DIR/results/native_round3.npz" \
  --require-baseline-match \
  --output "$DIR/results/native_ab.json"
```

## Known original-frame measurements

| Arm | Committed minimum |
|---|---:|
| Original | +0.082681 mm |
| Rotation FD frame conversion only | +0.080878 mm |
| Solver tolerance tightened only | +0.085538 mm |

Original: worst linear row gap about -0.002672 mm; nonlinear prediction discrepancy up to 0.014872 mm. Neither fixes the 0.100 mm margin. Actual historical -0.766410 mm failure remains **unreproduced**.

## Guardrails

- All new code/logs under `testing/beam_feasible_domain/nonlinear_same_frame/`.
- No edits to `mcr_sim/`, `training/`, original tests, PPO, reward, observations, actions, curriculum, physics substep number, Beam margin or other production parameters.
- Do not launch training or longer rollouts.
- Report PASS/FAIL/INCONCLUSIVE separately for **(i) arithmetic reconstruction**, **(ii) actual native repeated solver availability**, **(iii) native same-frame safety improvement**.
- Do not claim historical deep penetration root cause from this safe sample.
