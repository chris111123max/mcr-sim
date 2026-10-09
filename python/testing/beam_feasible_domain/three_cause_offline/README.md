# Three-cause Beam unilateral offline diagnostics

**Scope:** test-only. Do not modify production modules, training, reward, state, action, curriculum, or solver configuration. No PPO, no SOFA stepping.

This package is intended to investigate the three distinct errors observed after the formal training hard stop:
1. **Rotation Jacobian frame:** local-axis finite differences provided to a potentially world-axis SOFA constraint Jacobian.
2. **Linear row satisfaction:** whether `g_free + J * (q_committed-q_free) >= 0` holds for each actual CSR row.
3. **Nonlinear SDF discrepancy:** whether independently captured committed dense clearance differs from row-linearized prediction and whether the global minimum moves to a different point.

The original -0.766410143 mm active-row failure was **not** reproduced by the bounded continuation. The available B02/target_04 step 2009 / substep 1 snapshot is **safe** (committed +0.082681 mm). Do not call this a reproduction of the original failure.

## Run

From repository `python/` on the SOFA server:

```bash
python testing/beam_feasible_domain/three_cause_offline/verify_three_causes.py \
  --snapshot /ABSOLUTE/PATH/TO/EXISTING_CAPTURE.npz \
  --metadata /ABSOLUTE/PATH/TO/EXISTING_CAPTURE.json \
  --output testing/beam_feasible_domain/three_cause_offline/results/report.json
```

`--metadata` is optional. The script prints the exact fields read, three independent MEASURED / INCONCLUSIVE sections, and writes JSON. It never calls the simulation or policy and refuses pickle data.

## Required snapshot arrays (and aliases)

Required for **row satisfaction**:
- `q_free`, `q_committed` = (N,7) Rigid3 world XYZ and xyzw quaternion
- `row_offsets` = (rows+1,) CSR
- `dof_indices` = (nnz,)
- `linear_jacobian`, `angular_jacobian` = (nnz,3)
- `free_violations` = (rows,)

Required for **nonlinear comparison**, plus the above:
- `q_free_dense_clearance`, `q_committed_dense_clearance` = *independent* full-profile 10 um SDF values in metres
- `selected_dense_indices` = (rows,)
- optional `requested_margin_m` (defaults to 0.100 mm)

Required for **rotation-frame reference**, plus `q_free`:
- `rotation_jacobian_local` = (K,3), saved local-axis rotation FD derivatives
- `rotation_jacobian_world_reference` = (K,3), independently recomputed **world-axis** FD derivatives
- `rotation_jacobian_nodes` = (K,), DOF index for each derivative block

The rotation-frame test does *not* invent reference derivatives by rotating the supplied Jacobian. If independently computed world-axis reference data was not captured, it explicitly reports INCONCLUSIVE. The other tests can still succeed.

**Important:** The production FD code uses `R * exp(delta_local)`. The numeric linear row check converts quaternions via `R_committed * R_free^-1` to express increments on world axes. It is a first-order diagnostic, not an authoritative read-back of the SOFA constraint-system residual or Lagrange multipliers. A native A/B test is still necessary for causal confirmation.

## Codex server task

1. `git pull origin master` from the repository root or `python/`.
2. Identify the actual saved B02/target_04, worker 23, episode 3, step 2009/substep 1 capture. Do **not** guess its file name; inspect existing `testing/` artifacts. Run this script against the *actual* capture.
3. If required arrays are named differently or nested, implement **test-only** schema adaptation within this folder; never fabricate missing values. In particular do not treat `q_prev` as `q_free`, do not derive a world-FD reference by simply converting the local Jacobian, and do not treat safety metric logs as a committed dense profile.
4. Execute and verify each test independently. Cross-check row counts, min clearance and selected indices against the original snapshot metadata. The baseline should reproduce +0.082681 mm at the relevant frame (allow only documented float tolerance).
5. For the three hypotheses separately, if native same-frame tests are requested, reuse the already proven fork-at-CollisionBeginEvent replay harness in **testing only** with exact q_prev/q_free matching (<=1e-10 m). A: original vs world-converted rotational Jacobian; B: original vs tighter solver tolerance; C: original vs *test-only* relinearization on the same frame. Never claim these independent effects can be inferred from offline JSON arithmetic alone.
6. Limit to one captured frame and its substep, no expanded rollout, no formal training, no production writes. Report evidence, MEASURED/INCONCLUSIVE and whether a change improves **nonlinear committed clearance**, separately from linear residual.
7. If actual existing failure-replay scripts are not in Git, make the dependency explicit. Do not silently rebuild an entire trainer or run additional PPO.

### Existing native A/B observations to preserve

| Native same-frame arm | Committed clearance |
|---|---:|
| Original | +0.082681 mm |
| Rotation frame corrected alone | +0.080878 mm |
| Tighter solver tolerance alone | +0.085538 mm |

These do **not** prove a repair for the original -0.766410143 mm failure. They show a residual nonlinear safety-margin shortfall even with tighter linear solve.

### Final result format

```text
SNAPSHOT ID/PATH =
BASELINE MATCH = PASS/FAIL/INCONCLUSIVE
ROTATION FRAME TEST = MEASURED/INCONCLUSIVE; before/after error
LINEAR ROW SATISFACTION = MEASURED/INCONCLUSIVE; worst gap mm
NONLINEAR SDF GAP = MEASURED/INCONCLUSIVE; max discrepancy mm; worst-point migration
NATIVE A/B = VERIFIED / NOT REPEATED, include original observations if applicable
ROOT CAUSE OF ORIGINAL -0.766410143 mm = UNPROVEN unless actual failure frame is reproduced
PRODUCTION MODIFIED = NO
```
