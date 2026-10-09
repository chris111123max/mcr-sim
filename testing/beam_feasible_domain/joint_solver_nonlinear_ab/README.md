# Joint native same-frame A/B — TEST ONLY

Goal: determine whether **three genuine SOFA-native same-substep relinearizations plus a tighter native constraint solver tolerance** reach the +0.100 mm independent 10 um dense SDF margin on the exact B02/target_04 RL step2009/substep1. The original -0.766410143 mm deep-penetration failure remains unexplained.

## Four arms (change only the variables stated)

| Arm | Native same-frame solves | Solver tolerance | Known independent committed min |
|---|---:|---:|---:|
| A baseline | 1 | 1e-6 | +0.082681 mm |
| B repeated | 3 | 1e-6 | +0.096711 mm |
| C tight alone | 1 | 1e-9 | +0.085538 mm |
| D joint, **NEW** | 3 | 1e-9 | unknown |

Reproduce A/B/C using existing **actual captures** and already validated native fork replay. The only new physics A/B arm is D. Reuse old captures instead of re-running when their metadata/initial state is complete. The read-only verifier cannot create native captures.

## Codex run instructions

1. Pull GitHub master from the repo (server cwd usually `.../mCR_simulator-master/python`). Locate existing actual artifacts and the working **test-only** same-frame native relinearization harness. Do **not** assume capture names or filenames.
2. Verify A baseline from source captured replay `action_prefix_sha256=6f534ad5155bfa635666f9d3237783681e65dc378a3d5608706db30bb00e181c`; `q_prev` and `q_free` must match stored snapshot within <=1e-10 m and quaternions exactly or <=1e-10 geodesic rad. Vessel B02 / target_04, RL step 2009 / substep 1.
3. Create branch D only, by forking the **same** CollisionBeginEvent real physical context (or replay just the same recorded prefix if unavoidable). Run genuine native solver **3 times** within the **same first 5ms physical substep**, rebuilding nonlinear rows at the physically valid relinearization points and refreshing mapping at most as in previous validated native scheme. Tighten ONLY GenericConstraintSolver tolerance from `1e-6` to `1e-9`. Keep the same solver max-iterations unless a separate test-only controlled change is explicitly labeled: tightening tolerance by itself doesn't guarantee the solver actually reaches it.
4. Do not call additional `animate` or execute the second 5ms substep. No manual position overwrite, rollback, projection, action shield, extra insertion, policy inference, PPO training, or production modification. Freeze at the target frame.
5. Per round capture **actual** re-built `row_count`, native `activeCount`, native iterations, native error (and units/semantics of error), independent committed dense profile and min, selected rows, min linear gap, max nonlinear-vs-linear difference, minimum-index migration. Do not reuse the initial 6-row expectation for stages with 5 rebuilt rows.
6. The original A/B/C reports may lack machine-readable provenance; if needed create **test-only** provenance adapters from genuine saved log/capture, with exact source references. Do not invent flags or infer solve count from filenames. Provenance fields below must be supported by native event/solver logs.
7. For each arm export a canonical NPZ accepted by `nonlinear_relinearization_test/verify_native_stages.py`, and matching JSON metadata with: `native_solve_count`, `native_tolerance`, `native_solver_iterations`, `native_solver_error`, `native_relinearization_count`, `native_mapping_refresh_count`, `native_substeps_executed`, `native_extra_animate_count`, `native_position_overwrite_count`, `active_count`. Preserve vessel/target/seed/step/substep/action_prefix SHA; if a field cannot be grounded in a real trace, state unavailable and let checker return INCONCLUSIVE.
8. Read-only joint verifier:
```bash
cd /data/home/3220251075/mcr_sim/mcr_project/mCR_simulator-master/python
PYTHON=/data/home/3220251075/mcr_sim/mcr_env/miniforge3/envs/mcr_sofa/bin/python
DIR=testing/beam_feasible_domain/joint_solver_nonlinear_ab
"$PYTHON" "$DIR/verify_joint_ab.py" \
 --a /REAL/PATH/baseline.npz --a-meta /REAL/PATH/A.json \
 --b /REAL/PATH/three_original.npz --b-meta /REAL/PATH/B.json \
 --c /REAL/PATH/one_tight.npz --c-meta /REAL/PATH/C.json \
 --d /REAL/PATH/three_tight.npz --d-meta /REAL/PATH/D.json \
 --output "$DIR/results/joint_native_ab.json"
```
Names above are placeholders. The same verifier may run with only --a first; it must report INCONCLUSIVE until arms B/C/D and real evidence exist.

9. Validate the **native logs** separately: the script only checks NPZ/metadata consistency; metadata is **self reported**, not proof of executed native solves. Report whether actual solver error <= tolerance; if not, do not claim solver converged, even if clearance meets target.
10. Audit `git diff -- mcr_sim training` and `git status`. Keep new scripts, provenance logs and reports strictly inside `testing/`. Do not launch PPO or expand rollout.

## Expected report

```text
FRAME = B02 / target_04 / step 2009 / substep 1
ACTION SHA =
q_prev/q_free match =
A: clearance, rows/active, min linear gap, nonlinear error, solver iterations/error/tolerance
B: same (3 solves, 1e-6)
C: same (1 solve, 1e-9)
D: same (3 solves, 1e-9)
D >= +0.100 mm = PASS/FAIL
D all linear rows satisfied = YES/NO
D solver error <= 1e-9 = YES/NO
D solve and mapping refresh counts confirmed =
D runtime vs A and B =
PRODUCTION MODIFIED = NO
ORIGINAL DEEP PENETRATION ROOT CAUSE = UNPROVEN
NEXT DIRECTION = one evidence-grounded proposal
```

Do not treat a PASS on this **safe** frame as a proof the original real training deep penetration is solved.
