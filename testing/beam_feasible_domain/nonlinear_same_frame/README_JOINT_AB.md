# Three-pass native relinearization × solver-tolerance joint A/B

This experiment keeps **three genuine SOFA native constraint solves inside one 5 ms physics substep** and changes **only** the native solver tolerance: original `1e-6` versus tighter `1e-9`.

## Tested frame

- B02 / target_04, diagnostic worker 23, step 2009 / substep 1.
- Raw action prefix SHA256: `6f534ad515bfa635666f9d3237783681e65dc378a3d5608706db30bb00e181c`.
- q_prev, q_free must match the captured baseline to `1e-10`.
- Independent dense committed reference +0.082681 mm; intended margin +0.100 mm.
- Prior real native 3-pass result, default tolerance: +0.096711 mm. Previous 1-pass tightened tolerance result: +0.085538 mm. **No result for the combined 3-pass + tightened tolerance yet.**

## Files

- `joint_tolerance_three_pass_ab.py`: executable native-runner orchestration/trace checking script. It does **not** implement or impersonate a native SOFA solver and does not run any training itself.
- `README_JOINT_AB.md`: this contract.

## Server instructions for Codex

1. Pull master. Locate the *actual saved* normalized baseline NPZ and previously successful test-only native fork-at-CollisionBeginEvent harness with repeat solves and mapping refresh. It may be in uncommitted server-side `testing/` files. Do not assume the harness is included in GitHub; inspect the actual server. Do not overwrite any captures.
2. Integrate the existing harness as a **test-only CLI** (only under `testing/`) accepting `--solver-tolerance` and `--output-dir` and `--baseline`, with exactly three same-substep genuine native solves and exactly two genuine mapping refreshes. Keep original exact action replay unchanged. This integration **is necessary**: the GitHub script validates actual output and cannot synthesize the native solves itself.
3. For each arm, save an actual `native_trace.json` in the output directory and referenced .npz captures (per-pass `q_prev`, `q_free`, `q_committed`, full independent 10 um `q_committed_dense_clearance`, `free_violations` at minimum), plus native solver iterations/error. The trace schema follows `testing/beam_feasible_domain/nonlinear_relinearization/verify_native_trace.py` and adds the mandatory fields:
   - top-level: `action_prefix_sha256`, `solver_tolerance`
   - per pass: `rebuilt_row_count`, `native_active_count`, `solver_iterations`, `solver_error`.
   - also keep native solver invocation count, row rebuild flag and `relinearized_about_committed_state_sha256`; no position writes/projection/rollback/shielding.
4. Correct earlier *test-only* audit logic: compare activeCount against **that pass's rebuilt row count**, not original six. Do not touch production row-selection logic.
5. Verify both arms' pass-1 clearance equals +0.082681 mm within `1e-5 mm`, and that both arms' original state and action SHA match. All native solver paths must remain exactly those used in previous successful fork replay. If the existing native trace labels iterations/error under other names, use real values with documented provenance; do not invent them or treat a reported error as converged if it exceeds tolerance.
6. Run from server repo `python/` directory:

```bash
PYTHON=/data/home/3220251075/mcr_sim/mcr_env/miniforge3/envs/mcr_sofa/bin/python
DIR=testing/beam_feasible_domain/nonlinear_same_frame
"$PYTHON" "$DIR/joint_tolerance_three_pass_ab.py" \
  --baseline /ACTUAL/PATH/TO/NORMALIZED_BASELINE.npz \
  --results "$DIR/results/joint_ab" \
  --command-template '"$PYTHON" /REAL/TESTING/ONLY/NATIVE_HARNESS.py --solver-tolerance {tolerance} --output-dir {output_dir} --baseline {baseline}'
```

The command-template above illustrates the interface and should be filled using the **actual harness path and Python executable**. Since this is passed as an argument, the harness process must be invoked with the genuine test-only Python binary; do not pass the literal `$PYTHON` inside single quotes expecting shell expansion. Use an explicit absolute Python path or a safe test-only wrapper.

Alternatively, if genuine arm data already exists, skip `--command-template` to audit `results/joint_ab/three_default` and `results/joint_ab/three_tight` without executing any SOFA.

7. Verify original 3-pass default result should be approximately **+0.096711 mm**. New tighter 3-pass arm must report final nonlinear clearance and native solver error/iterations, and whether each stage truly satisfies its configured tolerance. **Do not assume tight tolerance will pass.**
8. Report comparison with the same frame / no extra substep / no PPO, all three pass row counts, final committed margin, lowest row gap where saved, nonlinear error where saved, speed cost; classify PASS/FAIL/INCONCLUSIVE. Fail requested safety margin if final dense clearance < +0.100 mm, even if row-gap convergence is good.
9. Do not modify `mcr_sim/`, `training/` or any production setting. All additional test code and results under `testing/`. No extra rollout, no new RL step, no extra `animate`, no direct q position overwrite, no projection/rollback/action shielding. The historical original -0.766410143 mm failure is still **unproven**.

**Evidence interpretation:** This program confirms internal consistency of a test-only native capture; a Codex human-level audit of actual native trace and test-only harness code is mandatory before claiming causality. Missing native capture means INCONCLUSIVE, not numerical success.
