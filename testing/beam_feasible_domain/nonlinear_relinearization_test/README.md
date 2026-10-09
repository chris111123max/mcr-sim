# Nonlinear relinearization: native same-frame verification (TEST ONLY)

This suite checks whether **genuine SOFA-native iterative relinearization** improves the
already captured B02 / target_04 / step 2009 / substep 1 state, compared with its
one-shot baseline. There must be **no policy inference, PPO, extra RL step, or extra
physics substep**. The original -0.766410143 mm crash has not been recovered.

## Existing evidence

- Original same-frame native solve: committed clearance +0.082681 mm.
- Converted rotational Jacobian alone: +0.080878 mm.
- Solver tolerance 1e-9 alone: +0.085538 mm, linear row residual about -0.00000336 mm.
- Observed nonlinear selected-row error: up to 0.014872 mm.
- Requested margin: +0.100 mm. Positive clearance is safe relative to the vessel wall, yet fails the intended positive margin.

## What is implemented

`verify_native_stages.py` is a **real read-only executable checker** of native
captures. It imports the existing validated `three_factor_validation` baseline
schema. It checks canonical CSR row data, independently saved free/committed dense
SDF, selected minimum point, reconstructed linear gaps, nonlinear differences,
identical `q_prev/q_free`, metadata compatibility and capture SHA256. It can
compare baseline with at most two more native-stage captures.

It does **NOT** perform SOFA integration, native constraint solve, change the
live state, synthesize a missing solver step, or claim native repeatability from
an NPZ alone. If there are no additional genuine native-stage captures, the
proper result is `INCONCLUSIVE_NO_NATIVE_RELINEARIZATION_CAPTURES`.

## Run

The **actual existing** three-factor suite is located in the repository root's
`testing/beam_feasible_domain/three_factor_validation/`. From the server's
repository `python/` directory, run:

```bash
PYTHON=/data/home/3220251075/mcr_sim/mcr_env/miniforge3/envs/mcr_sofa/bin/python
DIR=testing/beam_feasible_domain/nonlinear_relinearization_test
"$PYTHON" "$DIR/verify_native_stages.py" \
  --baseline /actual/path/to/normalized_baseline.npz \
  --output "$DIR/results/nonlinear_baseline_report.json"
```

For **genuine** native test-only iterations, if possible:

```bash
"$PYTHON" "$DIR/verify_native_stages.py" \
  --baseline /actual/path/to/baseline.npz \
  --stage /actual/path/to/relinearized_2.npz \
  --stage /actual/path/to/relinearized_3.npz \
  --output "$DIR/results/native_stage_comparison.json"
```

Paths are examples only; do not assume these files exist. No need to run the full action prefix merely for this read-only check.

## Codex execution steps on server

1. Pull the commit, inspect existing real captures and the previously successful
   fork-at-`CollisionBeginEvent` native A/B tests. Determine the exact capture
   paths and whether any **same-substep nonlinear relinearization native mechanism**
   actually exists. The old A/B branches each solved only once and are **not**
   valid substitutes for iterative relinearization.
2. Run `verify_native_stages.py` on the real baseline; ensure its independent
   minimum is +0.082681 mm (10 nm baseline match bound) and original row/SDF
   findings match the saved report. If input schema is different, adapt only
   under this `testing/` directory; never fabricate fields.
3. If the current SOFA API supports repeated linearization and native constraint
   solve **without integrating another 5ms substep, overwriting positions,
   projection, action shield, or rollback**, implement a **separate test-only**
   native harness based on the existing fork replay. Preserve the real physical
   environment at `CollisionBeginEvent`, verify saved `q_prev` and `q_free`
   with translation difference <=1e-10m and quaternions <=1e-10.
   Produce captures for 2nd/3rd genuinely native inner solve, and log each native
   solver iteration count, residual, active rows, independent dense SDF profile,
   mapping refresh and event stage. **Never use extra SOFA.animate()** to create
   fake "inner solve" improvements.
4. If the API cannot perform a coherent native re-solve within the same physical
   substep, **stop with INCONCLUSIVE_NATIVE_RELINEARIZATION_UNAVAILABLE**. Do not
   substitute standalone offline projection or the earlier feasible-domain
   optimization solver. Clearly describe the API limitation and evidence.
5. Feed real native captures to `verify_native_stages.py` and independently
   inspect the native trace to establish provenance. Only then label the
   test VALIDATED. Otherwise numeric A/B is not a causal native result.
6. Output baseline, stage2, stage3 committed clearance, min raw/converted linear
   residual, nonlinear mismatch, new worst dense index, SOFA native solve evidence,
   and whether requested margin +0.100mm is met. Classify clearly:
   `PASS`, `FAIL`, or `INCONCLUSIVE`.
7. Do not edit production. Do not resume PPO or run broader replay.

**Important caveat:** The historical -0.766410143 mm failure mechanism remains
unknown. A safe B02 test-frame result cannot establish it.
