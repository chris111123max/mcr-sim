# Diagnostic tests

This directory is the canonical home for **new Python diagnostic/experiment tests**.

Rules going forward:

- Put new bounded replay, same-state fork, solver, collision, and safety diagnostics here.
- Do not add new diagnostic scripts directly under `testing/py/`.
- Historical scripts already under `testing/py/` are intentionally left in place for reproducibility.
- Diagnostic result artifacts belong under the relevant training run's `diagnostics/` directory, not in the source tree.
- These scripts are evaluation/diagnostic only: they must not start training unless explicitly named as a training test.

Current diagnostics:

- `sdf_unilateral_full_ab.py`: full-horizon GenericConstraintSolver A/B for the SDF unilateral constraint using an identical replayed raw-action sequence.
