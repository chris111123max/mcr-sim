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
- `sdf_unilateral_targeted_step776.py`: targeted step 775-776 replay that compares the actual unilateral rows/tangent planes with the dense post-solve SDF worst point.
- `sdf_dense_adaptive_sampling.py`: test-only voxel-derived dense edge sampler used to prototype denser unilateral constraints without changing training behavior.
- `sdf_safety_aligned_sampling.py`: test-only bridge that reuses the production body-safety sample arclengths on the CollisionDOF chain and reports representation error.
