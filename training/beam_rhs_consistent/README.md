# Standalone RHS-consistent PPO training

This directory packages the adaptive implementation used by the independent
32/64-environment experiment. Existing production training sources are unchanged.

Use the existing mcr_sofa environment, SOFA 21.12 runtime/plugin paths, and
Ascend/CANN environment. No installation or environment upgrade is performed.
Set SOFA_ROOT, SOFA_SRC, and EIGEN3_INCLUDE_DIR to existing development files.

```bash
bash training/beam_rhs_consistent/build.sh
N_ENVS=64 bash training/beam_rhs_consistent/launch.sh
```

Build the existing production Beam unilateral plugin separately if it is missing.
The launcher checks both plugins before starting. It starts a fresh NPU PPO run
with 100 epochs, 100 episodes/epoch, 256 rollout steps, batch size 1024, five PPO
training epochs, curriculum, and two 5 ms physics substeps. N_ENVS supports 32/64.
config.json contains the preserved scientific defaults; no prior experiment files
or checkpoints are required. PYTHON, EXP_NAME, and MCR_RHS_PLUGIN can be supplied.
The legacy run manager adds its own timestamp to the model/log directory; the
printed startup.out records the actual model and console-log paths.

Physics uses world-frame angular Jacobians, native Contact plus Beam unilateral
rows, and one native solve by default. Free clearance <=0.150 mm or detected
first-pass risk enables a second solve with updated Contact reference coordinates
and native mapping propagation. The candidate margin remains 0.100 mm.

Near-wall/active/two-pass states are audited independently; far-wall states are
audited every 16 substeps. Skipped committed frames are null/unknown. Negative
committed clearances are counted; severe >0.100 mm penetration or invalid SDF
terminates the *current episode* with the baseline out-of-vessel penalty.
This is not a nonpenetration guarantee. In the historical 64-env run, large
penetration caused SDF samples to leave the finite grid; existing invalid-profile
checks stopped that experiment. Packaging does not repair that known failure.

Outputs, failure snapshots, runtime journals and checkpoints are written under
training_runs, never committed. Building creates native/_build, which is ignored.
No production files or running processes are edited by source packaging.

## Fast train-time safety policy (2026-10-12)

The candidate only (not production) now uses distance-only dense SDF lookups
for row planning and committed clearance: full-grid normals were previously
computed but never used by either operation. Normals are only checked for
actually penetrating samples in the independent audit. The deprecated
`invalid_normal_count` field is null rather than misleadingly reported as 0.

Per-step native JSON/Jacobian/contact-geometry auditing is disabled away from
the 0.150 mm free-clearance risk band. It remains available for near-wall
frames and failure snapshots. Ordinary full JSONL frame journaling is sampled
every 16 physics substeps, plus risky/double-solve/penetrating events. Compact
counters remain on every substep.

The candidate terminates the **current episode**, not the PPO job, if any
independently checked committed Beam surface clearance is below -0.100 mm.
The original out-of-vessel penalty (-30 by default) applies once, and a
simultaneous success reward is suppressed. Negative clearance between 0 and
-0.100 mm is still counted but does not immediately terminate the episode.

If any *free* SDF samples leave the finite grid, there is no legal local Beam
SDF linearization: leave native collision Contact active, disable Beam SDF
rows only for that substep, mark the SDF invalid, finish the scheduled physics
step, and terminate that episode with the same penalty. If committed samples
leave the grid, classify as `SDF_OUT_OF_GRID` rather than manufacturing a
finite clearance. Unexpected failures continue to fail fast.

This code has not been run in the SOFA 21.12 / Ascend server runtime following
packaging. Rebuild the candidate native plugin before use, and verify one PPO
update plus severe-penetration episode reset before a long training run.

Finite-grid limitation: a local-FD perturbation that exits the SDF grid is also treated as a per-episode invalid-SDF failure. Unexpected native errors still stop the experiment.

## Candidate coarse-to-fine Beam free-state gate (V2)

This change is isolated to training/beam_rhs_consistent and does not alter the
production Beam/Contact solver, PPO observation, reward coefficients, action,
curriculum, 2 x 5 ms physics substeps, or the independent fine auditor.

Every physics substep first reconstructs the deployed Beam centerline at
approximately 100 um **spatial** spacing and queries signed clearance only.
Per-Beam spatial spacing is bounded from the cubic Bezier derivative, rather
than assuming rest-length t steps always represent 100 um of arc length.

A free Beam is considered safely far from the wall ONLY when:
- every coarse SDF value is finite; and
- coarse minimum minus (2 x half the maximum inter-sample arc upper bound)
  minus an extra 50 um grid/interpolation cushion is >= 0.25 mm.

These are conservative empirical sampling allowances, not a proof of SDF
Lipschitz regularity or mathematical nonpenetration. Every ambiguous or
possibly near-wall free state still runs the unchanged production 10 um full
profile and world-frame finite-difference Beam unilateral row builder.
The original native Contact is always enabled. A first-pass Contact correction
with active native constraints still runs the existing post-solve movement risk
check; if needed it executes the second real native same-substep solve with
RHS reference consistency and mapping refresh.

Independent committed 10 um audits still execute for near-wall, active,
two-pass or periodic states. Unchecked committed states remain **unknown**.
Episode termination for independently observed penetration deeper than
0.100 mm (or out-of-grid states) is unchanged. Coarse-only free clearance in
the status stream is marked as a *conservative lower bound*, not an exact
10 um measurement. Three cumulative per-worker counters distinguish
coarse_safe_substeps, fine_fallback_substeps and coarse_invalid_substeps.

N_ENVS=64 bash training/beam_rhs_consistent/launch.sh

For a throughput comparison under the same code, launch a separate 32-env
run using N_ENVS=32 and a distinct EXP_NAME. Old processes do not reload
Python code after git pull. This patch needs no native-plugin rebuild because
it edits only Python and README/launch files; the launcher still checks its
previous native plugin binary for staleness.
