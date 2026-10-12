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
committed clearances are counted; original environment done/reward are retained.
This is not a nonpenetration guarantee. In the historical 64-env run, large
penetration caused SDF samples to leave the finite grid; existing invalid-profile
checks stopped that experiment. Packaging does not repair that known failure.

Outputs, failure snapshots, runtime journals and checkpoints are written under
training_runs, never committed. Building creates native/_build, which is ignored.
No production files or running processes are edited by source packaging.
