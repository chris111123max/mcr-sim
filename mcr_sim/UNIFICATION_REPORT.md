# mcr_sim source unification report

## Outcome and limits

The morning stEVE task/algorithm implementation now lives in mcr_sim. One
production SOFA21 Nelson scene and one formal LSTM-SAC launcher remain.
This is a source-integration delivery, NOT a completed physical validation.
No training, full catheter/vessel initialization, animate call or integration
smoke was executed. No system/Python/SOFA installation, Git operation or download
was performed. Existing checkpoints, logs, results, meshes and native plugins
were not deleted.

## CURRENT_CODE_AUDIT / cleanup

- Deleted old RL/launcher/source files: **46**.
- Deleted migrated duplicate stEVE Python modules/placeholders: **20**.
- Total explicit source deletions: **66**.
- Source-only backup: `/data/home/3220251075/mcr_sim/mcr_project/mCR_simulator-master/source_backups/mcr_unify_20261019_160904.tar.gz`; all **77** members hash-verified.

Removed V11/V12/V13/V15 reward/curriculum/state logic from mcr_rl_env.py and
training_config.py; route tracking/ordered-point helpers; goal-conditioned/HER
wrappers; distributed PPO, LSTM-PPO, ordinary/goal SAC implementations; old
training and evaluation launchers; contrastive-recovery/Goal-TQC training source;
and candidate PPO environment/launchers. These are not retained as compatibility
modes. A missed nested candidate launch.sh was inspected, added to the same
source backup and removed before the final inventory.

Retained Nelson sources: mcr_simulator.py, mcr_instrument.py,
mcr_environment.py, mcr_mag_controller.py, mcr_magnet.py; generic centerline,
SDF, calibration and native-plugin resources remain. The first four now accept
one shared configuration and have no V15 training imports. Generic device and
logging helpers remain. Standalone ROS/hardware and independent physical
experiment directories are outside the cleanup boundary; they are not imported
by the new production environment. Historical documents/results/vendor snapshots
are preserved as provenance, not runnable production entry points.

Changed/new modules and every exact deletion are in
reports/CURRENT_CODE_AUDIT.json and reports/final_source_inventory.json.

## Migration mapping

| Morning module | Production module |
|---|---|
| config | config |
| action_controller | magnetic_controller |
| centerline_adapter | pathfinder |
| vessel_adapter | vessel_bundle |
| instrument layout | instrument_layout |
| environment | rl_env |
| evaluate metrics | metrics |
| sac_training / networks | recurrent_sac / networks |
| train_sac | train |
| standalone sofa_scene | replaced by sofa_simulation delegating Nelson modules |
| reward / task / observation / magnetics / diagnostics | same names under mcr_sim |

## Effective parameter evidence

SOFA beam values below were read from real SOFA21 component Data **before full
beam init**. Root dt/LCP values were read **after minimal pipeline init**.
Pure action/torque/network contracts were tested independently. These distinctions
must not be interpreted as executed physical-model validation.

| Parameter | Target | Effective evidence | Match |
|---|---|---|---|
| body/tip/total length | 0.500 / 0.034 / 0.534 m | WireRestShape straightLength=0.5, length=0.534; tip difference 0.034 | YES, pre-init Data |
| body/tip beams | 100 / 34 | [100, 34] | YES, pre-init Data |
| mechanical nodes | 135 | [135, 1, 1] | YES, grid Data; actual init NOT_RUN |
| collision edges | 100 / 68 | [100, 68] | YES, pre-init Data |
| visual/rest-shape edges | 600 | 600 | YES, headless scene |
| Young modulus body/tip | 100e6 / 21e6 Pa | 100000000.0 / 21000000.0 | YES, pre-init Data |
| catheter OD/ID | 1.33 / 0.8 mm | equivalent stiffness radius read back 0.0006420791471049021 m | YES, derived sectional stiffness |
| magnet length/OD/ID/Br | 4 / 1.33 / 0.86 mm / 1.45 T | moment 0.0037312125 A m^2; original Nelson formula checked | YES, math tests |
| distributed torque | one full magnet total, last four 1 mm elements | actual CFF indices [131, 132, 133, 134]; normalized quarter weights | YES, Data + torque conservation; execution NOT_RUN |
| field | [0.01,0.01,0] T; 14.142 mT | action and controller setup; constant-norm interpolation tested | YES, pure tests; execution NOT_RUN |
| field action | 0.001 quantization / <=24 deg composite | 22 interpolated commands, joint norm limit | YES, pure tests |
| insertion action | +/-4 mm / 22 substeps | increments sum to +/-0.004 m; IRC speed=0, listening=1 | YES, mocked execution + real controller Data |
| dt / substeps / control period | 0.006 s / 22 / 0.132 s | root dt=0.006; mocked loop exactly22; real animate count0 | dt YES; real22 NOT_RUN |
| LCP mu/tolerance/maxIt/build_lcp | 0.1 / 1e-4 / 2000 / False | 0.1 / 0.0001 / 2000 / 0 | YES, post-init Data, actual LCPConstraintSolver |
| intersection contact/alarm/cone | 0.0003 / 0.0005 m / 0.02 | shared config forwarded to LocalMinDistance; source wiring tested | YES, construction; contact execution NOT_RUN |
| collision models | catheter Line+Point, vessel Triangle+Line | real catheter objects created; vessel builder wiring tested and factory registered | YES in construction; full scene NOT_RUN |
| material damping/density | morning 0.2 / 0.1 / 155 kg m^-3 | actual Euler Rayleigh 0.2 / 0.1; density forwarded to Nelson mass field | YES, construction |
| observation/action | 20 / 3 | reset/step, network and action contracts tested | YES |
| reward | success + .001*path_delta_mm - .005 | exact three terms; no other penalties/curriculum | YES |
| target / horizons | <3 mm / 150 train / 450 eval | task boundary and truncation tests | YES |
| LSTM / actor / twin critics | 1x900 / 3x400 / 3x400 each | actual tensors/layer dimensions and optimizer update tested | YES |
| gamma / batch / replay / LR | .99 / 32 / 10000 / .0003218 | same morning config used by learner; batch32 backward test | YES |
| other morning SAC schedule | heatup500000; update1/20; eval250000; tau.005; sequence8; entropy-3 | preserved; no training launched | YES |
| random vessels | AorticArchRandom / TypeI / diameter.85 | official local source; seeds101/1101 geometry reproducible, zero branch gaps | YES geometry; meshes NOT_RUN |

### Explicit physics corrections to match the morning baseline

The original local Instrument treated a computed equivalent **radius** as if it
were a diameter and divided it by two again. The unified builder uses the
morning formula `(outer_radius**4-inner_radius**4)**0.25` directly, as shown by
real Data readback. It also has 135 mechanical nodes for 134 elements, rather
than the original elements-as-nodes grid count. Both changes implement the
specified morning model; they are not attempts to improve stability by changing
the requested material or beam count. Rayleigh stiffness is the morning 0.2,
not the old Instrument's 1.0. No V15 SDF forces, projection, GenericConstraintSolver
fallback or environment-variable physical overrides are attached.

## SOFA21 adaptation and actual component provenance

Sources are actual ObjectFactory registrations in reports/sofa21_components.json,
including file locations; not a guessed plugin-name list. All 16 explicitly
required traditional plugins returned True when loaded.

| Modern/canonical role | SOFA21 spelling used | Observed source |
|---|---|---|
| animation loop | FreeMotionAnimationLoop | SofaConstraint |
| detection pipeline | CollisionPipeline (registered alias of DefaultPipeline) | SofaBaseCollision |
| broad/narrow phase | BruteForceBroadPhase / BVHNarrowPhase | SofaBaseCollision |
| intersection | LocalMinDistance | SofaConstraint |
| response | CollisionResponse (alias of DefaultContactManager) | SofaBaseCollision |
| collision groups | DefaultCollisionGroupManager | SofaMiscCollision |
| constrained solver/correction | LCPConstraintSolver / LinearSolverConstraintCorrection | SofaConstraint |
| integration | EulerImplicitSolver | SofaImplicitOdeSolver |
| beam linear solve | BTDLinearSolver | SofaGeneralLinearSolver |
| rest shape/interpolation/force/IRC/mapping | original Nelson component names | BeamAdapter (source locations verified even where factory Target says None) |
| STL loader | MeshSTLLoader | SofaGeneralLoader source location |
| mesh / regular grid | Mesh / RegularGrid (registered aliases) | SofaBaseTopology |
| Triangle/Line/Point models | unchanged names | SofaMeshCollision |
| magnetic CFF / fixed nodes | ConstantForceField / FixedConstraint | SofaBoundaryCondition |
| entry anchoring springs | RestShapeSpringsForceField | SofaDeformable |
| visual manager loop | DefaultVisualManagerLoop | Sofa.SimulationCore |

No unreliable or non-equivalent solver substitution was made. An initial
readback check failed because SOFA21 exposes grid resolution as Data `n`, not
construction keyword `nx`. Only the readback check was corrected; the complete
attempt log/traceback is retained as sofa21_components.readback_attempt1.*.
The final minimal/component check has zero captured warnings/errors. An earlier
minimal root check warned about the implicit visual manager; the original
Nelson DefaultVisualManagerLoop was then made explicit. Import-only shutdown
may emit SOFA library-cleanup warnings; that is not evidence of a physical
scene or contact check passing.

## Core source tree and unique entries

```text
python/mcr_sim/
  config.py
  mcr_simulator.py / mcr_instrument.py / mcr_environment.py / mcr_mag_controller.py
  sofa_compat.py
  sofa_simulation.py             # ONLY production scene + backend
  instrument_layout.py / magnetics.py / magnetic_controller.py
  vessel_generator.py / vessel_bundle.py / pathfinder.py
  observation.py / reward.py / task.py / diagnostics.py / metrics.py
  rl_env.py                      # ONLY production Gymnasium environment
  networks.py / recurrent_sac.py / sampling.py
  train.py                       # ONLY formal LSTM-SAC launcher
  evaluate.py                    # evaluation launcher
  preflight.py / compatibility_check.py / integration_smoke.py
  mcr_centerline.py / vessel_assets.py / diagnostic_config.py
  sdf_* / beam_sdf_unilateral_fast.py / native/  # opt-in diagnostics, not baseline
python/scene/example_aortic_arch.py            # facade calling the same scene
python/testing/mcr_sim/test_unified.py / test_assets.py
```

Historical ROS/hardware and native experiment directories are intentionally
not represented as new production RL alternatives. All duplicate top-level
stEVE_magnetic_rl Python modules were removed; reports/licenses/vendor sources
remain read-only provenance.

## Final checks

| Check | Status | Scope |
|---|---|---|
| Syntax and relative import references | PASS | all final checked source files |
| Production imports | PASS | 25 modules; no environment constructed |
| Pure/mocked regression suite | PASS | 24 tests, 6.250 seconds, exit0 |
| Magnetic moment, weights and torque conservation | PASS | math against original Nelson Magnet |
| 22-substep distribution, field interpolation, insertion total | PASS | pure/mocked backend only |
| Observation, targets, graph reward, terminal/truncation semantics | PASS | pure tests; SimError cannot count as success |
| LSTM-SAC forward/backward/hidden state/checkpoint | PASS | CPU batch32 update; no rollout training |
| Local official random geometry | PASS | seeds101/1101 reproducible, zero attachment gaps |
| Existing 1.0 asset loader/mm conversion and content-integrity guard | PASS | old assets read only; not accepted as new0.85 caches |
| New0.85 full mesh generation/enclosure | NOT_RUN | no mesh assets generated in this task |
| SOFA21 minimal pipeline init and LCP values | PASS | real root init, dt/LCP readback, no animate |
| SOFA21 beam component creation/Data | PASS | pre-full-init; not full beam validation |
| Full catheter+vessel Simulation.init | NOT_RUN | Luna handoff |
| Real SOFA step / 22 integrations | NOT_RUN | actual animate calls0 |
| Contact response | NOT_RUN | measurements NOT_OBSERVED; cannot claim PASS |
| Independent penetration/safety validation | NOT_RUN | no SDF/new mesh measurement; NOT_OBSERVED |
| NPU learner / multi-process physical collection | NOT_RUN | utilities retained; no new distributed system |
| Formal training / multi-episode integration | NOT_RUN | prohibited and not launched |
| Source backup integrity | PASS | every saved member SHA256 checked |

Termination reasons retain the morning names: target_reached is terminated;
max_steps, vessel_end and simulation_error are truncated. Replay resets sequence
boundaries on either flag but uses only terminated for the SAC bootstrap mask.

## Runtime limitations and next gate

Base mcr_sofa lacks PyVista. The already-existing anatomical `.venv` imports
all required generation/meshing dependencies successfully; the handoff uses it
only for assets and then the existing SOFA21 Python for integration. Neither
environment was modified. Full mesh validation might still reject geometry;
that failure must be inspected, not bypassed or replaced with adult1.0 assets.
BeamAdapter initialization/mapping and true contact/LCP behavior remain risks
until the bounded integration passes. No contact count or penetration result
is invented. The source is ready for Luna's one-action integration gate, not
for formal training.

## Luna single-seed, single-action command (NOT executed)

```bash
cd /data/home/3220251075/mcr_sim/mcr_project/mCR_simulator-master
source ../../setup_mcr_sofa.sh
set -euo pipefail
export PYTHONPATH="$PWD/python:${PYTHONPATH:-}"
python/stEVE_anatomical_vessels/.venv/bin/python -m mcr_sim.vessel_generator --seed 101
python -m mcr_sim.integration_smoke --seed 101 --steps 1 --action 0 0 0 \
  2>&1 | tee python/mcr_sim/reports/luna_seed101_step1.log
```

The generator refuses incompatible caches, never silently overwrites existing
assets and only creates seed101 here. Smoke takes exactly one zero RL action,
attempts22 SOFA substeps, records actual Data/stages/messages/positions/time and
exits nonzero on a solver/simulation error. It does not import a training runner.
The command is also in reports/LUNA_SEED101_STEP1.sh. Do not increase its scale
or start training as part of this handoff.
