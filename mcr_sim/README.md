# Unified mcr_sim: Nelson + SOFA21 + stEVE + LSTM-SAC

The authoritative code is the server-local project. No Git download, environment
installation, formal training or physical integration was performed for this refactor.

## Only supported production entry points

- Environment: `mcr_sim.rl_env.MagneticNavigationEnv`.
- Physics: `mcr_sim.sofa_simulation.create_sofa_scene` / `SofaBackend`, delegating
  to the existing Nelson simulator, instrument, environment and magnetic-controller modules.
- Training: `python -m mcr_sim.train`; training requires `--execute`.
- Evaluation: `python -m mcr_sim.evaluate --checkpoint <trusted checkpoint>`.
- Preflight: `python -m mcr_sim.train --preflight` or `python -m mcr_sim.preflight`.
- Bounded integration handoff: `python -m mcr_sim.integration_smoke --seed 101 --steps 1`.

## Runtime and assets

From the repository root, source `../../setup_mcr_sofa.sh` and export
`PYTHONPATH="$PWD/python:${PYTHONPATH:-}"`.

SOFA physics and PyTorch use the existing `mcr_sofa` environment. Official stEVE
geometry/meshing uses the already-existing anatomical generator `.venv`; base
`mcr_sofa` lacks PyVista. No environment was changed or newly installed.

The new generator uses actual local `AorticArchRandom`, Type I, diameter scale
0.85, default width/height sampling 0.7..1.3. Existing adult-radius 1.0 assets
are preserved and explicitly rejected by the production environment.

Explicit generation (not executed in this delivery):

```bash
python/stEVE_anatomical_vessels/.venv/bin/python -m mcr_sim.vessel_generator --seed 101
# Or, only when deliberately requested:
# python/stEVE_anatomical_vessels/.venv/bin/python -m mcr_sim.vessel_generator --split all
```

The configured split has 10 training seeds and 2 disjoint evaluation seeds.
Cache: `python/vessel_cache/archvariety_type_I_d085/`. Geometry configurations and
file hashes prevent silently mixing cached mesh, centerline or seed. New mesh
validation uses an enclosed-point test for all centerline points. Failed meshes
are retained under a pending directory and are not accepted as valid RL caches.
No new vessel meshes were generated in this source-integration task.

## Task and algorithm

Observation 41 (tip-local), action 3, target threshold strictly below 3 mm. Reward is
`1.0*success + 0.001*(previous_path_mm-current_path_mm) - 0.005`.
The 41D state has no manual frame history; temporal memory is held by the LSTM.
All geometric quantities use the current SOFA tip frame (position vectors subtract
its origin; field directions and displacement vectors are rotated only).

| Indices | Feature | Fixed normalization |
| --- | --- | --- |
| 0-2 | Current unit magnetic direction, tip-local | unit vector |
| 3-17 | Five points on the existing target BFS route at 0/8/20/36/56 mm | 100 mm |
| 18 | Remaining BFS path arc length | 500 mm |
| 19-27 | Real SOFA catheter polyline points 10/30/60 mm behind the tip | 100 mm |
| 28 | Episode time remaining | remaining actions / train or eval limit |
| 29 | Inserted catheter length | 534 mm |
| 30 | Radius interpolated at the current centerline projection | 20 mm |
| 31-33 | Final target vector, tip-local | 500 mm |
| 34-36 | Last applied bounded magnetic command and measured insertion action | normalized actions |
| 37-39 | Measured tip displacement, expressed in the current tip frame | 10 mm |
| 40 | Measured insertion/retraction during the last action | 4 mm |

Every component is clipped to its fixed Gym range (signed [-1,1], unsigned [0,1]).
Short routes and short/duplicate catheter polylines clamp to their real endpoint;
there is no geometry extrapolation or centerline substitution for catheter shape.
Magnetic action feedback uses the quantized bounded command fraction scheduled;
insertion feedback uses actual backend length change, including physical limits.
Reset clears previous action, displacement and insertion feedback to zero.
Old 20D/27D checkpoints and replay cannot be mixed with this 41D model.

Targets are sampled on allowed centerlines. The migrated graph adapter computes
weighted centerline paths (Dijkstra equivalent for the saved tree's positive arc
lengths), never a Euclidean-distance replacement. A new graph/target/reward state
is initialized at each reset. Train/eval horizons: 150/450 actions.

The recurrent SAC keeps 1x900 LSTM and 3x400 actor/twin critics, with the morning
learning rate, warmup and update schedule. It now explicitly keeps one online
hidden state per environment, clears it on reset and avoids cross-episode or
cross-environment replay sequences. Time-limit truncation is not a terminal
bootstrap mask. Checkpoints include optimizers, replay and RNG states. Multi-env
collection is process-isolated and opt-in; its real SOFA operation remains untested.
Ascend device selection is reused, not a new distributed learner.

## Diagnostics and boundaries

No V15 reward, curriculum, HER, PPO or old SAC entry remains in the production
path. SDF and Beam unilateral-constraint helpers were removed from this package;
separate experimental sources remain under python/cpp and
python/training/beam_rhs_consistent and are not attached to the production scene.
Native source/assets outside this package, ROS/hardware code and separate
physical experiment directories remain independent of production RL.

No stable contact-count/penetration measurement was established in this task.
Those diagnostics remain null/NOT_OBSERVED, not zero or PASS. Full beam init,
real 22-substep integration, contact response and penetration require Luna.

## Validation and recovery

```bash
PYTHONPATH=python:$PYTHONPATH OMP_NUM_THREADS=2 python -m unittest discover -s python/testing/mcr_sim -v
PYTHONPATH=python:$PYTHONPATH python -m mcr_sim.compatibility_check
```

The second command initializes only a minimal SOFA21 root pipeline and separately
constructs beam component Data without initializing the full beam/vessel scene.
It never calls `Sofa.Simulation.animate`.

See `UNIFICATION_REPORT.md`, `reports/CURRENT_CODE_AUDIT.json` and
`reports/LUNA_SEED101_STEP1.sh`. Source-only backup is recorded in the audit.
