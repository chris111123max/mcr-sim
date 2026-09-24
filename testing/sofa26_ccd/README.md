# SOFA v26.06 CCD feasibility test

This directory is isolated from the production MCR/SOFA 21.12 environment.

## Purpose

Validate a new physical-safety architecture before touching training:

- SOFA v26.06
- SofaPython3 v26.06
- BeamAdapter v26.06
- FreeMotionAnimationLoop
- CCDTightInclusionIntersection
- FrictionContactConstraint
- BlockGaussSeidelConstraintSolver

The first test is deliberately smaller than the catheter environment. A point
moves far enough to cross a static triangle wall in one 5 ms step. The script
runs the exact same motion with discrete detection and Tight-Inclusion CCD.

Expected result:

- discrete: tunnels through the wall;
- CCD: detects the continuous crossing and keeps the point on the original side.

BeamAdapter is also loaded during the preflight, so the build is useful for the
next catheter-specific test.

## Isolation

All generated artifacts are ignored by Git and live under:

`testing/sofa26_ccd/_runtime/`

The existing `mcr_sofa`, SOFA 21.12 build, training scene, rewards, actions and
checkpoints are not modified.

## Install and run

From the repository Python root:

```bash
cd /data/home/3220251075/mcr_sim/mcr_project/mCR_simulator-master/python
git pull origin master

bash testing/sofa26_ccd/install_and_test_v26_06.sh
```

Artifacts:

- `testing/sofa26_ccd/_runtime/build_manifest.txt`
- `testing/sofa26_ccd/_runtime/ccd_preflight.json`
- `testing/sofa26_ccd/_runtime/logs/configure.log`
- `testing/sofa26_ccd/_runtime/logs/build.log`
- `testing/sofa26_ccd/_runtime/logs/install.log`
- `testing/sofa26_ccd/_runtime/logs/ccd_preflight.log`

To rerun only the preflight after a successful build:

```bash
bash testing/sofa26_ccd/run_preflight.sh
```

The default build uses 10 parallel jobs to stay below the server's effective
22-CPU quota. Override with `SOFA26_JOBS=<n>` if needed.
