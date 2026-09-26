# Codex / Luna XHigh Task: B02 step 776 substep 1 real-beam feasible-domain solve

## 0. Scope

Validate exactly one real dangerous B02 free-motion state.

Do not run training. Do not modify production logic. Do not run B02 10/20/100/1000-step windows.

All new source/test files must stay under `testing/beam_feasible_domain/`.
All runtime outputs must stay under `testing/beam_feasible_domain/_runtime/`.

Do not revive C-IPC, SOFA26 CCD, SDF penalty walls, CollisionDOF unilateral constraints, rollback, projection, or action shielding.

The single question is:

> At B02 / target04 / seed 15204 / RL step 776 / physics substep 1, can the existing Beam feasible-domain algorithm use the actual BeamAdapter free-motion Rigid3d state plus the production B02 SDF and produce a finite feasible accepted beam state without using native contact correction?

If this one frame cannot be validated, stop. Do not continue to longer B02 tests.

## 1. Existing evidence

Existing PoC:
- `testing/beam_feasible_domain/tests/beam_feasible_poc.py`
- `testing/beam_feasible_domain/tests/test_segment_certification.py`

The PoC has already passed A-H algorithmic regression and 1/20/100/1000 pseudo-physics tests. It is not production validation because it uses a hand-written two-element Bezier geometry and does not use the production B02 SDF or a real BeamAdapter pre-solve state.

Existing deterministic replay infrastructure:
- `testing/py/diagnostics/sdf_unilateral_targeted_step776.py`

Production files to inspect, not modify:
- `mcr_sim/mcr_instrument.py`
- `mcr_sim/mcr_rl_env.py`

Historical target:
- vessel: B02
- target: `target_04_centerline.vtk`
- seed: 15204
- checkpoint: epoch 74
- physics: 2 x 0.005 s substeps

Checkpoint:
`/data/home/3220251075/mcr_sim/mcr_project/mCR_simulator-master/training_runs/ppo_v15_2b_physics_wall_10Nm_20260922_215738/models/ppo_base_epoch_074_episodes_07400.zip`

Previously observed at RL step 776 / substep 1:
- free-state radius-adjusted clearance: about `-0.2853 mm`
- native-contact post-solve clearance: about `+0.4792 mm`

The native-contact value is baseline only and must never be counted as feasible-domain success.

## 2. Reuse the existing replay

Reuse the deterministic replay/action-prefix logic from `testing/py/diagnostics/sdf_unilateral_targeted_step776.py` wherever practical.

That script already provides:
- B02/target04/seed15204 scene construction
- deterministic epoch74 action generation/replay
- float32 action SHA256
- production environment and SDF metadata
- `beam_dofs = instrument.getObject('DOFs')`
- CollisionDOFs access for diagnostics only

Do not modify the existing diagnostic file.

Suggested new test-only files:
- `testing/beam_feasible_domain/tests/b02_beam_adapter.py`
- `testing/beam_feasible_domain/tests/b02_step776_free_state.py`

Names may differ if justified.

## 3. Do not use updateVisual injection

The previous attempt to inject a saved state through `Sofa.Simulation.updateVisual(...)` did not correctly update mapped beam geometry. Do not use that path again.

Instead capture the real states produced by SOFA at the dangerous physical substep:
- previous committed Rigid3d state
- `beam_dofs.free_position`
- `beam_dofs.position`

Use `beam_dofs.free_position` as the feasible-domain solver input.

This task is an offline same-frame feasible solve. Do not overwrite the live SOFA scene.

## 4. Hard architecture rule: CollisionDOFs are diagnostic only

The feasible-domain constraint source must not be `CollisionDOFs.position` or `CollisionDOFs.free_position`.

Historical testing found safety-geometry-to-CollisionDOF representation error up to about 0.5 mm, larger than the current dangerous penetration.

CollisionDOFs may only be recorded for comparison.

The new adapter must derive beam geometry from:
- actual Rigid3d Beam DOFs
- production BeamAdapter interpolation/controller state

Relevant production components to inspect:
- MechanicalObject `DOFs`
- `WireBeamInterpolation`
- `InterventionalRadiologyController`
- `MultiAdaptiveBeamMapping`
- MechanicalObject `CollisionDOFs`

The final report must state exactly how points on the beam are generated from an arbitrary copied Rigid3d state.

If the required BeamAdapter interpolation API is not Python-bound, inspect component methods/source/examples and implement a test-only helper that preserves the same geometry definition.

Do not silently fall back to CollisionDOFs.

If real beam geometry cannot be reconstructed reliably, stop with:
`INCONCLUSIVE: REAL BEAM GEOMETRY ADAPTER NOT AVAILABLE`

## 5. Adapter contract

Implement a test-only adapter conceptually equivalent to:

```python
adapter = BeamFeasibleAdapter(env, instrument)
points = adapter.sample_state(beam_rigid_state, inserted_length=...)
clearance, inward, valid = adapter.query_production_sdf(points)
certificate = adapter.certify_state(beam_rigid_state)
```

Requirements:
1. Evaluate an arbitrary copied Rigid3d beam state without committing it to the live production scene.
2. Use the production BeamAdapter geometry definition.
3. Use the production B02 SDF and transforms.
4. Support finite-difference Jacobian evaluation in beam DOF space if no analytical Jacobian is available.
5. Evaluate only the physically inserted beam.

## 6. Production SDF audit

Before solving, verify from real code/data:
- catheter radius
- SDF sign convention
- units
- `asset_source_to_sim_scale`
- `asset_T_env_sim`
- `asset_offset_sim`
- B02 SDF asset
- inserted catheter length

Empirically check at least one known-inside and one known-outside point to verify sign.

The SDF is only a feasibility geometry query. It must not generate penalty forces.

A sign, unit, transform, or radius mismatch is a structural FAIL.

## 7. Geometry sampling and certification

Use beam element/arclength structure. Do not use an arbitrary fixed Cartesian point cloud.

The first implementation may reuse the PoC architecture:
- element endpoints
- midpoint/interior checks
- recursive subdivision where safety cannot be certified

But geometry must come from the actual BeamAdapter state.

Record:
- inserted length
- active beam element count
- initial sample count
- subdivision count
- max subdivision depth
- minimum sampled/certified clearance
- worst arclength/location
- unresolved segment count

If no rigorous Lipschitz bound is implemented, call the result `adaptive sampled certification`, not a mathematical proof.

## 8. Deterministic capture gate

Replay only as far as needed to reach RL step 776 / physics substep 1.

Use the same epoch74 checkpoint and seed15204.

Record:
- action count
- action shape
- dtype `float32`
- action SHA256
- raw action at step 776

The SHA256 must match the historical deterministic prefix. If not, stop with `FAIL: ACTION PREFIX MISMATCH` before running the feasible solve.

## 9. Capture three states

At the dangerous substep capture:

A. Previous committed state: Rigid3d beam state immediately before the dangerous physical substep.

B. Free state: `beam_dofs.free_position`. This is the feasible-domain solver input.

C. Native solved state: `beam_dofs.position`. This is baseline only.

Use the same new real-beam adapter + production SDF to measure A/B/C:
- minimum radius-adjusted clearance
- worst point
- worst arclength
- certification statistics

Do not compare values from different geometry samplers as if they were equivalent.

## 10. Free-state consistency gate

The new real-beam adapter must show the free state is genuinely unsafe.

Expected sign: `free clearance < 0`.

It should be reasonably consistent with the prior `-0.2853 mm`, but exact equality is not required because the new adapter measures real beam geometry rather than CollisionDOFs.

If the new real-beam adapter says the free state is clearly safe, for example `> +0.05 mm`, stop and diagnose geometry-definition mismatch. Do not tune the solver first.

## 11. Solver state and rotation parameterization

Use:
- `q_prev`: previous committed Rigid3d state
- `q_free`: captured Rigid3d free state

The desired unconstrained update is the physical change from `q_prev` toward `q_free`.

Quaternion coefficients must not be subtracted as Euclidean rotational DOFs.

Use a proper SO(3) increment:
- relative quaternion
- convert to rotation vector/tangent increment
- apply updates through quaternion composition

The final report must state the rotation parameterization.

Incorrect quaternion treatment makes the test INVALID.

## 12. Feasible solver

Reuse the validated PoC ideas:
- active constraints
- nonlinear relinearization
- finite-difference or analytical Jacobian
- feasible line search
- adaptive geometry certification

The objective is to stay as close as practical to `q_free` while satisfying production radius-adjusted clearance.

Do not use:
- large penalty forces
- position projection
- rollback to `q_prev`
- clamp to zero motion
- native contact result as initialization or final answer

If no feasible update can be found, report failure.

## 13. Nonlinear trace

For every nonlinear iteration record:
- iteration
- active constraint count
- minimum clearance before update
- worst arclength/location
- Jacobian row count
- solver success/residual
- line-search alpha
- candidate minimum clearance
- accepted minimum clearance

When more than one nonlinear iteration occurs, at least one of active set, worst location, normal, or Jacobian should actually change.

## 14. Feasible line search

Evaluate alpha=1 first.

For every candidate record:
- alpha
- candidate sampled/certified clearance
- accepted/rejected

If alpha=1 is feasible after the constrained update, accept it. Do not artificially force alpha<1.

If alpha is repeatedly reduced toward zero without progress, report `FAIL: FEASIBLE LINE SEARCH STAGNATION`.

Do not replace the failure with rollback.

## 15. Independent safety check

The accepted state must be checked by a second measurement path denser than the active constraint set.

At minimum:
- denser arclength sampling
- production SDF queried independently
- beam element interiors included

If an existing direct beam/mesh distance helper is readily available, record it too, but do not build a new geometry library for this one-frame task.

Report separately:
- solver constraint minimum clearance
- independent dense minimum clearance

## 16. Native contact contamination check

The feasible solver must start from the free state, not the native post-contact state.

Every result file must contain:

```json
{"used_native_post_contact_state_as_solver_input": false}
```

If true, the test is invalid.

Do not write the accepted candidate back into the live SOFA scene in this task.

## 17. Required runtime outputs

Write only under `testing/beam_feasible_domain/_runtime/results/`:
- `b02_step776_adapter_audit.json`
- `b02_step776_capture.json`
- `b02_step776_feasible_solve.json`
- `b02_step776_independent_check.json`
- `b02_step776_final_report.md`

Do not add runtime outputs to Git.

Adapter audit must include:
- beam state source
- beam geometry source
- whether CollisionDOFs are used as constraint source
- production B02 SDF usage
- native-contact-input contamination flag
- catheter radius
- SDF sign/units
- rotation parameterization
- production files modified flag

Capture must include:
- seed/model/target/checkpoint
- action SHA256
- RL step/substep
- raw action
- previous/free/native state summaries
- previous/free/native real-beam clearances
- CollisionDOF clearances only as diagnostics

Feasible solve must include:
- free min clearance
- accepted solver min clearance
- nonlinear iterations
- active constraints per iteration
- Jacobian rows
- complete line-search trace
- minimum alpha
- solver residual/error
- max/RMS/tip translation correction
- max/RMS/tip rotation correction in degrees
- subdivisions/depth/unresolved segments
- finite/NaN/Inf
- runtime breakdown

Single-frame runtime breakdown:
- beam geometry evaluation
- production SDF queries
- adaptive certification
- Jacobian finite differences
- optimization solve
- relinearization
- line search
- independent safety check
- total

Do not optimize performance yet.

Independent check must include:
- sample count
- minimum clearance in m and mm
- worst point/arclength
- count deeper than 0.01 mm
- count deeper than 0.05 mm
- count deeper than 0.1 mm
- finite

## 18. Numerical classification

Keep raw values.

For classification only, treat absolute violation <= `1e-6 m` as the numerical-tolerance region.

Do not widen tolerance to obtain PASS.

Any residual penetration > `0.05 mm` is a meaningful single-frame failure.

## 19. PASS gate

PASS only if all are true:
1. Action prefix SHA256 matches.
2. Production B02 SDF and real production transforms are used.
3. Actual Rigid3d beam free state is used.
4. CollisionDOFs are not the constraint source.
5. New real-beam adapter shows the free state is genuinely unsafe.
6. Feasible solver actually runs.
7. Accepted solver clearance is within/above numerical tolerance.
8. Independent dense check is also within/above numerical tolerance.
9. No projection/rollback/clamp.
10. Native post-contact state is not used as solver input.
11. Quaternion/rotation increments are handled correctly.
12. All outputs are finite.
13. No unresolved unsafe beam segment remains.

PARTIAL:
- real adapter and production SDF work;
- solver runs;
- but residual violation remains <= 0.05 mm or certification remains unresolved.

FAIL examples:
- action mismatch
- wrong geometry/SDF source
- CollisionDOF used as constraint source
- free state cannot be demonstrated unsafe
- nonlinear solve failure
- line-search stagnation
- accepted penetration > 0.05 mm
- NaN/Inf
- invalid quaternion handling
- hidden rollback/projection/clamp required

INCONCLUSIVE:
- production BeamAdapter geometry cannot be evaluated from an arbitrary copied Rigid3d state with sufficient fidelity.

Do not replace INCONCLUSIVE with a CollisionDOF surrogate.

## 20. Mandatory stop

After this one-frame test, STOP.

Do not run 10/20-step B02 windows.
Do not start training.
Do not integrate into production SOFA dynamics.

## 21. Final report format

`b02_step776_final_report.md` must end with:

```text
B02 STEP776 REAL-BEAM FEASIBLE-DOMAIN TEST
==========================================

Action prefix:
PASS / FAIL
SHA256: ...

Real BeamAdapter geometry adapter:
PASS / PARTIAL / FAIL / INCONCLUSIVE
Geometry source: ...

Production B02 SDF:
PASS / FAIL

Free-state hazard:
PASS / FAIL
Free clearance: ... mm

Native contact baseline:
clearance: ... mm
NOTE: baseline only, not feasible-domain result

Feasible solve:
PASS / PARTIAL / FAIL / NOT RUN
Accepted solver clearance: ... mm
Nonlinear iterations: ...
Minimum alpha: ...
Max translation correction: ... mm
Max rotation correction: ... deg

Independent safety check:
PASS / PARTIAL / FAIL / NOT RUN
Minimum clearance: ... mm

Runtime:
Total: ... ms
Top cost: ...

FINAL DECISION:
PASS / PARTIAL / FAIL / INCONCLUSIVE

NEXT STEP:
<exactly one recommended next task>
```

If PASS, the only next task is:
`Integrate the validated adapter/solver as a test-only pre-commit hook in the real SOFA BeamAdapter physics step, then repeat step 776/substep 1 before any multi-step replay.`

If not PASS, recommend fixing only the layer that failed.

## 22. Execution order for Luna XHigh

Follow this order strictly:
1. Read the current PoC and targeted step776 replay.
2. Audit production BeamAdapter and production SDF geometry.
3. Implement the test-only real-beam adapter.
4. Verify adapter output on a known current/native state.
5. Replay deterministic prefix.
6. Capture previous/free/native states at step776/substep1.
7. Verify the free-state hazard with the new adapter.
8. Only if valid, run one offline feasible solve from the free state.
9. Run the independent dense safety check.
10. Write JSON and Markdown results.
11. Stop.

Ordinary Python/path/API errors may be fixed autonomously.

If actual BeamAdapter geometry access is unavailable, do not invent a surrogate. Report INCONCLUSIVE with evidence.

Avoid high-frequency polling and giant logs.

Do not modify production files.
Do not start training.