# Fast FD V2 test-only acceptance

Scope: `testing/beam_feasible_domain/` only. V1 and production files remain unchanged.

V2 retains 10 µm dense Beam/SDF geometry, +0.100 mm margin, the same local-point FD perturbations, row selection, Beam unilateral constraint, and GenericConstraintSolver. It omits the diagnostic q_prev dense measure and passes the one external q_free dense profile into the builder. Independent 10 µm committed Beam/SDF validation remains enabled on every substep.

Run stages in order and stop on the first scientific/safety failure:

1. `tests/diagnose_step665_fast_local_fd_v2.py`: same protected step665 state; V1 and V2 rows built from the same q_prev/q_free; only V2 rows sent to Generic; compare archived V1 committed clearance.
2. `tests/b02_full_episode_beam_unilateral_fast_fd_v2.py`: B02 target_04 seed15204 epoch74 deterministic, 2048 RL steps, 2 x 5 ms, fail-fast at >0.001 mm committed penetration or row/telemetry/solver failure.
3. `tests/profile_fast_fd_v2_32env_components.py`: 32 spawn workers, NPU policy, 700 vector steps including 100 warm-up, same component timing boundary as V1 profiler.

All outputs belong under `_runtime/beam_unilateral_fast_fd_v2/` and are not committed. No training, production integration, or optimizer updates.
