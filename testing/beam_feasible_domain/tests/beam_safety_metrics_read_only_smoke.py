"""Test-only smoke for read-only Beam safety metric aggregation."""
from __future__ import annotations

import copy

from mcr_sim.rl_core.beam_safety_metrics import BeamSafetyMetricsCollector


class FakeLogger:
    def __init__(self):
        self.values = {}

    def record(self, key, value, **kwargs):
        self.values[str(key)] = value


def _snapshot(
    physics,
    safe,
    active,
    near,
    rows,
    penetration,
    audits,
    skips,
    worst_free,
    worst_committed,
):
    return {
        "beam_safety": {
            "physics_substeps_seen": physics,
            "safe_free_substeps": safe,
            "active_substeps": active,
            "near_wall_substeps": near,
            "unilateral_rows_built": rows,
            "q_free_penetration_substeps": penetration,
            "committed_audits": audits,
            "committed_audit_skips": skips,
            "worst_free_clearance_m": worst_free,
            "worst_audited_committed_clearance_m": worst_committed,
        }
    }


def main():
    collector = BeamSafetyMetricsCollector(episode_window=8)

    first = [_snapshot(2, 2, 0, 0, 0, 0, 1, 2, 0.0010, 0.0005)]
    first_before = copy.deepcopy(first)
    collector.observe(first, [False])
    assert first == first_before, "collector mutated environment info"

    terminal = [_snapshot(4, 3, 1, 1, 5, 1, 3, 2, -0.0001, 0.0002)]
    terminal_before = copy.deepcopy(terminal)
    collector.observe(terminal, [True])
    assert terminal == terminal_before, "collector mutated terminal info"

    fresh = [_snapshot(2, 2, 0, 0, 0, 0, 1, 2, 0.0012, 0.0006)]
    fresh_before = copy.deepcopy(fresh)
    collector.observe(fresh, [False])
    assert fresh == fresh_before, "collector mutated fresh-episode info"

    snap = collector.snapshot()
    total = snap["total"]
    assert total["physics_substeps_seen"] == 6.0
    assert total["safe_free_substeps"] == 5.0
    assert total["active_substeps"] == 1.0
    assert total["near_wall_substeps"] == 1.0
    assert total["unilateral_rows_built"] == 5.0
    assert total["q_free_penetration_substeps"] == 1.0
    assert total["committed_audits"] == 4.0
    assert total["committed_audit_skips"] == 4.0
    assert snap["episode_count"] == 1

    logger = FakeLogger()
    collector.log_rollout(logger)

    assert logger.values["beam_safety/physics_substeps_seen_rollout"] == 6.0
    assert logger.values["beam_safety/active_substeps_rollout"] == 1.0
    assert logger.values["beam_safety/unilateral_rows_built_rollout"] == 5.0
    assert (
        logger.values[
            "beam_safety/committed_verification_fraction_rollout"
        ]
        == 0.5
    )
    assert abs(
        logger.values["beam_safety/worst_free_clearance_mm_rollout"] + 0.1
    ) < 1.0e-12
    assert abs(
        logger.values[
            "beam_safety/worst_audited_committed_clearance_mm_rollout"
        ]
        - 0.2
    ) < 1.0e-12

    print("BEAM_SAFETY_METRICS_READ_ONLY=PASS")


if __name__ == "__main__":
    main()
