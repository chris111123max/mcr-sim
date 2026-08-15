"""Fast invariants for the shared SAC/PPO validation unlock gate."""

from __future__ import annotations

import unittest

from mcr_sim.training_config import update_validation_unlocked


class ValidationGateTest(unittest.TestCase):
    def test_gate_stays_locked_below_twenty_percent(self) -> None:
        self.assertFalse(update_validation_unlocked(False, 0.199999, 0.20))

    def test_gate_unlocks_at_twenty_percent(self) -> None:
        self.assertTrue(update_validation_unlocked(False, 0.20, 0.20))

    def test_gate_stays_unlocked_after_later_regression(self) -> None:
        self.assertTrue(update_validation_unlocked(True, 0.0, 0.20))

    def test_zero_threshold_disables_gate(self) -> None:
        self.assertTrue(update_validation_unlocked(False, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
