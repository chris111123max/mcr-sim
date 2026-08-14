"""Fast invariants for the shared SAC/PPO reward profile."""

from __future__ import annotations

import unittest

from mcr_sim.training_config import (
    MAX_INSERTION_PER_ACTION_M,
    NO_PROGRESS_CONFIRM_STEPS,
    NO_PROGRESS_GRACE_STEPS,
    REWARD_NO_PROGRESS,
    REWARD_NO_PROGRESS_TERMINAL,
    REWARD_OUT_OF_VESSEL,
    REWARD_PROGRESS_NORMALIZATION_M,
    REWARD_RETRACTION,
    REWARD_STEP,
    REWARD_WALL_PROXIMITY,
    REWARD_WAYPOINT_APPROACH,
)


class RewardProfileTest(unittest.TestCase):
    def test_one_full_safe_insert_is_positive_even_near_wall(self) -> None:
        progress_feature = (
            MAX_INSERTION_PER_ACTION_M / REWARD_PROGRESS_NORMALIZATION_M
        )
        reward = (
            REWARD_WAYPOINT_APPROACH * progress_feature
            + REWARD_WALL_PROXIMITY
            + REWARD_STEP
        )
        self.assertGreater(reward, 0.0)

    def test_stationary_failure_is_costly_but_safer_than_crashing(self) -> None:
        stationary_return = (
            REWARD_STEP * (NO_PROGRESS_GRACE_STEPS + NO_PROGRESS_CONFIRM_STEPS)
            + REWARD_NO_PROGRESS * NO_PROGRESS_CONFIRM_STEPS
            + REWARD_NO_PROGRESS_TERMINAL
        )
        self.assertLess(stationary_return, -10.0)
        self.assertGreater(stationary_return, REWARD_OUT_OF_VESSEL)

    def test_continuous_full_retraction_is_not_a_stationary_escape(self) -> None:
        stationary_return = (
            REWARD_STEP * (NO_PROGRESS_GRACE_STEPS + NO_PROGRESS_CONFIRM_STEPS)
            + REWARD_NO_PROGRESS * NO_PROGRESS_CONFIRM_STEPS
            + REWARD_NO_PROGRESS_TERMINAL
        )
        retraction_return = stationary_return + REWARD_RETRACTION * (
            NO_PROGRESS_GRACE_STEPS + NO_PROGRESS_CONFIRM_STEPS
        )
        self.assertLess(retraction_return, REWARD_OUT_OF_VESSEL)


if __name__ == "__main__":
    unittest.main()
