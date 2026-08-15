"""Fast trajectory-level invariants for the shared SAC/PPO reward profile."""

from __future__ import annotations

import unittest

from mcr_sim.training_config import (
    MAX_INSERTION_PER_ACTION_M,
    MAX_EPISODE_STEPS,
    NO_PROGRESS_CONFIRM_STEPS,
    NO_PROGRESS_GRACE_STEPS,
    REWARD_NON_FINITE,
    REWARD_NO_PROGRESS,
    REWARD_NO_PROGRESS_TERMINAL,
    REWARD_OUT_OF_VESSEL,
    REWARD_PROGRESS_BUDGET,
    REWARD_RETRACTION,
    REWARD_SUCCESS,
    REWARD_STEP,
    REWARD_TIMEOUT,
    REWARD_WALL_PROXIMITY,
    REWARD_WAYPOINT_BUDGET,
    REWARD_WRONG_BRANCH,
    TRAIN_ROUTE_MAX_LENGTH_M,
    bounded_progress_feature,
    bounded_waypoint_feature,
)


class RewardProfileTest(unittest.TestCase):
    def test_one_full_safe_insert_is_positive_on_longest_route(self) -> None:
        reward = (
            REWARD_PROGRESS_BUDGET
            * MAX_INSERTION_PER_ACTION_M
            / TRAIN_ROUTE_MAX_LENGTH_M
            + REWARD_WALL_PROXIMITY
            + REWARD_STEP
        )
        self.assertGreater(reward, 0.0)

    def test_positive_progress_and_waypoint_credit_are_hard_capped(self) -> None:
        progress_fraction = 0.0
        progress_total = 0.0
        for _ in range(10_000):
            feature, progress_fraction = bounded_progress_feature(0.001, progress_fraction)
            progress_total += feature
        self.assertAlmostEqual(progress_fraction, 1.0)
        self.assertAlmostEqual(progress_total, 1.0)

        waypoint_fraction = 0.0
        waypoint_total = 0.0
        for _ in range(100):
            feature, waypoint_fraction = bounded_waypoint_feature(20, waypoint_fraction)
            waypoint_total += feature
        self.assertAlmostEqual(waypoint_fraction, 1.0)
        self.assertAlmostEqual(waypoint_total, 1.0)

    def test_regression_does_not_restore_positive_credit(self) -> None:
        fraction = 0.0
        total = 0.0
        for _ in range(20):
            positive, fraction = bounded_progress_feature(0.1, fraction)
            negative, fraction = bounded_progress_feature(-0.1, fraction)
            total += positive + negative
        self.assertAlmostEqual(fraction, 1.0)
        self.assertLess(total, 0.0)

    def test_every_terminal_failure_is_negative_after_maximum_credit(self) -> None:
        maximum_credit = REWARD_PROGRESS_BUDGET + REWARD_WAYPOINT_BUDGET
        self.assertLess(maximum_credit + REWARD_OUT_OF_VESSEL, 0.0)
        self.assertLess(maximum_credit + REWARD_WRONG_BRANCH, 0.0)
        self.assertLess(maximum_credit + REWARD_NON_FINITE, 0.0)
        self.assertLess(
            maximum_credit + REWARD_TIMEOUT + REWARD_STEP * MAX_EPISODE_STEPS,
            0.0,
        )

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

    def test_success_has_a_large_margin_over_best_failure(self) -> None:
        maximum_credit = REWARD_PROGRESS_BUDGET + REWARD_WAYPOINT_BUDGET
        successful_return = maximum_credit + REWARD_SUCCESS
        best_timeout_failure = (
            maximum_credit + REWARD_TIMEOUT + REWARD_STEP * MAX_EPISODE_STEPS
        )
        self.assertGreater(successful_return - best_timeout_failure, 200.0)


if __name__ == "__main__":
    unittest.main()
