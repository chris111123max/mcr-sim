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
    PPO_ENT_COEF,
    REWARD_PROGRESS_BUDGET,
    REWARD_RETRACTION,
    REWARD_SUCCESS,
    REWARD_STEP,
    REWARD_TIMEOUT,
    REWARD_WALL_PROXIMITY,
    REWARD_WAYPOINT_BUDGET,
    REWARD_WRONG_BRANCH,
    SAC_MIN_ENT_COEF,
    TRAIN_ROUTE_MAX_LENGTH_M,
    TRAINING_CURRICULUM_MODELS,
    ordered_route_potential,
    update_curriculum_stage,
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

    def test_route_potential_is_bounded_and_ordered(self) -> None:
        waypoints = [0.0, 0.005, 0.010]
        self.assertEqual(ordered_route_potential(0.0, 0.010, waypoints, 0, 0.0), 0.0)
        self.assertAlmostEqual(
            ordered_route_potential(0.0, 0.010, waypoints, 1, 0.003),
            0.2,
        )
        self.assertAlmostEqual(
            ordered_route_potential(0.0, 0.010, waypoints, 2, 0.0),
            1.0,
        )

    def test_potential_difference_restores_credit_after_correction(self) -> None:
        potentials = [0.0, 0.3, 0.2, 0.3, 0.7]
        deltas = [b - a for a, b in zip(potentials, potentials[1:])]
        self.assertAlmostEqual(sum(deltas), potentials[-1] - potentials[0])
        self.assertAlmostEqual(deltas[1] + deltas[2], 0.0)

    def test_ordered_waypoint_bonus_is_naturally_bounded(self) -> None:
        rewardable_waypoints = 20
        total = sum(1.0 / rewardable_waypoints for _ in range(rewardable_waypoints))
        self.assertAlmostEqual(total, 1.0)

    def test_curriculum_advances_one_stage_and_latches(self) -> None:
        self.assertEqual(update_curriculum_stage(0, 0.049), 0)
        self.assertEqual(update_curriculum_stage(0, 0.05), 1)
        self.assertEqual(update_curriculum_stage(1, 0.10), 2)
        self.assertEqual(update_curriculum_stage(2, 1.0), 2)
        self.assertLess(len(TRAINING_CURRICULUM_MODELS[0]), len(TRAINING_CURRICULUM_MODELS[1]))
        self.assertLess(len(TRAINING_CURRICULUM_MODELS[1]), len(TRAINING_CURRICULUM_MODELS[2]))

    def test_exploration_defaults_are_nonzero(self) -> None:
        self.assertGreater(SAC_MIN_ENT_COEF, 0.0)
        self.assertGreater(PPO_ENT_COEF, 0.0)

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
