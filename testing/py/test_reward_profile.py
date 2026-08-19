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
    REWARD_PROFILE_VERSION,
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
    PPO_MAX_ACTION_STD,
    PPO_MIN_ACTION_STD,
    TRAINING_CURRICULUM_MODELS,
    TRAINING_CURRICULUM_CONSECUTIVE_EPOCHS,
    TRAINING_CURRICULUM_MIN_EPISODES_PER_VESSEL,
    body_sdf_risk_features,
    ordered_route_potential,
    update_curriculum_progress,
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

    def test_curriculum_requires_five_consecutive_successful_epochs(self) -> None:
        stage, streak = update_curriculum_progress(0, 0, 0.20)
        self.assertEqual((stage, streak), (0, 1))
        stage, streak = update_curriculum_progress(stage, streak, 0.20)
        self.assertEqual((stage, streak), (0, 2))
        stage, streak = update_curriculum_progress(stage, streak, 0.199)
        self.assertEqual((stage, streak), (0, 0))
        for _ in range(TRAINING_CURRICULUM_CONSECUTIVE_EPOCHS):
            stage, streak = update_curriculum_progress(stage, streak, 0.20)
        self.assertEqual((stage, streak), (1, 0))

    def test_curriculum_advances_only_one_of_four_stages(self) -> None:
        stage, streak = update_curriculum_progress(
            2,
            TRAINING_CURRICULUM_CONSECUTIVE_EPOCHS - 1,
            1.0,
        )
        self.assertEqual((stage, streak), (3, 0))
        stage, streak = update_curriculum_progress(stage, streak, 1.0)
        self.assertEqual((stage, streak), (3, 0))
        self.assertLess(len(TRAINING_CURRICULUM_MODELS[0]), len(TRAINING_CURRICULUM_MODELS[1]))
        self.assertLess(len(TRAINING_CURRICULUM_MODELS[1]), len(TRAINING_CURRICULUM_MODELS[2]))
        self.assertLess(len(TRAINING_CURRICULUM_MODELS[2]), len(TRAINING_CURRICULUM_MODELS[3]))

    def test_curriculum_requires_every_active_vessel_to_meet_threshold(self) -> None:
        active_models = TRAINING_CURRICULUM_MODELS[2]
        rates = {model_id: 0.20 for model_id in active_models}
        rates["C02"] = 0.0
        stage, streak = update_curriculum_progress(2, 2, 0.50, rates)
        self.assertEqual((stage, streak), (2, 0))

        rates["C02"] = 0.20
        for _ in range(TRAINING_CURRICULUM_CONSECUTIVE_EPOCHS):
            stage, streak = update_curriculum_progress(stage, streak, 0.50, rates)
        self.assertEqual((stage, streak), (3, 0))

    def test_missing_active_vessel_blocks_curriculum_advance(self) -> None:
        rates = {"B01": 1.0}
        stage, streak = update_curriculum_progress(0, 2, 1.0, rates)
        self.assertEqual((stage, streak), (0, 0))

    def test_curriculum_requires_enough_rolling_samples_per_vessel(self) -> None:
        active_models = TRAINING_CURRICULUM_MODELS[0]
        rates = {model_id: 1.0 for model_id in active_models}
        too_few = {
            model_id: TRAINING_CURRICULUM_MIN_EPISODES_PER_VESSEL - 1
            for model_id in active_models
        }
        stage, streak = update_curriculum_progress(
            0,
            TRAINING_CURRICULUM_CONSECUTIVE_EPOCHS - 1,
            1.0,
            rates,
            too_few,
        )
        self.assertEqual((stage, streak), (0, 0))

        enough = {
            model_id: TRAINING_CURRICULUM_MIN_EPISODES_PER_VESSEL
            for model_id in active_models
        }
        stage, streak = update_curriculum_progress(
            0,
            TRAINING_CURRICULUM_CONSECUTIVE_EPOCHS - 1,
            1.0,
            rates,
            enough,
        )
        self.assertEqual((stage, streak), (1, 0))

    def test_body_sdf_risk_warns_before_confirmed_outside(self) -> None:
        safe = body_sdf_risk_features(-0.0005)
        wall = body_sdf_risk_features(0.0)
        terminal = body_sdf_risk_features(0.0005)
        self.assertEqual(safe, (0.0, 0.0))
        self.assertEqual(wall, (0.5, 0.0))
        self.assertEqual(terminal, (1.0, 1.0))

    def test_exploration_defaults_are_nonzero(self) -> None:
        self.assertGreater(SAC_MIN_ENT_COEF, 0.0)
        self.assertGreater(PPO_ENT_COEF, 0.0)
        self.assertLess(PPO_MIN_ACTION_STD, PPO_MAX_ACTION_STD)
        self.assertLessEqual(PPO_MAX_ACTION_STD, 1.0)

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

    def test_continuous_full_retraction_is_penalized_without_being_worse_than_crash(self) -> None:
        stationary_return = (
            REWARD_STEP * (NO_PROGRESS_GRACE_STEPS + NO_PROGRESS_CONFIRM_STEPS)
            + REWARD_NO_PROGRESS * NO_PROGRESS_CONFIRM_STEPS
            + REWARD_NO_PROGRESS_TERMINAL
        )
        retraction_return = stationary_return + REWARD_RETRACTION * (
            NO_PROGRESS_GRACE_STEPS + NO_PROGRESS_CONFIRM_STEPS
        )
        self.assertLess(retraction_return, stationary_return)
        self.assertLess(retraction_return, 0.0)
        self.assertGreater(retraction_return, REWARD_OUT_OF_VESSEL)

    def test_reward_profile_is_v6(self) -> None:
        self.assertEqual(REWARD_PROFILE_VERSION, 6)

    def test_success_has_a_large_margin_over_best_failure(self) -> None:
        maximum_credit = REWARD_PROGRESS_BUDGET + REWARD_WAYPOINT_BUDGET
        successful_return = maximum_credit + REWARD_SUCCESS
        best_timeout_failure = (
            maximum_credit + REWARD_TIMEOUT + REWARD_STEP * MAX_EPISODE_STEPS
        )
        self.assertGreater(successful_return - best_timeout_failure, 200.0)


if __name__ == "__main__":
    unittest.main()
