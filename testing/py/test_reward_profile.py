"""Fast trajectory-level invariants for the shared SAC/PPO reward profile."""

from __future__ import annotations

import math
import unittest

from mcr_sim.training_config import (
    MAX_INSERTION_PER_ACTION_M,
    LOCAL_FIELD_ACTION_ANGLE_RAD,
    MAX_EPISODE_STEPS,
    NO_PROGRESS_GRACE_STEPS,
    REWARD_NON_FINITE,
    REWARD_NO_PROGRESS,
    REWARD_NO_PROGRESS_TERMINAL,
    REWARD_OUT_OF_VESSEL,
    REWARD_PROFILE_VERSION,
    PPO_ENT_COEF,
    PPO_BATCH_SIZE,
    PPO_GAE_LAMBDA,
    PPO_N_STEPS,
    REWARD_PROGRESS_BUDGET,
    REWARD_RETRACTION,
    REWARD_SUCCESS,
    REWARD_STEP,
    REWARD_TIMEOUT,
    REWARD_WALL_PROXIMITY,
    REWARD_UNSAFE_CURVE_INSERTION,
    REWARD_WRONG_BRANCH,
    SAC_MIN_ENT_COEF,
    SAC_BATCH_SIZE,
    SAC_GAMMA,
    SAC_GRADIENT_STEPS,
    TRAIN_ROUTE_MAX_LENGTH_M,
    PPO_MAX_ACTION_STD,
    PPO_MIN_ACTION_STD,
    TRAINING_CURRICULUM_MODELS,
    TRAINING_CURRICULUM_ALL_MODELS,
    TRAINING_CURRICULUM_BRANCH_MODELS,
    TRAINING_CURRICULUM_CONSECUTIVE_SUCCESS_EPISODES,
    TRAINING_CURRICULUM_CURVED_MODELS,
    TRAINING_CURRICULUM_DR_FRACTIONS,
    TRAINING_CURRICULUM_MIN_EPISODES_PER_VESSEL,
    TRAINING_CURRICULUM_SIMPLE_MODELS,
    TRAINING_CURRICULUM_STAGE_NAMES,
    TRAINING_CURRICULUM_TARGET_FRACTIONS,
    body_sdf_risk_features,
    curriculum_domain_randomization_profile,
    curriculum_exploration_profile,
    curriculum_sampling_weights,
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

    def test_potential_difference_restores_credit_after_correction(self) -> None:
        potentials = [0.0, 0.3, 0.2, 0.3, 0.7]
        deltas = [b - a for a, b in zip(potentials, potentials[1:])]
        self.assertAlmostEqual(sum(deltas), potentials[-1] - potentials[0])
        self.assertAlmostEqual(deltas[1] + deltas[2], 0.0)

    def test_curriculum_requires_three_consecutive_success_episodes(self) -> None:
        required = TRAINING_CURRICULUM_CONSECUTIVE_SUCCESS_EPISODES
        stage, streak = update_curriculum_progress(0, required - 1, 0.90)
        self.assertEqual((stage, streak), (0, required - 1))
        stage, streak = update_curriculum_progress(0, required, 0.899)
        self.assertEqual((stage, streak), (0, 0))
        stage, streak = update_curriculum_progress(0, required, 0.90)
        self.assertEqual((stage, streak), (1, 0))

    def test_curriculum_advances_only_one_of_five_stages(self) -> None:
        stage, streak = update_curriculum_progress(
            3,
            TRAINING_CURRICULUM_CONSECUTIVE_SUCCESS_EPISODES,
            1.0,
        )
        self.assertEqual((stage, streak), (4, 0))
        stage, streak = update_curriculum_progress(stage, streak, 1.0)
        self.assertEqual((stage, streak), (4, 0))
        self.assertEqual(len(TRAINING_CURRICULUM_MODELS), 5)
        self.assertEqual(
            TRAINING_CURRICULUM_MODELS,
            (
                TRAINING_CURRICULUM_BRANCH_MODELS,
                TRAINING_CURRICULUM_CURVED_MODELS,
                TRAINING_CURRICULUM_SIMPLE_MODELS,
                TRAINING_CURRICULUM_ALL_MODELS,
                TRAINING_CURRICULUM_ALL_MODELS,
            ),
        )
        self.assertEqual(
            TRAINING_CURRICULUM_STAGE_NAMES,
            (
                "branch_fixed",
                "curved_fixed",
                "simple_full_dr",
                "all_fixed",
                "all_full_dr",
            ),
        )
        self.assertEqual(
            TRAINING_CURRICULUM_TARGET_FRACTIONS,
            (1.00, 1.00, 1.00, 1.00, 1.00),
        )
        self.assertEqual(
            TRAINING_CURRICULUM_DR_FRACTIONS,
            (0.0, 0.0, 1.0, 0.0, 1.0),
        )

    def test_first_stage_uses_aggregate_ninety_percent_success(self) -> None:
        rates = {"B01": 1.0, "B02": 0.80}
        counts = {"B01": 100, "B02": 100}
        stage, streak = update_curriculum_progress(
            0,
            TRAINING_CURRICULUM_CONSECUTIVE_SUCCESS_EPISODES,
            0.0,
            rates,
            counts,
        )
        self.assertEqual((stage, streak), (1, 0))

        rates["B02"] = 0.79
        stage, streak = update_curriculum_progress(
            0,
            TRAINING_CURRICULUM_CONSECUTIVE_SUCCESS_EPISODES,
            0.0,
            rates,
            counts,
        )
        self.assertEqual(
            (stage, streak),
            (0, 0),
        )

    def test_curriculum_requires_every_active_vessel_to_meet_threshold(self) -> None:
        active_models = TRAINING_CURRICULUM_MODELS[3]
        rates = {model_id: 0.50 for model_id in active_models}
        rates["C02"] = 0.0
        stage, streak = update_curriculum_progress(
            3,
            TRAINING_CURRICULUM_CONSECUTIVE_SUCCESS_EPISODES,
            0.50,
            rates,
        )
        self.assertEqual((stage, streak), (3, 0))

        rates["C02"] = 0.50
        stage, streak = update_curriculum_progress(
            stage,
            TRAINING_CURRICULUM_CONSECUTIVE_SUCCESS_EPISODES,
            0.50,
            rates,
        )
        self.assertEqual((stage, streak), (4, 0))

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
            TRAINING_CURRICULUM_CONSECUTIVE_SUCCESS_EPISODES,
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
            TRAINING_CURRICULUM_CONSECUTIVE_SUCCESS_EPISODES,
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

    def test_stable_ppo_credit_assignment_defaults(self) -> None:
        self.assertEqual(SAC_GAMMA, 0.995)
        self.assertEqual(PPO_N_STEPS, 256)
        self.assertEqual(PPO_BATCH_SIZE, 1024)
        self.assertEqual(PPO_GAE_LAMBDA, 0.95)

    def test_stable_sac_update_to_data_defaults(self) -> None:
        self.assertEqual(SAC_BATCH_SIZE, 1024)
        self.assertEqual(SAC_GRADIENT_STEPS, 1)

    def test_insertion_step_and_episode_limit_preserve_motion_budget(self) -> None:
        self.assertEqual(MAX_INSERTION_PER_ACTION_M, 0.0004)
        self.assertEqual(MAX_EPISODE_STEPS, 2048)
        self.assertAlmostEqual(
            MAX_INSERTION_PER_ACTION_M * MAX_EPISODE_STEPS,
            0.8192,
        )

    def test_domain_randomization_resets_when_hard_vessels_are_added(self) -> None:
        stage0 = curriculum_domain_randomization_profile(0)
        stage1 = curriculum_domain_randomization_profile(1)
        stage2 = curriculum_domain_randomization_profile(2)
        stage3 = curriculum_domain_randomization_profile(3)
        self.assertAlmostEqual(stage0["fraction"], 0.0)
        self.assertAlmostEqual(stage0["vessel_scale_min"], 1.0)
        self.assertAlmostEqual(stage0["start_window_distance_m"], 0.0)
        self.assertAlmostEqual(stage0["initial_orientation_max_angle_deg"], 0.0)
        stage4 = curriculum_domain_randomization_profile(4)
        self.assertAlmostEqual(stage1["fraction"], 0.0)
        self.assertAlmostEqual(stage2["fraction"], 1.0)
        self.assertAlmostEqual(stage3["fraction"], 0.0)
        self.assertAlmostEqual(stage4["fraction"], 1.0)

    def test_adaptive_sampling_favors_hard_vessels_without_starvation(self) -> None:
        weights = curriculum_sampling_weights(
            ("B01", "B02"),
            {"B01": 0.80, "B02": 0.10},
        )
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertGreater(weights["B02"], weights["B01"])
        self.assertGreater(weights["B01"], 0.0)

        capped = curriculum_sampling_weights(
            ("B01", "B02", "B03", "B04", "B05"),
            {"B01": 0.0, "B02": 1.0, "B03": 1.0, "B04": 1.0, "B05": 1.0},
        )
        self.assertLessEqual(max(capped.values()), 0.40 + 1e-9)

    def test_exploration_floors_use_stable_values(self) -> None:
        stage0 = curriculum_exploration_profile(0)
        stage3 = curriculum_exploration_profile(4)
        self.assertEqual(stage0["ppo_min_action_std"], 0.25)
        self.assertEqual(stage0, stage3)

    def test_every_terminal_failure_is_negative_after_maximum_credit(self) -> None:
        maximum_credit = REWARD_PROGRESS_BUDGET
        self.assertLess(maximum_credit + REWARD_OUT_OF_VESSEL, 0.0)
        self.assertLess(maximum_credit + REWARD_NON_FINITE, 0.0)
        self.assertLess(
            maximum_credit + REWARD_TIMEOUT + REWARD_STEP * MAX_EPISODE_STEPS,
            0.0,
        )

    def test_stationary_episode_reaches_timeout_instead_of_no_progress_terminal(self) -> None:
        stationary_return = (
            REWARD_STEP * MAX_EPISODE_STEPS
            + REWARD_NO_PROGRESS * (MAX_EPISODE_STEPS - NO_PROGRESS_GRACE_STEPS)
            + REWARD_TIMEOUT
        )
        self.assertEqual(REWARD_NO_PROGRESS_TERMINAL, 0.0)
        self.assertLess(stationary_return, -10.0)
        self.assertGreater(stationary_return, REWARD_OUT_OF_VESSEL)

    def test_continuous_full_retraction_is_penalized_without_being_worse_than_crash(self) -> None:
        stationary_return = (
            REWARD_STEP * MAX_EPISODE_STEPS
            + REWARD_NO_PROGRESS * (MAX_EPISODE_STEPS - NO_PROGRESS_GRACE_STEPS)
            + REWARD_TIMEOUT
        )
        retraction_return = stationary_return + REWARD_RETRACTION * MAX_EPISODE_STEPS
        self.assertLess(retraction_return, stationary_return)
        self.assertLess(retraction_return, 0.0)
        self.assertGreater(retraction_return, REWARD_OUT_OF_VESSEL)

    def test_reward_profile_is_v9(self) -> None:
        self.assertEqual(REWARD_PROFILE_VERSION, 9)
        self.assertLess(REWARD_UNSAFE_CURVE_INSERTION, 0.0)
        self.assertAlmostEqual(math.degrees(LOCAL_FIELD_ACTION_ANGLE_RAD), 3.0)

    def test_wrong_branch_is_recoverable_dense_cost(self) -> None:
        self.assertGreater(REWARD_WRONG_BRANCH, -10.0)
        self.assertLess(REWARD_WRONG_BRANCH, 0.0)

    def test_more_progress_can_beat_waiting_for_timeout(self) -> None:
        waiting_return = (
            0.23 * REWARD_PROGRESS_BUDGET
            + REWARD_TIMEOUT
            + REWARD_STEP * MAX_EPISODE_STEPS
        )
        later_exit_return = (
            0.30 * REWARD_PROGRESS_BUDGET
            + REWARD_OUT_OF_VESSEL
            + REWARD_STEP * MAX_EPISODE_STEPS
        )
        self.assertGreater(later_exit_return, waiting_return)

    def test_success_has_a_large_margin_over_best_failure(self) -> None:
        maximum_credit = REWARD_PROGRESS_BUDGET
        successful_return = maximum_credit + REWARD_SUCCESS
        best_timeout_failure = (
            maximum_credit + REWARD_TIMEOUT + REWARD_STEP * MAX_EPISODE_STEPS
        )
        self.assertGreater(successful_return - best_timeout_failure, 400.0)


if __name__ == "__main__":
    unittest.main()
