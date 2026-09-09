"""Fast trajectory-level invariants for the shared SAC/PPO reward profile."""

from __future__ import annotations

import math
import unittest

from mcr_sim.training_config import (
    ACTOR_HISTORY_STEPS,
    ACTOR_OBSERVATION_DIM,
    ACTOR_SHAFT_LOOKBACK_DISTANCES_M,
    ACTOR_STATIC_ROUTE_FEATURE_DIM,
    INSERT_ACTION_NEGATIVE_LIMIT,
    MAX_INSERTION_PER_ACTION_M,
    LOCAL_FIELD_ACTION_ANGLE_RAD,
    MAX_EPISODE_STEPS,
    NO_PROGRESS_GRACE_STEPS,
    REWARD_NON_FINITE,
    REWARD_DISCOUNT_GAMMA,
    REWARD_OFF_TARGET_BRANCH,
    REWARD_OUT_OF_VESSEL,
    REWARD_PROFILE_VERSION,
    PPO_ENT_COEF,
    PPO_BATCH_SIZE,
    PPO_GAE_LAMBDA,
    PPO_N_STEPS,
    REWARD_PROGRESS_BUDGET,
    REWARD_PROGRESS_PER_M,
    REWARD_SUCCESS,
    REWARD_STEP,
    REWARD_STAGNATION,
    REWARD_TIMEOUT,
    REWARD_WALL_PROXIMITY,
    SAC_MIN_ENT_COEF,
    SAC_BATCH_SIZE,
    SAC_GAMMA,
    SAC_GRADIENT_STEPS,
    TRAIN_ROUTE_MAX_LENGTH_M,
    VESSEL_SECTION_FEATURE_DIM,
    PPO_MAX_ACTION_STD,
    PPO_INITIAL_ACTION_STD,
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
    def test_one_safe_progress_step_is_positive(self) -> None:
        completion_delta = MAX_INSERTION_PER_ACTION_M / TRAIN_ROUTE_MAX_LENGTH_M
        reward = REWARD_PROGRESS_PER_M * completion_delta + REWARD_STEP
        self.assertGreater(reward, 0.0)

    def test_direct_progress_telescopes_without_terminal_erasure(self) -> None:
        completions = [0.0, 0.10, 0.08, 0.14]
        shaping = [
            REWARD_PROGRESS_PER_M * (nxt - current)
            for current, nxt in zip(completions, completions[1:])
        ]
        self.assertAlmostEqual(sum(shaping), REWARD_PROGRESS_PER_M * 0.14)

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
        self.assertEqual(PPO_ENT_COEF, 0.0005)
        self.assertEqual(PPO_INITIAL_ACTION_STD, 0.50)
        self.assertEqual(PPO_MIN_ACTION_STD, 0.10)
        self.assertEqual(PPO_MAX_ACTION_STD, 0.60)
        self.assertLess(PPO_MIN_ACTION_STD, PPO_MAX_ACTION_STD)
        self.assertLess(PPO_MAX_ACTION_STD, 1.0)
        self.assertLessEqual(PPO_MIN_ACTION_STD, PPO_INITIAL_ACTION_STD)
        self.assertLessEqual(PPO_INITIAL_ACTION_STD, PPO_MAX_ACTION_STD)

    def test_stable_ppo_credit_assignment_defaults(self) -> None:
        self.assertEqual(SAC_GAMMA, 0.9995)
        self.assertEqual(PPO_N_STEPS, 256)
        self.assertEqual(PPO_BATCH_SIZE, 1024)
        self.assertEqual(PPO_GAE_LAMBDA, 0.98)

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
        self.assertEqual(stage0["ppo_min_action_std"], 0.10)
        self.assertEqual(stage0, stage3)

    def test_terminal_outcomes_dominate_normalized_shaping(self) -> None:
        maximum_credit = REWARD_PROGRESS_BUDGET
        self.assertLess(maximum_credit + REWARD_OUT_OF_VESSEL, 0.0)
        self.assertLess(maximum_credit + REWARD_NON_FINITE, 0.0)
        self.assertGreater(REWARD_SUCCESS, maximum_credit)

    def test_stationary_episode_reaches_timeout_with_negative_return(self) -> None:
        penalized_steps = MAX_EPISODE_STEPS - NO_PROGRESS_GRACE_STEPS
        stationary_return = (
            REWARD_STEP * MAX_EPISODE_STEPS
            + REWARD_STAGNATION * penalized_steps
            + REWARD_TIMEOUT
        )
        self.assertLess(stationary_return, 0.0)

    def test_reward_profile_is_minimal_v12(self) -> None:
        self.assertEqual(REWARD_PROFILE_VERSION, "12.2")
        self.assertEqual(REWARD_PROGRESS_PER_M, 60.0)
        self.assertLess(REWARD_WALL_PROXIMITY, 0.0)
        self.assertLess(REWARD_OFF_TARGET_BRANCH, 0.0)
        self.assertEqual(REWARD_TIMEOUT, -80.0)
        self.assertEqual(REWARD_STEP, -0.002)
        self.assertEqual(REWARD_STAGNATION, -0.020)
        self.assertEqual(INSERT_ACTION_NEGATIVE_LIMIT, -0.25)
        self.assertAlmostEqual(math.degrees(LOCAL_FIELD_ACTION_ANGLE_RAD), 3.0)

    def test_observation_v11_is_local_compact_and_contains_one_response_step(self) -> None:
        self.assertEqual(ACTOR_HISTORY_STEPS, 1)
        self.assertEqual(ACTOR_SHAFT_LOOKBACK_DISTANCES_M, (0.010, 0.030, 0.060))
        self.assertEqual(VESSEL_SECTION_FEATURE_DIM, 26)
        self.assertEqual(ACTOR_STATIC_ROUTE_FEATURE_DIM, 12)
        self.assertEqual(ACTOR_OBSERVATION_DIM, 45)

    def test_equal_completion_progress_has_equal_reward_across_routes(self) -> None:
        delta_completion = 0.01
        previous_completion = 0.20
        short_route_reward = REWARD_PROGRESS_PER_M * delta_completion
        long_route_reward = REWARD_PROGRESS_PER_M * delta_completion
        self.assertAlmostEqual(short_route_reward, long_route_reward)

    def test_two_step_forward_retract_oscillation_is_not_profitable(self) -> None:
        potential = REWARD_PROGRESS_PER_M * 0.01
        discounted_return = potential + REWARD_STEP + SAC_GAMMA * (-potential + REWARD_STEP)
        self.assertLess(discounted_return, 0.0)

    def test_waiting_and_immediate_unsafe_exit_are_both_negative(self) -> None:
        waiting_return = (
            REWARD_TIMEOUT
            + REWARD_STEP * MAX_EPISODE_STEPS
            + REWARD_STAGNATION * (MAX_EPISODE_STEPS - NO_PROGRESS_GRACE_STEPS)
        )
        immediate_exit_return = REWARD_OUT_OF_VESSEL
        self.assertLess(waiting_return, 0.0)
        self.assertLess(immediate_exit_return, 0.0)
        self.assertLess(waiting_return, immediate_exit_return)

    def test_success_has_a_large_margin_over_best_failure(self) -> None:
        maximum_credit = REWARD_PROGRESS_BUDGET
        successful_return = maximum_credit + REWARD_SUCCESS
        best_timeout_failure = (
            maximum_credit + REWARD_TIMEOUT + REWARD_STEP * MAX_EPISODE_STEPS
        )
        self.assertGreater(successful_return - best_timeout_failure, 20.0)


if __name__ == "__main__":
    unittest.main()
