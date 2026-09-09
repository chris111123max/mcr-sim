"""CPU-only validation-state tests; no SOFA or accelerator is required."""

from __future__ import annotations

import unittest

import numpy as np

from mcr_sim.rl_core.evaluation import (
    ValidationEpisodeResult,
    evaluate_policy,
    evaluate_vector_policy,
    summarize_validation,
    validation_selection_key,
)


class _OneStepEnv:
    def reset(self, seed=None):
        return np.zeros((1, 3), dtype=np.float32)

    def step(self, action):
        return (
            np.zeros((1, 3), dtype=np.float32),
            np.array([1.0], dtype=np.float32),
            np.array([True]),
            [
                {
                    "done_by_target": True,
                    "terminal_reason": "target",
                    "final_dist_to_goal": 0.002,
                    "min_dist_to_goal": 0.0015,
                    "route_progress_ratio": 1.0,
                    "route_potential": 1.0,
                }
            ],
        )

    def close(self):
        return None


class _StatefulPredictor:
    def __init__(self):
        self.reset_count = 0
        self.steps_since_reset = 99

    def reset(self):
        self.reset_count += 1
        self.steps_since_reset = 0

    def __call__(self, observation):
        if self.steps_since_reset != 0:
            raise AssertionError("recurrent state leaked across validation episodes")
        self.steps_since_reset += 1
        return np.zeros((1, 2), dtype=np.float32)


class _OneStepVectorEnv:
    def __init__(self, n_envs):
        self.n_envs = int(n_envs)
        self.base_seed = 0

    def seed(self, seed):
        self.base_seed = int(seed)

    def reset(self):
        return np.zeros((self.n_envs, 3), dtype=np.float32)

    def step(self, action):
        infos = [
            {
                "done_by_target": True,
                "terminal_reason": "target",
                "final_dist_to_goal": 0.002,
                "min_dist_to_goal": 0.0015,
                "route_progress_ratio": 1.0,
                "route_potential": 1.0,
                "route_projection_segment": 7,
                "centerline_safety_ratio_max_episode": 0.8,
                "centerline_safety_margin_min_episode": 0.001,
                "sdf_body_surface_clearance_min_episode": 0.0005,
                "terminal_diagnostic_trace": [
                    {
                        "step": 1,
                        "route_completion": 1.0,
                        "rot_n": 0.25,
                    }
                ],
            }
            for _ in range(self.n_envs)
        ]
        return (
            np.zeros((self.n_envs, 3), dtype=np.float32),
            np.ones(self.n_envs, dtype=np.float32),
            np.ones(self.n_envs, dtype=np.bool_),
            infos,
        )

    def close(self):
        return None


class RecurrentValidationTest(unittest.TestCase):
    def test_action_state_resets_once_per_validation_episode(self):
        predictor = _StatefulPredictor()
        result = evaluate_policy(
            ["V01"],
            lambda vessel_id: _OneStepEnv(),
            predictor,
            episodes_per_vessel=3,
            max_episode_steps=1,
        )
        self.assertEqual(predictor.reset_count, 3)
        self.assertEqual(result.valid_episodes, 3)
        self.assertEqual(result.valid_success_count, 3)
        self.assertEqual(result.valid_route_completion_mean, 1.0)
        self.assertEqual(result.valid_route_potential_mean, 1.0)
        self.assertAlmostEqual(result.valid_final_distance_mm_mean, 2.0)
        self.assertAlmostEqual(result.valid_min_distance_mm_mean, 1.5)

    def test_plain_mlp_callable_remains_supported(self):
        result = evaluate_policy(
            ["V01"],
            lambda vessel_id: _OneStepEnv(),
            lambda observation: np.zeros((1, 2), dtype=np.float32),
            episodes_per_vessel=2,
            max_episode_steps=1,
        )
        self.assertEqual(result.valid_success_count, 2)

    def test_vector_evaluation_batches_without_losing_episodes(self):
        batch_sizes = []

        def factory(vessel_id, n_envs):
            batch_sizes.append(int(n_envs))
            return _OneStepVectorEnv(n_envs)

        result = evaluate_vector_policy(
            ["B01"],
            factory,
            lambda observation: np.zeros((len(observation), 3), dtype=np.float32),
            episodes_per_vessel=5,
            max_parallel_envs=2,
            max_episode_steps=1,
            base_seed=123,
        )
        self.assertEqual(batch_sizes, [2, 2, 1])
        self.assertEqual(result.valid_episodes, 5)
        self.assertEqual(result.valid_success_count, 5)
        self.assertEqual([item.seed for item in result.episodes], [123, 124, 125, 126, 127])
        first = result.episodes[0]
        self.assertEqual(first.diagnostics["route_projection_segment"], 7)
        self.assertAlmostEqual(
            first.diagnostics["centerline_safety_margin_min_episode"], 0.001
        )
        self.assertAlmostEqual(
            first.diagnostics["sdf_body_clearance_min_episode_mm"], 0.5
        )
        self.assertEqual(first.diagnostics["terminal_trace"][0]["step"], 1)
        self.assertAlmostEqual(first.diagnostics["model_action_rot_n_abs_mean"], 0.0)

    def test_best_selection_uses_progress_only_to_break_success_ties(self):
        weak_failure = summarize_validation(
            [
                ValidationEpisodeResult(
                    "V01",
                    0,
                    1,
                    False,
                    "out_of_vessel",
                    10,
                    -140.0,
                    route_completion=0.1,
                    route_potential=0.2,
                    final_distance_mm=100.0,
                    min_distance_mm=90.0,
                )
            ]
        )
        better_failure = summarize_validation(
            [
                ValidationEpisodeResult(
                    "V01",
                    0,
                    1,
                    False,
                    "out_of_vessel",
                    20,
                    -120.0,
                    route_completion=0.4,
                    route_potential=0.5,
                    final_distance_mm=60.0,
                    min_distance_mm=50.0,
                )
            ]
        )
        one_success = summarize_validation(
            [
                ValidationEpisodeResult(
                    "V01",
                    0,
                    1,
                    True,
                    "target",
                    30,
                    200.0,
                    route_completion=0.0,
                    route_potential=0.0,
                    final_distance_mm=3.0,
                    min_distance_mm=2.0,
                )
            ]
        )
        self.assertGreater(
            validation_selection_key(better_failure),
            validation_selection_key(weak_failure),
        )
        self.assertGreater(
            validation_selection_key(one_success),
            validation_selection_key(better_failure),
        )


if __name__ == "__main__":
    unittest.main()
