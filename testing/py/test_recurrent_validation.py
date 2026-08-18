"""CPU-only validation-state tests; no SOFA or accelerator is required."""

from __future__ import annotations

import unittest

import numpy as np

from mcr_sim.rl_core.evaluation import evaluate_policy


class _OneStepEnv:
    def reset(self, seed=None):
        return np.zeros((1, 3), dtype=np.float32)

    def step(self, action):
        return (
            np.zeros((1, 3), dtype=np.float32),
            np.array([1.0], dtype=np.float32),
            np.array([True]),
            [{"done_by_target": True, "terminal_reason": "target"}],
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

    def test_plain_mlp_callable_remains_supported(self):
        result = evaluate_policy(
            ["V01"],
            lambda vessel_id: _OneStepEnv(),
            lambda observation: np.zeros((1, 2), dtype=np.float32),
            episodes_per_vessel=2,
            max_episode_steps=1,
        )
        self.assertEqual(result.valid_success_count, 2)


if __name__ == "__main__":
    unittest.main()
