"""Static contract checks that do not import SB3, SOFA, or torch_npu."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _keyword_map(call: ast.Call):
    return {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in call.keywords
        if keyword.arg is not None
    }


class LstmPpoSourceContractTest(unittest.TestCase):
    def test_new_algorithm_uses_official_recurrent_ppo_and_mask(self):
        source = _source("mcr_sim/distributed/recurrent_ppo.py")
        self.assertIn("from sb3_contrib import RecurrentPPO", source)
        self.assertIn("class DistributedRecurrentPPO(RecurrentPPO)", source)
        self.assertIn("mask = rollout_data.mask > 1e-8", source)
        self.assertIn("rollout_data.lstm_states", source)
        self.assertIn("rollout_data.episode_starts", source)
        self.assertIn("context.average_gradients(self.policy.parameters())", source)

    def test_formal_common_constructor_parameters_match_mlp_ppo(self):
        mlp_tree = ast.parse(_source("training/py/train_ppo.py"))
        lstm_tree = ast.parse(_source("training/py/train_lstm_ppo.py"))

        mlp_call = next(
            node.value
            for node in ast.walk(mlp_tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "kwargs" for target in node.targets)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "dict"
            and any(keyword.arg == "policy" for keyword in node.value.keywords)
        )
        lstm_call = next(
            node
            for node in ast.walk(lstm_tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "DistributedRecurrentPPO"
            and any(keyword.arg == "policy" for keyword in node.keywords)
        )
        mlp_kwargs = _keyword_map(mlp_call)
        lstm_kwargs = _keyword_map(lstm_call)
        common = (
            "env",
            "learning_rate",
            "n_steps",
            "batch_size",
            "n_epochs",
            "gamma",
            "gae_lambda",
            "clip_range",
            "ent_coef",
            "vf_coef",
            "max_grad_norm",
            "tensorboard_log",
            "seed",
            "device",
            "verbose",
            "distributed_context",
            "min_action_std",
            "max_action_std",
        )
        for name in common:
            self.assertEqual(mlp_kwargs[name], lstm_kwargs[name], name)
        self.assertEqual(mlp_kwargs["policy"], "'MlpPolicy'")
        self.assertEqual(lstm_kwargs["policy"], "'MlpLstmPolicy'")

    def test_only_requested_lstm_structure_is_configured(self):
        source = _source("training/py/train_lstm_ppo.py")
        self.assertIn("LSTM_HIDDEN_SIZE_DEFAULT = 128", source)
        self.assertIn("LSTM_NUM_LAYERS_DEFAULT = 1", source)
        self.assertIn('"bidirectional": False', source)
        self.assertNotIn("GRU", source)
        self.assertNotIn("Transformer", source)

    def test_existing_mlp_distributed_algorithm_was_not_modified(self):
        # Its source remains a separate implementation and contains no
        # recurrent dependency, so installing sb3-contrib is LSTM-only.
        source = _source("mcr_sim/distributed/ppo.py")
        self.assertNotIn("sb3_contrib", source)
        self.assertNotIn("Recurrent", source)


if __name__ == "__main__":
    unittest.main()
