"""CPU-only wiring checks for the shared Reward V6 body-safety state."""

from __future__ import annotations

from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class BodySafetySourceContractTest(unittest.TestCase):
    def test_all_algorithms_share_the_62_dimensional_environment_state(self):
        source = (PROJECT_ROOT / "mcr_sim" / "mcr_rl_env.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("self.vessel_section_feature_dim = 14", source)
        self.assertIn("self.actor_current_geometry_dim = 34", source)
        self.assertIn("body_clearance_feature", source)
        self.assertIn("outside_counter_feature", source)

    def test_whole_body_risk_drives_existing_wall_reward_features(self):
        source = (PROJECT_ROOT / "mcr_sim" / "mcr_rl_env.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("self.current_sdf_body_warning_feature", source)
        self.assertIn("self.current_sdf_body_outside_depth_feature", source)
        self.assertIn(
            "near_wall_feature = max(\n"
            "                tip_near_wall_feature,\n"
            "                float(self.current_sdf_body_warning_feature)",
            source,
        )
        self.assertIn(
            "penetration_feature = max(\n"
            "                tip_penetration_feature,\n"
            "                float(self.current_sdf_body_outside_depth_feature)",
            source,
        )

    def test_all_training_entrypoints_record_the_observation_schema(self):
        for relative_path in (
            "training/py/train_sac.py",
            "training/py/train_ppo.py",
            "training/py/train_lstm_ppo.py",
        ):
            source = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
            self.assertIn("args.observation_space_shape", source)
            self.assertIn("args.observation_space_dtype", source)


if __name__ == "__main__":
    unittest.main()
