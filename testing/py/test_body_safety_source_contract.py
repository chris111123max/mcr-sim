"""CPU-only wiring checks for the shared Reward V6 body-safety state."""

from __future__ import annotations

from pathlib import Path
import unittest

from mcr_sim.training_config import (
    ACTOR_CURRENT_GEOMETRY_DIM,
    ACTOR_OBSERVATION_DIM,
    VESSEL_SECTION_FEATURE_DIM,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class BodySafetySourceContractTest(unittest.TestCase):
    def test_all_algorithms_share_the_78_dimensional_environment_state(self):
        source = (PROJECT_ROOT / "mcr_sim" / "mcr_rl_env.py").read_text(
            encoding="utf-8"
        )
        self.assertEqual(VESSEL_SECTION_FEATURE_DIM, 30)
        self.assertEqual(ACTOR_CURRENT_GEOMETRY_DIM, 50)
        self.assertEqual(ACTOR_OBSERVATION_DIM, 78)
        self.assertIn("self.vessel_section_feature_dim = VESSEL_SECTION_FEATURE_DIM", source)
        self.assertIn("body_clearance_feature", source)
        self.assertIn("outside_counter_feature", source)
        self.assertIn("current_sdf_worst_arc_fraction", source)
        self.assertIn("worst_position_local", source)
        self.assertIn("worst_inward_local", source)
        self.assertIn("_get_centerline_lookahead_tangent_features", source)

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
            self.assertIn("args.curriculum_protocol", source)

    def test_training_target_curriculum_preserves_forced_validation_routes(self):
        source = (PROJECT_ROOT / "mcr_sim" / "mcr_rl_env.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("def _apply_curriculum_target_position", source)
        self.assertIn('or bool(getattr(self, "_explicit_force_model", ""))', source)
        self.assertIn("self.current_full_target_position", source)
        self.assertIn("self._apply_curriculum_target_position()", source)


if __name__ == "__main__":
    unittest.main()
