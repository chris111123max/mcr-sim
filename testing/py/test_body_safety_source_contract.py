"""CPU-only wiring checks for the shared Reward/Observation V11 state."""

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
    def test_all_algorithms_share_the_45_dimensional_environment_state(self):
        source = (PROJECT_ROOT / "mcr_sim" / "mcr_rl_env.py").read_text(
            encoding="utf-8"
        )
        self.assertEqual(VESSEL_SECTION_FEATURE_DIM, 26)
        self.assertEqual(ACTOR_CURRENT_GEOMETRY_DIM, 38)
        self.assertEqual(ACTOR_OBSERVATION_DIM, 45)
        self.assertIn("self.vessel_section_feature_dim = VESSEL_SECTION_FEATURE_DIM", source)
        self.assertIn("body_clearance_feature", source)
        self.assertIn("current_sdf_worst_arc_fraction", source)
        self.assertIn("worst_inward_local", source)
        self.assertIn("shaft_landmarks_local", source)
        self.assertIn("_get_centerline_lookahead_tangent_features", source)
        actor_builder = source.split(
            "def _build_actor_current_geometry_observation", 1
        )[1].split("def _build_actor_dynamic_step_observation", 1)[0]
        self.assertIn("remaining_route_distance_norm", actor_builder)
        self.assertIn("time_remaining_norm", actor_builder)
        self.assertIn("inserted_length_norm", actor_builder)
        self.assertNotIn("bend_severity_features", actor_builder)
        self.assertNotIn("tip_forward_local", actor_builder)
        self.assertNotIn("route_progress_ratio", actor_builder)

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
        self.assertNotIn('"wall_penetration_penalty"', source)

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

    def test_vector_env_episodes_latch_their_curriculum_stage(self):
        env_source = (PROJECT_ROOT / "mcr_sim" / "mcr_rl_env.py").read_text(
            encoding="utf-8"
        )
        experiment_source = (
            PROJECT_ROOT / "mcr_sim" / "rl_core" / "experiment.py"
        ).read_text(encoding="utf-8")
        self.assertIn("self.episode_curriculum_stage = int(self.curriculum_stage)", env_source)
        self.assertIn('info.get("curriculum_stage", self.curriculum_stage)', experiment_source)
        self.assertIn("episode_curriculum_stage == self.curriculum_stage", experiment_source)
        self.assertIn("self.curriculum_success_streak += 1", experiment_source)
        self.assertIn("self.curriculum_success_streak = 0", experiment_source)

    def test_reward_v11_uses_only_ordinary_terminal_failures(self):
        source = (PROJECT_ROOT / "mcr_sim" / "mcr_rl_env.py").read_text(
            encoding="utf-8"
        )
        done_block = source.split("def _get_done", 1)[1].split("def _get_info", 1)[0]
        self.assertIn('getattr(self, "out_of_vessel_failure", False)', done_block)
        self.assertIn('getattr(self, "non_finite_failure", False)', done_block)
        self.assertNotIn("wrong_branch_failure", done_block)
        self.assertNotIn("no_progress_failure", done_block)
        self.assertIn("self.no_progress_failure = False", source)
        self.assertIn('"done_by_wrong_branch": False', source)
        self.assertIn('"done_by_no_progress": False', source)
        self.assertNotIn('"unsafe_curve_insertion_penalty"', source)
        self.assertNotIn('"no_progress_penalty"', source)
        self.assertIn('"step_penalty"', source)
        self.assertIn("def _terminalize_route_progress_shaping", source)
        timeout_block = source.split("if truncated:", 1)[1].split(
            "info = self._get_info", 1
        )[0]
        self.assertIn("_terminalize_route_progress_shaping", timeout_block)
        self.assertIn("_get_curve_control_features", source)


if __name__ == "__main__":
    unittest.main()
