"""CPU-only source-contract checks for V15.2-A/B."""

from __future__ import annotations

from pathlib import Path
import unittest

from mcr_sim.training_config import (
    ACTOR_CURRENT_GEOMETRY_DIM,
    ACTOR_OBSERVATION_DIM,
    CATHETER_COLLISION_BODY_EDGES,
    CATHETER_COLLISION_TIP_EDGES,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class BodySafetySourceContractTest(unittest.TestCase):
    def test_actor_observation_remains_255_dimensional(self) -> None:
        self.assertEqual(ACTOR_CURRENT_GEOMETRY_DIM, 31)
        self.assertEqual(ACTOR_OBSERVATION_DIM, 255)

    def test_physics_wall_does_not_enter_actor_observation(self) -> None:
        source = (PROJECT_ROOT / "mcr_sim" / "mcr_rl_env.py").read_text(
            encoding="utf-8"
        )
        observation_block = source.split(
            "def _get_observation", 1
        )[1].split("def _get_reward_features", 1)[0]
        self.assertNotIn("sdf_wall_", observation_block)
        self.assertNotIn("sdf_physics_wall", observation_block)

    def test_physics_wall_does_not_enter_reward(self) -> None:
        source = (PROJECT_ROOT / "mcr_sim" / "mcr_rl_env.py").read_text(
            encoding="utf-8"
        )
        reward_block = source.split(
            "def _get_reward_features", 1
        )[1].split("def _get_observation", 1)[0]
        self.assertNotIn("sdf_wall_", reward_block)
        self.assertNotIn("sdf_physics_wall", reward_block)
        self.assertIn('"route_progress"', reward_block)
        self.assertIn('"successful_task"', reward_block)
        self.assertIn('"out_of_vessel_penalty"', reward_block)

    def test_sdf_wall_is_applied_as_force_not_position_projection(self) -> None:
        source = (PROJECT_ROOT / "mcr_sim" / "sdf_physics_wall.py").read_text(
            encoding="utf-8"
        )
        instrument = (PROJECT_ROOT / "mcr_sim" / "mcr_instrument.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("ConstantForceField", instrument)
        self.assertIn("SDFWallCFF", instrument)
        self.assertIn("force_rows[node_indices, :3] = forces", source)
        self.assertNotIn("position +=", source)
        self.assertNotIn("MO.position.value =", source)
        self.assertNotIn("IRC.xtip", source)

    def test_collision_sampling_and_two_sided_vessel_wall(self) -> None:
        self.assertEqual(CATHETER_COLLISION_BODY_EDGES, 80)
        self.assertEqual(CATHETER_COLLISION_TIP_EDGES, 12)
        environment = (
            PROJECT_ROOT / "mcr_sim" / "mcr_environment.py"
        ).read_text(encoding="utf-8")
        self.assertIn("bothSide=True", environment)

    def test_wall_diagnostics_are_diagnostic_only(self) -> None:
        source = (PROJECT_ROOT / "mcr_sim" / "mcr_rl_env.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"sdf_wall_active_nodes"', source)
        self.assertIn('"sdf_wall_max_force_N"', source)
        self.assertIn('"sdf_wall_min_clearance_m"', source)
        self.assertIn("self._terminal_diagnostic_trace.append", source)

    def test_distributed_event_schema_matches_current_width(self) -> None:
        source = (
            PROJECT_ROOT / "mcr_sim" / "rl_core" / "experiment.py"
        ).read_text(encoding="utf-8")
        self.assertIn("events = np.zeros((len(dones), 62)", source)
        self.assertIn('"sdf_wall_min_clearance_m"', source)
        self.assertIn('"sdf_wall_max_force_N"', source)

    def test_terminal_trace_is_diagnostic_only(self) -> None:
        source = (PROJECT_ROOT / "mcr_sim" / "mcr_rl_env.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('info["terminal_diagnostic_trace"]', source)
        observation_block = source.split(
            "def _get_observation", 1
        )[1].split("def _get_reward_features", 1)[0]
        self.assertNotIn("terminal_diagnostic_trace", observation_block)


if __name__ == "__main__":
    unittest.main()
