"""Fast invariants for the current V15.2-A reward/navigation profile."""

from __future__ import annotations

import math
import unittest

from mcr_sim.training_config import (
    ACTOR_CURRENT_GEOMETRY_DIM,
    ACTOR_DYNAMIC_STEP_DIM,
    ACTOR_HISTORY_STEPS,
    ACTOR_OBSERVATION_DIM,
    ACTOR_SHAFT_LOOKBACK_DISTANCES_M,
    CATHETER_COLLISION_BODY_EDGES,
    CATHETER_COLLISION_TIP_EDGES,
    INSERT_ACTION_NEGATIVE_LIMIT,
    LOCAL_FIELD_ACTION_ANGLE_RAD,
    MAX_EPISODE_STEPS,
    MAX_INSERTION_PER_ACTION_M,
    PPO_BATCH_SIZE,
    PPO_ENT_COEF,
    PPO_GAE_LAMBDA,
    PPO_INITIAL_ACTION_STD,
    PPO_MAX_ACTION_STD,
    PPO_MIN_ACTION_STD,
    PPO_N_ENVS,
    PPO_N_STEPS,
    REWARD_NON_FINITE,
    REWARD_OFF_TARGET_BRANCH,
    REWARD_OUT_OF_VESSEL,
    REWARD_PROFILE_VERSION,
    REWARD_PROGRESS_SCALE,
    REWARD_STEP,
    REWARD_STAGNATION,
    REWARD_SUCCESS,
    REWARD_TIMEOUT,
    REWARD_WALL_PROXIMITY,
    SAC_BATCH_SIZE,
    SAC_GRADIENT_STEPS,
    SAC_GAMMA,
    SDF_PHYSICS_WALL_ENABLED,
    SDF_WALL_ACTIVATION_CLEARANCE_M,
    SDF_WALL_MAX_FORCE_N,
    SDF_WALL_STIFFNESS_N_PER_M,
    TRAINING_CURRICULUM_BRANCH_MODELS,
    TRAINING_CURRICULUM_DR_FRACTIONS,
    TRAINING_CURRICULUM_MODELS,
    TRAINING_CURRICULUM_ROUTES,
    TRAINING_CURRICULUM_STAGE_NAMES,
    TRAINING_CURRICULUM_TARGET_FRACTIONS,
    map_insert_action,
)


class RewardProfileTest(unittest.TestCase):
    def test_v15_2a_reward_profile(self) -> None:
        self.assertEqual(
            REWARD_PROFILE_VERSION,
            "15.2A-discrete-forward-relaxed-waypoints",
        )
        self.assertEqual(REWARD_PROGRESS_SCALE, 1000.0)
        self.assertEqual(REWARD_SUCCESS, 300.0)
        self.assertEqual(REWARD_OUT_OF_VESSEL, -30.0)
        self.assertEqual(REWARD_NON_FINITE, -30.0)
        self.assertEqual(REWARD_TIMEOUT, -10.0)
        self.assertEqual(REWARD_STEP, -0.0005)
        self.assertEqual(REWARD_WALL_PROXIMITY, 0.0)
        self.assertEqual(REWARD_OFF_TARGET_BRANCH, 0.0)
        self.assertEqual(REWARD_STAGNATION, 0.0)

    def test_actor_state_dimension_is_unchanged_by_physics_wall(self) -> None:
        self.assertEqual(ACTOR_CURRENT_GEOMETRY_DIM, 31)
        self.assertEqual(ACTOR_DYNAMIC_STEP_DIM, 7)
        self.assertEqual(ACTOR_HISTORY_STEPS, 32)
        self.assertEqual(ACTOR_OBSERVATION_DIM, 255)
        self.assertEqual(
            ACTOR_CURRENT_GEOMETRY_DIM
            + ACTOR_HISTORY_STEPS * ACTOR_DYNAMIC_STEP_DIM,
            ACTOR_OBSERVATION_DIM,
        )
        self.assertEqual(
            ACTOR_SHAFT_LOOKBACK_DISTANCES_M,
            (0.010, 0.030, 0.060),
        )

    def test_action_and_motion_budget(self) -> None:
        self.assertEqual(MAX_INSERTION_PER_ACTION_M, 0.0008)
        self.assertEqual(MAX_EPISODE_STEPS, 2048)
        self.assertEqual(INSERT_ACTION_NEGATIVE_LIMIT, -1.0)
        self.assertEqual(map_insert_action(-1.0), -1.0)
        self.assertEqual(map_insert_action(0.0), 0.0)
        self.assertEqual(map_insert_action(1.0), 1.0)
        self.assertAlmostEqual(
            MAX_INSERTION_PER_ACTION_M * MAX_EPISODE_STEPS,
            1.6384,
        )

    def test_ppo_defaults_match_current_baseline(self) -> None:
        self.assertEqual(PPO_N_ENVS, 32)
        self.assertEqual(PPO_N_STEPS, 256)
        self.assertEqual(PPO_BATCH_SIZE, 1024)
        self.assertEqual(PPO_GAE_LAMBDA, 0.98)
        self.assertEqual(PPO_ENT_COEF, 0.002)
        self.assertEqual(PPO_INITIAL_ACTION_STD, 0.50)
        self.assertEqual(PPO_MIN_ACTION_STD, 0.20)
        self.assertEqual(PPO_MAX_ACTION_STD, 0.60)
        self.assertAlmostEqual(math.degrees(LOCAL_FIELD_ACTION_ANGLE_RAD), 3.0)

    def test_sac_defaults_match_current_baseline(self) -> None:
        self.assertEqual(SAC_GAMMA, 0.9995)
        self.assertEqual(SAC_BATCH_SIZE, 1024)
        self.assertEqual(SAC_GRADIENT_STEPS, 1)

    def test_route_curriculum_is_b01_b02_only(self) -> None:
        self.assertEqual(TRAINING_CURRICULUM_BRANCH_MODELS, ("B01", "B02"))
        self.assertEqual(len(TRAINING_CURRICULUM_MODELS), 5)
        self.assertTrue(
            all(
                tuple(models) == TRAINING_CURRICULUM_BRANCH_MODELS
                for models in TRAINING_CURRICULUM_MODELS
            )
        )
        self.assertEqual(TRAINING_CURRICULUM_DR_FRACTIONS, (0.0,) * 5)
        self.assertEqual(TRAINING_CURRICULUM_TARGET_FRACTIONS, (1.0,) * 5)
        self.assertEqual(
            TRAINING_CURRICULUM_STAGE_NAMES,
            (
                "shallow_routes",
                "b01_medium_routes",
                "both_medium_routes",
                "b01_all_routes",
                "b01_b02_all_routes",
            ),
        )
        self.assertEqual(
            TRAINING_CURRICULUM_ROUTES[0],
            {"B01": (1, 4), "B02": (1, 4)},
        )

    def test_physics_wall_defaults_are_physical_only(self) -> None:
        self.assertTrue(SDF_PHYSICS_WALL_ENABLED)
        self.assertEqual(CATHETER_COLLISION_BODY_EDGES, 80)
        self.assertEqual(CATHETER_COLLISION_TIP_EDGES, 12)
        self.assertEqual(SDF_WALL_ACTIVATION_CLEARANCE_M, 0.0003)
        self.assertEqual(SDF_WALL_STIFFNESS_N_PER_M, 10.0)
        self.assertEqual(SDF_WALL_MAX_FORCE_N, 0.010)


if __name__ == "__main__":
    unittest.main()
