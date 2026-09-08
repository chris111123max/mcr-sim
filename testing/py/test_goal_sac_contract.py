"""CPU regression tests, no SOFA/NPU required. Run from the Python root:

    python -m unittest discover -s testing/py -p test_goal_sac_contract.py -v
"""
import tempfile
import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from mcr_sim.goal_contract import GoalReward, condition_observation, relabel_observation
from mcr_sim.goal_conditioned_env import GoalConditionedVecEnv, SafeHerReplayBuffer
from mcr_sim.distributed.goal_sac import GoalConditionedSAC


class ToyRoute(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, shape=(45,), dtype=np.float32)
    action_space = gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32)

    def __init__(self, success=False):
        self.success = success
        self.current_route_progress_ratio = 0.0

    def state(self):
        obs = np.zeros(45, np.float32)
        obs[9] = 1 - self.current_route_progress_ratio
        obs[10] = 1 - self.steps / 10
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.current_route_progress_ratio = 0.0
        return self.state(), {}

    def step(self, action):
        self.steps += 1
        self.current_route_progress_ratio = self.steps * .02
        done = self.steps == 10
        info = dict(route_progress_ratio=self.current_route_progress_ratio,
                    done_by_target=done and self.success,
                    sdf_body_min_surface_clearance=.002)
        return self.state(), 0., done and self.success, done and not self.success, info


class GoalContractTests(unittest.TestCase):
    def test_relabel_updates_all_goal_fields(self):
        base = np.zeros((2, 45), np.float32)
        original = condition_observation(base, [.2, .3], [1., 1.])
        changed = relabel_observation(original, [.4, .5])
        np.testing.assert_allclose(changed[:, 9], [.2, .2], atol=1e-7)
        np.testing.assert_allclose(changed[:, -2:], [[.2, .4], [.3, .5]])
        np.testing.assert_array_equal(changed[:, :9], base[:, :9])
        self.assertEqual(changed.shape, (2, 47))

    def test_potential_telescopes_for_success_failure_and_backtracking(self):
        reward = GoalReward()
        for success in (True, False):
            path = [0., .2, .1, .4, .4]
            discounted = 0.
            for t, (prev, nxt) in enumerate(zip(path, path[1:])):
                terminal = t == len(path) - 2
                base = 20 if terminal and success else -20 if terminal else -.002
                discounted += reward.gamma ** t * (float(reward(
                    prev, nxt, 1., success and terminal, terminal)) - base)
            self.assertAlmostEqual(discounted, 0., places=5)
        self.assertLess(float(reward(0., 0., 1., False, False)), 0.)
        self.assertLess(float(reward(.4, .4, 1., False, True)), 0.)

    def test_auto_reset_and_terminal_observation(self):
        env = GoalConditionedVecEnv(DummyVecEnv([ToyRoute]))
        try:
            obs = env.reset()
            self.assertEqual(obs.shape, (1, 47))
            for _ in range(10):
                obs, rewards, dones, infos = env.step(np.zeros((1, 2)))
            self.assertTrue(dones[0])
            self.assertFalse(infos[0]['TimeLimit.truncated'])
            self.assertAlmostEqual(float(obs[0, -2]), 0.)
            self.assertAlmostEqual(float(infos[0]['terminal_observation'][-2]), .2)
            self.assertAlmostEqual(float(rewards[0]), float(GoalReward()(
                .18, .2, 1., False, True)), places=5)
        finally:
            env.close()

    def make_buffer(self, size=32):
        return SafeHerReplayBuffer(size, gym.spaces.Box(-1, 1, (47,), dtype=np.float32),
                                   ToyRoute.action_space, device='cpu', her_ratio=1.)

    def add_transition(self, replay, prev, nxt, done=False, safe=True):
        obs = condition_observation(np.zeros((1, 45)), [prev], [1.])
        next_obs = condition_observation(np.zeros((1, 45)), [nxt], [1.])
        reward = GoalReward()(prev, nxt, 1., False, done)
        replay.add(obs, next_obs, np.zeros((1, 2)), np.array([reward]),
                   np.array([done]), [dict(achieved_goal=nxt, her_safe=safe,
                                          sdf_body_min_surface_clearance=.002)])

    def test_her_reward_and_terminal_match_contract(self):
        replay = self.make_buffer()
        for t in range(10):
            self.add_transition(replay, t * .02, (t + 1) * .02, t == 9)
        batch = replay.sample(1000)
        obs, nxt = batch.observations.numpy(), batch.next_observations.numpy()
        goals = obs[:, -1]
        self.assertTrue(np.all(goals < 1.))
        self.assertTrue(np.all(goals > obs[:, -2] + replay.goal_tolerance))
        success = nxt[:, -2] + replay.goal_tolerance >= goals
        terminal = success | (nxt[:, -2] >= .19999)
        expected = GoalReward()(obs[:, -2], nxt[:, -2], goals, success, terminal)
        np.testing.assert_allclose(batch.rewards.numpy()[:, 0], expected, atol=1e-6)
        np.testing.assert_allclose(batch.dones.numpy()[:, 0], terminal)
        np.testing.assert_allclose(obs[:, 9], goals - obs[:, -2], atol=1e-7)
        np.testing.assert_allclose(nxt[:, 9], goals - nxt[:, -2], atol=1e-7)
        self.assertGreater(replay.her_stats['her_successes'], 0)
        self.assertGreater(replay.pop_window_stats()['samples'], 0)
        self.assertEqual(replay.her_stats['samples'], 0)

    def test_stall_unsafe_and_ring_overwrite(self):
        replay = self.make_buffer(size=4)
        for _ in range(4):
            self.add_transition(replay, 0., 0.)
        self.assertTrue(torch.all(replay.sample(32).observations[:, -1] == 1.))
        self.add_transition(replay, 0., .5, done=True, safe=False)
        for _ in range(4):
            self.add_transition(replay, 0., 0.)
        batch = replay.sample(32)
        self.assertTrue(torch.all(batch.observations[:, -1] == 1.))
        self.assertTrue(torch.all(batch.dones == 0.))

    def test_original_failure_is_not_erased(self):
        replay = self.make_buffer()
        self.add_transition(replay, 0., .2, done=True, safe=False)
        batch = replay.sample(64)
        self.assertTrue(torch.all(batch.dones == 1.))
        self.assertTrue(torch.all(batch.observations[:, -1] == 1.))
        np.testing.assert_allclose(batch.rewards.numpy(), -20.)

    def test_reward_configuration_mismatch_fails_early(self):
        env = GoalConditionedVecEnv(DummyVecEnv([ToyRoute]))
        try:
            with self.assertRaisesRegex(ValueError, 'gamma'):
                GoalConditionedSAC('MlpPolicy', env, gamma=.99, device='cpu',
                    buffer_size=32, replay_buffer_class=SafeHerReplayBuffer)
        finally:
            env.close()

    def test_twin_q_update_save_load(self):
        torch.set_num_threads(1)
        env = GoalConditionedVecEnv(DummyVecEnv([ToyRoute]))
        try:
            model = GoalConditionedSAC('MlpPolicy', env, device='cpu', seed=5,
                gamma=.999,
                buffer_size=128, batch_size=8, learning_starts=10, utd_ratio=1,
                actor_update_interval=2, policy_kwargs=dict(net_arch=[16, 16]),
                replay_buffer_class=SafeHerReplayBuffer)
            model.learn(total_timesteps=30)
            self.assertEqual(model._n_updates, 20)
            self.assertEqual(model._goal_sac_actor_updates, 10)
            self.assertEqual(len(model.critic.q_networks), 2)
            heads = list(model.critic.q_networks)
            self.assertFalse(torch.equal(next(heads[0].parameters()), next(heads[1].parameters())))
            self.assertTrue(all(not p.requires_grad for p in model.critic_target.parameters()))
            observation = env.reset()
            expected = model.predict(observation, deterministic=True)[0]
            with tempfile.TemporaryDirectory() as tmp:
                path = str(Path(tmp) / 'model')
                model.save(path)
                restored = GoalConditionedSAC.load(path, env=env, device='cpu')
                np.testing.assert_allclose(restored.predict(observation, deterministic=True)[0], expected)
                restored.learn(total_timesteps=12)
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
