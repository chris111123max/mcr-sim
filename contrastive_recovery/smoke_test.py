#!/usr/bin/env python3
"""Pure-PyTorch structural smoke test; does not start SOFA."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from contrastive_recovery.agent import AgentConfig, ContrastiveRecoveryAgent
from contrastive_recovery.replay import EpisodeSequenceReplay


def main():
    torch.manual_seed(7); np.random.seed(7)
    envs, obs_dim, action_dim, length = 4, 18, 3, 8
    replay = EpisodeSequenceReplay(1_000, envs, obs_dim, action_dim)
    obs = np.zeros((envs, obs_dim), dtype=np.float32)
    for step in range(24):
        actions = np.tanh(np.random.randn(envs, action_dim)).astype(np.float32)
        next_obs = obs + np.random.randn(envs, obs_dim).astype(np.float32) * .01
        dones = np.full(envs, step % 12 == 11, dtype=bool)
        replay.add_batch(
            obs, actions, next_obs, dones,
            np.random.randn(envs).astype(np.float32),
            np.random.rand(envs).astype(np.float32),
            np.clip((step + 1) / 24., 0., 1.) * np.ones(envs, dtype=np.float32),
            task_rewards=np.full(envs, 0.02, dtype=np.float32),
        )
        obs = next_obs
    assert replay.can_sample(4, length), "sequence replay did not preserve complete episodes"
    agent = ContrastiveRecoveryAgent(AgentConfig(observation_dim=obs_dim, action_dim=action_dim, num_envs=envs, device="cpu", sequence_length=length, hidden_dim=32, embedding_dim=16))
    actions, risks, recovering = agent.act(obs)
    assert actions.shape == (envs, action_dim) and risks.shape == (envs,) and recovering.shape == (envs,)
    batch = replay.sample(4, length, 6, 4)
    assert batch.task_rewards.shape == (4, length)
    metrics = agent.update(batch)
    assert all(np.isfinite(value) for value in metrics.values()), metrics
    assert metrics["risk_weight_effective"] == 0.0
    assert "task_critic_loss" in metrics and "goal_support_rate" in metrics
    print("[PASS] contrastive_recovery structural smoke", metrics)


if __name__ == "__main__": main()
