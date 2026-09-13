#!/usr/bin/env python3
"""CPU structural test; no SOFA scene or NPU allocation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from mcr_sim.goal_contract import condition_observation
from recurrent_goal_tqc.agent import Config, RecurrentGoalTQC, quantile_huber
from recurrent_goal_tqc.replay import TopologyHerReplay, goal_reward


def main():
    torch.manual_seed(4)
    np.random.seed(4)
    replay = TopologyHerReplay(500, 2)
    for route_index in (1, 2):
        for step in range(12):
            before, after = step / 30., (step + 1) / 30.
            base = np.zeros(18, dtype=np.float32)
            obs = condition_observation(base, before, 1.0)
            next_obs = condition_observation(base, after, 1.0)
            info = {
                "chosen_model": "B01", "target_route_id": f"target_{route_index:02d}",
                "route_progress_ratio": after,
                "sdf_body_min_surface_clearance": .001,
                "sdf_body_warning_feature": 0.0,
                "done_by_out_of_vessel": False,
            }
            replay.add(route_index - 1, obs, np.zeros(3, dtype=np.float32),
                       next_obs, info, step == 11)
    batch = replay.sample(32, 8, 1.0, 5, 4)
    assert batch.obs.shape == (32, 8, 20)
    assert batch.actions.shape == (32, 8, 3)
    assert batch.her_mask.mean() > 0.0
    assert np.all(batch.obs[:, -1, -1] < 1.0)
    assert np.allclose(batch.obs[:, :, 9],
                       batch.obs[:, :, -1] - batch.obs[:, :, -2], atol=1e-6)
    # A failed terminal transition may not be recast as a safe HER success.
    failed = TopologyHerReplay(100, 1)
    for step in range(8):
        base = np.zeros(18, dtype=np.float32)
        info = {"chosen_model": "B01", "target_route_id": "target_01",
                "route_progress_ratio": (step + 1) / 20.,
                "sdf_body_min_surface_clearance": -.001 if step == 7 else .001,
                "done_by_out_of_vessel": step == 7}
        failed.add(0, condition_observation(base, step / 20., 1.),
                   np.zeros(3, dtype=np.float32),
                   condition_observation(base, (step + 1) / 20., 1.), info,
                   step == 7)
    unsafe_batch = failed.sample(8, 8, 1.0, 4, 4)
    assert unsafe_batch.her_mask.sum() == 0.0
    assert np.all(unsafe_batch.dones == 1.0)
    assert np.all(unsafe_batch.rewards < 0.0)
    assert goal_reward(.1, .2, 1., False, True) < 0.0
    assert 20.0 * 0.99 - 25.0 < 0.0  # Unsafe near-finish failure stays net-negative.

    agent = RecurrentGoalTQC(Config(observation_dim=20, action_dim=3,
                                     num_envs=2, device="cpu", sequence_length=8,
                                     hidden_dim=32, n_critics=3, n_quantiles=8,
                                     drop_top_quantiles_per_critic=1))
    actions, risk = agent.act(np.zeros((2, 20), dtype=np.float32))
    assert actions.shape == (2, 3) and risk.shape == (2,)
    metrics = agent.update(batch)
    assert all(np.isfinite(value) for value in metrics.values()), metrics
    predicted = torch.randn(4, 3, 8, requires_grad=True)
    loss = quantile_huber(predicted, torch.randn(4, 21))
    loss.backward()
    assert torch.isfinite(predicted.grad).all()
    print("[PASS] recurrent_goal_tqc replay/HER/GRU/TQC/risk", metrics)


if __name__ == "__main__":
    main()
