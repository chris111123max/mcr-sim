#!/usr/bin/env python3
"""Deterministic, parallel evaluation of one vessel/target route."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from mcr_sim.distributed import initialize_distributed
from recurrent_goal_tqc.agent import Config, RecurrentGoalTQC
from recurrent_goal_tqc.train import append_csv, build_env, parser


def main():
    p = parser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--eval-episodes", type=int, default=64)
    p.add_argument("--output", default="")
    args = p.parse_args()
    if args.eval_episodes < 1 or args.n_envs < 1:
        raise ValueError("Evaluation requires positive episodes and environments")
    args.distributed = False
    args.world_size = 1
    args.distributed_rank = 0
    args.training_curriculum = False
    if not args.force_model:
        raise ValueError("Evaluation requires --force-model")
    context = initialize_distributed(False, args.device, 1, 0, "")
    checkpoint = Path(args.checkpoint).resolve()
    state = torch.load(str(checkpoint), map_location=context.device.resolved)
    env = build_env(args)
    try:
        observations = env.reset()
        config = dict(state["config"])
        config.update(observation_dim=observations.shape[1],
                      action_dim=int(env.action_space.shape[0]),
                      num_envs=env.num_envs, device=context.device.resolved,
                      distributed=False)
        agent = RecurrentGoalTQC(Config(**config))
        agent.load_state_dict(state)
        steps = np.zeros(env.num_envs, dtype=np.int32)
        risk_sums = np.zeros(env.num_envs, dtype=np.float64)
        output = Path(args.output) if args.output else (
            checkpoint.parent.parent / "diagnostics" /
            f"eval_{args.force_model}_{Path(args.centerline_file).stem or 'default'}.csv")
        output.parent.mkdir(parents=True, exist_ok=True)
        fields = ["episode", "vessel_id", "target_route_id", "success", "terminal_reason",
                  "steps", "route_completion", "final_distance_mm", "predicted_risk_mean"]
        results = []
        while len(results) < args.eval_episodes:
            actions, risk = agent.act(observations, deterministic=True)
            observations, _, dones, infos = env.step(actions)
            steps += 1
            risk_sums += risk
            agent.reset_envs(dones)
            for i, done in enumerate(dones):
                if not done:
                    continue
                if len(results) < args.eval_episodes:
                    info = infos[i]
                    row = {
                        "episode": len(results) + 1,
                        "vessel_id": info.get("chosen_model", "unknown"),
                        "target_route_id": info.get("target_route_id", "default"),
                        "success": int(bool(info.get("done_by_target", False))),
                        "terminal_reason": info.get("terminal_reason", "unknown"),
                        "steps": int(steps[i]),
                        "route_completion": float(info.get("route_progress_ratio", np.nan)),
                        "final_distance_mm": 1000. * float(info.get("final_dist_to_goal", np.nan)),
                        "predicted_risk_mean": float(risk_sums[i] / max(1, steps[i])),
                    }
                    append_csv(output, fields, row)
                    results.append(row)
                steps[i] = 0
                risk_sums[i] = 0.0
        summary = {"episodes": len(results), "success_rate": float(np.mean(
            [row["success"] for row in results])), "route_completion_mean": float(np.mean(
            [row["route_completion"] for row in results])), "output": str(output)}
        print("[RGTQC][EVAL] " + json.dumps(summary), flush=True)
    finally:
        env.close()
        context.close()


if __name__ == "__main__":
    main()
