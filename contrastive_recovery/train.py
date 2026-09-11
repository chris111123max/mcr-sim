#!/usr/bin/env python3
"""Standalone two-NPU trainer for Contrastive Goal RL + Recovery RL.

This deliberately does not import, subclass, or patch the existing PPO/SAC
algorithms. It reuses only the stable SOFA environment factory.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnvWrapper
from torch.utils.tensorboard import SummaryWriter

from contrastive_recovery.agent import AgentConfig, ContrastiveRecoveryAgent
from contrastive_recovery.replay import EpisodeSequenceReplay
from mcr_sim.distributed import configure_npu_execution, initialize_distributed
from mcr_sim.paths import TRAINING_RUNS_DIR
from mcr_sim.rl_core.run_logging import start_run_log_capture, write_run_config
from training.py.train_sac import build_env


class ProgressGoalVecEnv(VecEnvWrapper):
    """Append continuous achieved/final goal without exposing a route identifier.

    The base environment supplies local centreline geometry and target-derived
    guidance. ``target_route_id`` remains diagnostic metadata only; neither
    actor gets it as an input feature.
    """

    def __init__(self, venv):
        super().__init__(venv)
        self.base_dim = int(venv.observation_space.shape[0])
        low = np.concatenate((np.asarray(venv.observation_space.low, dtype=np.float32), np.array([0., 0.], dtype=np.float32)))
        high = np.concatenate((np.asarray(venv.observation_space.high, dtype=np.float32), np.array([1., 1.], dtype=np.float32)))
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.achieved = np.zeros(self.num_envs, dtype=np.float32)

    def _augment(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        return np.concatenate((obs, self.achieved[:, None], np.ones((self.num_envs, 1), dtype=np.float32)), axis=1)

    def reset(self):
        obs = self.venv.reset()
        try:
            self.achieved[:] = np.asarray(self.venv.get_attr("current_route_progress_ratio"), dtype=np.float32)
        except Exception:
            self.achieved.fill(0.)
        return self._augment(obs)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        out_infos = []
        terminal_augmented = []
        for index, info in enumerate(infos):
            info = dict(info or {})
            value = info.get("route_progress_ratio", self.achieved[index])
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = float(self.achieved[index])
            self.achieved[index] = float(np.clip(value, 0., 1.)) if np.isfinite(value) else self.achieved[index]
            terminal = info.get("terminal_observation")
            if terminal is not None:
                terminal = np.asarray(terminal, dtype=np.float32).reshape(-1)
                if terminal.shape[0] == self.base_dim:
                    info["terminal_observation"] = np.concatenate((terminal, np.array([self.achieved[index], 1.], dtype=np.float32)))
            info["contrastive_achieved_goal"] = float(self.achieved[index])
            # raw SDF warning is observable only as a risk feature, never a
            # hand-crafted steering action.
            info["contrastive_risk"] = float(_risk_from_info(info))
            out_infos.append(info)
        augmented = self._augment(obs)
        # SubprocVecEnv has already reset a done slot. Do not make its next
        # episode inherit the previous endpoint as its achieved goal.
        for index, done in enumerate(dones):
            if done:
                self.achieved[index] = 0.
                augmented[index, -2] = 0.
        return augmented, rewards, dones, out_infos


def _risk_from_info(info) -> float:
    if bool(info.get("done_by_out_of_vessel", False) or info.get("out_of_vessel", False) or info.get("done_by_non_finite", False)):
        return 1.0
    values = []
    for key in ("sdf_body_warning_feature", "sdf_tip_warning_feature", "off_target_branch_feature"):
        try:
            values.append(float(info.get(key, 0.0)))
        except (TypeError, ValueError):
            pass
    return float(np.clip(max(values) if values else 0., 0., 1.))


def _clearance_from_info(info) -> float:
    for key in ("sdf_body_surface_clearance", "sdf_surface_clearance", "sdf_tip_surface_clearance"):
        try:
            value = float(info.get(key))
            if np.isfinite(value):
                return value
        except (TypeError, ValueError):
            pass
    return 0.0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    # hardware / distributed
    p.add_argument("--device", default="npu")
    p.add_argument("--distributed", action="store_true")
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument("--local-rank", type=int, default=0)
    p.add_argument("--dist-backend", default="")
    p.add_argument("--n-envs", type=int, default=32)
    p.add_argument("--npu-fast-execution", action="store_true", default=True)
    # experiment
    p.add_argument("--exp-name", default="contrastive_recovery_2npu_32env")
    p.add_argument("--log-root", default=str(TRAINING_RUNS_DIR))
    p.add_argument("--run-dir", default="", help="Explicit run directory; used by the managed --nohup launcher.")
    p.add_argument("--timesteps", type=int, default=0, help="Optional global safety cap; 0 means stop by epochs × completed episodes.")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--episodes-per-epoch", type=int, default=100)
    p.add_argument("--save-freq", type=int, default=1_024_000)
    p.add_argument("--seed", type=int, default=42)
    # algorithm: global batch is divided exactly over the two devices.
    p.add_argument("--sequence-batch-size", type=int, default=256)
    p.add_argument("--sequence-length", type=int, default=32)
    p.add_argument("--future-horizon", type=int, default=64)
    p.add_argument("--risk-horizon", type=int, default=24)
    p.add_argument("--replay-capacity", type=int, default=300_000)
    p.add_argument("--learning-starts", type=int, default=20_000)
    p.add_argument("--updates-per-rollout", type=int, default=2)
    p.add_argument("--rollout-steps", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--embedding-dim", type=int, default=128)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--entropy-weight", type=float, default=0.05)
    p.add_argument("--risk-weight", type=float, default=1.0)
    p.add_argument("--recovery-gate-threshold", type=float, default=0.40)
    p.add_argument("--recovery-min-steps", type=int, default=12)
    p.add_argument("--metric-sync-interval", type=int, default=50)
    # stable SOFA environment contract (same names as train_sac.build_env).
    p.add_argument("--env-type", default="aortic", choices=("aortic", "flat"))
    p.add_argument("--force-model", default="")
    p.add_argument("--render", default="headless", choices=("headless", "human"))
    p.add_argument("--time-step", type=float, default=0.01)
    p.add_argument("--frame-skip", type=int, default=1)
    p.add_argument("--settle-steps", type=int, default=8)
    p.add_argument("--target-threshold", type=float, default=0.003)
    p.add_argument("--max-episode-steps", type=int, default=2048)
    p.add_argument("--radius-observation-scale", type=float, default=0.005)
    p.add_argument("--randomize-start-target", action="store_true")
    p.add_argument("--start-window-mm", type=float, default=0.)
    p.add_argument("--target-window-mm", type=float, default=0.)
    p.add_argument("--randomize-initial-orientation", action="store_true")
    p.add_argument("--initial-orientation-max-angle-deg", type=float, default=0.)
    p.add_argument("--entry-tangent-points", type=int, default=5)
    p.add_argument("--soft-randomize-single-vessel", action="store_true")
    p.add_argument("--vessel-scale-min", type=float, default=1.)
    p.add_argument("--vessel-scale-max", type=float, default=1.)
    p.add_argument("--training-curriculum", action="store_true")
    p.add_argument("--scene-verbose", action="store_true")
    p.add_argument("--asset-root", default="")
    return p


def _append_csv(path: Path, fields, row):
    fresh = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fresh:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fields})


def _episode_row(global_step, rank, info):
    return {
        "global_env_steps_local": global_step, "rank": rank,
        "vessel_id": info.get("chosen_model", info.get("task_id", "unknown")),
        "terminal_reason": info.get("terminal_reason", "unknown"),
        "success": int(bool(info.get("done_by_target", False))),
        "route_completion": float(info.get("route_progress_ratio", 0.)),
        "final_distance_mm": 1000. * float(info.get("final_dist_to_goal", np.nan)),
        "min_distance_mm": 1000. * float(info.get("min_dist_to_goal", np.nan)),
        "sdf_body_clearance_mm": 1000. * _clearance_from_info(info),
        "risk": _risk_from_info(info),
    }


def main():
    args = parser().parse_args()
    if args.n_envs < 1 or args.n_envs % args.world_size:
        raise ValueError("--n-envs must be a positive multiple of --world-size.")
    if args.sequence_batch_size < args.world_size or args.sequence_batch_size % args.world_size:
        raise ValueError("--sequence-batch-size must divide evenly across the ranks.")
    if args.render == "human" and args.n_envs > 1:
        raise ValueError("SOFA GUI cannot be used with parallel environments.")
    os.environ["MCR_SOFA_DT"] = str(args.time_step)
    context = initialize_distributed(args.distributed, args.device, args.world_size, args.local_rank, args.dist_backend)
    args.distributed_rank = context.rank
    args.local_n_envs = args.n_envs // context.world_size
    args.rank_seed = args.seed + context.rank * args.local_n_envs
    args.npu_execution = configure_npu_execution(context.device.accelerator, args.npu_fast_execution)
    random.seed(args.rank_seed); np.random.seed(args.rank_seed); torch.manual_seed(args.rank_seed)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S") if context.is_main else ""
    stamp = context.broadcast_text(stamp) if context.enabled else stamp
    run_dir = Path(args.run_dir) if args.run_dir else Path(args.log_root) / f"{args.exp_name}_{stamp}"
    for directory in (run_dir / "models", run_dir / "tb", run_dir / "logs", run_dir / "diagnostics"):
        directory.mkdir(parents=True, exist_ok=True)
    capture = start_run_log_capture(run_dir / "logs", context.rank)
    write_run_config(run_dir / "run_config.json", args, algorithm="contrastive_goal_recovery_rl", run_dir=str(run_dir), local_n_envs=args.local_n_envs, global_sequence_batch=args.sequence_batch_size, local_sequence_batch=args.sequence_batch_size // context.world_size, topology_contract="selected centreline geometry + target only; target_route_id excluded from actor")
    print(f"[CRRL] rank={context.rank}/{context.world_size} device={context.device.resolved} envs={args.local_n_envs}/{args.n_envs} seq_batch={args.sequence_batch_size // context.world_size}/{args.sequence_batch_size} sequence={args.sequence_length} run={run_dir}", flush=True)
    try:
        env = ProgressGoalVecEnv(build_env(args))
        obs = env.reset()
        agent = ContrastiveRecoveryAgent(AgentConfig(observation_dim=obs.shape[1], action_dim=int(env.action_space.shape[0]), num_envs=env.num_envs, device=context.device.resolved, sequence_length=args.sequence_length, hidden_dim=args.hidden_dim, embedding_dim=args.embedding_dim, learning_rate=args.learning_rate, gamma=args.gamma, tau=args.tau, entropy_weight=args.entropy_weight, risk_weight=args.risk_weight, recovery_gate_threshold=args.recovery_gate_threshold, recovery_min_steps=args.recovery_min_steps, distributed=context.enabled))
        print(f"[CRRL] npu_execution={args.npu_execution} fused_adam={agent.fused_adam_status}", flush=True)
        replay = EpisodeSequenceReplay(args.replay_capacity, env.num_envs, obs.shape[1], int(env.action_space.shape[0]))
        writer = SummaryWriter(str(run_dir / "tb" / f"rank_{context.rank}"))
        local_batch = args.sequence_batch_size // context.world_size
        local_limit = int(np.ceil(args.timesteps / context.world_size)) if args.timesteps > 0 else None
        target_global_episodes = int(args.epochs) * int(args.episodes_per_epoch)
        total_steps = 0; update_metrics = defaultdict(float); update_count = 0
        global_completed_episodes = 0; local_unsynced_episodes = 0; rollout_loops = 0; last_saved_epoch = 0
        warnings = np.zeros(env.num_envs, dtype=np.float32); clearances = np.zeros(env.num_envs, dtype=np.float32)
        episodes = deque(maxlen=100); active_recovery = deque(maxlen=1000)
        episode_fields = ["global_env_steps_local", "rank", "vessel_id", "terminal_reason", "success", "route_completion", "final_distance_mm", "min_distance_mm", "sdf_body_clearance_mm", "risk"]
        while global_completed_episodes < target_global_episodes and (local_limit is None or total_steps < local_limit):
            actions, predicted_risk, recovery_active = agent.act(obs, warnings)
            next_obs, _, dones, infos = env.step(actions)
            terminal_next = np.asarray(next_obs, dtype=np.float32).copy()
            rewards = np.zeros(env.num_envs, dtype=np.float32); risks = np.zeros(env.num_envs, dtype=np.float32); goals = np.zeros(env.num_envs, dtype=np.float32)
            for index, info in enumerate(infos):
                info = dict(info or {})
                if dones[index] and info.get("terminal_observation") is not None:
                    terminal_next[index] = np.asarray(info["terminal_observation"], dtype=np.float32)
                risk = _risk_from_info(info); clearance = _clearance_from_info(info)
                clearance_delta = float(np.clip((clearance - clearances[index]) / 0.002, -1., 1.))
                rewards[index] = clearance_delta - 1.5 * risk - (20. if bool(info.get("done_by_out_of_vessel", False)) else 0.) + (3. if bool(dones[index]) and risk < .1 else 0.)
                risks[index] = risk; goals[index] = float(info.get("contrastive_achieved_goal", 0.))
                warnings[index] = risk; clearances[index] = clearance
                if dones[index]:
                    local_unsynced_episodes += 1
                    row = _episode_row(total_steps, context.rank, info)
                    episodes.append(row); _append_csv(run_dir / "diagnostics" / f"episode_events_rank_{context.rank}.csv", episode_fields, row)
                    clearances[index] = 0.; warnings[index] = 0.
            replay.add_batch(obs, actions, terminal_next, dones, rewards, risks, goals)
            agent.reset_envs(dones); obs = next_obs; total_steps += env.num_envs; rollout_loops += 1; active_recovery.extend(recovery_active.tolist())
            if total_steps >= args.learning_starts and replay.can_sample(local_batch, args.sequence_length):
                for _ in range(args.updates_per_rollout):
                    metrics = agent.update(replay.sample(local_batch, args.sequence_length, args.future_horizon, args.risk_horizon))
                    update_count += 1
                    for key, value in metrics.items(): update_metrics[key] += value
                    if update_count % args.metric_sync_interval == 0:
                        vector = torch.tensor([update_metrics[k] / args.metric_sync_interval for k in sorted(update_metrics)], device=context.device.resolved)
                        vector = context.average_metric_tensor(vector)
                        merged = dict(zip(sorted(update_metrics), vector.detach().cpu().tolist()))
                        if context.is_main:
                            for key, value in merged.items(): writer.add_scalar(f"train/{key}", value, update_count)
                            _append_csv(run_dir / "diagnostics" / "update_metrics.csv", ["update"] + sorted(merged), {"update": update_count, **merged})
                        update_metrics.clear()
            # One scalar per 32 vector rollout steps: this controls genuine
            # epoch semantics without an all-reduce on every SOFA step.
            if rollout_loops % max(1, args.rollout_steps) == 0:
                episode_counter = torch.tensor([float(local_unsynced_episodes)], device=context.device.resolved)
                context.all_reduce(episode_counter)
                global_completed_episodes += int(episode_counter.detach().cpu().item())
                local_unsynced_episodes = 0
                global_steps = total_steps * context.world_size
                if context.is_main:
                    writer.add_scalar("rollout/replay_transitions", replay.size, global_steps)
                    writer.add_scalar("recovery/gate_fraction", float(np.mean(active_recovery)) if active_recovery else 0., global_steps)
                    writer.add_scalar("rollout/global_completed_episodes", global_completed_episodes, global_steps)
                completed_epoch = global_completed_episodes // max(1, args.episodes_per_epoch)
                if completed_epoch > last_saved_epoch:
                    last_saved_epoch = completed_epoch
                    if context.is_main:
                        checkpoint = run_dir / "models" / f"contrastive_recovery_epoch_{completed_epoch:03d}_episodes_{global_completed_episodes:05d}.pt"
                        torch.save(agent.state_dict(), checkpoint)
                        recent = list(episodes)
                        summary = {"epoch": completed_epoch, "global_completed_episodes": global_completed_episodes, "global_env_steps": global_steps, "episodes_window_rank0": len(recent), "success_rate_rank0_window": float(np.mean([x["success"] for x in recent])) if recent else np.nan, "route_completion_mean_rank0_window": float(np.nanmean([x["route_completion"] for x in recent])) if recent else np.nan, "out_of_vessel_rate_rank0_window": float(np.mean([x["terminal_reason"] == "out_of_vessel" for x in recent])) if recent else np.nan, "recovery_gate_fraction_rank0": float(np.mean(active_recovery)) if active_recovery else 0., "checkpoint": str(checkpoint)}
                        _append_csv(run_dir / "train_summary.csv", list(summary), summary)
                        print("[CRRL][CHECKPOINT] " + json.dumps(summary), flush=True)
                    context.barrier()
        if context.is_main:
            torch.save(agent.state_dict(), run_dir / "models" / "contrastive_recovery_final.pt")
            print(f"[CRRL][DONE] global_episodes={global_completed_episodes} local_steps={total_steps} updates={agent.update_count}", flush=True)
        writer.close(); env.close()
    finally:
        capture.close(); context.close()


if __name__ == "__main__":
    main()
