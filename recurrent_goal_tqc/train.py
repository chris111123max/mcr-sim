#!/usr/bin/env python3
"""Independent two-NPU recurrent Goal-TQC trainer with topology-safe HER."""

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
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrapper
from torch.utils.tensorboard import SummaryWriter

from mcr_sim.distributed import configure_npu_execution, initialize_distributed
from mcr_sim.goal_contract import condition_observation
from mcr_sim.mcr_rl_env import ActionType, EnvType, MCREnv, ObservationType
from mcr_sim.paths import TRAINING_RUNS_DIR
from mcr_sim.rl_core.base import RenderFramework, RenderMode
from mcr_sim.rl_core.run_logging import start_run_log_capture, write_run_config
from recurrent_goal_tqc.agent import Config, RecurrentGoalTQC
from recurrent_goal_tqc.replay import TopologyHerReplay


class RouteGoalVecEnv(VecEnvWrapper):
    """Rollout always seeks the real endpoint; HER changes replay only."""

    def __init__(self, venv):
        super().__init__(venv)
        self.base_dim = int(venv.observation_space.shape[0])
        low = np.concatenate((np.asarray(venv.observation_space.low, dtype=np.float32),
                              np.array([0., 0.], dtype=np.float32)))
        high = np.concatenate((np.asarray(venv.observation_space.high, dtype=np.float32),
                               np.array([1., 1.], dtype=np.float32)))
        low[9], high[9] = -1., 1.
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.achieved = np.zeros(self.num_envs, dtype=np.float32)

    def reset(self):
        base = self.venv.reset()
        self.achieved[:] = np.asarray(
            self.venv.get_attr("current_route_progress_ratio"), dtype=np.float32)
        return condition_observation(base, self.achieved, np.ones(self.num_envs, dtype=np.float32))

    def step_wait(self):
        base, reward, dones, infos = self.venv.step_wait()
        result = []
        for i, item in enumerate(infos):
            info = dict(item or {})
            value = float(info.get("route_progress_ratio", self.achieved[i]))
            if np.isfinite(value):
                self.achieved[i] = float(np.clip(value, 0., 1.))
            terminal = info.get("terminal_observation")
            if terminal is not None:
                terminal = np.asarray(terminal, dtype=np.float32).reshape(-1)
                if terminal.shape[0] == self.base_dim:
                    info["terminal_observation"] = condition_observation(
                        terminal, self.achieved[i], 1.0)
            result.append(info)
        augmented = condition_observation(
            base, self.achieved, np.ones(self.num_envs, dtype=np.float32))
        # VecEnv has already reset a finished worker. Its next observation
        # belongs to a fresh episode, never to the just-finished HER segment.
        for i, done in enumerate(dones):
            if done:
                self.achieved[i] = 0.0
                augmented[i] = condition_observation(base[i], 0.0, 1.0)
        return augmented, reward, dones, result


def build_env(args):
    env_type = EnvType.AORTIC if args.env_type == "aortic" else EnvType.FLAT
    render = RenderMode.HUMAN if args.render == "human" else RenderMode.NONE
    local_envs = args.n_envs // args.world_size if args.distributed else args.n_envs
    offset = args.distributed_rank * local_envs

    def make(slot):
        def create():
            seed = args.seed + offset + slot
            np.random.seed(seed); random.seed(seed)
            kwargs = {
                "force_model": args.force_model,
                "radius_observation_scale": args.radius_observation_scale,
                "randomize_start_target": args.randomize_start_target,
                "start_window_distance_m": args.start_window_mm / 1000.0,
                "target_window_distance_m": args.target_window_mm / 1000.0,
                "randomize_initial_orientation": args.randomize_initial_orientation,
                "initial_orientation_max_angle_deg": args.initial_orientation_max_angle_deg,
                "entry_tangent_points": args.entry_tangent_points,
                "soft_randomize_single_vessel": args.soft_randomize_single_vessel,
                "vessel_scale_min": args.vessel_scale_min,
                "vessel_scale_max": args.vessel_scale_max,
                "training_curriculum_enabled": args.training_curriculum,
                "training_curriculum_stage": args.curriculum_stage,
                "sampling_slot": offset + slot,
                "verbose_scene": args.scene_verbose,
                "debug_rendering": args.render == "human",
                "positioning_camera": args.render == "human",
            }
            if args.centerline_file:
                kwargs["centerline_file"] = args.centerline_file
            if args.asset_root:
                kwargs["asset_root"] = args.asset_root
            env = MCREnv(
                env_type=env_type, observation_type=ObservationType.STATE,
                action_type=ActionType.CONTINUOUS, time_step=args.time_step,
                frame_skip=args.frame_skip, settle_steps=args.settle_steps,
                render_mode=render, render_framework=RenderFramework.PYGLET,
                target_distance_threshold=args.target_threshold,
                max_episode_steps=args.max_episode_steps,
                create_scene_kwargs=kwargs,
            )
            return Monitor(env)
        return create

    if local_envs == 1:
        return RouteGoalVecEnv(DummyVecEnv([make(0)]))
    return RouteGoalVecEnv(SubprocVecEnv([make(i) for i in range(local_envs)],
                                         start_method="spawn"))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="npu")
    p.add_argument("--distributed", action="store_true")
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument("--local-rank", type=int, default=0)
    p.add_argument("--dist-backend", default="")
    p.add_argument("--n-envs", type=int, default=32)
    p.add_argument("--npu-fast-execution", action="store_true", default=True)
    p.add_argument("--exp-name", default="recurrent_goal_tqc_pilot")
    p.add_argument("--log-root", default=str(TRAINING_RUNS_DIR))
    p.add_argument("--run-dir", default="")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--episodes-per-epoch", type=int, default=100)
    p.add_argument("--timesteps", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sequence-length", type=int, default=16)
    p.add_argument("--sequence-batch-size", type=int, default=256)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--n-critics", type=int, default=3)
    p.add_argument("--n-quantiles", type=int, default=16)
    p.add_argument("--drop-top-quantiles-per-critic", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=.995)
    p.add_argument("--tau", type=float, default=.005)
    p.add_argument("--entropy-coef", type=float, default=.05)
    p.add_argument("--risk-weight", type=float, default=.05)
    p.add_argument("--risk-warmup-updates", type=int, default=2000)
    p.add_argument("--her-ratio", type=float, default=.5)
    p.add_argument("--future-horizon", type=int, default=64)
    p.add_argument("--risk-horizon", type=int, default=24)
    p.add_argument("--replay-capacity", type=int, default=300000)
    p.add_argument("--learning-starts", type=int, default=20000)
    p.add_argument("--updates-per-rollout", type=int, default=1)
    p.add_argument("--rollout-steps", type=int, default=32)
    p.add_argument("--metric-sync-interval", type=int, default=50)
    p.add_argument("--env-type", default="aortic", choices=("aortic", "flat"))
    p.add_argument("--force-model", default="B01")
    p.add_argument("--centerline-file", default="target_01_centerline.vtk")
    p.add_argument("--render", default="headless", choices=("headless", "human"))
    p.add_argument("--time-step", type=float, default=.01)
    p.add_argument("--frame-skip", type=int, default=1)
    p.add_argument("--settle-steps", type=int, default=8)
    p.add_argument("--target-threshold", type=float, default=.003)
    p.add_argument("--max-episode-steps", type=int, default=2048)
    p.add_argument("--radius-observation-scale", type=float, default=.005)
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
    p.add_argument("--curriculum-stage", type=int, default=0)
    p.add_argument("--scene-verbose", action="store_true")
    p.add_argument("--asset-root", default="")
    return p


def append_csv(path, fields, row):
    fresh = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        if fresh:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fields})


def main():
    args = parser().parse_args()
    if not args.distributed:
        args.world_size = 1
    if args.n_envs < 1 or args.n_envs % args.world_size or args.sequence_batch_size % args.world_size:
        raise ValueError("n-envs and sequence-batch-size must divide evenly over world-size")
    if not 0 <= args.her_ratio <= 1 or args.n_critics < 2 or args.n_quantiles < 2:
        raise ValueError("Invalid HER or quantile-critic configuration")
    if not 0 <= args.drop_top_quantiles_per_critic < args.n_quantiles:
        raise ValueError("Must retain at least one quantile per critic")
    if args.centerline_file and not args.force_model:
        raise ValueError("--centerline-file requires --force-model")
    if args.render == "human" and args.n_envs > 1:
        raise ValueError("GUI requires one environment")
    os.environ["MCR_SOFA_DT"] = str(args.time_step)
    context = initialize_distributed(args.distributed, args.device,
                                     args.world_size, args.local_rank, args.dist_backend)
    args.distributed_rank = context.rank
    args.local_n_envs = args.n_envs // context.world_size
    args.rank_seed = args.seed + context.rank * args.local_n_envs
    args.npu_execution = configure_npu_execution(context.device.accelerator,
                                                  args.npu_fast_execution)
    random.seed(args.rank_seed); np.random.seed(args.rank_seed); torch.manual_seed(args.rank_seed)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S") if context.is_main else ""
    stamp = context.broadcast_text(stamp) if context.enabled else stamp
    run_dir = Path(args.run_dir) if args.run_dir else Path(args.log_root) / f"{args.exp_name}_{stamp}"
    for name in ("models", "tb", "logs", "diagnostics"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    capture = start_run_log_capture(run_dir / "logs", context.rank)
    if context.is_main:
        write_run_config(run_dir / "run_config.json", args, algorithm="recurrent_goal_tqc",
                         run_dir=str(run_dir), local_n_envs=args.local_n_envs)
    print(f"[RGTQC] rank={context.rank}/{context.world_size} device={context.device.resolved} "
          f"envs={args.local_n_envs}/{args.n_envs} batch={args.sequence_batch_size // context.world_size}/"
          f"{args.sequence_batch_size} route={args.force_model}/{args.centerline_file or 'cycling'} "
          f"run={run_dir}", flush=True)
    try:
        env = build_env(args)
        obs = env.reset()
        agent = RecurrentGoalTQC(Config(
            observation_dim=obs.shape[1], action_dim=int(env.action_space.shape[0]),
            num_envs=env.num_envs, device=context.device.resolved,
            sequence_length=args.sequence_length, hidden_dim=args.hidden_dim,
            n_critics=args.n_critics, n_quantiles=args.n_quantiles,
            drop_top_quantiles_per_critic=args.drop_top_quantiles_per_critic,
            learning_rate=args.learning_rate, gamma=args.gamma, tau=args.tau,
            entropy_coef=args.entropy_coef, risk_weight=args.risk_weight,
            risk_warmup_updates=args.risk_warmup_updates,
            distributed=context.enabled,
        ))
        print(f"[RGTQC] npu={args.npu_execution} fused_adam={agent.fused_adam}", flush=True)
        replay = TopologyHerReplay(args.replay_capacity, env.num_envs)
        writer = SummaryWriter(str(run_dir / "tb" / f"rank_{context.rank}"))
        local_batch = args.sequence_batch_size // context.world_size
        target_episodes = args.epochs * args.episodes_per_epoch
        global_episodes = 0; pending_episodes = 0; total_steps = 0; loops = 0
        ready = False; last_epoch = 0; update_sums = defaultdict(float)
        recent = deque(maxlen=100); recent_rank_episodes = deque(maxlen=100)
        episode_fields = ["rank", "global_env_steps_local", "vessel_id", "target_route_id",
                          "success", "terminal_reason", "route_completion", "final_distance_mm",
                          "sdf_body_clearance_mm", "route_consistency_error", "steps"]
        step_counts = np.zeros(env.num_envs, dtype=np.int32)
        started = time.perf_counter()
        while global_episodes < target_episodes and (args.timesteps == 0 or
              total_steps * context.world_size < args.timesteps):
            actions, predicted_risk = agent.act(obs)
            next_obs, _, dones, infos = env.step(actions)
            terminal_next = np.asarray(next_obs, dtype=np.float32).copy()
            for i, info in enumerate(infos):
                step_counts[i] += 1
                if dones[i] and info.get("terminal_observation") is not None:
                    terminal_next[i] = np.asarray(info["terminal_observation"], dtype=np.float32)
                replay.add(i, obs[i], actions[i], terminal_next[i], info, bool(dones[i]))
                if dones[i]:
                    pending_episodes += 1
                    row = {
                        "rank": context.rank, "global_env_steps_local": total_steps + env.num_envs,
                        "vessel_id": info.get("chosen_model", "unknown"),
                        "target_route_id": info.get("target_route_id", "default"),
                        "success": int(bool(info.get("done_by_target", False))),
                        "terminal_reason": info.get("terminal_reason", "unknown"),
                        "route_completion": float(info.get("route_progress_ratio", np.nan)),
                        "final_distance_mm": 1000. * float(info.get("final_dist_to_goal", np.nan)),
                        "sdf_body_clearance_mm": 1000. * float(info.get("sdf_body_min_surface_clearance", np.nan)),
                        "route_consistency_error": float(info.get("route_completion_consistency_error", np.nan)),
                        "steps": int(step_counts[i]),
                    }
                    append_csv(run_dir / "diagnostics" / f"episodes_rank_{context.rank}.csv",
                               episode_fields, row)
                    recent.append(row); recent_rank_episodes.append(row)
                    step_counts[i] = 0
            agent.reset_envs(dones)
            obs = next_obs; total_steps += env.num_envs; loops += 1
            if loops % args.rollout_steps == 0:
                eligibility = torch.tensor([float(total_steps >= args.learning_starts and
                                                   replay.can_sample(args.sequence_length))],
                                           device=context.device.resolved)
                context.all_reduce(eligibility, op=torch.distributed.ReduceOp.MIN)
                ready = bool(eligibility.detach().cpu().item())
            if ready:
                for _ in range(args.updates_per_rollout):
                    sample = replay.sample(local_batch, args.sequence_length,
                                           args.her_ratio, args.future_horizon, args.risk_horizon)
                    metrics = agent.update(sample)
                    for key, value in metrics.items():
                        update_sums[key] += value
                    if agent.update_count % args.metric_sync_interval == 0:
                        keys = sorted(update_sums)
                        values = torch.tensor([update_sums[key] / args.metric_sync_interval for key in keys],
                                              device=context.device.resolved)
                        values = context.average_metric_tensor(values).detach().cpu().tolist()
                        if context.is_main:
                            merged = dict(zip(keys, values))
                            append_csv(run_dir / "diagnostics" / "updates.csv",
                                       ["update"] + keys, {"update": agent.update_count, **merged})
                            for key, value in merged.items():
                                writer.add_scalar("train/" + key, value, agent.update_count)
                        update_sums.clear()
            if loops % args.rollout_steps == 0:
                counter = torch.tensor([float(pending_episodes)], device=context.device.resolved)
                context.all_reduce(counter)
                global_episodes += int(counter.detach().cpu().item())
                pending_episodes = 0
                epoch = global_episodes // args.episodes_per_epoch
                if epoch > last_epoch:
                    last_epoch = epoch
                    if context.is_main:
                        checkpoint = run_dir / "models" / f"rgtqc_epoch_{epoch:03d}_episodes_{global_episodes:05d}.pt"
                        torch.save(agent.state_dict(), checkpoint)
                        rows = list(recent)
                        summary = {
                            "epoch": epoch, "global_completed_episodes": global_episodes,
                            "global_env_steps": total_steps * context.world_size,
                            "episodes_rank0_window": len(rows),
                            "success_rate_rank0_window": float(np.mean([r["success"] for r in rows])) if rows else np.nan,
                            "route_completion_rank0_window": float(np.mean([r["route_completion"] for r in rows])) if rows else np.nan,
                            "out_of_vessel_rate_rank0_window": float(np.mean([r["terminal_reason"] == "out_of_vessel" for r in rows])) if rows else np.nan,
                            "updates": agent.update_count, "elapsed_s": time.perf_counter() - started,
                            "checkpoint": str(checkpoint),
                        }
                        append_csv(run_dir / "train_summary.csv", list(summary), summary)
                        print("[RGTQC][CHECKPOINT] " + json.dumps(summary), flush=True)
                    context.barrier()
        if context.is_main:
            torch.save(agent.state_dict(), run_dir / "models" / "rgtqc_final.pt")
            print(f"[RGTQC][DONE] episodes={global_episodes} steps={total_steps * context.world_size} "
                  f"updates={agent.update_count}", flush=True)
        writer.close(); env.close()
    finally:
        capture.close(); context.close()


if __name__ == "__main__":
    main()
