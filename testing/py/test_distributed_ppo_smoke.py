"""Two-rank CPU/Gloo smoke test for DistributedPPO and portable resume."""

from __future__ import annotations

import os
import socket
import sys
import tempfile
from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parents[2]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

import gymnasium as gym
import torch as th
import torch.distributed as dist
from stable_baselines3 import PPO

from mcr_sim.distributed.context import initialize_distributed
from mcr_sim.distributed.ppo import DistributedPPO


def _max_parameter_difference(module: th.nn.Module, source: int = 0) -> float:
    difference = th.zeros(1, dtype=th.float32)
    for parameter in module.parameters():
        reference = parameter.detach().clone()
        dist.broadcast(reference, src=source)
        difference = th.maximum(
            difference, (parameter.detach() - reference).abs().max().cpu()
        )
    dist.all_reduce(difference, op=dist.ReduceOp.MAX)
    return float(difference.item())


def run_rank() -> None:
    context = initialize_distributed(
        enabled=True,
        requested_device="cpu",
        requested_world_size=2,
        requested_backend="gloo",
    )
    env = gym.make("Pendulum-v1")
    try:
        model = DistributedPPO(
            "MlpPolicy",
            env,
            distributed_context=context,
            device="cpu",
            n_steps=16,
            batch_size=8,
            n_epochs=2,
            seed=123 + context.rank,
            verbose=0,
        )
        model.synchronize_parameters()
        model.learn(total_timesteps=32)
        first_diff = _max_parameter_difference(model.policy)
        if first_diff > 1e-6:
            raise AssertionError(f"Distributed PPO parameters diverged: {first_diff}")

        checkpoint = Path(tempfile.gettempdir()) / "mcr_distributed_ppo_gloo_smoke"
        context.barrier()
        if context.is_main:
            model.save(str(checkpoint))
        context.barrier()
        resumed = DistributedPPO.load(str(checkpoint) + ".zip", env=env, device="cpu")
        resumed.set_distributed_context(context)
        resumed.synchronize_parameters()
        resumed.learn(total_timesteps=16, reset_num_timesteps=False)
        resumed_diff = _max_parameter_difference(resumed.policy)
        if resumed_diff > 1e-6:
            raise AssertionError(f"Resumed PPO parameters diverged: {resumed_diff}")

        if context.is_main:
            loaded = PPO.load(str(checkpoint) + ".zip", device="cpu")
            observation, _ = env.reset(seed=123)
            action, _ = loaded.predict(observation, deterministic=True)
            print(
                "[PASS] two-rank Gloo PPO gradients and resume synchronized; "
                f"native PPO.load action_shape={action.shape} "
                f"max_diff={max(first_diff, resumed_diff):.3g}"
            )
        context.barrier()
    finally:
        env.close()
        context.close()


def _spawn_rank(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    run_rank()


def main() -> None:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        run_rank()
        return
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    th.multiprocessing.spawn(_spawn_rank, args=(2, port), nprocs=2, join=True)


if __name__ == "__main__":
    main()
