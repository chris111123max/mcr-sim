"""Two-rank CPU/Gloo smoke test for the project-side DistributedSAC layer.

Run from the Python/Git root with the target training environment active:

    torchrun --standalone --nproc_per_node=2 testing/py/test_distributed_sac_smoke.py
"""

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
from stable_baselines3 import SAC

from mcr_sim.distributed.context import initialize_distributed
from mcr_sim.distributed.sac import DistributedSAC


def _max_parameter_difference(module: th.nn.Module, source: int = 0) -> float:
    difference = th.zeros(1, dtype=th.float32)
    for parameter in module.parameters():
        reference = parameter.detach().clone()
        dist.broadcast(reference, src=source)
        difference = th.maximum(difference, (parameter.detach() - reference).abs().max().cpu())
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
        env.reset(seed=1000 + context.rank)
        model = DistributedSAC(
            "MlpPolicy",
            env,
            distributed_context=context,
            device="cpu",
            buffer_size=256,
            learning_starts=8,
            batch_size=8,
            train_freq=1,
            gradient_steps=1,
            seed=123 + context.rank,
            verbose=0,
        )
        model.synchronize_parameters()
        model.learn(total_timesteps=32)

        actor_diff = _max_parameter_difference(model.actor)
        critic_diff = _max_parameter_difference(model.critic)
        target_diff = _max_parameter_difference(model.critic_target)
        entropy = getattr(model, "log_ent_coef", None)
        entropy_diff = 0.0
        if entropy is not None:
            reference = entropy.detach().clone()
            dist.broadcast(reference, src=0)
            entropy_diff = float((entropy.detach() - reference).abs().max().item())
            entropy_diff_tensor = th.tensor([entropy_diff], dtype=th.float32)
            dist.all_reduce(entropy_diff_tensor, op=dist.ReduceOp.MAX)
            entropy_diff = float(entropy_diff_tensor.item())

        max_diff = max(actor_diff, critic_diff, target_diff, entropy_diff)
        if max_diff > 1e-6:
            raise AssertionError(f"Distributed parameters diverged: max_diff={max_diff}")

        checkpoint = Path(tempfile.gettempdir()) / "mcr_distributed_sac_gloo_smoke"
        context.barrier()
        if context.is_main:
            model.save(str(checkpoint))
        context.barrier()

        resumed = DistributedSAC.load(
            str(checkpoint) + ".zip",
            env=env,
            device="cpu",
            custom_objects={"batch_size": 8, "learning_starts": 8},
        )
        resumed.set_distributed_context(context)
        resumed.synchronize_parameters()
        resumed.learn(total_timesteps=8, reset_num_timesteps=False)
        resumed_diff = max(
            _max_parameter_difference(resumed.actor),
            _max_parameter_difference(resumed.critic),
            _max_parameter_difference(resumed.critic_target),
        )
        if resumed_diff > 1e-6:
            raise AssertionError(
                f"Resumed distributed parameters diverged: max_diff={resumed_diff}"
            )

        if context.is_main:
            loaded = SAC.load(str(checkpoint) + ".zip", device="cpu")
            observation, _ = env.reset(seed=123)
            action, _ = loaded.predict(observation, deterministic=True)
            print(
                "[PASS] two-rank Gloo gradients and distributed resume synchronized; "
                f"native SAC.load action_shape={action.shape} "
                f"max_diff={max(max_diff, resumed_diff):.3g}"
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
