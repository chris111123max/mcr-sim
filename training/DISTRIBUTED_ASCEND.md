# Distributed SAC on 4 x Ascend 910B3

Target platform:

- Debian 12, aarch64, 88 CPU cores
- four process-visible logical devices: `npu:0` through `npu:3`
- Python 3.8.20, SOFA 21.12, Gymnasium 1.0.0, SB3 2.4.0
- a CANN-compatible `torch` and `torch_npu` pair

The code never uses physical `npu-smi` indices. `torchrun` assigns one logical
device to each rank through `LOCAL_RANK`.

## Architecture

Each rank owns an equal share of CPU SOFA environments and its own SB3 replay
buffer in CPU RAM. It samples a different local minibatch and runs SAC forward
and backward on its local NPU. Actor, critic, and automatic entropy gradients
are averaged before their optimizer steps. Initial actor, critic, target critic,
and entropy state are broadcast from rank 0. The target critic then stays equal
because every rank applies the same Polyak update to synchronized critic weights.

This is intentionally implemented in a project-side `DistributedSAC` subclass.
Wrapping only `policy` in PyTorch DDP would not cover SB3 SAC's separate critic
and entropy optimizer paths.

## CLI semantics

- `--n-envs`: global SOFA environment count; must divide by world size.
- `--batch-size`: global SAC batch; must divide by world size.
- `--timesteps`: global transition budget (SB3 may overshoot by one global
  vectorized step, as it does in single-process vector environments).
- `--buffer-size`: capacity of each rank-local CPU replay buffer.
- `--learning-starts`: rank-local warm-up transitions.
- checkpoint, TensorBoard, progress bar, experiment directory, and final model
  writes are rank-0 only.

For four ranks, `--n-envs 64 --batch-size 512` means 16 SOFA environments and
a 128-transition minibatch on every rank. Gradient averaging makes the effective
global minibatch 512.

## Launch

The normal launcher inserts `torchrun` automatically:

```bash
./training/sh/run_train_sac.sh \
  --device npu \
  --distributed \
  --world-size 4 \
  --n-envs 64 \
  --batch-size 512 \
  --force-model 0237 \
  --timesteps 5000000 \
  --target-threshold 0.003 \
  --time-step 0.01 \
  --frame-skip 1 \
  --render headless \
  --exp-name test_4x910b3
```

Single-device CPU, CUDA, and NPU commands remain supported by omitting
`--distributed`. `--device auto` prefers CUDA, then Ascend NPU, then CPU.

## Checkpoint and resume

Only rank 0 writes the normal SB3 `.zip`. Distributed process state is excluded,
so the result is loadable using upstream `SAC.load()` for CPU, CUDA, or single-NPU
inference. During distributed resume, every rank loads the same `.zip`, rank 0
broadcasts model state, and synchronized training continues. The saved world
size is used to retain global timestep meaning when changing world size.

## Server smoke test

Do not start the full five-million-step run first. Use four ranks, a smaller
environment count, and 2,000-5,000 global timesteps. Confirm all rank/device
lines appear, HCCL initializes, training losses advance, and exactly one
checkpoint/TensorBoard run is written.

The repository includes a backend-independent CPU test:

```bash
torchrun --standalone --nproc_per_node=2 \
  testing/py/test_distributed_sac_smoke.py
```

## NEEDS ASCEND SERVER VALIDATION

- import and availability of the installed CANN-matched `torch_npu`
- HCCL initialization across `npu:0` through `npu:3`
- SB3/PyTorch operations used by SAC on Ascend
- four-rank SOFA subprocess stability and CPU/memory sizing
- checkpoint save and distributed resume on the shared filesystem
- throughput benchmark for 16, 32, 48, and 64 total environments
