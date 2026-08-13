# Distributed SAC/PPO on 4 x Ascend 910B3

Target platform:

- Debian 12, aarch64, 88 CPU cores
- four process-visible logical devices: `npu:0` through `npu:3`
- Python 3.8.20, SOFA 21.12, Gymnasium 1.0.0, SB3 2.4.0
- a CANN-compatible `torch` and `torch_npu` pair

The code never uses physical `npu-smi` indices. `torchrun` assigns one logical
device to each rank through `LOCAL_RANK`.

## Architecture

For SAC, each rank owns an equal share of CPU SOFA environments and its own SB3 replay
buffer in CPU RAM. It samples a different local minibatch and runs SAC forward
and backward on its local NPU. Actor, critic, and automatic entropy gradients
are averaged before their optimizer steps. Initial actor, critic, target critic,
and entropy state are broadcast from rank 0. The target critic then stays equal
because every rank applies the same Polyak update to synchronized critic weights.
Training metrics stay on the NPU until the end of each update block.

This is intentionally implemented in a project-side `DistributedSAC` subclass.
Wrapping only `policy` in PyTorch DDP would not cover SB3 SAC's separate critic
and entropy optimizer paths.

PPO uses a separate project-side `DistributedPPO`. Each rank owns an on-policy
rollout buffer; policy gradients are averaged before PPO gradient clipping and
the optimizer step. PPO uses the same device-resident metric infrastructure as
SAC, and never uses the SAC replay buffer.

## CLI semantics

- `--n-envs`: global SOFA environment count; must divide by world size.
- `--batch-size`: global SAC batch; must divide by world size.
- `--gradient-steps`: synchronized optimizer updates per rollout. The four-NPU
  default is 4. `-1` follows SB3's rank-local collected-transition count and is
  deliberately not multiplied by world size.
- `--epochs` and `--episodes-per-epoch`: the primary global episode budget.
  The formal default is 50 x 100 = 5,000 completed episodes. All ranks participate
  in the episode counter and rank 0 saves one checkpoint per epoch.
- `--steps-per-epoch`: legacy transition-budget setting, used only when
  `--timesteps` is supplied.
- `--timesteps`: optional compatibility override for the global transition
  budget (SB3 may overshoot by one global vectorized step).
- `--buffer-size`: capacity of each rank-local CPU replay buffer.
- `--learning-starts`: rank-local warm-up transitions.
- checkpoint, TensorBoard, progress bar, experiment directory, and final model
  writes are rank-0 only.

Every run directory contains three persistent output groups:

- `logs/`: `console.log`, per-rank console logs, the effective
  `run_config.json`, and train/validation CSV summaries;
- `tb/`: TensorBoard event files;
- `models/`: per-epoch, best-validation, and final model checkpoints.

The console capture is performed by the training process itself, so these logs
remain available when the outer `nohup` output is redirected to `/dev/null`.

Episode outcomes and terminal statistics are accumulated locally and packed
into one collective every 256 vector steps instead of synchronizing every
environment step. SAC loss metrics are reduced every 64 train blocks. Each
rank also writes a `[PERF]` window every 64 vector steps with update time,
collective time/call count, and remaining rollout/IPC/callback time.

For the balanced four-NPU default, `--n-envs 32 --batch-size 1024
--gradient-steps 4` means eight SOFA environments and a 256-transition minibatch
on every rank. Gradient averaging makes the effective global minibatch 1024.
Each rollout collects 32 new transitions and processes `4 x 1024 = 4096` replay
samples, retaining the old single-NPU `128 replay samples / new transition`
ratio while cutting synchronized optimizer steps. A 2048 global batch with only
two updates is faster in principle, but changes SAC's optimizer and target-update
cadence more aggressively and is not the quality-first default.

## Launch

The normal launcher inserts `torchrun` automatically:

```bash
./training/sh/run_train_sac.sh \
  --device npu \
  --distributed \
  --world-size 4 \
  --n-envs 32 \
  --batch-size 1024 \
  --gradient-steps 4 \
  --epochs 50 \
  --episodes-per-epoch 100 \
  --target-threshold 0.003 \
  --time-step 0.01 \
  --frame-skip 1 \
  --render headless \
  --exp-name test_4x910b3
```

Single-device CPU, CUDA, and NPU commands remain supported by omitting
`--distributed`. `--device auto` prefers CUDA, then Ascend NPU, then CPU.

The equivalent PPO launcher is `training/sh/run_train_ppo.sh`. Both launchers
validate five unseen vessels twice each on rank 0 after epochs 2, 4, ..., 50.
Ranks 1-3 wait at barriers. `best_valid.zip` is replaced only when
`valid_success_rate` strictly increases.

## Checkpoint and resume

Only rank 0 writes the normal SB3 `.zip`. Distributed process state is excluded,
so the result is loadable using upstream `SAC.load()` for CPU, CUDA, or single-NPU
inference. During distributed resume, every rank loads the same `.zip`, rank 0
broadcasts model state, and synchronized training continues. The saved world
size is used to retain global timestep meaning when changing world size.

## Server smoke test

Do not start the full run first. Use four ranks, 32 environments, and
`--epochs 1 --episodes-per-epoch 2`. Confirm all rank/device
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
