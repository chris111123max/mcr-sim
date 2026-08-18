# Distributed SAC/PPO on 4 x Ascend 910B3

Target platform:

- Debian 12, aarch64, 88 CPU cores
- four process-visible logical devices: `npu:0` through `npu:3`
- Python 3.8.20, SOFA 21.12, Gymnasium 1.0.0, SB3 2.4.0
- a CANN-compatible `torch` and `torch_npu` pair

The code never uses physical `npu-smi` indices. `torchrun` assigns one logical
device to each rank through `LOCAL_RANK`.

## Architecture

For SAC, each rank owns an equal share of CPU SOFA environments and its own replay
buffer. On NPU the default buffer storage and minibatch gather remain on the
local accelerator; CPU/CUDA or an explicit fallback uses the standard SB3 CPU
buffer. Each rank samples a different local minibatch and runs SAC forward
and backward on its local NPU. Actor, critic, and automatic entropy gradients
are averaged before their optimizer steps. Initial actor, critic, target critic,
and entropy state are broadcast from rank 0. The target critic then stays equal
because every rank applies the same Polyak update to synchronized critic weights.
Training metrics stay on the NPU until the end of each update block.
During the SAC actor phase, critic parameters are temporarily frozen: gradients
still flow through Q to the action and actor, but unused critic parameter
gradients are not constructed. Standard optimizers clear gradients with
`set_to_none` to avoid redundant device writes; fused optimizers use their
required compatible clearing path. This handling is shared by PPO.

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
- `--training-curriculum` (default): keep domain randomization active while the
  geometry pool expands B01/B02 -> all B -> all B plus C01/C02 -> all B/C.
  Every promotion requires every active vessel to reach 10% success in three
  consecutive global epochs. A missing or under-threshold vessel resets the
  streak, so easier vessels cannot hide an unlearned C vessel. Per-vessel
  episode counts/rates, the stage, and the streak are synchronized, logged, and
  saved with the checkpoint. Forced vessels and validation bypass it.
- `--min-ent-coef`: lower bound for SAC automatic entropy tuning (default 0.02).
  PPO uses `ent_coef=0.001` and clamps Gaussian action std to `0.25..1.0`,
  preventing both premature exploration collapse and saturated random actions.
- `--npu-fused-adam` (default): replace SAC actor/critic Adam and the PPO policy
  Adam with `torch_npu.optim.NpuFusedAdam`, or the matching Ascend Apex class on
  older installations. A disposable optimizer step is tested first; all ranks
  retain standard Adam if any rank cannot enable it.
- `--npu-fast-execution` (default): request precompiled eager operators and the
  native ND matrix format through APIs provided by the installed torch_npu.
  Missing version-specific APIs are recorded and safely skipped.
- `--epochs` and `--episodes-per-epoch`: the primary global episode budget.
  The formal default is 100 x 100 = 10,000 completed episodes. All ranks participate
  in the episode counter and rank 0 saves one checkpoint per epoch.
- `--steps-per-epoch`: legacy transition-budget setting, used only when
  `--timesteps` is supplied.
- `--timesteps`: optional compatibility override for the global transition
  budget (SB3 may overshoot by one global vectorized step).
- `--buffer-size`: capacity of each rank-local replay buffer.
- `--npu-replay-buffer` (default on NPU): store each rank's replay tensors on
  its local NPU and gather sampled minibatches without repeated bulk CPU-to-NPU
  copies. Startup probes required indexing support; every rank collectively
  falls back to standard SB3 replay if any rank fails.
- `--learning-starts`: rank-local warm-up transitions.
- checkpoint, TensorBoard, progress bar, experiment directory, and final model
  writes are rank-0 only.

Every run directory contains three persistent output groups:

- `logs/`: `console.log`, per-rank console logs, the effective
  `run_config.json`, and train/validation CSV summaries;
- `tb/`: TensorBoard event files;
- `models/`: per-epoch, best-validation, and final model checkpoints.

`logs/run_config.json` is generated from the final effective arguments for each
run; it is not a hand-maintained template. A formal default run records
`epochs=100`, `episodes_per_epoch=100`, and the corresponding upper-bound
`timesteps=40960000`. Explicit CLI values still override these defaults.

The console capture is performed by the training process itself. The launchers'
managed `--nohup` mode additionally writes outer torchrun, HCCL, and native
runtime output to `logs/launcher.log` in the same timestamped run directory.
`run_config.json` and the `[MCR NPU]` startup line record the requested and
actually enabled optimizer/execution paths. Use `--no-npu-fused-adam` or
`--no-npu-fast-execution` to obtain an explicit standard-path comparison.

Episode outcomes and terminal statistics are accumulated locally and packed
into one collective every 256 vector steps instead of synchronizing every
environment step. SAC loss metrics are reduced every 64 train blocks. Each
rank also writes a `[PERF]` window every 64 vector steps with update time,
collective time/call count, and remaining rollout/IPC/callback time.

For the quality-oriented four-NPU default, `--n-envs 64 --batch-size 2048
--gradient-steps 4` means 16 SOFA environments and a 512-transition minibatch
on every rank. Gradient averaging makes the effective global minibatch 2048.
Each rollout collects 64 new transitions and processes `4 x 2048 = 8192` replay
samples. Compared with the 32-environment, 2048-batch, two-update setting, this
preserves both replay samples and optimizer updates per newly collected sample.
Learning rate, tau, gamma, and entropy settings remain unchanged.

## Launch

The normal launcher inserts `torchrun` automatically:

```bash
./training/sh/run_train_sac.sh \
  --nohup \
  --device npu \
  --distributed \
  --world-size 4 \
  --n-envs 64 \
  --batch-size 2048 \
  --gradient-steps 4 \
  --epochs 100 \
  --episodes-per-epoch 100 \
  --valid-min-train-success-rate 0.20 \
  --target-threshold 0.003 \
  --time-step 0.01 \
  --frame-skip 1 \
  --render headless \
  --exp-name test_4x910b3
```

Single-device CPU, CUDA, and NPU commands remain supported by omitting
`--distributed`. `--device auto` prefers CUDA, then Ascend NPU, then CPU.

The launcher-owned `--nohup` flag requires an explicit `--exp-name`, starts the
job in the background, and prints its PID and run directory. Do not combine it
with shell-level `nohup`, `&`, or output redirection. The equivalent PPO launcher
is `training/sh/run_train_ppo.sh` and implements the same managed mode. Both launchers
keep validation locked until a completed training epoch first reaches success rate
`0.20`. The gate stays unlocked; validation starts immediately when that epoch is
even, otherwise on the next even epoch, then runs every two epochs.
The ten
fixed-seed episode tasks are distributed 3/3/2/2 over four ranks, then gathered
through the active distributed backend. Rank 0 alone writes the unchanged CSV
summaries and checkpoints. `best_valid.zip` is replaced only when
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
