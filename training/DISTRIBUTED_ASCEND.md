# Distributed SAC/PPO on 4 x Ascend 910B3

Target platform:

- Debian 12, aarch64, 88 CPU cores
- four process-visible logical devices: `npu:0` through `npu:3`
- Python 3.8.20, SOFA 21.12, Gymnasium 1.0.0, SB3 2.4.0
- SB3-Contrib 2.4.x for LSTM-PPO only; SAC/MLP-PPO do not import it
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

LSTM-PPO uses official SB3-Contrib 2.4 `RecurrentPPO`, `MlpLstmPolicy`,
`RecurrentRolloutBuffer`, and `RNNStates`. The project-side
`DistributedRecurrentPPO` changes only the optimizer step to reuse the same
gradient averaging, clipping, fused Adam, and metric reduction as MLP-PPO.
Actor and critic each have a one-layer, 128-unit, unidirectional LSTM. Official
`episode_starts` reset only the completed environment's state; sequence padding
and masks prevent padded samples, environment boundaries, and episode
boundaries from entering the PPO loss.

## CLI semantics

- `--n-envs`: global SOFA environment count; must divide by world size.
- `--batch-size`: global SAC/PPO optimizer minibatch; it must divide by world
  size and is not changed to accommodate recurrent padding.
- `--gradient-steps`: synchronized optimizer updates per rollout. The four-NPU
  default is 1. `-1` follows SB3's rank-local collected-transition count and is
  deliberately not multiplied by world size.
- `--training-curriculum` (default): use four complete-route stages:
  `simple_fixed` (B01/B02/C01/C02, no DR), `simple_full_dr` (the same four with
  the complete requested DR envelope), `all_fixed` (all ten vessels, no DR), and
  `all_full_dr` (all ten with complete DR). Each active vessel keeps a 100-episode
  rolling 3 mm success window. Promotion is eligible only after every active
  vessel has 100 samples and at least 50% rolling success for three consecutive
  global epochs. A missing,
  under-sampled, or under-threshold vessel resets the streak, so easier vessels
  cannot hide an unlearned C vessel. Per-vessel epoch and rolling counts/rates,
  the stage, and the streak are synchronized, logged, and saved with the
  checkpoint. Forced vessels and validation bypass it. DR is intentionally reset
  to zero when the six harder vessels are first introduced.
  Sampling mixes 50% uniform probability with 50% squared failure-rate weight
  and caps one vessel at twice its uniform probability. Stage transitions clear
  old outcome windows. Validation remains locked until the final all-vessel/full-DR stage.
- `--min-ent-coef`: lower bound for SAC automatic entropy tuning (0.02).
  PPO and recurrent PPO retain the stable action-std range `0.25..1.0`.
- `--npu-fused-adam` (default): replace SAC actor/critic Adam and the PPO policy
  Adam with `torch_npu.optim.NpuFusedAdam`, or the matching Ascend Apex class on
  older installations. A disposable optimizer step is tested first; all ranks
  retain standard Adam if any rank cannot enable it.
- `--npu-fast-execution` (default): request precompiled eager operators and the
  native ND matrix format through APIs provided by the installed torch_npu.
  Missing version-specific APIs are recorded and safely skipped.
- `--epochs` and `--episodes-per-epoch`: the primary global episode budget.
  The formal default is 100 x 100 = 10,000 completed episodes. All ranks
  participate in this global count, and rank 0 saves one checkpoint per epoch.
  Equal epoch counts do not imply equal transition counts across PPO and
  LSTM-PPO because their episode lengths can differ; use the logged
  `global_env_steps` when making sample-efficiency comparisons.
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

For the quality-oriented four-NPU SAC default, `--n-envs 64 --batch-size 1024
--gradient-steps 1` means 16 SOFA environments and a 256-transition minibatch
on every rank. Gradient averaging makes the effective global minibatch 1024.
Each vector step collects 64 new transitions and processes 1024 replay samples,
reducing the replay-sample/new-transition ratio from 128 to 16. Learning rate
and tau remain unchanged; gamma stays at the stable 0.995. PPO and recurrent
PPO use `n_steps=256`, so a global policy update begins after 16,384 fresh
transitions rather than 65,536 while preserving the same optimizer work per
65,536 environment steps.

## Launch

The normal launcher inserts `torchrun` automatically:

```bash
./training/sh/run_train_sac.sh \
  --nohup \
  --device npu \
  --distributed \
  --world-size 4 \
  --n-envs 64 \
  --batch-size 1024 \
  --gradient-steps 1 \
  --epochs 200 \
  --episodes-per-epoch 100 \
  --valid-min-train-success-rate 0.20 \
  --target-threshold 0.003 \
  --time-step 0.01 \
  --frame-skip 1 \
  --render headless \
  --exp-name sac_allvessel_targetdr_200ep
```

Single-device CPU, CUDA, and NPU commands remain supported by omitting
`--distributed`. `--device auto` prefers CUDA, then Ascend NPU, then CPU.

The launcher-owned `--nohup` flag requires an explicit `--exp-name`, starts the
job in the background, and prints its PID and run directory. Do not combine it
with shell-level `nohup`, `&`, or output redirection. The equivalent PPO launcher
is `training/sh/run_train_ppo.sh`; recurrent PPO uses
`training/sh/run_train_lstm_ppo.sh`. All three implement the same managed mode. Both PPO launchers
keep validation locked until the final `all_full_dr` stage reaches a completed
training-epoch success rate of `0.20`. The gate stays unlocked; validation starts immediately when that epoch is
even, otherwise on the next even epoch, then runs every two epochs.
The ten fixed-seed episode tasks are distributed 3/3/2/2 over four ranks, then
gathered through the active distributed backend. Rank 0 alone writes CSV
summaries and checkpoints. Validation CSVs include continuous route completion, route
potential, and final/minimum target distance. `best_valid.zip` uses success rate
as the primary criterion; ties are resolved by route completion, route potential,
then smaller final target distance. Thus a 0%-success validation phase can still
retain the checkpoint with the strongest measurable progress.

Reward V8 exposes the same 78-dimensional observation to SAC, PPO, and
LSTM-PPO. In addition to tip SDF probes, it includes whole-body minimum surface
clearance, outside-confirmation progress, the worst shaft point and inward
direction, moving route guidance 10/20 mm ahead, plus selected-route tangents
5/10/20 mm ahead. Continuous progress uses recurrent local projection with a
physical step gate, so adjacent U-turn arms cannot create an arc-length jump. The existing wall
proximity and penetration reward components take the maximum of tip and
whole-body risk, so unsafe shaft contact is visible before the terminal
whole-body SDF check fires. Every actor vector is expressed in the catheter-tip
frame. Absolute XYZ and route completion percentage are absent; the single route
horizon scalar is remaining centerline distance normalized by the longest training
route. Navigation semantics changed even though the state remains 78-dimensional.
Reward V8 also removes no-progress/wrong-branch
termination and rebalances the reward scale, so older checkpoints must not be resumed.

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
- LSTM operator forward/backward support on the installed torch_npu/CANN pair
- recurrent state, padding/mask, and checkpoint round-trip under four-rank HCCL
- throughput benchmark for 16, 32, 48, and 64 total environments
