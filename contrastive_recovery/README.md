# Contrastive Goal-Conditioned RL + Recovery RL

This folder is a self-contained experiment and does not modify PPO, SAC, Goal-SAC, their replay buffers, or their launchers.

## Architecture

- **Task actor**: a fixed-history, direct 3-action policy (`rot_n`, `rot_b`, `insert`). It receives the existing local-centreline/target state plus achieved and final goal fractions. It never receives `target_route_id`. The ordered 32-step context is encoded with Linear/SiLU rather than GRU because the deployed torch-npu DynamicGRUV2 kernel fails on 910B3.
- **Contrastive critic**: samples a reachable future progress from the *same complete trajectory* as a positive and every other batch goal as a negative (InfoNCE). The task actor maximizes reachability and minimizes learned risk instead of relying on a large hand-shaped dense reward.
- **Risk critic**: predicts whether the following short horizon contains SDF warning, wrong-branch warning, out-of-vessel, or non-finite state.
- **Recovery actor + twin critic**: a second direct history-aware SAC actor, trained from clearance change and unsafe termination. It is gated briefly by measured SDF warning or learned risk; it is not a scripted controller and does not write a nominal action over the task actor.

The recovery gate is deliberately disabled for the first 2,000 learner updates. A new risk critic is initially uncalibrated and would otherwise produce approximately 0.5 risk everywhere; after warm-up the default gate threshold is 0.65.

The current environment supplies the selected centreline geometry as local guide vectors and a final target. This experimental policy does **not** receive a discrete route/branch ID. The route ID is retained only in diagnostics for per-route failure analysis.

## Two-910B3 default

`run_train.sh` starts `torchrun --nproc_per_node=2`, HCCL/DDP, 32 total SOFA environments (16/rank), sequence batch 256 globally (128/rank), sequence length 32, and two learner updates per 32-env rollout. DDP performs native gradient bucketing; scalar learner diagnostics are all-reduced only every 50 updates, and a single episode counter is reduced every 32 vector steps. An epoch is genuinely 100 globally completed episodes by default.

## Logs

Each run contains `models/`, `tb/rank_*/`, `logs/console*.log`, `diagnostics/episode_events_rank_*.csv`, `diagnostics/update_metrics.csv`, `train_summary.csv`, and `run_config.json`.

## Tests

Pure code test: `python contrastive_recovery/smoke_test.py`.

SOFA/HCCL smoke test is described in the main task response; it uses 32 total environments and two NPUs but a very short timestep budget.
