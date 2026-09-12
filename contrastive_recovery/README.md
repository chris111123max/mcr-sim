# Contrastive Goal-Conditioned RL + Recovery RL

This folder is a self-contained experiment and does not modify PPO, SAC, Goal-SAC, their replay buffers, or their launchers.

## Architecture

- **Task actor**: a fixed-history, direct 3-action policy (`rot_n`, `rot_b`, `insert`). It receives the existing local-centreline/target state plus achieved and final goal fractions. It never receives `target_route_id`. The ordered 32-step context is encoded with Linear/SiLU rather than GRU because the deployed torch-npu DynamicGRUV2 kernel fails on 910B3.
- **Contrastive critic**: samples a reachable future progress from the *same complete trajectory* as a positive and different-progress batch goals as negatives (InfoNCE). Near-identical scalar goals are masked to avoid contradictory false negatives. The actor's auxiliary term compares a sampled forward goal (at least 0.005 ahead) against current progress and squashes the score margin to [0, 1]. Its default weight is 0.05, so an ever-growing absolute contrastive logit cannot dominate task Q. The final destination remains the environment's true target, not a fabricated training success.
- **Task twin critic**: learns a Bellman value from measured route progress change, true target success, unsafe termination, and a small step cost. Its action gradient remains available while the contrastive critic lacks endpoint examples.
- **Risk critic**: predicts whether the following short horizon contains SDF warning, wrong-branch warning, out-of-vessel, or non-finite state.
- **Recovery actor + twin critic**: a second direct history-aware SAC actor, trained from clearance change and unsafe termination. Warning opens a short eligibility window, but the second actor controls only when its proposed action has at least 0.05 lower predicted risk than the main actor's action. The comparison is repeated each step; a previous handoff cannot force a riskier recovery action. Neither actor is replaced by a scripted steering action.

The recovery gate is deliberately disabled for the first 2,000 learner updates. A new risk critic is initially uncalibrated and would otherwise produce approximately 0.5 risk everywhere; after warm-up the default gate threshold is 0.65. The main actor's risk penalty starts at zero, then ramps to its default weight of 0.10 over the following 4,000 updates. The task actor uses a separate, smaller entropy coefficient (0.002) so its entropy bonus does not swamp per-step progress; Recovery SAC retains 0.05.

The current environment supplies the selected centreline geometry as local guide vectors and a final target. This experimental policy does **not** receive a discrete route/branch ID. The route ID is retained only in diagnostics for per-route failure analysis.

## Two-910B3 default

`run_train.sh` starts `torchrun --nproc_per_node=2`, HCCL/DDP, 32 total SOFA environments (16/rank), sequence batch 256 globally (128/rank), sequence length 32, and two learner updates per 32-env rollout. DDP performs native gradient bucketing; scalar learner diagnostics are all-reduced only every 50 updates, and a single episode counter is reduced every 32 vector steps. An epoch is genuinely 100 globally completed episodes by default.

## Logs

Each run contains `models/`, `tb/rank_*/`, `logs/console*.log`, `diagnostics/episode_events_rank_*.csv`, `diagnostics/update_metrics.csv`, `train_summary.csv`, and `run_config.json`. Episode CSVs include forward/backward insertion fractions, action magnitude, recovery and rejected-recovery fractions, actual SDF risk and terminal reason. `diagnostics/recovery_events_rank_*.csv` records each recovery onset; `diagnostics/terminal_trace_rank_*.csv` records the final 24 steps before every out-of-vessel event, including before/after SDF clearance, warning, both candidates' predicted risks, active policy, rejected recovery and action. These event files appear only when the corresponding event occurs. Update metrics include task critic loss/value, bounded relative contrastive signal and weight, and the current risk weight.

## Tests

Pure code test: `python contrastive_recovery/smoke_test.py`.

SOFA/HCCL smoke test is described in the main task response; it uses 32 total environments and two NPUs but a very short timestep budget.
