# Recurrent Goal-TQC（独立实验）

此目录是一条与现有 PPO、SAC、Goal-SAC 和 contrastive-recovery 分离的实验路线，不改动它们的训练入口或模型。默认先在固定的 `B01 / target_01_centerline.vtk` 上做门槛实验，不直接混合多个目标路线。策略只接收原环境的观测和目标进度条件；`target_route_id` 只用于 replay 校验，不作为策略输入。

## 模块

- `train.py`：SOFA 并行环境、两卡 HCCL/DDP、训练、分开的日志与 checkpoint。
- `agent.py`：显式 GRU 单元的循环 actor、3 个分位数 critic 的 TQC、辅助风险 critic。风险 critic 只提供可微的软代价，不覆盖 RL 动作。显式 GRU 是为了避开当前 torch-npu 2.2 在 `nn.GRU` 上报出的 DynamicGRUV2 问题。
- `replay.py`：按完整 episode 保存序列，只从同一血管、同一目标路线、连续安全的未来片段选择 HER 目标；碰撞、投影跳变或错误分支不能重标记为成功。原始环境奖励不进入此实验的训练，实验使用统一的目标条件奖励。
- `evaluate.py`：固定路线上的确定性并行评估，逐 episode 保存结果。
- `smoke_test.py`：不启动 SOFA 的 CPU 结构测试。

## 先运行门槛测试

在服务器项目的 `python/` 目录中：

```bash
python recurrent_goal_tqc/smoke_test.py

bash recurrent_goal_tqc/run_train.sh \
  --nohup \
  --device npu \
  --distributed --world-size 2 \
  --n-envs 32 \
  --epochs 5 --episodes-per-epoch 100 \
  --force-model B01 \
  --centerline-file target_01_centerline.vtk \
  --exp-name rgtqc_b01_route01_pilot_2npu_32env
```

默认全局序列 batch 为 256（每卡 128），序列长 16；`learning_starts=20000`，每 32 个 vector steps 同步一次训练就绪与 episode 计数。5 个 epoch 是 pilot，不是成功率保证。只有在固定路线能产生安全前进和非零成功率后，才考虑增加 B01 的其他目标路线、其他血管和域随机化；目前没有自动课程升级，以免未经验证就扩大任务。

当前目标条件奖励按路线进度比例计：安全推进 `20 × Δprogress`，真实或 HER 目标完成 `+20`，出血管或非有限状态 `-25`，每步 `-0.002`。事故步不拿正向进度奖励，因此即使在 99% 路程出血管，整条轨迹的正向进度与终止惩罚之和仍为负；风险 critic 另以较小软代价影响 actor。

单独评估 checkpoint，例如：

```bash
bash recurrent_goal_tqc/run_train.sh --eval \
  --device npu --n-envs 32 \
  --checkpoint /path/to/rgtqc_epoch_005_episodes_00500.pt \
  --force-model B01 --centerline-file target_01_centerline.vtk \
  --eval-episodes 64
```

每次训练写入独立 `training_runs/<exp-name>_<timestamp>/`；`logs/launcher.log` 保存启动失败，`logs/console*.log` 保存各 rank，`diagnostics/episodes_rank_*.csv` 保存每条轨迹，`diagnostics/updates.csv` 保存损失、HER 比例和风险预测，`train_summary.csv`、`tb/`、`models/` 分别保存摘要、TensorBoard 与模型。训练日志中的窗口成功率明确是 rank 0 最近 100 条，不应误读为全局成功率；离线评估以 `evaluate.py` 的结果为准。

## 当前边界

本地工作区不能运行 SOFA 或两卡 Ascend；必须在服务器上先做结构测试、pilot 和固定种子的离线评估。HER 的“拓扑安全”依赖环境 `chosen_model`、`target_route_id`、路线投影与 SDF 指标正确，不会修复原始目标路线或物理控制错误。checkpoint 保存模型与优化器；replay 未随 checkpoint 保存，因此当前版本不支持严格无损的中断续训。不要把这条实验的 pilot 结果与已经训练数百 epoch 的 PPO 直接比较。
