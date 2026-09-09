# 非 ROS 版训练参数基线

本文件记录 B01..B05、C01..C05 人工血管对应的训练参数。唯一配置源是
`mcr_sim/training_config.py`；训练脚本、SOFA 场景和 Web 预检均从该文件取默认值。

例外：独立 Goal-SAC 使用 `mcr_sim/goal_sac_config.py`；v3 的双 Q、47 维目标观测、奖励与启动说明见 [GOAL_SAC_V3.md](GOAL_SAC_V3.md)，不使用下文 baseline reward 默认值。

## 血管与导管尺寸包络

生成器中的长度单位转换为 SOFA 米制单位后，十条血管的有效范围为：

| 项目 | 数值 | 作用 |
|---|---:|---|
| B 系列目标路径长度 | 160.9–239.3 mm | 多分叉任务的导航距离 |
| C 系列路径长度 | 287.5–494.6 mm | 多弯道任务的导航距离 |
| 血管内半径 | 2.8–5.2 mm | 决定导管中心允许的安全空间 |
| 导管外径/半径 | 1.33/0.665 mm | 导管 Line/Point 碰撞半径 |
| 血管缩放 | 0.90–1.00 | 每回合只缩小、不放大全部几何与中心线半径 |
| 最小缩放后中心净空 | 1.855 mm | `2.8×0.90−0.665`，用于校验阈值和动作步长 |

## 环境和任务

| 参数 | 默认值 | 功能与理由 |
|---|---:|---|
| `time_step` | 0.01 s | 每次 SOFA 物理推进的时间；场景和 Gym 环境保持一致 |
| `frame_skip` | 1 | 每个 RL step 推进一次物理求解，避免一次动作跨过碰撞面 |
| 最大单步插入 | 0.4 mm | 保持每个 RL step 一次碰撞求解，同时提高采样吞吐 |
| `max_episode_steps` | 2048 | 最大纯插入预算 819.2 mm，覆盖最长 C05 并留出转向/回撤余量 |
| 目标成功阈值 | 3 mm | 还要求连续路线进度进入目标前 5 mm 且导管处于血管内 |
| 动态路线引导 | 前方 10/30 mm | 连续移动，不要求命中离散 waypoint 球 |
| 局部磁场转向 | 最大 3°/step | 提高连续长弯中的响应能力，同时保留动作变化率限制 |
| 起点/目标窗口 | 各 10 mm 弧长 | 固定物理距离，不受 B/C 原始中心线采样密度影响 |
| 初始方向扰动 | 最大 10° | 提供方向泛化，避免 20° 在最窄分叉入口产生过大横向偏差 |
| 血管缩放 | 均匀采样 0.90–1.00 | 域随机化只提高难度，不制造比母版更宽、更长的简单任务 |

2048 是最大步数，不要求每个回合都运行满：到达目标或确认出血管会提前结束；
非有限状态作为仿真异常紧急结束。偏离目标分支只产生连续风险代价，允许策略恢复；
长期无进展会产生有上限的逐步代价，但不额外终止。

## 奖励

Reward profile v13.0 使用选定路线的归一化完成度增量：
`60×(completion_next-completion_previous)`。终止时不再清零已经获得的真实净进度；
回撤仍产生等量负增量，因此往返振荡不能制造进度收益。
首次定位允许搜索完整目标路线；之后只在上一进度前后
20/40 mm 内投影，并拒绝超过 2 mm 单步物理进度的候选。空间相邻的 180° 回头弯或
其他分支因此不能制造进度跳变。往返振荡净奖励为 0，必要回退后再次前进会恢复奖励。

| 奖励项 | 权重 | 含义 |
|---|---:|---|
| 连续路线进度 | 完整路线最多 +60 | `60×completion_delta`；回撤为负 |
| tip/整段导管接近越界 | -0.005×risk | tip 持续贴壁和 whole-body SDF 警告取最大值，risk 限制在 0..1 |
| 偏离目标分支 | -0.005×risk | 选定路线相对完整中心线图的距离差，risk 限制在 0..1 |
| 最终成功 | +100 | 进入 3 mm 目标、路线剩余小于 5 mm，且导管在血管内 |
| 出血管 | -80 | 整段导管中心超出 SDF 管壁 0.5 mm，连续 3 步时终止 |
| 非有限状态 | -100 | observation/reward 出现非有限值时紧急终止 |
| 超时 | -80 | 2048 步仍未完成 |
| 每步代价 | -0.002 | 小型时间成本 |
| 持续停滞 | -0.02×risk | 64 步宽限后按最近 32 步净进度计算；只扣分、不终止 |

插入通道采用保持零点的非对称分段映射：策略原始动作 `-1/0/+1` 分别映射到物理动作
`-0.25/0/+1`。正向保持完整插入力，负向为低速回撤；零动作不会暗中推动导管前进。

SAC 在跨 rank 梯度平均之后统一使用 `max_grad_norm=10`；PPO 使用
`max_grad_norm=0.5` 和 `ent_coef=0.0005`。PPO/LSTM-PPO 的动作标准差初始值为
`0.50`、下限 `0.10`、上限 `0.60`；SAC 自动熵系数下限为 `0.02`。
`run_config.json` 会完整保存 Reward v13.0、实际 observation shape/dtype，`train_summary.csv`
同时记录四类奖励分项（progress/terminal/safety/step）、无进展/错误分支诊断事件、正回报失败率、终止路线势、
中心线跳变拒绝次数、课程阶段、当前阶段每根血管的回合数与成功率、无进展次数、
正负插入比例和最终插入长度。每个新 run 还会生成 `train_episodes.csv`，逐回合记录
血管、终止原因、路线完成度、目标距离、插入长度、最大 SDF 穿透、路线进度和投影跳变次数；
该文件只用于诊断，不参与 observation、reward 或梯度更新。
正式训练中 `positive_failure_rate` 必须保持为 0。

`train_episodes.csv` 还会逐回合保存 progress、wall、branch、stagnation、step
和 terminal 奖励分量，以及由路线起终弧长和完成度重建的累计路线位置。环境仅在内存中
保留最近 64 步诊断，训练回调按血管和终止原因每 20 回合抽样一次，写入
`diagnostics/terminal_traces/rank_<rank>_part_<part>.jsonl`，每 100 条轨迹自动换文件。
`diagnostics/safety_summary.csv` 保存逐 epoch 安全趋势，`vessel_summary.csv` 保存逐血管趋势，
`failure_episodes.csv` 只保存失败回合。这些数据不进入 observation 或 PPO 更新。

可使用 `training/py/evaluate_train_policy.py` 对训练 checkpoint 进行固定 B01/B02 的
deterministic/stochastic 对照评估。结果写入 run 的 `diagnostics/`，独立于 valid 解锁条件。

B01..B05 和 C01..C05 全部使用 `vessel_sdf.vti` 直接判断管壁关系。
每步从导管尖端向入口遍历已插入的导管段，并按不大于半个 VTI
网格的间距加密采样。body 可以接触和依靠管壁滑动；whole-body 警告从导管表面
距管壁 0.5 mm 时开始线性启用，并与连续 3 步越界终止使用同一 SDF 状态。
确认越界由终止惩罚处理，不再叠加第二个穿透 shaping 项。tip 净空仍用于贴壁风险和成功质量统计；
旧中心线安全比只作为缺少 VTI 的旧血管兼容后备，不参与这十条训练血管的判定。

Observation V12 为 45 维：38 维当前局部状态加最近一个 7 维动作-响应元组。
当前状态包含磁场、前方 10/30 mm 引导、中心线纠偏向量、剩余路程、剩余时间、插入长度，
以及 tip/whole-body SDF 风险、最危险导管段、tip 后方 10/30/60 mm 三个局部形状采样点、
前方 5/15/30 mm 路线切向等。删除重复的弯曲标量、接触位、前向 SDF 探针、近远两套
重复引导和四帧混合局部坐标历史。PPO、LSTM-PPO、SAC 使用完全相同的状态；所有空间
向量位于当前导管尖端局部坐标系，策略不接收绝对 XYZ 或 route completion percentage。
本次明确不兼容旧 checkpoint，旧 78 维模型不能载入 52 维环境继续训练。

## 碰撞模型

训练和 Web 预检均采用同一套模型：血管只启用静态
`TriangleCollisionModel`，导管启用 `LineCollisionModel + PointCollisionModel`。
视觉壁面与碰撞网格分离，因此可以显示平滑血管而不增加训练碰撞三角形。

| 参数 | 默认值 | 功能 |
|---|---:|---|
| 血管三角形 proximity | 0.2 mm | 给低面数碰撞网格保留很小检测裕量 |
| 导管 Line/Point proximity | 0.665 mm | 用中心线碰撞元表达实际导管半径 |
| LocalMinDistance contact | 0.2 mm | 接触响应距离 |
| LocalMinDistance alarm | 1.0 mm | 窄相检测预警范围；必须大于 contact |
| LCP 摩擦系数 | 0.01 | 保留低摩擦导管-血管接触 |
| LCP tolerance/maxIt | `1e-6` / `20000` | 保留高精度约束求解上限，避免因减面而放松求解 |

不启用血管 Line/Point 碰撞：它会显著增加接触对，既拖慢训练，也曾导致 LCP
非有限值。碰撞网格减面仍需单独保证封闭、法向一致、无自交；上述数值不能修复坏网格。

## SAC 与 epoch 语义

一个 epoch 定义为 **100 个全局完成回合**。正式 SAC/PPO 实验默认训练 100 epoch，即 10000 个回合。
不同算法即使 epoch 数相同，回合长度也可能不同，因此总 transition 数并不相同；比较
MLP-PPO、LSTM-PPO 与其他策略的样本效率时，应以 `global_env_steps` 对齐或至少同时报告，
不能只比较 epoch。
训练默认启用五阶段完整路线课程：

1. `branch_fixed`：仅 B01、B02，固定母版几何；
2. `curved_fixed`：仅 C01、C02，固定母版几何，专项学习持续弯曲控制；
3. `simple_full_dr`：B01、B02、C01、C02，启用完整域随机化；
4. `all_fixed`：全部十条血管，重新关闭域随机化；
5. `all_full_dr`：全部十条血管，重新启用完整域随机化。

每根当前活跃血管保存最近 100 个 episode 的滚动结果。第一阶段要求 B01/B02 都有完整
100 个样本，合并滚动 3 mm 成功率达到 90%；后续阶段要求每根活跃血管达到 50%。
达到阶段成功率门槛后，最近连续 3 个完成的 episode 都成功即可在下一 epoch 安全边界升级，
不再要求连续 3 个 epoch。缺少样本或未达标都会清零连续 episode 计数。阶段升级时所有旧窗口清空，固定几何成绩不能证明随机化任务已掌握，
简单血管成绩也不能替代新增困难血管成绩。阶段、名称、连续计数、滚动结果和逐血管
成功率都保存在 checkpoint，只前进不回退；强制单血管和 V01..V05 validation 不受影响。
当前池内采样由
50% 均匀分布和 50% 平方失败率权重混合，且单根血管概率不超过均匀概率的 2 倍，
困难血管获得更多回合，但不会造成对其他血管的灾难性遗忘。只有完整路线阶段的训练
还不够：必须已经在最终 `all_full_dr` 阶段，且训练成功率达到 20%，才能解锁
V01–V05 validation。
四卡分布式训练时，所有 rank 同步累计回合数，默认每 epoch 保存一次 checkpoint。
`--steps-per-epoch` 仅保留给旧的 transition-budget 命令；传入 `--timesteps` 时启用旧模式。
最终阶段训练成功率第一次达到 `0.20` 前不创建 valid 环境；达到后永久解锁，该轮若为偶数
就立即验证，否则从下一个偶数 epoch 开始每 2 个 epoch 在 `mesh/valid` 的 5 条 unseen
血管上各运行 2 个确定性回合。四卡将
10 个固定种子验证任务按 3/3/2/2 并行执行，再由 rank 0 汇总；验证覆盖、CSV 与
`best_valid.zip` 以成功率为第一排序；成功率相同时依次比较 route completion、route
potential 和更小的 final distance，避免所有验证成功率都是 0 时永久保留第一次模型。
`valid_episodes.csv` 同步记录这些指标。SAC 在 NPU 上默认使用每 rank accelerator-resident
replay buffer，避免每次更新重复搬运大批量 observation；不支持时四个 rank 一致回退。

| 参数 | 默认值 | 说明 |
|---|---:|---|
| epoch / episodes per epoch | 100 / 100 | 正式训练预算与保存周期 |
| 全局环境数 | 64 | 四卡时每卡 16 个 SOFA 环境 |
| 全局/local batch（四卡） | 1024 / 256 | 每卡独立采样，梯度同步后等效全局 1024 |
| replay buffer | 每卡 500000 | NPU 默认驻留本卡；CPU/CUDA 或探针失败时使用标准 SB3 buffer |
| learning starts | 每卡 50000 | 先收集较多、多血管经验再更新 |
| gradient steps | 1 | 每个 vector step 执行一次同步更新，降低 SAC critic 的 replay 过拟合风险 |
| gamma / tau / lr | 0.995 / 0.005 / 3e-4 | 恢复已验证设置，避免初期失败终止项压倒稠密进度信号 |

推荐四卡启动方式：

```bash
bash training/sh/run_train_sac.sh \
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
  --render headless \
  --exp-name sac_allvessel_targetdr_200ep
```

`--nohup` 由启动脚本处理，不会传入 Python。它自动创建带时间戳的统一运行目录，
在 `logs/launcher.log` 保存 nohup、torchrun、HCCL 和原生运行时输出，同时保留
`console*.log`、CSV、`tb/` 和 `models/`。使用该模式必须显式提供 `--exp-name`；
脚本启动后会直接打印后台 PID、运行目录和 launcher 日志路径，不需要再手写
`nohup`、`&` 或输出重定向。

`logs/run_config.json` 不是固定模板，而是在每次运行时根据最终生效参数自动生成。
正式默认运行会记录 `"epochs": 100`、`"episodes_per_epoch": 100` 和上限
`"timesteps": 40960000`；命令行显式传入的值仍会覆盖默认值。

旧的 `--timesteps` 仍可使用，并会切换到 transition-budget 模式，以兼容已有
启动命令。第一次上服务器应先运行短 smoke test，例如
`--epochs 1 --episodes-per-epoch 2`，确认 SOFA、HCCL、日志和 checkpoint 后再开始
完整训练。

## LSTM-PPO 对照实验

LSTM-PPO 使用官方 SB3-Contrib 2.4 的 `RecurrentPPO`、`MlpLstmPolicy`、
`RecurrentRolloutBuffer` 和 `RNNStates`，分布式更新仍复用项目的 HCCL
梯度平均。现有 MLP-PPO 的 reward、环境、课程、验证和所有共同 PPO 参数不变。

| 参数 | MLP-PPO | LSTM-PPO |
|---|---:|---:|
| policy | `MlpPolicy` | `MlpLstmPolicy` |
| learning rate | 3e-4 | 3e-4 |
| n_steps | 256 | 256 |
| global batch | 1024 | 1024 |
| PPO n_epochs | 10 | 10 |
| gamma / GAE lambda | 0.9995 / 0.98 | 0.9995 / 0.98 |
| clip / entropy / value coef | 0.2 / 0.0005 / 0.5 | 0.2 / 0.0005 / 0.5 |
| max grad norm | 0.5 | 0.5 |
| action std initial/bounds | 0.50 / 0.10–0.60 | 0.50 / 0.10–0.60 |
| LSTM hidden/layers | N/A | 128 / 1 |
| bidirectional | N/A | false |

训练时每个 rank 的 actor 与 critic state 分别为
`(n_lstm_layers, envs_per_rank, hidden_size)`；正式四卡 64 环境配置下即
`(1, 16, 128)`。rollout buffer 另外保存每一步的 actor/critic hidden 和 cell
state，形状为 `(256, 1, 16, 128)`。minibatch 参数仍是每 rank 128 个真实
transition；官方 buffer 内部产生的 padding 只通过 mask 排除，不会静默改写
`batch_size`。
