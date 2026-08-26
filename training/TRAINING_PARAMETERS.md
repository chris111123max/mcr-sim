# 非 ROS 版训练参数基线

本文件记录 B01..B05、C01..C05 人工血管对应的训练参数。唯一配置源是
`mcr_sim/training_config.py`；训练脚本、SOFA 场景和 Web 预检均从该文件取默认值。

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
| 最大单步插入 | 0.2 mm | 约为最小中心净空的 1/9，兼顾碰撞稳定与训练速度 |
| `max_episode_steps` | 4096 | 最大纯插入预算 819.2 mm，覆盖最长 C05 并留出转向/回撤余量 |
| 目标成功阈值 | 3 mm | 最终目标比中间 waypoint 略宽松；成功仍要求导管处于血管内 |
| waypoint 间距/阈值 | 5/2 mm | 给长血管提供密集引导，同时要求真实接近 waypoint |
| 起点/目标窗口 | 各 10 mm 弧长 | 固定物理距离，不受 B/C 原始中心线采样密度影响 |
| 初始方向扰动 | 最大 10° | 提供方向泛化，避免 20° 在最窄分叉入口产生过大横向偏差 |
| 血管缩放 | 均匀采样 0.90–1.00 | 域随机化只提高难度，不制造比母版更宽、更长的简单任务 |

4096 是整数，也选为 2 的幂，便于配置和日志识别。它不是要求每个回合都运行
4096 步：到达目标、出血管或出现非有限状态时仍会提前结束。

## 奖励

Reward profile v6 使用严格有序 waypoint 的路线势函数 `Phi in [0,1]`，连续进度项为
`100 × (Phi_t-Phi_{t-1})`。它在整条轨迹上自动望远镜求和：往返振荡净奖励为 0，
但大弯道中必要回退后再次前进会恢复奖励，不再被一次性正向额度永久截断。全部
有序 waypoint 奖励仍自然封顶为 `+20`；handoff 不奖励。任何终止失败即使发生在
目标前也保持负回报，数值继续控制在 `O(1)..O(100)`。

| 奖励项 | 权重 | 含义 |
|---|---:|---|
| waypoint/目标连续接近 | 净额范围 -100..+100 | `100 × ΔPhi`；远离对称扣分，纠偏后可恢复，振荡净值为零 |
| 到达中间 waypoint | 总额最多 +20 | 在全部有序 waypoint 之间平均分配，handoff 不奖励 |
| tip/整段导管持续接近越界 | -0.02 | tip 贴壁与 whole-body 中心距越界阈值 0.5 mm 内取较大风险；安全插入仍为正 |
| tip 穿壁/整段中心越界深度 | -0.50 | tip 穿入深度与 whole-body 中心越过管壁深度取较大归一化值 |
| 偏离目标分支 | -0.10 | 选定路径与完整中心线图的距离差形成连续扣分 |
| 请求回撤 | -0.005 | 按负插入动作幅度扣分；保留约束，但不再压倒长弯道中必要的纠偏 |
| 持续无进展 | -0.005 | 256 步窗口净接近不足 1 mm 时逐步启用；持续 512 步仍由终止项处理 |
| 进入错误分支 | -150 | 路径差超过 2 mm 并持续 5 步时终止 |
| 最终成功 | +150 | 进入 3 mm 目标且导管中心未在内腔外 |
| 出血管 | -150 | 整段导管中心超出 SDF 管壁 0.5 mm，连续 3 步时终止 |
| 非有限状态 | -150 | observation/reward 出现非有限值时终止并结算 |
| 无进展终止 | -120 | 经过 grace 后再持续 512 步几乎没有净接近 |
| 超时 | -120 | 4096 步仍未完成；步数代价保证最大导航额度下仍为负 |
| 每步代价 | -0.002 | 鼓励更短路径，但不压倒最长路线上的安全连续进度 |

SAC 在跨 rank 梯度平均之后统一使用 `max_grad_norm=10`；PPO 使用
`max_grad_norm=0.5` 和 `ent_coef=0.001`。PPO/LSTM-PPO 保留已经验证过的动作标准差
下限 `0.25`，SAC 自动熵系数下限为 `0.02`；策略参数自身仍可自然退火到这些下限。
`run_config.json` 会完整保存 Reward v6、实际 observation shape/dtype，`train_summary.csv`
同时记录奖励分项（包括独立的回撤/稠密无进展项）、正回报失败率、终止路线势、
课程阶段、当前阶段每根血管的回合数与成功率、无进展次数、正负插入比例和最终插入长度。
正式训练中 `positive_failure_rate` 必须保持为 0。

B01..B05 和 C01..C05 全部使用 `vessel_sdf.vti` 直接判断管壁关系。
每步从导管尖端向入口遍历已插入的导管段，并按不大于半个 VTI
网格的间距加密采样。body 可以接触和依靠管壁滑动；whole-body 警告从最差导管中心
距管壁 0.5 mm 时开始线性启用，中心真正越过管壁后再启用较强穿透项，并与连续 3 步
越界终止使用同一 SDF 状态。tip 净空仍用于贴壁/穿壁风险和成功质量统计；
旧中心线安全比只作为缺少 VTI 的旧血管兼容后备，不参与这十条训练血管的判定。

SDF 还向 78 维状态观测提供：tip 净空、whole-body 最小表面净空、连续越界确认进度、
tip 指向内腔的 3 维方向、尖端前方 1/2/4 mm 的净空探针、最危险导管段的弧长位置、
相对 tip 的局部坐标及其回正方向。另加入目标路径前方 5/10/20 mm 的局部切向，使 MLP
也能预见即将到来的弯道。PPO、LSTM-PPO、SAC 使用完全相同的状态。观测维度从 62
变为 78，旧 checkpoint 不能直接续训。

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
训练默认启用九阶段任务课程，所有阶段都使用 B01..B05、C01..C05：目标路线先按
`40% → 55% → 70% → 85% → 100%`延长，在固定完整任务学会后，域随机化再按
`0% → 10% → 30% → 60% → 100%`逐步加入。每根血管保存最近 100 个 episode 的滚动结果；
当前阶段每一根训练血管都积累完整 100 个样本、成功率达到 50%，并连续维持 3 个
全局 epoch 才会升级。缺少样本或任一血管未达标都会清零连续计数。阶段升级时旧阶段
窗口会清空，短目标成功不能用于证明下一阶段已经掌握。阶段、连续计数、滚动
结果和逐血管成功率都保存在 checkpoint，
只前进不回退；强制单血管和 V01..V05 validation 不受影响。
前五个目标长度阶段域随机化均为 0%。当前池内采样由
50% 均匀分布和 50% 平方失败率权重混合，且单根血管概率不超过均匀概率的 2 倍，
困难血管获得更多回合，但不会造成对其他血管的灾难性遗忘。只有完整路线阶段的训练
成功率达到 20% 才能解锁 V01–V05 validation。
四卡分布式训练时，所有 rank 同步累计回合数，默认每 epoch 保存一次 checkpoint。
`--steps-per-epoch` 仅保留给旧的 transition-budget 命令；传入 `--timesteps` 时启用旧模式。
训练成功率第一次达到 `0.20` 前不创建 valid 环境；达到后永久解锁，该轮若为偶数
就立即验证，否则从下一个偶数 epoch 开始每 2 个 epoch 在 `mesh/valid` 的 5 条 unseen
血管上各运行 2 个确定性回合。四卡将
10 个固定种子验证任务按 3/3/2/2 并行执行，再由 rank 0 汇总；验证覆盖、CSV 与
`best_valid.zip` 以成功率为第一排序；成功率相同时依次比较 waypoint ratio、route
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
| global/local batch（四卡） | 512 / 128 | 512 / 128 |
| PPO n_epochs | 10 | 10 |
| gamma / GAE lambda | 0.995 / 0.95 | 0.995 / 0.95 |
| clip / entropy / value coef | 0.2 / 0.001 / 0.5 | 0.2 / 0.001 / 0.5 |
| max grad norm | 0.5 | 0.5 |
| action std bounds | 0.25–1.0 | 0.25–1.0 |
| LSTM hidden/layers | N/A | 128 / 1 |
| bidirectional | N/A | false |

训练时每个 rank 的 actor 与 critic state 分别为
`(n_lstm_layers, envs_per_rank, hidden_size)`；正式四卡 64 环境配置下即
`(1, 16, 128)`。rollout buffer 另外保存每一步的 actor/critic hidden 和 cell
state，形状为 `(256, 1, 16, 128)`。minibatch 参数仍是每 rank 128 个真实
transition；官方 buffer 内部产生的 padding 只通过 mask 排除，不会静默改写
`batch_size`。
