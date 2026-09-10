# mCR 血管内导管强化学习项目概述

> 用途：作为与 GPT 讨论状态、奖励、课程和算法问题时的背景材料。基于当前源码整理，更新时间：2026-09-04。

## 1. 项目与任务

项目使用 SOFA 模拟磁驱导管在三维血管内导航，通过强化学习控制导管到达目标。

策略输出 3 维连续动作：两个磁场局部旋转分量，以及导管插入/回撤。当前每步最大约为 `±3°` 和 `±0.8 mm`，相邻动作还有限速，以避免控制突变。

一个 episode 最多 2048 step。成功必须同时满足：

- 导管尖端进入最终目标 3 mm 内；
- 沿目标路线也已进入终点前约 5 mm；
- 当前没有出血管或进入错误分支。

训练集为 `mesh/train/B01～B05、C01～C05`：B 类以分叉为主，C 类以弯曲和急弯为主。验证集为未参与训练的 `mesh/valid/V01～V05`。

每根血管包含表面模型、中心线或目标路线、SDF 和元数据；分叉血管还需要中心线图。SDF 用于检查整根已插入导管与血管壁的关系。

## 2. 代码结构

```text
mcr_sim/
├─ mcr_rl_env.py              环境、状态、奖励、成功和终止
├─ training_config.py         训练参数、Reward 和 curriculum
├─ route_tracking.py          路线投影与 route progress
├─ vessel_assets.py           血管资产读取和检查
├─ mcr_controller_sofa.py     插入/回撤控制
├─ distributed/               PPO、LSTM-PPO、SAC 多卡实现
└─ rl_core/                   训练循环、验证、日志、checkpoint

scene/example_aortic_arch.py  SOFA 场景和血管加载
training/py/                  三种算法的 Python 入口
training/sh/                  三种算法的标准 nohup 启动脚本
testing/                      GUI 和各类测试
tools/                        人工血管生成与检查
```

## 3. 当前状态：45 维局部观测

状态主要采用导管尖端局部坐标，不直接提供绝对世界坐标。包含：

- 当前磁场方向；
- 路线前方 10 mm、20 mm 的目标方向；
- 路线前方 5/15/30 mm 的切线方向；
- 剩余路线距离和剩余时间；
- 当前插入长度和局部血管半径；
- 尖端及整根导管最危险位置的壁面 clearance；
- 最危险位置、SDF 向内梯度；
- 偏离目标分支的程度；
- 距尖端约 10/30/60 mm 的导管轴形状点；
- 上一步实际动作、尖端位移和 route progress 变化。

设计目的是依赖局部几何并泛化到新血管。潜在不足是：MLP 没有长期历史；20 mm 前视对急弯可能不足；路线投影跳变会污染状态；近壁信号可能触发偏晚。

## 4. 当前奖励：Reward V13.0

```text
reward = route progress
       + wall penalty
       + wrong branch penalty
       + step cost
       + terminal reward/penalty
```

| 奖励项 | 当前值 |
|---|---:|
| route progress scale | `60.0` |
| near-wall | `-0.005 × risk/step` |
| wrong branch | `-0.005/step` |
| step cost | `-0.002/step` |
| success | `+100` |
| out of vessel | `-80` |
| non-finite | `-100` |
| timeout | `-80` |
| no progress | `-0.020 × risk/step`，不终止 |

progress 公式：

```text
completion_t = route_progress_t / target_route_length
r_progress = 60 × (completion_t - completion_(t-1))
```

它不是“每前进一步奖励 30”。前进和后退按净路线完成度严格抵消，但失败或超时时不再一次性抹掉此前的真实净进度，从而给 PPO 保留连续学习信号。

Reward V13.0 不使用 waypoint 一次性奖励，也不使用 no-progress 提前终止。停滞在 64 步宽限后按 32 步净路线进度产生最高 `-0.020/step` 的可恢复代价。若走满 2048 步，step cost 累计为 `-4.096`，之后再加 timeout `-80`。体部 SDF 预警从导管表面距管壁 0.5 mm 开始，在接触时达到 1；插入动作采用保持零点的分段映射。

当前风险是：progress 高度依赖路线投影；近壁惩罚较小且可能出现偏晚；参数尚未通过高成功率实验验证。

## 5. 安全与终止

episode 在成功、确认出血管或 non-finite 时终止；达到 2048 step 时按 timeout 截断。wrong branch 可以恢复，只产生小惩罚；no-progress 只用于诊断。

出血管主要使用整根导管的 SDF 采样判断：最差采样中心点穿出 SDF 边界超过约 0.5 mm，并连续 3 step 后确认。

值得继续检查的是：状态中的 clearance 已考虑导管半径，但部分危险和出界判断仍基于采样中心点 SDF，安全信号可能晚于导管表面实际碰壁。

## 6. 五阶段课程学习

所有阶段都要求走完整路线，不再使用 40% 目标距离课程。

| 阶段 | 血管 | 域随机化 | 晋级要求 |
|---:|---|---|---|
| 0 | B01、B02 | 关闭 | 总体成功率 ≥ 0.90 |
| 1 | C01、C02 | 关闭 | 每根成功率 ≥ 0.50 |
| 2 | B01、B02、C01、C02 | 开启 | 每根成功率 ≥ 0.50 |
| 3 | 全部 10 根 | 关闭 | 每根成功率 ≥ 0.50 |
| 4 | 全部 10 根 | 开启 | 最终阶段 |

晋级还要求每根相关血管至少完成 100 episode，按最近 100 episode 计算成功率，并满足最近连续 3 episode 成功。

采样概率由 50% 均匀采样和 50% 困难度采样组成。低成功率血管会被增加采样，但最多约为均匀概率的 2 倍。域随机化包括血管尺度 `0.9～1.0`、起终点小范围变化和最大约 10° 的初始朝向变化。

## 7. 算法与训练

当前支持：

- **MLP-PPO**：45 维单步状态输入 MLP；
- **LSTM-PPO**：状态相同，用循环隐藏状态处理物理滞后；
- **SAC**：off-policy，使用 replay buffer。

三者原则上共享相同的任务、状态、Reward V13、课程、成功标准和验证集，以便公平比较 MLP、RNN 和未来 Transformer。

训练支持单 NPU、多环境和 HCCL 多 NPU。多卡时各 rank 收集本地环境并同步梯度。由于网络较小，SOFA、进程调度和通信占比较高，多卡 FPS 不会简单等于单卡乘卡数。

验证使用 `V01～V05`，每根通常运行 2 个 deterministic episode。只有 curriculum 进入最终阶段且训练成功率达到 0.20 后才开启。验证异常只记录失败，不应终止训练。

每次实验保存 console/launcher 日志、`run_config.json`、训练/验证 CSV、TensorBoard 和模型 checkpoint，统一放在 `training_runs/<实验名_时间>/`。

## 8. 当前训练现象

- 策略能学会向前和局部导航；
- route completion 有时能达到并长期停在 0.5～0.7；
- 最终 3 mm success 仍偏低；
- 主要失败是急弯处出血管和 timeout；
- C 类，尤其曲率变化大的血管，明显更难；
- MLP-PPO、LSTM-PPO 和 SAC 都尚未稳定达到理想成功率。

`route_completion=0.7` 只表示走完约 70% 的目标路线，不等于成功。长期卡住说明策略已学会局部推进，但在困难后段、急弯控制、状态信息或安全约束上存在瓶颈。

## 9. 最值得继续讨论的问题

1. 45 维状态是否足以表达急弯前的几何和导管动态；
2. MLP 是否因缺少历史而无法处理磁控制和形变滞后；
3. route projection 是否在急弯或邻近中心线段间跳变；
4. SDF wall-risk 是否能在真正碰壁前及时出现；
5. 插入速度和动作限速是否导致来不及提前转向；
6. Reward V13 的净路线增量是否稳定提升且不被路线投影噪声污染；
7. curriculum 是否过早把任务从 B 类扩展到 C 类；
8. C03/C05 是否超出当前控制器和观测的可学习范围；
9. 低成功率主要来自 reward、partial observability、控制可达性，还是任务难度；
10. 是否需要自动示范、规划引导、安全动作筛选或少量 model-based 方法。

更详细的历史问题见 `training/RL_TRAINING_PROBLEM_SUMMARY.md`。

## 11. 独立 Goal-conditioned SAC 实验路线

当前还新增了一条与 V12 baseline 隔离的实验路线：
`training/py/train_goal_sac.py` + `training/sh/run_train_goal_sac.sh`。

它复用 SOFA、血管资产、route tracking、SDF 和原环境控制动作，并在 `MCREnv` 的 Reward V12 之外使用独立 Goal reward。Goal-SAC v3 使用 47 维观测：45 维基础状态中 index 9 重算为目标剩余路线比例，末尾附加 achieved/desired goal。真实目标恒为 `1.0`，成功仍要求原环境的物理到达条件；HER 使用同一安全轨迹上的内部路线目标。

奖励使用共享 `GoalReward`：普通步 `-0.01`、成功 `+10`、失败（包括 horizon）`-20`，加 `gamma*Phi(next)-Phi(previous)` 与小安全代价；`Phi=-5*max(goal-achieved,0)`，终止势能为零。真实和 HER 使用相同函数，不再补扣剩余步数。此版本不再是纯 sparse reward。Safe HER 保留连续安全前缀过滤，去掉相对 episode 首个保留样本的 2% 门槛，改为目标不能在当前 transition 开始前就被达到。申请 HER 比例为 50%，实际比例由安全候选决定，不能假设始终 50:50。

默认标准双 Q，无输入 LayerNorm，actor/target 均取 min(Q1,Q2)。32 环境、batch 1024、每 vector step 更新 1→2 次，actor 每次更新，300 epoch。详细参数、验证命令与已知边界见 [GOAL_SAC_V3.md](GOAL_SAC_V3.md)。日志仍保存至独立 `training_runs/<exp_name>_<timestamp>/`。

训练结束后新入口可调用公共 validation evaluator 对 `V01～V05` 做独立 deterministic 评估；缺失验证资产或单次验证异常只写入 `[VALID][ERROR]`，不会改变已完成的训练结果。用 `--skip-validation` 可显式跳过。

新旧观测/奖励不兼容，应从头训练。不能用负回报或低 alpha 单独判断失败：应结合真实物理成功率、当前窗口 HER 比例、策略熵和各血管结果。该实现修复不代表已经验证了 SOFA 中的收敛效果。
