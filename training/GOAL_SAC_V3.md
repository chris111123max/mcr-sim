# Goal-SAC v3 / Reward V4：一致性修复与防停滞基线

算法依据：[SAC 双 Q 更新](https://spinningup.openai.com/en/latest/algorithms/sac.html)、[势能奖励变换原论文](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf)。它们支持更新/奖励形式，不是本项目成功率保证。

## 范围

只改独立 Goal-SAC 路线；不改基础 PPO、LSTM-PPO、SAC 的状态/奖励、血管模型、动作或物理成功门槛。不加载旧 checkpoint。保留已有五阶段血管/DR 课程，不引入短目标课程。HER 内部虚拟目标不计入真实训练成功率。当前默认 `her_ratio=0`；只有显式设置正值才启用。

## 本次解决的问题

- HER 过去只改 desired goal，却保留原终点剩余距离；现在重算前后状态的 index 9，并显式提供 achieved goal。
- 真实失败与 HER 的奖励/终止重算不一致；现在共同调用 `mcr_sim/goal_contract.py`。
- 过去 critic 副本从同一参数复制，且输入 LayerNorm 没有独立慢更新 target。现在取消 ensemble 扩展及 LayerNorm，回到 SB3 两个独立初始化 Q。
- actor 从 Q 均值改成标准双 Q 最小值；actor/target 更新间隔依据累计更新数，不在每次 train() 重置。
- 保存时不再把 optimizer 放入裸 tensor 列表；恢复时正确执行模型构建。
- HER 日志改为日志窗口计数，增加 `train/policy_entropy`。alpha 是温度，不是策略熵。

## 观测与目标

45 维基础状态的路线前视向量描述选定路径，保留为局部地图信息，并不冒充虚拟目标向量。index 9 替换为 `clip(desired-achieved,-1,1)`；末尾依次添加 achieved/desired 路线比例，合计 47 维。未添加未来真实动作、专家轨迹或模型专属特征。

真实 goal=1：完全沿用基础环境 `done_by_target`（包括 3 mm、路线进度及安全条件）。HER 只选内部 goal < 1-tolerance，使用路线比例容差 .01；它是虚拟子任务，不等同于真实 3 mm 成功。

HER 候选必须来自同环境、同 episode、当前或未来 transition 的连续安全前缀；投影无效、wrong branch、出界或 clearance 不足后不再产生候选。clearance 缺失/非有限值默认拒绝。目标还必须大于当前 transition 起点 achieved+tolerance，避免已经成功状态继续被当作未终止样本。保留单调里程碑索引以控制搜索成本，非均匀标准 future-HER；该采样偏置应在报告中说明。

## 奖励

```
Phi(s,g) = 10 * min(max(achieved(s),0), g)
r = base + .999*Phi(next,g) - Phi(s,g) - .002*(wall_risk+branch_risk) - .0001*mean((action-prev_action)^2)
base = +100 (success), -120 (timeout), -150 (out), -200 (non-finite), -.005 (otherwise)
Phi(terminal,g) = 0
```

wall/branch risk 各截断至 [0,1]。真实与 HER 共用函数与参数。horizon 属于有时间状态的有限任务终止，不跨 timeout bootstrap；HER 成功提前终止，但不能抹去其它物理终止。保存的 run_config 明确记录这些约定。

势能项在折扣 episode 中望远镜相消为 `-Phi(initial)`，不是重复刷 waypoint 的奖励；也不保证有限样本的优化一定成功。势能采用非负已完成进度，因此静止的 shaping 不会产生正 living reward。安全和动作平滑项都有界，不再用未折扣剩余步数作补偿。若改 gamma 或代价，需要重新检查尺度。

评判请看真实成功率，而不是“return 必须为正”。按 `gamma=.999`，第 2048 步的权重约为 .129，因此 timeout 使用 -120，避免延迟失败被折扣成几乎无影响；出界和非有限状态更重。完整成功并不是每步给 +100，只在终止给一次。

## 默认值与日志

单张 910B3：32 env、batch 1024、2 Q、gamma .999、lr 3e-4、buffer 500000、learning_starts 50000、每 vector step 1→2 次更新、actor interval 1、300 epoch、每 epoch 100 episode。熵系数从 .001 自动调整，下限 .001；按当前约 2.0 的策略熵，对应每步约 .002 的 soft bonus，低于普通步代价 .005。

这里 UTD=2 是每 32 条新 transition 更新 2 次，不是每条 transition 更新 2 次。保留 fused Adam、现有低频性能统计。温度下限 .001 不再被公共 curriculum callback 隐式覆盖。HER/replay 仍在 CPU，不能宣称全流程 NPU 化或保证特定 FPS。Goal-SAC 的 CSV 奖励分量由 wrapper 单独累计，不再误读基础 V11 reward。

## 验证与启动

在仓库 Python 根目录、已安装 torch/gymnasium/SB3 的环境执行无 SOFA 回归：

```bash
python -m unittest discover -s testing/py -p test_goal_sac_contract.py -v
```

通过后先在服务器验证至少跨过 learning_starts、有 update 的短跑（不是只有环境启动）：

```bash
bash training/sh/run_train_goal_sac.sh --device npu --n-envs 4 \
  --timesteps 4096 --learning-starts 512 --batch-size 64 \
  --buffer-size 8192 --skip-validation --exp-name goal_sac_v3_smoke
```

确认无异常、Q/TD 有限、奖励分量有限之后，正式独立新训练：

```bash
bash training/sh/run_train_goal_sac.sh \
  --nohup --device npu --n-envs 32 --epochs 300 \
  --exp-name goal_sac_v3_twinq_32env_300ep
```

保留标准 logs（含 launcher/console）、tb、models 目录。不要接旧 sparsev2 模型。CPU 玩具回归不验证 SOFA 磁控弯道的可控性、910B3 算子兼容或真实收敛。正式效果仍需按血管查看 success、out_of_vessel、timeout、路线完成度，并用固定验证任务评估；若安全 HER 候选接近零，先查拒绝与物理推进，不能只继续增加 epoch。
