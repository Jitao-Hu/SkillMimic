## CTDE vs Old RL：Dual Humanoid Pass-and-Catch 训练架构对比报告

本文从代码实现层面，对比 `HRL-CTDE` 与 `Old HRL` 两套训练架构在以下维度的差异，并尽量“每一个细节都讲清楚”：
- reward function
- observations（包括 task_obs / ball history / role encoding）
- NN 数量与子模块（actor/critic/predictor/LLC 等）
- HLC / LLC（时间尺度、动作解码与执行路径）
- 以及最终形成的训练-执行逻辑差异

--------------------------------

## 1) CTDE RL 训练架构（`HRLCTDEHumanoid` + `HRLCTDEAgent` + `HRLCTDEBuilder`）

CTDE 的核心思想是：**集中式训练时用联合信息做 value / critic 评估**，而**去中心化执行时，actor 对 A/B 的选择只使用各自的局部观测**。

### 1.1 任务类：`HRLCTDEHumanoid`（观测与 reward 的落点）

任务实现文件：`skillmimic/env/tasks/hrl_ctde_humanoid.py`

该类继承 `SkillMimicDualHumanoid`，并通过重写：
- 观测：构造对称 per-agent obs，并拼成 `obs = [obs_a || obs_b]` 供 centralized critic 使用
- action decoding：HLC 输出联合离散动作（0..8），随后由 agent 解码为 A/B 两侧各自的技能选择（pass/run/idle vs catch/run/idle）
- reward：仍复用同一套 coop reward（在 `skillmimic/env/tasks/skillmimic_dual.py` 的 `compute_coop_reward`）
- 额外：加入 `ball_history` 缓冲，并支持（可选）trajectory predictor 辅助学习

--------------------------------

### 1.2 Reward function（CTDE 的奖励细节）

CTDE 的 reward 在 `HRLCTDEHumanoid._compute_reward()` 中计算：
- `self.rew_buf[:] = compute_coop_reward(...)`
- reward 的数学定义来自：`skillmimic/env/tasks/skillmimic_dual.py` 中的 `compute_coop_reward(...)`

`compute_coop_reward` 的总体结构是：**alive + standing + upright + ground_contact_penalty + catch_fail + coop**，其中 coop 为 **pass 与 catch 的乘法耦合**。

下面按实现顺序拆解每一项（所有阈值/常数均来自代码）：

#### 1.2.1 Alive 奖励（生存门控）
- `alive_a = (height_a > termination_heights).float()`
- `alive_b = (height_b > termination_heights).float()`
- `r_alive = alive_a * alive_b * w_alive`

含义：A 和 B 同时都在阈值以上才获得 alive 奖励（否则 coop 部分也会受到其它因素影响，但 alive 会被置零）。

#### 1.2.2 Pass 奖励（球在飞行阶段的塑形）
常数：
- `K_PASS = 2.0`
- `BALL_IN_FLIGHT_SPEED = 0.5`
- `BALL_LEFT_HAND_DIST = 0.2`

定义球是否处于“飞行/离开手之后”状态：
- `ball_in_flight = (ball_speed > BALL_IN_FLIGHT_SPEED) & (dist_ball_to_hand_a > BALL_LEFT_HAND_DIST)`

谁是接球者由“谁更近球”决定：
- `b_is_catcher = (ball_to_a < ball_to_b).float()`

接球手距离（用于 pass 质量评估）：
- `catcher_hand_dist = dist_ball_to_hand_b * b_is_catcher + dist_ball_to_hand_a * (1.0 - b_is_catcher)`
- `dist_to_catcher_hand` 在实现中通过上述逻辑得到

塑形形式：
- 若 `ball_in_flight`：`R_pass = exp(-K_PASS * dist_to_catcher_hand)`
- 否则：`R_pass = 1`

实现中 `R_pass` 用于后续 coop 的乘法耦合。

#### 1.2.3 Catch 奖励（软接塑形 + 硬接奖励）
常数：
- `K_CATCH_SOFT = 2.0`
- `CATCH_HAND_DIST = 0.15`
- `CG2_CONTACT_THRESH = 1.0`
- `MIN_STANDING_HEIGHT = 0.8`

硬接由三部分相乘构成：
1. 是否发生接触力阈值：
   - `ball_has_contact = (norm(ball_contact_force) > CG2_CONTACT_THRESH)`
2. 接球手距离阈值：
   - `hard_catch_dist = (catcher_hand_dist < CATCH_HAND_DIST)`
3. 接球者姿态站立门控：
   - `catcher_is_standing = (catcher_height > MIN_STANDING_HEIGHT)`

得到：
- `hard_catch = ball_has_contact * hard_catch_dist * catcher_is_standing`

软接塑形：
- `soft_catch = exp(-K_CATCH_SOFT * catcher_hand_dist)`

最终 catch 的合成：
- `R_catch = (1.0 - hard_catch) * soft_catch + hard_catch * 1.0`

#### 1.2.4 Cooperative（乘法耦合，pass x catch）
核心耦合：
- `R_coop = R_pass * R_catch`
- `r_coop = R_coop * w_catch_success`

这意味着：
- 仅有 pass 好不够，catch 也必须进入合理范围
- pass 与 catch 在 reward 上被强耦合（乘法），使策略必须同时学会“传得像样 + 接得像样”

#### 1.2.5 Standing（高度裁剪到 0..1）
实现：
- `standing_reward_a = clamp((height_a - 0.8)/0.5, 0, 1)`
- `standing_reward_b = clamp((height_b - 0.8)/0.5, 0, 1)`
- `r_standing = (standing_reward_a + standing_reward_b) * 0.5 * w_standing`

#### 1.2.6 Upright（四元数推导 up_z，姿态垂直度）
- 用 quaternion 的 x/y 分量计算 `up_z = 1 - 2*(x*x + y*y)`
- 分别得到 `upright_a, upright_b` 并 clamp 到 `[0,1]`
- `r_upright = (upright_a + upright_b) * 0.5 * w_upright`

#### 1.2.7 Ground contact penalty（非足接触惩罚，带 clamp 保底不爆负）
实现要点：
- `non_foot_contact` 只统计 `non_foot_body_ids`
- 接触阈值 `contact_threshold=10.0`
- 对每个 humanoid：
  - 先对每个非足 body 算 `clamp(norm - threshold, min=0)`
  - 再 sum，再除以 100
  - 再 clamp 到 `max_contact_per_humanoid=2.0`
- 最终：
  - `r_ground_contact = (gc_a + gc_b) * w_ground_contact_penalty`

#### 1.2.8 Catch fail（球落地附近、且未硬接）
常数：
- `BALL_GROUND_Z = 0.3`
- `CATCH_FAIL_RADIUS = 2.0`

条件：
- `ball_near_ground = (ball_pos.z < 0.3)`
- `ball_near_b = (ball_to_b < 2.0)`
- `catch_not_made = (1.0 - hard_catch)`

得到：
- `r_catch_fail = ball_near_ground * ball_near_b * catch_not_made * w_catch_fail`

#### 1.2.9 CTDE 的奖励观测增强点
虽然 reward 数学本体与 old 基本一致，但 CTDE 的环境在 `HRLCTDEHumanoid` 里会把 reward components 存入 `self.extras["reward_components"]`，用于更细粒度的训练日志记录（component breakdown）。

--------------------------------

### 1.3 Observations（CTDE：对称 per-agent obs + 联合拼接 + role/ball history）

CTDE 任务类核心观测组成逻辑在：
- `HRLCTDEHumanoid._get_per_agent_obs_size()`
- `HRLCTDEHumanoid._compute_observations()`
- `HRLCTDEHumanoid._compute_task_obs_pair()`
- `HRLCTDEHumanoid._update_ball_history()`

#### 1.3.1 维度与拼接（非常关键）
配置中：
- `goalSize: 15`
- `ballHistoryLength: 10`
- `role_size = 2`
- `condition_size = 64`
- parent self_body / humanoid state：`_num_obs = 823`（在 CTDE 类 docstring 中也明确）

每个 agent（A 或 B）的 obs =：
- self_body：823
- ball：15
- partner：15
- task_obs：goal_size = 15
- ball_history：T=10 帧 * 6 = 60
- role：2
- condition：64

所以每 agent obs 维度：
- `823 + 15 + 15 + 15 + 60 + 2 + 64 = 994`

最终拼接给 HLC / centralized critic：
- `obs = [obs_a || obs_b]` => 总维度 `1988`

#### 1.3.2 CTDE 的 task_obs：对称 15D（每个 agent 一份）
实现：`HRLCTDEHumanoid._compute_task_obs_pair()`
- 它调用 `_agent_task_obs(...)` 构造对称 15D：

每个 agent 的 15D 由 5 段，每段 3 维组成：
1. `ball → self_hand_center`（3）
2. `ball velocity`（在自身 heading frame）（3）
3. `ball → partner_hand_center`（3）
4. `partner velocity`（在自身 heading frame）（3）
5. `predicted future ball position`（~0.5s 预测，relative to self root）（3）

预测公式：
- `future_ball = ball_pos + ball_vel * PREDICT_DT + 0.5 * GRAVITY * (PREDICT_DT^2)`
- `PREDICT_DT = 0.5`
- `GRAVITY = [0,0,-9.81]`

#### 1.3.3 role encoding（passer/catcher 明确区分）
CTDE 通过 `_role_a` 和 `_role_b` 在 obs 里编码：
- `_role_a = [1,0]`（passer）
- `_role_b = [0,1]`（catcher）

这让同一套网络能够处理 A=pass/B=catch 与 A=certain role/B=certain role 的语义差异。

#### 1.3.4 ball history（时序输入，供 trajectory predictor 与 actor 融合）
实现：`HRLCTDEHumanoid._update_ball_history()`
- 每一步把球的 pos/vel 从 world 转到 agent 自己 heading 的局部坐标
- 每 agent 的历史缓冲是 `[num_envs, T, 6]`，其中 `T=ballHistoryLength`
- 更新方式：shift-left + append newest `[bp, bv]`

最终 ball_history reshape 成 `T*6=60`，拼进每 agent obs。

--------------------------------

### 1.4 HLC / LLC（CTDE：9 joint 离散动作解码 + LLC 执行）

CTDE 的 HLC 输出是一个单离散值 `0..8`，代表 joint action（9 种组合）。

在 `HRLCTDEAgent._compute_llc_action()`：
1. `K = skills_per_agent = 3`
2. 解码：
   - `idx_a = action_1d // K`
   - `idx_b = action_1d % K`
   - `skill_a = control_mapping_a[idx_a]`
   - `skill_b = control_mapping_b[idx_b]`

其中：
- A 技能集合：`control_mapping_a = [4, 13, 31]`（pass/run/idle）
- B 技能集合：`control_mapping_b = [3, 13, 31]`（catch/run/idle）

3. LLC 执行：
   - 从任务拿 `llc_obs_a, llc_obs_b = task.get_llc_obs_pair()`
   - 对 A/B 分别构造 64维控制 one-hot（技能 embedding / control signal），并拼到 LLC 输入
   - 通过冻结 LLC actor 输出连续动作，最后拼成 `[batch, 312]`（A 的 156 + B 的 156）

### 1.4.1 时间尺度（llc_steps）
两套 YAML 中 `llc_steps` 都是 3：
- `env_step` 中：对同一个 HLC action，运行 LLC 执行 `llc_steps` 次物理步
- `rewards /= llc_steps` 做平均

CTDE agent 在这段循环之后额外做：
- guidance penalty
- skill monitor update
- catch stats log
- reward components logging
- per-agent entropy logging（基于网络 stash 的 `_last_logits_a/_last_logits_b`）

--------------------------------

### 1.5 CTDE 的 NN 数量与结构（模块级别）

结合文件：
- `skillmimic/learning/hrl_ctde_network_builder.py`：HLC actor/critic（factorized actor + centralized critic + 可选 trajectory predictor）
- `skillmimic/learning/hrl_ctde_trajectory_predictor.py`：trajectory predictor 子模块
- `skillmimic/learning/hrl_ctde_agent.py`：训练时的辅助损失计算（MSE）注入到 PPO loss
- `HRLCTDEAgent` / `HRLCTDEPlayer`：动作解码与执行路径

#### CTDE 模块清单（“NN 数量”按子网络实例化统计）
1. **Frozen LLC 网络**：1 套（`skillmimic_llc.pth`）
2. **HLC Actor（共享 factorized actor）**：1 套共享 MLP + 1 个线性 head（最终输出 A/B 各自 K=3 logits）
   - A/B 两次前向，但使用同一套网络权重（共享 actor）
   - joint logits 通过 outer-sum 合成（K*K=9）
3. **HLC Critic（centralized value）**：1 套（基于联合 obs 的 value 网络）
4. **Trajectory Predictor（可选）**：0 或 1 套
   - 本配置启用时：1 套
   - GRU-based：`GRU(input_dim=6, hidden=64) + Linear(hidden->H*3)`
   - `H=len(traj_pred_horizons)=3 => 输出维度 9`

此外：
- CTDE agent 在训练时会为 predictor 引入辅助损失：
  - 通过 `traj_pred_loss_weight=0.1` 把 `traj_loss` 加到 PPO 主损失中

--------------------------------

## 2) Old RL 训练架构（`HRLDualHumanoid` + `HRLDualAgent`）

old 版本的“训练架构”与 CTDE 在奖励数学上相当接近，但在：
- action 解码耦合方式
- task_obs（非对称 vs 对称）
- 是否引入 role encoding 与 ball_history 时序注入
上有明显差异。

--------------------------------

### 2.1 任务类：`HRLDualHumanoid`

任务文件：`skillmimic/env/tasks/hrl_dual_humanoid.py`

关键点（来自代码与配置）：
- HLC 观测：仍是“组合式 obs”，但 task_obs 是 **goalSize=21** 且分为 A(6) + B(15) 两段语义
- HLC action：训练 agent 输出离散动作，old agent 会把离散动作映射到技能（但技能组合存在绑定/折叠效应）

--------------------------------

### 2.2 Reward function（old 的 reward）

old 环境在奖励上并没有另写一套 coop reward 数学式，而是复用与 CTDE 相同的 `compute_coop_reward`（来自 `skillmimic/env/tasks/skillmimic_dual.py`）。

也就是说：
- reward 的每一项（alive/pass/catch/coop/standing/upright/ground_contact_penalty/catch_fail）数学结构保持一致
- 差异主要来自：
  - `terminationHeight` 等配置导致的 alive/termination gating 不同
  - 训练过程中的 extra penalty（guidance penalty）与日志项差异（old agent 是否记录相同 component 取决于实现，但数学本体一致）

--------------------------------

### 2.3 Observations（old：A-centric + 非对称 21D task_obs）

old 的 HLC obs 维度组成来自 `HRLDualHumanoid.get_obs_size()`：
- humanoid_obs（parent 给定 `_num_obs`，即 823）
- obj_obs：15
- other_humanoid_obs：15
- task_obs：goalSize（old 默认是 21）
- condition embedding：64

因此 task_obs 的具体维度与语义来自：
- `skillmimic/data/cfg/hrl_dual_humanoid.yaml`：`goalSize: 21`

--------------------------------

### 2.3.1 old 的 task_obs：21D（A:6 + B:15）

实现：`HRLDualHumanoid._compute_task_obs()`

代码注释非常明确：
- Humanoid A：6 dims（在 A 的 heading frame）
- Humanoid B：15 dims（在 B 的 heading frame）

old 的 21D 每一部分语义：

Humanoid A（6 dims）：
1. ball position relative to B hand center（3）
2. B linear velocity（3，旋转到 A 的 heading frame）

Humanoid B（15 dims）：
1. ball position relative to B root（3）
2. ball linear velocity（3）
3. ball position relative to B hand center（3）
4. ball velocity relative to B hand velocity（3）
5. predicted ball position ~0.5s（3，relative to B root）

预测球位置同样使用抛物线近似（`PREDICT_DT=0.5` + gravity）。

--------------------------------

### 2.4 HLC / LLC（old：离散动作折叠与技能绑定）

old 的离散动作空间在配置里是 6，但在 agent 解码时存在 `actions % 3` 折叠：
- 在 `HRLDualAgent._compute_llc_action()`：
  - `num_skills_per_humanoid = 3`
  - `skill_idx = actions % 3`
  - `skill_a = controlmapping[skill_idx]`
  - `skill_b = controlmapping[3 + skill_idx]`

因此 6 个离散动作实际上只对应 3 种技能对组合：
1. idx=0：A=pass(4), B=catch(3)
2. idx=1：A=run(13), B=run(13)
3. idx=2：A=idle(31), B=idle(31)

这意味着 old 的 action 模型更“绑定”，无法自然产生 CTDE 那种 9 种任意组合（例如 A=pass 与 B=run 的组合）。

LLC 执行仍是冻结 LLC：
- 根据 `(skill_a, skill_b)` 分别构造 64维 one-hot 控制信号
- 拼入 LLC 输入
- 分别前向得到 A/B 的连续关节控制
- 拼成 `[batch, 312]`

--------------------------------

### 2.5 old 的 HLC：guidance penalty 与 skill monitor

old 的 `HRLDualAgent._apply_guidance_penalty()` 会在 env rewards 的基础上做减法惩罚。

惩罚条件（按代码逻辑）：
1. 如果 A_holding 成立（A 持球：dist_to_hand_a < thresh 且 ball_has_contact）但 HLC 选择的 A skill 不是 PASS，则 penalty 增加
2. 如果 ball_to_b 成立（球朝 B 飞行），但 HLC 选择的 B skill 不是 CATCH(3) 或 RUN(13)，则 penalty 增加

该惩罚权重由：
- `guidance_penalty_weight`（训练 YAML 中默认 1.0）

old 的 skill monitor 也记录技能选择比例、以及条件概率：
- `p_B_catch_given_ball_to_B`
- `p_A_pass_given_A_holding`

--------------------------------

## 3) CTDE vs old 的逐项对比（按你的要求：reward/obs/NN数量/HLC/LLC/etc.）

### 3.1 Reward function：数学结构基本一致，但 CTDE 有更强的 component 可观测性
- 相同点：
  - 两套都使用 `compute_coop_reward`：pass/catch soft+hard、coop 乘法耦合、standing/upright/ground_contact_penalty/catch_fail
  - LLC 与 environment reward 之间的接口一致：HLC 决策影响的是 skills/动作，reward 由仿真状态计算得到
- 不同点：
  - CTDE 在环境中将 reward components 写到 `extras["reward_components"]`，更细粒度用于训练期分析
  - 配置项差异（如 terminationHeight）会影响 alive gate 的触发频率，从而改变 reward 分布统计

### 3.2 Observations：CTDE 明确“对称化 + role + ball_history”，old 则是“非对称 21D task_obs + 缺少显式 role/ball_history”
- CTDE：
  - obs = `[obs_a || obs_b]`（central critic 使用联合信息）
  - 对称 task_obs：每个 agent 都有 15D（5段*3）
  - role encoding：2维明确区分 passer/catcher
  - ball_history：10帧 * 6 维 = 60维（提供时序线索）
- old：
  - obs = humanoid_a_obs + obj + other + task_obs + condition（仍是 A-centric 的拼接）
  - task_obs = 21D 但非对称：A(6) + B(15)，且 B 的语义显式在 B heading frame 下给出
  - 没有显式 role encoding（通过 task_obs / heading frames 隴出语义差异）
  - 没有 CTDE 那样的 ball_history 时序缓冲

### 3.3 HLC：CTDE 的联合动作真正因子化（9组合可覆盖），old 的动作技能组合被折叠成 3种
- CTDE：
  - joint action = 0..8（K*K=9）
  - A 与 B 的技能选择独立解码：`idx_a=action//K` 与 `idx_b=action%K`
  - 因此可覆盖如：A=pass + B=run 等跨组合
- old：
  - action space=6，但使用 `actions % 3` 折叠
  - 结果只有 3种技能对组合：pass/catch 绑定、run/run 绑定、idle/idle 绑定

### 3.4 LLC：两者都冻结 LLC，但 CTDE 的“选择方式”更独立
- 都是：HLC 输出离散技能索引 -> 构造 64维 skill control one-hot -> 前向 frozen LLC actor -> 输出连续关节动作
- 差异在于：
  - CTDE 的 (skill_a, skill_b) 来自 independent decoding
  - old 的 (skill_a, skill_b) 来自 actions%3 的折叠映射（技能相关性更强）

### 3.5 NN 数量与子模块（重点：CTDE 比 old 多了 trajectory predictor）
- CTDE 子模块：
  1. Frozen LLC：1套
  2. HLC actor：共享 factorized actor（shared MLP + per-agent logits head）
  3. HLC critic：centralized critic（联合 obs）
  4. Trajectory predictor（可选）：启用时 1套 GRU-based + linear head
  5. CTDE agent 在训练时把 `traj_pred_loss_weight * traj_loss` 加到 PPO loss 中
- old 子模块：
  1. Frozen LLC：1套
  2. HLC actor：单套离散 actor（输出 logits=6）
  3. HLC critic：单套 value/critic
  4. 没有 trajectory predictor 辅助网络

### 3.6 HLC/LLC 时间尺度（两者都用 llc_steps=3）
- 两者 yaml：
  - `llc_steps: 3`
- env_step：
  - 对每个 HLC action，执行 LLC action `llc_steps` 次
  - 平均 reward 并进入 PPO rollout/update

--------------------------------

## 小结：CTDE 主要通过“因子化联合动作 + 对称观测 + 显式 role + 可选时序预测辅助损失”提升表达能力

从代码实现角度，CTDE 相比 old 的关键增强点在于：
1. **动作空间表达能力**：从 old 的 3种绑定组合，扩展到 CTDE 的 9种可组合（因子化 joint logits）
2. **观测建模方式**：从 old 的非对称 21D task_obs，升级为 CTDE 的对称 15D + role encoding + ball_history 时序线索
3. **训练目标增强**：CTDE 可以启用 trajectory predictor，并把未来球位置预测误差以辅助损失注入 PPO

如果你希望我进一步“每一个细节”落到更细粒度（例如：per-layer MLP 单元数、critic 的具体 input 拼接方式、以及 guidance penalty/skill monitor 在 CTDE 与 old 中每个统计项的对应关系），我也可以继续把差异再往下钻。

