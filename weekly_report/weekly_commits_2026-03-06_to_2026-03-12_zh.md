## 逐 Commit 细粒度改动总结（2026-03-06 ~ 2026-03-12）

本文件按提交顺序（由早到晚）记录过去 7 天内每一个 commit 的具体改动和影响。

---

- Updated Hyperparameters for R.L.
- Added functions that can be used to monitor what skills are used
  - LLC has many skills ready like run, pass, catch, stand, etc.
  - HLC will need to choose to perform a skill for every few steps
- increased max episode from 60 to 150
  - observed that sometimes the episode will terminate too early even when i see a potential for the task to be completed.
- Added cap to the groun contact penalty, otherwise the rewards would be really small
- Expanded ths obs from 12 to 21 Diemensions
  - 

Next Step
- find some issues with the current reward function condition, it should be updated

- for example the hard_catch

### 1）4a1dd77 – HRL 超参与 Skill Monitor 初版（3 月 6 日）

- **训练配置更新（`hrl_dual_humanoid.yaml`）**  
  
  - `learning_rate: 2e-5 → 5e-5`：提高高层策略的学习率，加快收敛速度。  
  - `entropy_coef: 0.01 → 0.05`：显著增强探索，避免过早收敛到次优策略。  
  - `horizon_length: 32 → 64`：PPO rollout 时间长度加倍，更好建模长时序依赖。  
  - `minibatch_size: 512 → 1024`：增大 batch，降低梯度噪声。  
  - `imit_reward_w: 0.0 → 0.8`：开始显式引入 imitation reward，引导高层靠近 demonstration 行为。  
  - `llc_steps: 5 → 2`：HLC 决策频率提升（每 2 个低层步长决策一次），控制更细粒度。

- **高层 Agent 技能监控（`hrl_dual_agent.py`）**  
  - 新增 `_as_bool` 和 `_skill_monitor_reset_accumulators`，用于解析配置与重置统计量。  
  - 新增一组 `_skill_monitor_*` 变量，用于记录：  
    - A 的 pass / run / idle 选择次数；  
    - B 的 catch / run / idle 选择次数；  
    - 球朝 B 飞行时 B 是否选 catch；  
    - A 持球时是否选 pass 等条件事件。  
  - 实现 `_skill_monitor_update(actions)`：  
    - 根据 HLC 动作和 control mapping 反推 skill_id（A/B 各自的离散技能）。  
    - 利用球位置、速度与 A/B root 位置计算：  
      - 球是否朝 B 飞 (`ball_to_b`)；  
      - 球是否在 A 手中；  
      - 是否出现「球飞向 B 但 B 没用 catch」等情况。  
    - 周期性（间隔 N 次 HLC step）写入 TensorBoard 标量，并可选打印 `[SkillMonitor]` 日志。  
  - 在 `env_step` 中接入：在 guidance penalty 之后调用 `_skill_monitor_update(actions)`，确保监控与实际奖励一致。

---

### 2）bae92cb – 终止条件与 episode 长度 + 训练并行度（3 月 8 日）

- **推理脚本与日志（`run.sh`）**  
  - 创建并使用 `inference_log/` 目录，所有 `debug_*.log` 统一写入该目录，和训练日志隔离。  
  - 在注释中补充两套完整用法：  
    - 有显示设备的可视化 inference（不带 `--headless`）。  
    - 服务器/headless 环境下的推理解码命令（附 `--save_images`）。  
  - 更新示例 checkpoint 到较新的 `SkillMimicDualHRL_06-13-40-14`。

- **环境终止配置与 episode 长度**  
  - `hrl_dual_humanoid.yaml`：暂时将 `enableEarlyTermination: True → False`，关闭早停便于观察在新 reward 与 obs 设置下的完整行为分布。  
  - `skillmimic_dual.py`：`max_episode_length: 60 → 150`，episode 持续时间显著拉长，为传接球提供更多尝试机会。

- **训练脚本并行度（`train.sh`）**  
  - 将 `--num_envs 512 → 1024`，翻倍并行环境数，提高采样吞吐和训练效率。

---

### 3）4dffd4a – 再次开启 early termination + ground contact 惩罚上限（3 月 8 日）

- **重新启用 Early Termination（`hrl_dual_humanoid.yaml`）**  
  - 在上一 commit 放宽之后，将 `enableEarlyTermination: False → True`，  
    以在 reward/obs 调整后重新引入姿态约束（如严重跌倒时提前结束 episode）。

- **控制频率回调（训练 cfg：`hrl_dual_humanoid`）**  
  - `llc_steps: 2 → 5`，并在注释标明 HLC 决策约为 12 Hz：  
    - 高层不再过于频繁地切换离散技能，避免策略抖动。  

- **地面接触惩罚裁剪（`skillmimic_dual.py`）**  
  - 对非脚部接触地面的惩罚做上限限制：  
    - 为每个 humanoid 单独 clamp 到 `max_contact_per_humanoid = 2.0`。  
    - 保留惩罚作用，但避免跌倒状态因接触力过大导致 reward 无限负，从而让训练稳定性更好。

---

### 4）dcec82d – 实验目录命名规则 & WandB 导出脚本（3 月 9 日）

- **推理脚本 checkpoint 路径更新（`run.sh`）**  
  - 将示例 checkpoint 替换为更新的 `SkillMimicDualHRL_08-21-42-38`，与最近实验保持一致。

- **Experiment 命名与 resume 行为（`skillmimic/run.py`）**  
  - 引入 `datetime`：  
    - 若未 `resume_from`：  
      - 使用 `name + "_%Y%m%d-%H-%M-%S"` 作为 `full_experiment_name`，目录名包含年月日和具体时间。  
    - 若使用 `resume_from`：  
      - 从 checkpoint 路径中解析出已有 experiment 目录名，写回 `full_experiment_name`，保证 resume 时继续向原目录写日志和模型。

- **WandB 数据导出工具（`wandb_csv/wandb_csv.py`）**  
  - 使用 `wandb.Api()` 读取指定 run：  
    - `history(samples=100000)` → `full_history.csv`，记录训练全过程的曲线。  
    - `run.config` → `config.json`，记录超参数配置。  
    - `run.summary` → `summary.json`，记录最终指标。  
  - 统一导出到 `wandb_export_YYYYMMDD_HHMMSS/` 目录，便于分享和离线分析。

---

### 5）1c71d67 – Catch reward / Task obs / 控制频率联动调参（3 月 9 日）

- **协作 reward 细化（`hrl_dual_humanoid.yaml` & `skillmimic_dual.py`）**  
  - `catch_success: 10.0 → 15.0`，显著提高成功接球瞬间的奖励。  
  - 新增 `catch_fail: -5.0` 权重，并在 `compute_coop_reward` 中增加对应项：  
    - 条件：球高度 < 0.3m、距离 B 的 root < 2m，且未触发 `hard_catch`；  
    - 在这种「球在 B 附近落地且没接住」的情形下给出额外惩罚，强化 B 对失败接球的敏感度。  
  - 最终总奖励结构：`reward = r_coop + r_alive + r_standing + r_upright + r_ground_contact + r_catch_fail`。

- **任务观测空间扩展至 21 维（`hrl_dual_humanoid.py`）**  

- LLC Obs
  - 每个humannoid都有自己的信息+球在自己坐标系下的信息
    - 是否应当加入对方在自己坐标系下的信息？
  - 需要加入球的历史信息，以及prediction of trajectory

- HLC Obs
  - 

- in the observations, check the xyz orientation of ball, check the definition of the ball position

- chekc the ball velocity

- prediction the ball trajectory, training the model with this information (must try)
  - badminton robot
    - https://www.youtube.com/watch?v=zYuxOVQXVt8
    - https://arxiv.org/html/2509.21690v1
    - https://research.nvidia.com/labs/toronto-ai/vid2player3d/

- LLC & HLC
  - LLC 
    - might need to retrain the pass/catch skills (try it?)
  - HLC:    
    - big picture: for the hieracrhical acrh, for a  long term goal, hoew can steps be automately generated

- maybe use two different networks?
 - one network mayb be bad at generating two actions
 - maybe two networks for two actions?

  - A 侧（6D，A 的 heading 坐标系）：  
    - 球相对 B 手中心的位置（3D）；  
    - B root 的线速度（3D）。  
  - B 侧（15D，B 的 heading 坐标系）：  
    - 球相对 B root 的位置（3D） + 球速度（3D）；  
    - 球相对 B 手中心的位置（3D）；  
    - 球速度相对 B 手速度的相对速度（3D）；  
    - 使用简单弹道方程预测 0.5 秒后的球位置，并转换到 B 本地坐标（3D）。  
  - 对应地将 `goalSize` 从 12 更新为 21，保证配置与实现一致。

- **控制频率提升（训练 cfg）**  
  - `llc_steps: 5 → 3`，HLC 决策频率从约 12 Hz 提升到约 20 Hz：  
    - 结合更丰富的轨迹预测型 obs，使 B 在球飞行期间能多次修正 catch/run 决策。

---

### 6）2d1225d – 三份 SkillMimic 训练指南归档（3 月 9 日）

- **新增中文文档（`archive/*.txt`）**  
  - 双人传接球 HRL 分步实施计划：  
    - 按阶段列出修改顺序：扩展 obs → 重构协作奖励 → 引入技能引导惩罚 → 训练与调试验证；  
    - 明确每一步对应的文件位置和修改点。  
  - 提升 B 接球稳定性的专项指南：  
    - 围绕 `p_B_catch_given_ball_to_B` 指标，从 reward、obs、LLC 调用和 curriculum four 个维度给出建议。  
  - SkillMimic 传接球任务整体优化指南：  
    - 系统梳理任务背景、目标、需要扩展的 goalSize、reward 乘法逻辑和 skill guidance 设计动机。  

---

### 7）525dc9b – Per-Attempt Catch Stats（3 月 12 日）

- **配置开关（`hrl_dual_humanoid.yaml`）**  
  - 增加 `enableCatchStats`、`catchStatsPrint`、`catchStatsInterval`：  
    - 控制是否启用 per-attempt 统计、是否在控制台打印，以及每多少次 ball flight 打印一次。

- **环境侧统计逻辑（`skillmimic_dual.py`）**  
  - `_init_catch_stats`：  
    - 为每个 env 维护「当前 ball 是否在 flight 中、是否已经出现 hard_catch / good_pass」等状态标记；  
    - 初始化全局计数器：pass/catch attempt & success/fail，以及 alive/standing/upright/ground_contact 的 step 计数。  
  - `_update_catch_stats`：  
    - 每步根据球速度和与 A/B 距离判断 `ball_in_flight`，并识别 `hard_catch`、`good_pass` 和 `catch_fail`；  
    - 当某个 env 的 flight 结束（ball 不再 in-flight 或 env reset）时：  
      - 将该次 flight 的结果累积到全局统计；  
      - `flights_total` 增加，达到 `catchStatsInterval` 后打印 `[CatchStats]` 日志，包含 pass/catch 成功率、fail 数量和姿态相关比率。  
    - 提供 `get_catch_stats()` 和 `reset_catch_stats()`，用于 agent 侧读取与重置。  
  - 在 `post_physics_step` 的主流程中调用 `_update_catch_stats()`，与 reward/reset 同步执行。

- **Agent 侧读取与日志（`hrl_dual_agent.py`）**  
  - `_catch_stats_log`：  
    - 首次调用时从 env 读取开关、间隔与是否打印；  
    - 每隔 `N` 个 HLC step 调用 `task.get_catch_stats()`，并将以下指标写入 TensorBoard：  
      - `pass_success_rate`、`catch_success_rate`、`catch_fail_rate`；  
      - `alive_rate`、`standing_rate`、`upright_rate`、`ground_contact_rate`；  
      - 以及 pass attempt 和 catch success 的计数。  
    - 若开启打印，则在控制台输出一行 `[CatchStats] hlc_step=...` 的摘要。  
    - 调用 `task.reset_catch_stats()` 实现窗口化统计。  
  - 在 `env_step` 中与 `_skill_monitor_update` 串联，形成「技能行为监控 + per-attempt 结果监控」的完整闭环。

---

### 小结

这 7 个 commit 串联起来，形成了一个从 **超参数与 reward/obs 设计 → 控制频率与终止条件 → 行为与结果监控 → 工程与文档沉淀** 的完整优化周期：  
- 通过多轮调整 learning rate、entropy、horizon、llc_steps 和 early termination，使训练在稳定性与探索性之间找到更合适的平衡点；  
- 通过扩展 21D task obs、引入 catch_fail 惩罚与 ground contact 裁剪，针对性提升 B 的接球稳定性；  
- 通过 Skill Monitor 与 Per-Attempt Catch Stats，将「策略在做什么」和「结果有多好」都变成可量化、可视化的指标；  
- 同时完善运行脚本、实验目录命名和 WandB 导出工具，并撰写多份中文训练指南，将这些经验固化为可复用的知识。

