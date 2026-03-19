## 逐 Commit 细粒度改动总结（2026-03-12 ~ 2026-03-19）

本文件按提交顺序（由早到晚）记录过去一周内每一个 commit 的具体改动和影响。

---

### 1）fe5f5ca – test_episode control（3 月 13 日）

- **测试/回放相关控制逻辑补齐（`skillmimic/learning/hrl_dual_players.py`）**
  - 补齐测试场景下的 episode 迭代/控制分支，避免测试与训练逻辑在边界条件上不一致。

- **测试入口与参数分支调整（`skillmimic/run.py`）**
  - 调整测试入口的参数处理方式，使测试轮数与配置解析更稳定可控。

- **配置解析支持测试轮数控制（`skillmimic/utils/config.py`）**
  - 更新配置解析逻辑，保证测试相关字段能够正确生效。

---

### 2）9d74f5b – weekly commit（3 月 13 日）

- **周报文档补齐/更新（`weekly_report*`）**
  - 维护并补齐与上期周报相关的 Markdown 内容，保证周报产出链路完整。

---

### 3）987d816 – new design doc; old design doc（3 月 13 日）

- **CTDE 设计文档与索引（文档层面落地规划）**
  - 新增 `hrl_ctde_phase_plan.md`：记录 CTDE 从阶段化实现到验证的目标与里程碑。
  - 新增 `hrl_dual_framework.txt`：补充旧/新框架对比与关键设计点。
  - 新增 `ctde_files_index.txt`、`dual_files_index.txt`：形成新旧文件体系的索引，降低后续集成成本。

---

### 4）4288727 – ctde structure（3 月 15 日）

- **CTDE 代码骨架与工程接入（从“设计”走到“可运行结构”）**
  - 新增 CTDE 相关代码文件：
    - `skillmimic/env/tasks/hrl_ctde_humanoid.py`
    - `skillmimic/learning/hrl_ctde_agent.py`
    - `skillmimic/learning/hrl_ctde_models.py`
    - `skillmimic/learning/hrl_ctde_network_builder.py`
    - `skillmimic/learning/hrl_ctde_players.py`
    - `skillmimic/learning/hrl_ctde_trajectory_predictor.py`
  - 更新工程接入：
    - `skillmimic/run.py`、`skillmimic/utils/parse_task.py`：注册/解析 `HRLCTDEHumanoid` 相关入口。
    - `train.sh`：对 CTDE 默认训练脚本参数做对齐，保证训练能顺利 resume 并产出 checkpoint。

---

### 本周训练/推理记录（含关键数值）

1. 训练（`output/train_20260319_140910.log`）
   - 本次训练为 CTDE 结构相关实验，`resume_from output/SkillMimicCTDE_20260313-18-06-00/`。
   - 训练进度到 `epoch_num:2002` 时：`mean_rewards:[37.04]`。
   - 并在该阶段保存 checkpoint：
     - `output/SkillMimicCTDE_20260313-18-06-00/nn/SkillMimicCTDE.pth`

2. 推理（CTDE）（`inference_log/inference_ctde__20260315_210141.log`）
   - `test_episodes=500`。
   - 最终指标汇总：
     - `av reward: 14.7537`
     - `av steps: 20.642`
   - CatchStats（围绕 `flights_total=500` 的窗口）：
     - `pass=14/500(0.028)`
     - `catch=2/500(0.004)`
     - `fails=19`
   - 运行末尾出现异常：
     - `Segmentation fault (core dumped)`
     - 同时在脚本解析侧看到 `run_ctde.sh` 里 `--headless` 行触发 “command not found”，提示 bash 命令续写/换行存在问题。

3. 推理（旧版 Dual HRL）（`inference_log/inference_old__20260315_210219.log`）
   - `test_episodes=500`。
   - 最终指标汇总：
     - `av reward: 15.2358`
     - `av steps: 19.892`
   - CatchStats（围绕 `flights_total=500` 的窗口）：
     - `pass=31/500(0.062)`
     - `catch=2/500(0.004)`
     - `fails=19`

---

### 小结

本周重点完成了 CTDE 从设计到骨架落地的闭环：先补齐测试相关控制逻辑与文档产线，再引入 CTDE 设计文档与文件索引，最终在 `hrl_ctde_*` 一组核心模块上完成代码骨架与工程接入。训练侧已在指定 CTDE checkpoint 上跑到较高 epoch 并保存更新权重；推理侧对 CTDE 与旧版 Dual HRL 做了对比，CTDE 的平均 reward 与步数处于相近区间，但 catch 触发率仍较低。与此同时，当前推理仍暴露出 `segmentation fault` 与 `run_ctde.sh` 脚本解析异常两类风险点，需要在下周进一步定位修复。

---

### 总体小结

代码层面从文档索引与结构规划转入 CTDE 实装；工程层面完成任务注册与训练脚本对齐；实验层面则产出了本周 CTDE 训练 checkpoint，并在推理对比中记录了可量化的 pass/catch/fail 指标与异常日志。接下来会围绕“推理稳定性（segfault）”与“脚本命令解析正确性（headless 行）”两条主线继续迭代，同时再观察 catch_success 相关指标是否能在后续训练阶段改善。

