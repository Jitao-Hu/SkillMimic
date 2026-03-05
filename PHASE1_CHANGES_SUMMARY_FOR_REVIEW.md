# 第一阶段改动总结（供 Gemini 审查）

依据《SkillMimic 双人协作分步实施指令.txt》**第一阶段：扩展感知与环境配置**完成的修改。

---

## 1. 目标

- 将 HLC 的 task 观测从 5 维扩展到 **12 维**，使 HLC 能感知「队友与球的动态关系」。
- 所有相对量在**各自机器人的 root local 坐标系**下给出。
- 配置与代码一致：`goalSize: 12`，并注明 `llc_steps` 在 train 配置中为 5。

---

## 2. 修改文件清单

| 文件 | 修改类型 |
|------|----------|
| `skillmimic/env/tasks/hrl_dual_humanoid.py` | 逻辑修改：goalSize 默认值、`_compute_task_obs` 重写 |
| `skillmimic/data/cfg/hrl_dual_humanoid.yaml` | 配置：goalSize、llc_steps 说明注释 |
| `skillmimic/learning/hrl_dual_players.py` | 注释：obs 维度 922→929、task_obs 5→goalSize |
| `skillmimic/learning/hrl_dual_agent.py` | 注释：同上，与 players 保持一致 |

---

## 3. 观测空间变更（核心）

### 3.1 goalSize

- **原**：`goalSize = 5`（config 与代码默认均为 5）。
- **现**：`goalSize = 12`，代码默认 `cfg["env"].get("goalSize", 12)`，yaml 中 `goalSize: 12`。

### 3.2 task 观测结构（12 维）

HLC 的 `task_obs` 仍为**单向量**，由 A 的 6 维与 B 的 6 维拼接而成：

| 段落 | 维度 | 含义 | 坐标系 |
|------|------|------|--------|
| **Humanoid A** | 6 | 球相对 B 手部中心的偏移 (3) + B 的线速度 (3) | A 的 root local |
| **Humanoid B** | 6 | 球的位置 (3) + 球的速度 (3) | B 的 root local |

- **A 的 6 维**：便于 A（传球方）判断「往 B 手传」的目标与 B 的移动。
- **B 的 6 维**：便于 B（接球方）预判球轨迹与落点。
- 坐标系：A 部分用 `calc_heading_quat_inv(root_rot_a)` 转到 A 的 heading；B 部分用 `calc_heading_quat_inv(root_rot_b)` 转到 B 的 heading。

### 3.3 实现要点（`_compute_task_obs`）

- **B 手部中心**：`rigid_body_pos_b[:, _hand_body_ids, :].mean(dim=1)`，与现有 `_get_closest_hand_distance` 使用同一套 `_hand_body_ids`。
- **A 的 6 维**：
  - `ball_rel_B_hand_world = ball_pos - hand_center_b`，再 `quat_rotate(heading_rot_a, ball_rel_B_hand_world)` → 3 维。
  - `root_vel_b = _humanoid_b_root_states[:, 7:10]`，再 `quat_rotate(heading_rot_a, root_vel_b)` → 3 维。
- **B 的 6 维**：
  - `ball_pos - root_pos_b` 再 `quat_rotate(heading_rot_b, ...)` → 球在 B local 下的位置 3 维。
  - `quat_rotate(heading_rot_b, ball_vel)` → 球在 B local 下的速度 3 维。
- **env_ids**：当 `env_ids is not None` 时，所有用到的 state（root_pos/rot/vel、ball、rigid_body_pos_b）均按 `env_ids` 索引，保证 partial 更新正确。

### 3.4 观测总维度

- 原：823 + 15 + 15 + **5** + 64 = **922**。
- 现：823 + 15 + 15 + **12** + 64 = **929**。
- LLC 仍只使用前 838 维（humanoid_obs + obj_obs），不依赖 task_obs，因此无需改 LLC 或 checkpoint。

---

## 4. 配置变更

**`skillmimic/data/cfg/hrl_dual_humanoid.yaml`**

- `goalSize: 5` → `goalSize: 12`，并加注释说明 12 维的构成（A 6 + B 6）。
- 新增注释：`llc_steps` 在 `train/rlg/hrl_dual_humanoid.yaml` 中设置（当前为 5），本阶段未改 train 配置。

---

## 5. 附带注释更新（无行为变化）

- **hrl_dual_players.py**：`total_obs_size`、`llc_obs_size` 计算注释中的 922、5 改为 929、goalSize；`_extract_llc_obs` 的 docstring 中 task_obs 写为 goalSize 维。
- **hrl_dual_agent.py**：`_extract_llc_obs` 的 docstring 与上述一致，避免与 12 维 task_obs 混淆。

---

## 6. 审查时可重点看

1. **坐标系**：A 的 6 维是否全部在 A 的 heading 下、B 的 6 维是否全部在 B 的 heading 下（无混用）。
2. **手部中心**：是否与现有 hand body 定义一致，且支持 `env_ids` 子集。
3. **维度与兼容性**：HLC 输入是否为 929 维；LLC 是否仍只读前 838 维，无需重新训练或改 checkpoint。
4. **配置一致性**：env 的 `goalSize: 12` 与代码默认 12、与 `get_task_obs_size()` 返回值一致。

---

## 7. 未改动的部分

- 协作奖励、技能 ID、reset、LLC 调用逻辑、train 配置中的 `llc_steps: 5` 均未改动。
- 第二阶段（协作奖励乘法逻辑 + Contact Graph）尚未实施。
