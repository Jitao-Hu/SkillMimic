# 第三阶段：技能标签引导惩罚（Guidance Penalty）— 代码改动说明

文件：`skillmimic/learning/hrl_dual_agent.py`  
涉及：`__init__`（可配参数）、`_apply_guidance_penalty`（状态检测 + 惩罚）、`env_step`（在 HLC 步上对 reward 扣减）。

---

## 1. 设计要点（按截图指令）

- **目的**：通过显式监督，迫使 HLC 尽快建立「球权状态 → 离散技能」的映射，减少早期盲目探索。
- **传球引导**：A 持球时若 HLC 未选 pass (ID: 4)，则扣分。
- **接球预判**：球飞向 B 时若 B 未选 catch (ID: 3) 或 run (ID: 13)，则扣分。
- **惩罚权重**：建议为主协作奖励的 10%–20%，不宜过大以免干扰主奖励；默认 1.0（与 `w_catch_success=10` 量级匹配）。

---

## 2. 修改文件与入口

| 位置 | 修改内容 |
|------|----------|
| `HRLDualAgent.__init__` | 读取并保存 Phase 3 相关配置；打印 `guidance_penalty_weight`。 |
| `HRLDualAgent._apply_guidance_penalty` | **新增**：根据当前 sim 状态与 HLC 的 `actions` 计算惩罚并 `reward -= penalty`。 |
| `HRLDualAgent.env_step` | **覆盖** 父类：在 `rewards /= self._llc_steps` 之后、返回前调用 `_apply_guidance_penalty(rewards, actions)`。 |

---

## 3. 状态检测逻辑定义（逻辑触发条件）

### 3.1 A_holding（A 持球）

**定义**：同时满足以下两条。

| 条件 | 实现 | 默认/说明 |
|------|------|-----------|
| 球靠近 A 手 | `dist_ball_to_hand_a < holding_dist_thresh` | 默认 0.2 m |
| 球有接触（cg2 风格） | `norm(ball_contact_force) > holding_contact_thresh` | 默认 1.0 N，用 `task._tar_contact_forces` |

- `dist_ball_to_hand_a` 来自 `task._get_closest_hand_distance(ball_pos, 'a')`（球到 A 手部 body 的最小距离）。
- 球的「接触」用球的 net contact force 近似（与 Phase 2 / 单人体 cg2 一致），不区分与 A/B/地面；用「近手 + 有接触」联合判定「A 持球」。

### 3.2 Ball_to_B（球在飞行且飞向 B）

**定义**：同时满足以下四条。

| 条件 | 实现 | 默认/说明 |
|------|------|-----------|
| 球速足够 | `\|ball_vel\| > ball_to_b_speed_thresh` | 默认 0.5 m/s |
| 球朝向 B | `dot(ball_vel, root_pos_b - ball_pos) > 0` | 速度方向指向 B |
| 球远离 A | `dot(ball_vel, ball_pos - root_pos_a) > 0` | 速度方向背离 A |
| 球更近 B 手 | `dist_ball_to_hand_b < dist_ball_to_hand_a` | 排除「球还在 A 侧」 |

- `ball_pos` / `ball_vel` 来自 `task._target_states`；`root_pos_a/b` 来自 `task._humanoid_root_states` / `_humanoid_b_root_states`。

---

## 4. 惩罚项实现（标签引导惩罚）

- **Pass 引导**：若 `A_holding == True` 且 `selected_skill_a != 4`（pass），则 `reward -= penalty_weight`。
- **Catch/预判引导**：若 `Ball_to_B == True` 且 `selected_skill_b ∉ {3, 13}`（catch, run），则 `reward -= penalty_weight`。
- 两档惩罚可同时触发（各扣一次 `penalty_weight`）；`penalty` 按 env 维度计算，与 `rewards` 同 shape 后做 `rewards = rewards - penalty`。
- **Skill ID 与 control_mapping**：与 `train/rlg/hrl_dual_humanoid.yaml` 一致，pass=4、catch=3、run=13；`skill_a` / `skill_b` 由当前 HLC 离散 `actions` 经 `control_mapping` 映射得到（与 `_compute_llc_action` 中逻辑一致）。

---

## 5. 配置项（可选，从 agent config 读取）

在创建 HRL agent 时传入的 `config` 中可包含（均从 `config.get(..., default)` 读取）：

| 键 | 含义 | 默认值 |
|----|------|--------|
| `guidance_penalty_weight` | 每次触发的惩罚量 | 1.0 |
| `holding_dist_thresh` | A 持球：球与 A 手距离阈值 (m) | 0.2 |
| `holding_contact_thresh` | A 持球：球接触力范数阈值 (N) | 1.0 |
| `ball_to_b_speed_thresh` | Ball_to_B：球速阈值 (m/s) | 0.5 |

若使用 rl_games 的 train 配置，需在对应 algo/agent 的 config 里传入上述键（例如在 `train/rlg/hrl_dual_humanoid.yaml` 的 `config` 下增加），否则使用默认值。

---

## 6. 代码片段摘要

### 6.1 __init__ 中新增

```python
# Phase 3: Guidance penalty (skill label alignment)
self._guidance_penalty_weight = float(config.get('guidance_penalty_weight', 1.0))
self._holding_dist_thresh = float(config.get('holding_dist_thresh', 0.2))
self._holding_contact_thresh = float(config.get('holding_contact_thresh', 1.0))
self._ball_to_b_speed_thresh = float(config.get('ball_to_b_speed_thresh', 0.5))
# ... 以及打印 Guidance penalty weight
```

### 6.2 _apply_guidance_penalty 核心逻辑

```python
# 状态 1: A_holding
ball_has_contact = (torch.norm(ball_contact_force, dim=-1) > self._holding_contact_thresh)
dist_ball_to_hand_a = task._get_closest_hand_distance(ball_pos, 'a')
a_holding = (dist_ball_to_hand_a < self._holding_dist_thresh) & ball_has_contact

# 状态 2: Ball_to_B
ball_speed = torch.norm(ball_vel, dim=-1)
to_b = torch.sum(ball_vel * (root_pos_b - ball_pos), dim=-1) > 0.0
away_from_a = torch.sum(ball_vel * (ball_pos - root_pos_a), dim=-1) > 0.0
ball_to_b = (ball_speed > self._ball_to_b_speed_thresh) & to_b & away_from_a & (dist_ball_to_hand_b < dist_ball_to_hand_a)

# 惩罚
penalty += (a_holding & (skill_a != 4)).float() * self._guidance_penalty_weight
penalty += (ball_to_b & ~((skill_b == 3) | (skill_b == 13))).float() * self._guidance_penalty_weight
rewards = rewards - penalty
```

### 6.3 env_step 中调用时机

```python
rewards /= self._llc_steps
if self.is_tensor_obses:
    rewards = self._apply_guidance_penalty(rewards, actions)
# 再处理 dones / infos 并 return
```

---

## 7. 小结

- **状态**：A_holding = 球近 A 手且球有接触；Ball_to_B = 球速>0.5、朝 B、离 A、且更近 B 手。
- **惩罚**：A 持球却未选 pass(4) → 扣 `penalty_weight`；球飞向 B 却 B 未选 catch(3)/run(13) → 扣 `penalty_weight`。
- **作用**：在 HLC 的每个决策步对 env 返回的 reward 做一次门控扣减，促使「持球→传、球来→接/跑」的标签与技能快速对齐。
