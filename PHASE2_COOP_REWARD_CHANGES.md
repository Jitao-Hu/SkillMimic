# 第二阶段：协作奖励函数重构 — 代码改动说明

文件：`skillmimic/env/tasks/skillmimic_dual.py`  
涉及：`_compute_reward`（调用处）、`compute_coop_reward`（JIT 函数内部逻辑）。

---

## 1. 设计要点（按截图指令）

- **R_pass**：球离 A 手后的轨迹与 B 手部中心的距离 → `exp(-k * dist)`，k=2.0。
- **R_catch**：软引导（未接到时用距离得分）+ 硬判定（cg2 接触）；避免前 100 万步全是 0。
- **R_coop = R_pass × R_catch**：乘性逻辑，B 没接到则协作分为 0。
- **standing / upright**：与 R_coop 加法关系，独立保留，维持站立。

---

## 2. `compute_coop_reward` 内具体改动

### 2.1 删除的旧逻辑（原加性项）

- 删除：`r_ball_to_hand`、`r_pass_direction`、`r_catch_success`、`r_ball_height` 的单独计算与加和。
- 删除：原「3. Pass Direction」「4. Catch Success」「5. Ball Height」整段（对齐、squared alignment、纯 0/1 catch）。

### 2.2 新增：R_pass

```python
K_PASS = 2.0
BALL_IN_FLIGHT_SPEED = 0.5
BALL_LEFT_HAND_DIST = 0.2

# 球离手且在飞行：球速 > 0.5 且 球与 A 手距离 > 0.2
ball_in_flight = (ball_speed > BALL_IN_FLIGHT_SPEED) & (dist_ball_to_hand_a > BALL_LEFT_HAND_DIST)
dist_to_catcher_hand = dist_ball_to_hand_b * b_is_catcher + dist_ball_to_hand_a * (1.0 - b_is_catcher)
R_pass = torch.where(
    ball_in_flight,
    torch.exp(-K_PASS * dist_to_catcher_hand),
    torch.ones(num_envs, device=device, dtype=torch.float32)
)
```

- 仅在「球已离 A 手且在飞行」时用 `exp(-2*dist)` 衡量传准；否则给 1，不惩罚未传状态。

### 2.3 新增：R_catch（软 + 硬）

```python
K_CATCH_SOFT = 2.0
CATCH_HAND_DIST = 0.15
CG2_CONTACT_THRESH = 1.0

# cg2：球受力 > 阈值视为被接触
ball_has_contact = (torch.norm(ball_contact_force, dim=-1) > CG2_CONTACT_THRESH).float()
hard_catch = ball_has_contact * (catcher_hand_dist < CATCH_HAND_DIST).float() * catcher_is_standing
soft_catch = torch.exp(-K_CATCH_SOFT * catcher_hand_dist)
R_catch = (1.0 - hard_catch) * soft_catch + hard_catch * 1.0
```

- **未接到**：`R_catch = soft_catch = exp(-2 * catcher_hand_dist)`，用距离带梯度，避免早期全 0。
- **接到且符合条件**：`R_catch = 1.0`（球接触 + 手距 < 0.15 + 接球方站立），硬判定接管。

### 2.4 乘性协作 + 总奖励

```python
R_coop = R_pass * R_catch
r_coop = R_coop * w_catch_success

# 总奖励：R_coop 与稳定性项加法
reward = r_coop + r_alive + r_standing + r_upright + r_ground_contact
```

- 不再把 `r_ball_to_hand`、`r_pass_direction`、`r_catch_success`、`r_ball_height` 加入总奖励。
- standing、upright 保持为独立加项，不被乘性逻辑过滤。

---

## 3. `_compute_reward` 调用处

- 仍调用 `compute_coop_reward(...)`，传入参数未改（含 `dist_ball_to_hand_a/b`、`ball_contact_force`、各权重等）。
- 仅该 JIT 函数内部按上述方式计算并返回 `reward`；**调用方无需改**。

---

## 4. 常量汇总

| 常量 | 值 | 含义 |
|------|-----|------|
| K_PASS | 2.0 | R_pass 的 exp(-k*dist) 系数 |
| K_CATCH_SOFT | 2.0 | R_catch 软引导的 exp(-k*dist) 系数 |
| BALL_IN_FLIGHT_SPEED | 0.5 | 球速阈值，高于视为在飞行 |
| BALL_LEFT_HAND_DIST | 0.2 | 球与 A 手距离 > 0.2 视为离手 |
| CATCH_HAND_DIST | 0.15 | 手与球 < 0.15 且 cg2 接触 → 硬判定接住 |
| CG2_CONTACT_THRESH | 1.0 | 球所受接触力范数阈值（cg2 物体接触） |

---

## 5. 小结

- **R_pass**：球离手后在飞行时，用当前球与接球手的距离做 `exp(-2*dist)`，否则 1。
- **R_catch**：先用距离软引导，再在「球接触 + 手近 + 站立」时给 1，避免早期全 0。
- **R_coop = R_pass * R_catch**，再乘 `w_catch_success` 得到 `r_coop`。
- **总奖励** = `r_coop + r_alive + r_standing + r_upright + r_ground_contact`，standing/upright 独立加性保留。
