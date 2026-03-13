# HRL-CTDE Framework Implementation Plan
## Centralized Training, Decentralized Execution for Dual Pass-and-Catch

---

## Naming Convention

| Component          | Old (`hrl_dual`)         | New (`hrl_ctde`)          |
|--------------------|--------------------------|---------------------------|
| Task class         | `HRLDualHumanoid`        | `HRLCTDEHumanoid`         |
| Agent class        | `HRLDualAgent`           | `HRLCTDEAgent`            |
| Player class       | `HRLDualPlayer`          | `HRLCTDEPlayer`           |
| Network builder    | `HRLBuilder` (shared)    | `HRLCTDEBuilder`          |
| Model              | `ModelHRLDiscrete` (shared) | `ModelHRLCTDE`         |
| Algo name          | `hrl_dual`               | `hrl_ctde`                |
| Env YAML           | `hrl_dual_humanoid.yaml` | `hrl_ctde_humanoid.yaml`  |
| Train YAML         | `hrl_dual_humanoid.yaml` | `hrl_ctde_humanoid.yaml`  |

---

## Architecture Comparison

### Old Architecture (`hrl_dual`)
- **Single joint action** (6 discrete): HLC outputs 1 action → maps to (skill_A, skill_B) pair
  - Action 0 → A=pass, B=catch; Action 1 → A=run, B=run; Action 2 → A=idle, B=idle
  - A and B's choices are always correlated (3 fixed pairs)
- **Single obs from A's perspective** with B's relative info appended
- **Standard single-agent PPO critic** (not centralized)
- **No role encoding** (implicit)

### New Architecture (`hrl_ctde`)
- **Factorized independent actions** (3 × 3 = 9 joint): each agent independently selects from its own skill set
  - A chooses from {pass, run, idle}, B chooses from {catch, run, idle}
  - Joint probability factorizes: P(a_A, a_B) = π(a_A|o_A) × π(a_B|o_B)
  - Allows all 9 combinations (e.g., A=pass + B=run)
- **Symmetric per-agent observations** with role encoding
  - obs_A and obs_B have identical structure, differentiated by role=[1,0] vs [0,1]
  - obs_buf = [obs_A || obs_B] concatenated
- **Centralized critic**: sees combined obs from both agents (full information)
- **Shared actor network**: same MLP weights process obs_A and obs_B independently
- **Decentralized execution**: at inference, each agent uses only its own obs

### Factorized Policy (Network)
```
Input: combined_obs [N, 1868]
  ├── Split → obs_A [N, 934], obs_B [N, 934]
  ├── Shared Actor MLP(obs_A) → logits_A [N, 3]
  ├── Shared Actor MLP(obs_B) → logits_B [N, 3]  (same weights!)
  ├── Joint logits = outer_sum(logits_A, logits_B) → [N, 9]
  └── Critic MLP(combined_obs) → value [N, 1]
```

### Per-Agent Observation Structure (934 dims)
| Component          | Dims | Description                                    |
|--------------------|------|------------------------------------------------|
| self_body          | 823  | Full-body obs (positions, rotations, velocities, contacts) |
| ball_from_self     | 15   | Ball state in self's reference frame           |
| partner_relative   | 15   | Partner root state relative to self            |
| task_obs           | 15   | Task-specific features (see below)             |
| role               | 2    | One-hot role encoding ([1,0]=passer, [0,1]=catcher) |
| condition          | 64   | Skill/motion condition embedding               |

### Per-Agent Task Obs (15 dims, symmetric)
| Feature                     | Dims | Description                            |
|-----------------------------|------|----------------------------------------|
| ball→self_hand_center       | 3    | Ball to own hand center in heading frame |
| ball_velocity_local         | 3    | Ball velocity in own heading frame      |
| ball→partner_hand_center    | 3    | Ball to partner's hand in heading frame |
| partner_velocity_local      | 3    | Partner velocity in own heading frame   |
| predicted_future_ball       | 3    | Ball position in ~0.5s relative to self |

---

## Phase 1: Core Implementation (All files for first runnable version)

### Files to CREATE:
1. `skillmimic/env/tasks/hrl_ctde_humanoid.py` — Task environment
2. `skillmimic/learning/hrl_ctde_network_builder.py` — Factorized policy network
3. `skillmimic/learning/hrl_ctde_models.py` — Model wrapper
4. `skillmimic/learning/hrl_ctde_agent.py` — Training agent (CTDE)
5. `skillmimic/learning/hrl_ctde_players.py` — Inference player
6. `skillmimic/data/cfg/hrl_ctde_humanoid.yaml` — Environment config
7. `skillmimic/data/cfg/train/rlg/hrl_ctde_humanoid.yaml` — Training config

### Files to MODIFY:
8. `skillmimic/run.py` — Register hrl_ctde agent, player, model, network
9. `skillmimic/utils/parse_task.py` — Import HRLCTDEHumanoid

### Verification:
```bash
python skillmimic/run.py \
  --task HRLCTDEHumanoid \
  --cfg_env skillmimic/data/cfg/hrl_ctde_humanoid.yaml \
  --cfg_train skillmimic/data/cfg/train/rlg/hrl_ctde_humanoid.yaml \
  --motion_file skillmimic/data/motions/BallPlay-M/pass \
  --headless --num_envs 64
```
Should start training without errors.

---

## Phase 2: Tuning & Validation (Future)
- Hyperparameter tuning (entropy_coef, learning_rate, reward weights)
- Verify factorized policy produces diverse skill combinations
- Compare training curves with old `hrl_dual` framework
- Add TensorBoard logging for per-agent skill distributions

## Phase 3: Trajectory Predictor (Future)
- Add learned ball trajectory prediction module
- Integrate predictions into task_obs
- Can be standalone network or attention-based

---

## Key Design Decisions

1. **Why factorized joint logits instead of separate action heads?**
   Compatible with rl_games single-discrete framework (9-way Categorical).
   Action semantics: joint_action = action_A × 3 + action_B.

2. **Why not double the batch to 2N?**
   Too invasive for rl_games PPO loop. Factorized logits achieve the same independence
   while keeping N environments and standard PPO.

3. **Why role encoding instead of separate networks?**
   Parameter sharing with role conditioning is more sample-efficient.
   The same network learns both passing and catching policies.

4. **Old files preserved:**
   All files in `dual_files_index.txt` remain untouched.
   New `hrl_ctde_*` files are independent.
