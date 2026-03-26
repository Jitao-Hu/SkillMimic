# HRL_DUAL vs CTDE RL Training Architectures (Beginner-Friendly Report)

## 1) What this report is for

This report explains, from scratch, the difference between two reinforcement learning (RL) training architectures used in this project:

- `HRL_DUAL`
- `HRL_CTDE` (often referred to as `CTDE` in notes and logs)

It is written for readers who are new to RL architecture design.

---

## 2) RL background in plain language

Before comparing architectures, here are a few core ideas:

- **Agent**: the decision maker (here, two humanoids A and B).
- **Environment**: the physics simulation where agents act.
- **Observation**: what an agent can "see" at a timestep.
- **Action**: what an agent decides to do.
- **Policy (Actor)**: the neural network that maps observation -> action.
- **Value function (Critic)**: a neural network that estimates how good a state is.
- **PPO**: the optimization algorithm used to train policy/value.

In this project, training is hierarchical:

- **High-Level Controller (HLC)** chooses a skill (pass, catch, run, idle).
- **Low-Level Controller (LLC)** is pre-trained and executes motor control.

So both `HRL_DUAL` and `CTDE` train the HLC; they share the same frozen LLC idea.

---

## 3) Quick one-screen summary

### `HRL_DUAL` (older architecture)

- Uses one discrete high-level action space of size 6.
- Action semantics are coupled through a fixed mapping style.
- Policy/critic are closer to standard single-agent PPO usage.
- Coordination is learned, but with stronger structural constraints.

### `CTDE` / `HRL_CTDE` (newer architecture)

- Uses factorized per-agent decisions: A chooses skill independently from B.
- Joint action space is explicit: `3 x 3 = 9` combinations.
- Uses **centralized critic** during training with combined observations.
- Uses **decentralized execution** at inference (each agent acts from local obs).
- Adds role encoding and optional trajectory prediction auxiliary learning.

---

## 4) Architecture-level difference

## 4.1 Action parameterization (most important difference)

### In `HRL_DUAL`

- Config shows `actions_num: 6`.
- A single discrete index is interpreted through mapping logic.
- Skill mapping list contains both A-side and B-side skills:
  - `control_mapping: [4, 13, 31, 3, 13, 31]`
  - Common interpretation:
    - A-side skills: pass/run/idle
    - B-side skills: catch/run/idle
- This introduces coupling constraints via action decoding logic.

### In `CTDE`

- Config shows `actions_num: 9`.
- This explicitly represents all pairings from:
  - A chooses one of 3 skills (`control_mapping_a`)
  - B chooses one of 3 skills (`control_mapping_b`)
- Joint index decoding:
  - `joint_action = idx_A * 3 + idx_B`
- Policy factorization in network:
  - `P(a_A, a_B) = pi_A(a_A | o_A) * pi_B(a_B | o_B)`
  - Implemented by building per-agent logits, then outer-summing into joint logits.

**Why this matters:** `CTDE` can represent richer coordination patterns because it models each agent decision separately, then combines them.

---

## 4.2 Observation design

### `HRL_DUAL`

- Older style observation engineering, including self/body/object/opponent/task and condition features.
- Role differentiation is more implicit through structure/mapping conventions.

### `CTDE`

- Uses symmetric per-agent observation vectors (same structure for A and B).
- Adds explicit role encoding (`2` dims, one-hot).
- Per-agent obs size in config is `994`.
- Training input to policy is combined as `[obs_A || obs_B]`.

**Why this matters:** explicit role + symmetric design helps parameter sharing and can improve sample efficiency.

---

## 4.3 Critic strategy (core CTDE concept)

### `HRL_DUAL`

- Uses standard PPO-style critic behavior without CTDE-style explicit centralized design emphasis.

### `CTDE`

- Uses centralized critic during training:
  - Critic sees full combined information from both agents.
- Uses decentralized execution during inference:
  - Actor for each agent only needs local/per-agent observation.

**Why this matters:** centralized critic reduces non-stationarity in multi-agent learning and usually stabilizes cooperative learning.

---

## 4.4 Network design

### `HRL_DUAL`

- Uses existing HRL discrete model/network path.
- Less explicit decomposition into per-agent actor branches.

### `CTDE`

- Custom network builder (`hrl_ctde_network_builder.py`) implements:
  - split combined obs into per-agent obs
  - shared actor MLP applied to each agent
  - per-agent logits
  - outer-sum to joint logits
  - centralized critic on full obs

**Why this matters:** architecture encodes multi-agent structure directly in the network.

---

## 5) Training objective differences

Both use PPO + imitation/AMP style reward composition in your stack, but `CTDE` introduces additional training structure:

- Factorized policy representation for two agents.
- Centralized critic for cooperative value estimation.
- Optional auxiliary trajectory prediction loss:
  - history length, horizons, hidden size, and loss weight are configurable.

This auxiliary loss is meant to improve anticipation (especially catching/running behavior), but your notes indicate mixed empirical impact so far.

---

## 6) Practical trade-offs

## 6.1 When `HRL_DUAL` can look better

- Simpler action structure can be easier to optimize early.
- Lower modeling flexibility can sometimes reduce variance in small-data or short training regimes.
- Your logs show that around 2000 epochs, `HRL_DUAL` can outperform `CTDE`.

## 6.2 Why `CTDE` can win later

- Better credit assignment from centralized critic.
- Richer joint behavior space (9 combinations vs constrained coupling).
- Parameter sharing + role encoding often helps generalization.
- Your reported results show strong CTDE improvement as training length increases (e.g., 2000 -> 5000 epochs).

---

## 7) Reading your current experimental evidence

From your weekly notes:

- `CTDE, 2000 epochs, 500 inference epochs -> avg reward ~14.7`
- `HRL_DUAL, 2000 epochs, 500 inference epochs -> avg reward ~16.9`
- `CTDE, 5000 epochs, 500 inference epochs -> avg reward ~25.9`

Interpretation:

- At shorter training, `HRL_DUAL` may have better early learning behavior.
- With longer training, `CTDE` appears to scale better and surpass prior baseline.
- This is consistent with multi-agent CTDE behavior in many settings: better asymptotic coordination, but not always best early-phase sample efficiency.

---

## 8) Implementation mapping in this repository

### `HRL_DUAL` main files

- `skillmimic/env/tasks/hrl_dual_humanoid.py`
- `skillmimic/learning/hrl_dual_agent.py`
- `skillmimic/learning/hrl_dual_players.py`
- `skillmimic/data/cfg/train/rlg/hrl_dual_humanoid.yaml`

### `CTDE` main files

- `skillmimic/env/tasks/hrl_ctde_humanoid.py`
- `skillmimic/learning/hrl_ctde_agent.py`
- `skillmimic/learning/hrl_ctde_network_builder.py`
- `skillmimic/learning/hrl_ctde_models.py`
- `skillmimic/learning/hrl_ctde_players.py`
- `skillmimic/data/cfg/train/rlg/hrl_ctde_humanoid.yaml`

---

## 9) Beginner takeaway

If you remember only one thing:

- `HRL_DUAL` is a simpler dual-humanoid HRL setup with more coupled action handling.
- `CTDE` is a more principled multi-agent architecture:
  - independent per-agent decision modeling,
  - centralized training signal (critic),
  - decentralized runtime behavior.

So `CTDE` is usually the stronger long-term architecture for coordinated multi-agent behavior, while `HRL_DUAL` can still be competitive in shorter or simpler training conditions.

---

## 10) Suggested next experiments (optional)

To make the comparison scientifically cleaner:

1. Run both architectures with identical seeds and 3-5 repeats.
2. Compare learning curves, not only final average reward.
3. Report variance and confidence intervals.
4. Evaluate with and without trajectory prediction under same budget.
5. Add per-skill usage and pass->catch success-chain metrics.

These will show whether one architecture is truly better, or just more sensitive to training budget and randomness.
