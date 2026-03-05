# SkillMimic Fork Progress Summary (for Gemini)

## 1. Repo Context

- **Fork source**: [wyhuai/SkillMimic](https://github.com/wyhuai/SkillMimic) (upstream)
- **Your fork**: [Jitao-Hu/SkillMimic](https://github.com/Jitao-Hu/SkillMimic) (origin)
- **Base paper**: "SkillMimic: Learning Basketball Interaction Skills from Demonstrations" ([arXiv](https://arxiv.org/abs/2408.15270))
- **Stack**: Isaac Gym, PyTorch, humanoid + ball (HOI) simulation, motion data from BallPlay-M dataset; original task is single-humanoid basketball skills (shoot, dribble, layup, retrieve, etc.) with skill policy + optional HRL (high-level controller selects skills, low-level controller executes).

---

## 2. Your Goal

Extend the repo to support a **dual-humanoid cooperative pass-and-catch task** in an **HRL (Hierarchical Reinforcement Learning)** setting:

- **Two humanoids per env**: Humanoid A (passer), Humanoid B (catcher), plus ball and optional projectiles.
- **HRL**: High-level controller (HLC) outputs discrete skill choices for each humanoid; low-level controller (LLC) is the pre-trained SkillMimic policy that executes the selected skill.
- **Skills used**: pass (4), catch (3), run (13), idle (31), pick (1) from the existing skill set.
- **Rewards**: Imitation (from reference motion) plus cooperative task rewards (ball-to-hand, pass direction, catch success, standing/upright, ground-contact penalty, etc.).

---

## 3. What You Added (New/Modified Files)

### New files (your work)

| Path | Purpose |
|------|--------|
| `skillmimic/env/tasks/skillmimic_dual.py` | **SkillMimicDualHumanoid**: dual-humanoid env (two humanoids + ball); observation/reward for both; motion loading; coop reward terms (alive, ball_to_hand, pass_direction, catch_success, ball_height, standing, upright, ground_contact_penalty). |
| `skillmimic/env/tasks/hrl_dual_humanoid.py` | **HRLDualHumanoid**: extends SkillMimicDualHumanoid; HRL-specific obs/action; builds LLC obs for A and B; skill IDs for pass/catch/run/idle/pick; overrides `get_obs`, `get_actions`, `compute_observations`, etc. |
| `skillmimic/learning/hrl_dual_agent.py` | **HRLDualAgent**: HLC agent for dual humanoid; outputs discrete actions for both humanoids; uses shared LLC. |
| `skillmimic/learning/hrl_dual_players.py` | **HRLDualPlayer**: rollout/training player for HRL dual task; interfaces with HRLDualAgent and env. |
| `skillmimic/data/cfg/skillmimic_dual.yaml` | Env config for dual humanoid (numEnvs, humanoidBSpacing, rewardWeights, coopRewardWeights, keyBodies, handBodies, etc.). |
| `skillmimic/data/cfg/hrl_dual_humanoid.yaml` | Env config for HRL dual (enableTaskObs, goalSize, coop rewards, same structure as skillmimic_dual). |
| `skillmimic/data/cfg/train/rlg/hrl_dual_humanoid.yaml` | Train config: algo `hrl_dual`, model `hrl_discrete`, network `hrl` separate, discrete actions, LLC config/checkpoint, control_mapping for A/B skills, PPO params. |

### Modified upstream files

- `skillmimic/run.py`: register and run task `HRLDualHumanoid` (and dual env).
- `skillmimic/utils/parse_task.py`: add task parsing for HRL dual.
- `skillmimic/env/tasks/humanoid_task.py`: camera position/layout changes for dual humanoid view (in “camera position good” commit).
- `run.sh` / `train.sh`: your local run/train commands (checkpoint path, GPU, optional save_images).

---

## 4. Architecture (HRL Dual)

- **Per env**: Actor 0 = Humanoid A (passer), Actor 1 = Humanoid B (catcher), Actor 2 = ball, Actor 3+ = projectiles.
- **HLC**: Single policy outputs **one discrete action** that encodes choices for both A and B (e.g. 6 options: 3 for A × 3 for B, or flattened). Stored in `control_mapping`: [4, 13, 31, 3, 13, 31] → A: pass/run/idle, B: catch/run/idle.
- **LLC**: Shared pre-trained SkillMimic policy (`skillmimic_llc.pth`), loaded via `llc_config` and `llc_checkpoint`; runs at higher frequency (e.g. `llc_steps: 5` per HLC step).
- **Observation**: Each humanoid gets base obs (body, object) plus task/goal obs when `enableTaskObs: True` (e.g. goal size 5). HRLDualHumanoid builds `_llc_obs_a` and `_llc_obs_b` for feeding the LLC.
- **Reward**: Combination of imitation (from MotionDataHandler / reference motion) and cooperative task rewards; in HRL config you set `imit_reward_w: 0` and `task_reward_w: 1.0` to train purely on task reward.

---

## 5. Commit-by-Commit Progress (Your Fork vs Upstream)

All of these are **your commits** (not on upstream):

1. **adf00a0 – Draft: Add HRL dual humanoid pass-and-catch task**  
   - Added dual humanoid env (`skillmimic_dual.py`), HRL env (`hrl_dual_humanoid.py`), HRL agent/players, env and train configs, run script entries, task registration.  
   - ~1.8k lines new code.

2. **0aba71a – Updated reward function**  
   - Tuned coop reward (e.g. pass_direction, standing, upright, ground_contact_penalty).  
   - Changes in `skillmimic_dual.py` (reward logic), `hrl_dual_humanoid.py`, and both YAML configs; extended `train.sh`.

3. **7fef00c – correct axis**  
   - Fixed coordinate/axis conventions for dual humanoid (so pass/catch direction and positions are consistent).  
   - Touched envs, HRL agent obs/action, and train config.

4. **b946a6b – correct axis + correct orb**  
   - Further axis/orientation fixes; “orb” likely refers to ball or camera orb.  
   - Updated `hrl_dual_humanoid.py`, `skillmimic_dual.py`, `hrl_dual_players.py`.

5. **065d63b – camera position good**  
   - Finalized camera setup for viewing both humanoids.  
   - Modified `humanoid_task.py` (camera position/layout), `skillmimic_dual.py`, `hrl_dual_agent.py`.

---

## 6. Current State

- **Training**: `train.sh` runs HRL dual with `HRLDualHumanoid`, `hrl_dual_humanoid.yaml` (env) and `train/rlg/hrl_dual_humanoid.yaml` (train), motion file `skillmimic/data/motions/BallPlay-M/pass`, GPU 1.
- **Inference**: `run.sh` runs test with checkpoint `output/SkillMimicDualHRL_12-21-17-25/nn/SkillMimicDualHRL.pth` (and commented `--save_images`).
- **Uncommitted local changes**:  
  - `run.sh`: checkpoint path and commented `# --save_images`.  
  - `train.sh`: `CUDA_VISIBLE_DEVICES=1`.  
  - Many untracked `debug_*.log` files and `skillmimic/data/videos/` (likely debug/visualization outputs).

---

## 7. Config Snippets (Quick Reference)

**Coop reward weights** (in `hrl_dual_humanoid.yaml` / `skillmimic_dual.yaml`):

- alive, ball_to_hand, pass_direction, catch_success, ball_height, standing, upright, ground_contact_penalty.

**HRL train** (`train/rlg/hrl_dual_humanoid.yaml`):

- algo: `hrl_dual`, model: `hrl_discrete`, network: `hrl` with `separate: True`.
- discrete action count 6; `control_mapping: [4, 13, 31, 3, 13, 31]`.
- LLC: `llc_config: skillmimic/data/cfg/train/rlg/skillmimic_as_llc.yaml`, `llc_checkpoint: skillmimic/data/models/mixedskills/nn/skillmimic_llc.pth`, `llc_steps: 5`.
- `task_reward_w: 1.0`, `imit_reward_w: 0.0`.

---

## 8. Suggested Next Steps (for Gemini or You)

- **Stability / robustness**: Check for edge cases (ball out of bounds, early terminations, resets) and reward scaling.
- **Evaluation**: Define a clear metric (e.g. catch rate, pass accuracy, episode return) and log it.
- **Ablations**: Vary coop reward weights, `llc_steps`, or HLC architecture (e.g. separate vs shared policy for A/B).
- **Data**: Try other motion files (e.g. `BallPlay-M/catch`) or mixed pass/catch for richer behavior.
- **Code hygiene**: Remove or gitignore `debug_*.log` and optionally `skillmimic/data/videos/`; keep `run.sh`/`train.sh` changes in a branch or document desired checkpoint/GPU in README.

---

*Summary generated from repo history and current files. Use this document to onboard Gemini (or another agent) on your fork and continue development.*
