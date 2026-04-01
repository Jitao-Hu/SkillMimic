# Scientific Evaluation Plan (SkillMimic / CTDE / HRL-DUAL)

This document is a **repeatable protocol** for making inference results (e.g., `avg_reward`) scientifically defensible: comparable across runs, robust to randomness, and reported with uncertainty.

---

## 0) Goal and principles

We want to answer questions like:
- Does **CTDE (or HRL-DUAL)** perform better than baseline?
- Does increasing training epochs improve performance?
- Does a specific change (e.g., ball-trajectory observation, LLC strength, reward tweak) improve performance?

To be “scientific”, results should be:
- **Controlled**: only the intended factor changes between conditions.
- **Replicated**: multiple independent seeds / runs.
- **Quantified**: report uncertainty (confidence intervals) and effect sizes, not only a single mean.
- **Pre-specified**: define primary metrics and success criteria before running the sweep.

---

## 1) Define experiment: hypothesis, conditions, and scope (pre-registration lite)

Create a short “experiment card” for each study (copy/paste this block into your notes or a new file).

### Experiment card template
- **Hypothesis**: (e.g., “CTDE @5000 epochs > CTDE @2000 epochs on BallPlay-M/pass.”)
- **Independent variable(s)**: exactly what changes (checkpoint, config, architecture).
- **Control condition**: baseline to compare against.
- **Treatment condition(s)**: the change(s).
- **Task**: (e.g., `HRLCTDEHumanoid`).
- **Motion file(s)**: (e.g., `skillmimic/data/motions/BallPlay-M/pass`).
- **Evaluation horizon**: `test_episodes = N` (per seed).
- **Seeds**: explicit list (see Section 4).
- **Primary metric**: choose ONE primary (see Section 2).
- **Secondary metrics**: optional (see Section 2).
- **Decision rule**: what counts as “better” (see Section 6).

---

## 2) Metrics: choose primary + secondary (avoid metric shopping)

### Primary metric (pick one)
Choose **one** as the headline metric for each study:
- **Mean episodic return** (what your logs call `av reward` / `avg_reward`): mean over episodes for each seed; then aggregate across seeds.
- **Catch success rate** (often more interpretable for pass-and-catch): proportion of successful catches over episodes (or over flights).

If you pick mean return as primary, still track success rate as secondary to ensure reward isn’t being “hacked” by unintended behaviors.

### Secondary metrics (recommended)
- **Mean episode length** (your logs: `av steps`).
- **CatchStats** components (pass%, catch%, fails, ground contact, etc.).
- **Robustness**: evaluate multiple motions (or multiple initializations) if generalization matters.

### Metric definitions (must be fixed)
Write down exact definitions to prevent “moving goalposts”:
- What is an “episode” termination?
- What counts as “catch success” (event definition + time window)?
- Are stats measured **per episode** or **per flight** (your logs show `flights_total=...`)?

---

## 3) Standardize evaluation settings (control the protocol)

For scientific comparisons, keep these fixed across conditions:
- **Task**: same `--task`
- **Config files**: same `--cfg_env` and `--cfg_train` (unless the config change is the treatment)
- **Motion file**: same `--motion_file`
- **Number of environments**: same `--num_envs` (prefer `1` for deterministic-style evaluation; use larger only if you prove it does not change metrics)
- **Episode count**: same `--test_episodes`
- **Headless/render**: keep consistent; rendering can change performance/timing (and sometimes even behavior).
- **Code version**: record the git commit hash in the run log (see Section 7).
- **Checkpoint**: record exact path; don’t compare different checkpoints unless that’s the intended variable.

Also record (not necessarily control):
- GPU/driver/CUDA version, PyTorch version, IsaacGym build.

---

## 4) Replication plan: seeds and sample size

### Minimum recommendation
- Use **at least 5 seeds** per condition for preliminary conclusions.
- Use **10 seeds** if results are noisy or if you want stronger claims.

### Seed strategy (important)
Use the **same seed list** for each condition (paired design), e.g.:
`SEEDS = [0, 1, 2, 3, 4]` (or any fixed list you choose).

This reduces variance and makes comparisons cleaner.

### Episodes per seed
Keep `test_episodes` fixed across conditions. Common choices:
- **200–500** episodes per seed if evaluation is affordable.
- If expensive, consider **fewer episodes but more seeds**, then validate with a longer evaluation for the final comparison.

---

## 5) Execution steps (what to actually run)

### Step A — Create a single source of truth for evaluation commands
Create one script (or one section in an existing script) that takes:
- condition name
- checkpoint path
- seed
- test_episodes
- output log path

Goal: avoid ad-hoc command edits that accidentally change settings.

### Step B — Run evaluations for each condition × seed
For each condition (control and treatment):
- For each seed in `SEEDS`:
  - Run inference with fixed args
  - Save log to `inference_log/condition_seed_timestamp.log`
  - Append a row to `logs/inference_runs.csv` (you already do this)

### Step C — Sanity checks after each run batch
Before interpreting results, confirm:
- All runs **completed** successfully (no CUDA illegal access, segfault, interruption).
- Same task/motion/config/checkpoint as intended.
- Same `test_episodes` per seed.

---

## 6) Analysis plan (how to turn runs into scientific claims)

Let each seed produce one scalar \(x_s\) (e.g., mean return over that seed’s episodes).

### Primary comparison (two conditions)
Compute across seeds:
- **Mean and 95% CI** for each condition.
- **Effect size**: difference in means \( \Delta = \bar{x}_{treat} - \bar{x}_{ctrl} \).

Recommended inference:
- **Paired bootstrap CI** over seeds (best default, minimal assumptions) if you used the same seed list.
- Or **paired t-test** if the distribution across seeds looks approximately normal (often OK at n≥10, but bootstrap is safer).

### Report format (recommended)
For each condition:
- `mean ± 95% CI across seeds` (not across episodes)
And for the comparison:
- `Δ mean` with `95% CI` and optionally `p-value`.

### Don’t over-claim
If you only have one seed (or one run):
- You can report it as **anecdotal / preliminary**.
- You cannot claim it is statistically reliable.

---

## 7) Logging requirements (so you can audit and reproduce)

Each run should record at minimum:
- git commit hash
- timestamp
- condition name
- checkpoint path
- cfg_env, cfg_train
- task, motion_file
- seed
- test_episodes
- num_envs
- headless/render flag
- avg_reward (mean)
- avg_steps (mean)
- exit status + error detail

You already record many of these in `logs/inference_runs.csv`; add missing items if needed (especially git commit hash and a clean condition label).

---

## 8) Common pitfalls and how this plan prevents them

- **Comparing different seeds**: can exaggerate differences → use paired seed list.
- **Changing test_episodes midstream**: changes estimator variance → fix it per study.
- **Silent failures** (segfault/CUDA errors): biases results (only “good” runs survive) → require completion checks.
- **Protocol drift** (headless vs not, config changes): confounds comparisons → lock settings.
- **Reward hacking**: avg reward rises but success drops → track success as secondary.

---

## 9) Practical checklist (copy/paste before you announce results)

- [ ] Hypothesis written; primary metric chosen.
- [ ] Control and treatment differ by exactly one intended factor.
- [ ] Same task/motion/config/test_episodes/num_envs across conditions.
- [ ] Fixed seed list used for all conditions.
- [ ] ≥5 seeds per condition (≥10 preferred for strong claims).
- [ ] All runs completed (no errors/interruption).
- [ ] Report mean ± 95% CI across seeds.
- [ ] Report effect size \(Δ\) with CI; avoid “significant” language unless tested.
- [ ] Secondary metrics checked (catch success, steps, CatchStats).

