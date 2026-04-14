# Ablation study (CTDE) — minimal 2–3 variants

## Budget
- **Training budget**: 2000 epochs (first pass)
- **Eval**: `test_episodes=500`, seeds `0..4`, `motion_file=skillmimic/data/motions/BallPlay-M/pass`

## Variants
- **Baseline (Full CTDE)**: `skillmimic/data/cfg/train/rlg/hrl_ctde_humanoid.yaml`
- **Ablation 1 (No trajectory predictor / aux)**: `skillmimic/data/cfg/train/rlg/hrl_ctde_humanoid_ablate_no_traj_pred.yaml`
- **Ablation 2 (No guidance penalty)**: `skillmimic/data/cfg/train/rlg/hrl_ctde_humanoid_ablate_no_guidance.yaml`

## How to run (commands)

### Train

```bash
# Baseline
./scripts/train_ctde_baseline.sh 2000

# Ablation 1
./scripts/train_ctde_ablation_no_traj_pred.sh 2000

# Ablation 2
./scripts/train_ctde_ablation_no_guidance.sh 2000
```

Each will write to `output/<EXP>_<timestamp>/nn/*.pth`. Use the produced checkpoint path for inference.

### Evaluate (multiseed)

```bash
./run_multiseed.sh --algo ctde --ckpt <CKPT_PATH>
```

### Aggregate into augmented CSV

```bash
conda run -n skillmimic python scripts/analyze_inference_scientific_eval.py \
  --input logs/inference_runs.csv \
  --output logs/inference_scientific_eval_augmented.csv
```

## What to report (copy from augmented CSV)

For each condition, report:
- **Primary**: `group_mean_avg_reward` ± 95% CI (`group_ci95_low/high_avg_reward`)
- **Secondary**: `group_mean_catch_success_rate`, `group_mean_pass_success_rate` ± CIs
- **Paired deltas vs baseline**: use `pairwise_*_vs_first_milestone_*` columns if you set the baseline as the “first milestone” in the allowlist, or compute paired deltas by aligning seeds.

## Notes
- If you want a stronger claim with tighter uncertainty, rerun eval with 10 seeds (update `run_multiseed.sh --seeds`), then regenerate the augmented CSV.

