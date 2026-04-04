# Scientific evaluation report (primary and secondary metrics)

This report summarizes the protocol in `docs/scientific_eval_plan.md`, the instrumentation added for **catch / pass success rates**, and how to refresh numbers after re-running inference.

## Primary metric (mean episodic return)

Aggregates below are from `logs/inference_scientific_eval_augmented.csv`, produced by:

```bash
conda run -n skillmimic python scripts/analyze_inference_scientific_eval.py \
  --input logs/inference_runs.csv \
  --output logs/inference_scientific_eval_augmented.csv
```

| Condition | Mean `avg_reward` (n=5 seeds) | 95% CI (approx.) |
|-----------|-------------------------------|------------------|
| CTDE_2000 | 49.56 | see `group_ci95_*_avg_reward` in augmented CSV |
| CTDE_5000 | 49.90 | same |
| CTDE_8000 | 51.92 | same |

Paired deltas vs the first milestone (2000) and vs the previous milestone are in the augmented CSV columns `pairwise_*_milestone_reward`.

## Secondary metrics (catch / pass)

After the player change in `skillmimic/learning/hrl_players_discrete.py`, each completed inference run logs full-eval **catch** and **pass** rates into `logs/inference_runs.csv` (`catch_success_rate`, `pass_success_rate`, and raw counts).

**Historical rows** (before this change) have empty secondary columns. The analyze script reports `Secondary (catch/pass) present for all 5 seeds: False` until you re-run the multiseed matrix.

### Refresh secondary metrics

From the repository root (GPU required):

```bash
./scripts/rerun_scientific_eval_multiseed_all.sh
```

Optional HRL-DUAL baseline at the same protocol:

```bash
RUN_DUAL_BASELINE=1 ./scripts/rerun_scientific_eval_multiseed_all.sh
```

Then regenerate the augmented CSV (command above). New columns include `group_mean_catch_success_rate`, `group_mean_pass_success_rate`, and paired comparisons.

## Appendix: informal historical baseline

Early March 2026 rows in `logs/inference_runs.csv` used **different** checkpoints (often overwritten), **single seeds**, and sometimes **`test_episodes=50`**. Those numbers are useful for motivation only, not for statistical comparison to the current CTDE_2000/5000/8000 multiseed protocol.

## Trajectory-predictor ablation

Train from LLC with predictor disabled:

```bash
chmod +x scripts/train_ctde_ablation_no_traj_pred.sh
./scripts/train_ctde_ablation_no_traj_pred.sh 8000
```

Config: `skillmimic/data/cfg/train/rlg/hrl_ctde_humanoid_ablate_no_traj_pred.yaml`. Compare inference at the **same epoch budget** as full CTDE using `run_multiseed.sh` on the new checkpoint.
