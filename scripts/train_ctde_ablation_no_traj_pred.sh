#!/usr/bin/env bash
# Train CTDE without trajectory predictor / traj auxiliary loss (ablation).
# Start from LLC only; produces weights incompatible with full CTDE checkpoints.
#
# Usage (repo root):
#   ./scripts/train_ctde_ablation_no_traj_pred.sh [max_iterations]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
ITERS="${1:-2000}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
EXP="CTDE_NoTrajPred_${ITERS}"

echo "Training ablation to $ITERS epochs as experiment $EXP (GPU=$GPU)"

CUDA_VISIBLE_DEVICES="$GPU" python skillmimic/run.py \
  --task HRLCTDEHumanoid \
  --cfg_env skillmimic/data/cfg/hrl_ctde_humanoid.yaml \
  --cfg_train skillmimic/data/cfg/train/rlg/hrl_ctde_humanoid_ablate_no_traj_pred.yaml \
  --motion_file skillmimic/data/motions/BallPlay-M/pass \
  --checkpoint skillmimic/data/models/mixedskills/nn/skillmimic_llc.pth \
  --experiment "$EXP" \
  --max_iterations "$ITERS" \
  --headless --num_envs 512

echo "Evaluate with run_multiseed.sh on the new nn/*.pth under output/${EXP}_*/"
