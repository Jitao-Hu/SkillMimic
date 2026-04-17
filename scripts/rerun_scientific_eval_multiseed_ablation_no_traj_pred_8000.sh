#!/usr/bin/env bash
# Re-run scientific-eval style inference for the "no trajectory predictor" CTDE
# ablation checkpoint trained 8000 epochs (500 episodes × seeds 0–4) so logs/inference_runs.csv picks
# up catch/pass secondary metrics from the player.
#
# Run from repository root. Requires GPU + Isaac Gym (same as training).
#
# You can override paths/params via env vars:
#   ABLATE_CKPT=<path>  SEEDS="0 1 2 3 4"  EPISODES=500  CUDA_VISIBLE_DEVICES=1  HEADLESS=1
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

GPU="${CUDA_VISIBLE_DEVICES:-1}"
SEEDS="${SEEDS:-0 1 2 3 4}"
EPISODES="${EPISODES:-500}"
HEADLESS="${HEADLESS:-1}" # 1=headless, 0=render

ABLATE_CKPT="${ABLATE_CKPT:-output/CTDE_NoTrajPred_8000_20260415-17-44-57/nn/CTDE_NoTrajPred_8000.pth}"

TASK="HRLCTDEHumanoid"
CFG_ENV="skillmimic/data/cfg/hrl_ctde_humanoid.yaml"
CFG_TRAIN="skillmimic/data/cfg/train/rlg/hrl_ctde_humanoid_ablate_no_traj_pred.yaml"
MOTION_FILE="skillmimic/data/motions/BallPlay-M/pass"

if [[ ! -f "$ABLATE_CKPT" ]]; then
  echo "ERROR: ABLATE_CKPT not found: $ABLATE_CKPT" >&2
  echo "Set ABLATE_CKPT=<path/to/*.pth> and re-run." >&2
  exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p inference_log

LABEL="ablate_no_traj_pred_$(basename "$(dirname "$(dirname "$ABLATE_CKPT")")")"
echo "============================================================"
echo "  Multi-seed inference (ablation): $LABEL"
echo "  Checkpoint : $ABLATE_CKPT"
echo "  GPU        : $GPU"
echo "  Seeds      : $SEEDS"
echo "  Episodes   : $EPISODES per seed"
echo "  Headless   : $HEADLESS"
echo "============================================================"
echo ""

HEADLESS_ARG=""
if [[ "$HEADLESS" == "1" ]]; then
  HEADLESS_ARG="--headless"
fi

FAILED=0
COMPLETED=0
SEED_ARR=($SEEDS)
NUM_SEEDS=${#SEED_ARR[@]}

for SEED in "${SEED_ARR[@]}"; do
  RUN_LABEL="${LABEL}_seed${SEED}"
  LOG_FILE="inference_log/multiseed_${LABEL}_seed${SEED}_${TIMESTAMP}.log"

  echo "--- [$((COMPLETED + FAILED + 1))/$NUM_SEEDS] seed=$SEED ---"
  echo "  Log: $LOG_FILE"

  CMD="CUDA_VISIBLE_DEVICES=$GPU python skillmimic/run.py --test \
    --task $TASK --num_envs 1 \
    --cfg_env $CFG_ENV \
    --cfg_train $CFG_TRAIN \
    --motion_file $MOTION_FILE \
    --checkpoint $ABLATE_CKPT \
    --seed $SEED \
    --test_episodes $EPISODES \
    $HEADLESS_ARG"

  if eval "$CMD" 2>&1 | tee "$LOG_FILE"; then
    echo "  DONE seed=$SEED"
    COMPLETED=$((COMPLETED + 1))
  else
    echo "  FAILED seed=$SEED (exit code $?)"
    FAILED=$((FAILED + 1))
  fi
  echo ""
done

echo "============================================================"
echo "  Multi-seed inference complete: $LABEL"
echo "  Completed : $COMPLETED / $NUM_SEEDS"
if [[ $FAILED -gt 0 ]]; then
  echo "  Failed    : $FAILED / $NUM_SEEDS"
fi
echo "  Logs in   : inference_log/multiseed_${LABEL}_*_${TIMESTAMP}.log"
echo "============================================================"

if [[ $FAILED -gt 0 ]]; then
  exit 1
fi

echo "Done. Regenerate augmented CSV:"
echo "  conda run -n skillmimic python scripts/analyze_inference_scientific_eval.py"

