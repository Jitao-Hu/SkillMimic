#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Multi-seed inference runner
#
# Usage examples:
#
#   # CTDE checkpoint, 5 seeds, 500 episodes each (defaults):
#   ./run_multiseed.sh --ckpt output/CTDE_2000_20260331-13-48-07/nn/CTDE_2000.pth
#
#   # Same but with custom label for logs:
#   ./run_multiseed.sh --ckpt output/CTDE_2000_20260331-13-48-07/nn/CTDE_2000.pth --label ctde_2000
#
#   # HRL-DUAL checkpoint:
#   ./run_multiseed.sh --ckpt output/SkillMimicDualHRL_20260319-15-53-47/nn/SkillMimicDualHRL.pth --algo dual
#
#   # Custom seeds and episodes:
#   ./run_multiseed.sh --ckpt <path> --seeds "0 1 2 3 4 5 6 7 8 9" --episodes 1000
#
#   # Use a specific GPU:
#   ./run_multiseed.sh --ckpt <path> --gpu 1
#
#   # Dry-run (print commands without executing):
#   ./run_multiseed.sh --ckpt <path> --dry-run
# ============================================================================

# ----- Defaults (matching scientific_eval_plan.md) -----
SEEDS="0 1 2 3 4"
TEST_EPISODES=500
GPU=0
ALGO="ctde"        # "ctde" or "dual"
CKPT=""
LABEL=""
DRY_RUN=false
HEADLESS=true

# ----- Parse arguments -----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt)       CKPT="$2";          shift 2 ;;
    --algo)       ALGO="$2";          shift 2 ;;
    --seeds)      SEEDS="$2";         shift 2 ;;
    --episodes)   TEST_EPISODES="$2"; shift 2 ;;
    --gpu)        GPU="$2";           shift 2 ;;
    --label)      LABEL="$2";         shift 2 ;;
    --dry-run)    DRY_RUN=true;       shift   ;;
    --no-headless) HEADLESS=false;    shift   ;;
    -h|--help)
      sed -n '3,/^# =====/p' "$0" | head -n -1
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

# ----- Validate -----
if [[ -z "$CKPT" ]]; then
  echo "ERROR: --ckpt is required. Run with --help for usage." >&2
  exit 1
fi

if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: Checkpoint not found: $CKPT" >&2
  exit 1
fi

# ----- Algo-specific config -----
case "$ALGO" in
  ctde)
    TASK="HRLCTDEHumanoid"
    CFG_ENV="skillmimic/data/cfg/hrl_ctde_humanoid.yaml"
    CFG_TRAIN="skillmimic/data/cfg/train/rlg/hrl_ctde_humanoid.yaml"
    ;;
  dual)
    TASK="HRLDualHumanoid"
    CFG_ENV="skillmimic/data/cfg/hrl_dual_humanoid.yaml"
    CFG_TRAIN="skillmimic/data/cfg/train/rlg/hrl_dual_humanoid.yaml"
    ;;
  *)
    echo "ERROR: --algo must be 'ctde' or 'dual', got '$ALGO'" >&2
    exit 1
    ;;
esac

MOTION_FILE="skillmimic/data/motions/BallPlay-M/pass"

# ----- Label for logging -----
if [[ -z "$LABEL" ]]; then
  LABEL="${ALGO}_$(basename "$(dirname "$(dirname "$CKPT")")")"
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p inference_log

# ----- Header -----
SEED_ARR=($SEEDS)
NUM_SEEDS=${#SEED_ARR[@]}
echo "============================================================"
echo "  Multi-seed inference: $LABEL"
echo "  Checkpoint : $CKPT"
echo "  Algo       : $ALGO ($TASK)"
echo "  Seeds      : ${SEED_ARR[*]} ($NUM_SEEDS seeds)"
echo "  Episodes   : $TEST_EPISODES per seed"
echo "  GPU        : $GPU"
echo "  Headless   : $HEADLESS"
echo "  Total runs : $NUM_SEEDS"
echo "============================================================"
echo ""

HEADLESS_ARG=""
if $HEADLESS; then
  HEADLESS_ARG="--headless"
fi

# ----- Run loop -----
COMPLETED=0
FAILED=0

for SEED in "${SEED_ARR[@]}"; do
  RUN_LABEL="${LABEL}_seed${SEED}"
  LOG_FILE="inference_log/multiseed_${LABEL}_seed${SEED}_${TIMESTAMP}.log"

  echo "--- [$((COMPLETED + FAILED + 1))/$NUM_SEEDS] seed=$SEED ---"

  CMD="CUDA_VISIBLE_DEVICES=$GPU python skillmimic/run.py --test \
    --task $TASK --num_envs 1 \
    --cfg_env $CFG_ENV \
    --cfg_train $CFG_TRAIN \
    --motion_file $MOTION_FILE \
    --checkpoint $CKPT \
    --seed $SEED \
    --test_episodes $TEST_EPISODES \
    $HEADLESS_ARG"

  if $DRY_RUN; then
    echo "  [DRY-RUN] $CMD"
    echo ""
    COMPLETED=$((COMPLETED + 1))
    continue
  fi

  echo "  Log: $LOG_FILE"
  START_TIME=$(date +%s)

  if eval "$CMD" 2>&1 | tee "$LOG_FILE"; then
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    MINS=$((ELAPSED / 60))
    SECS=$((ELAPSED % 60))
    echo "  DONE seed=$SEED (${MINS}m ${SECS}s)"
    COMPLETED=$((COMPLETED + 1))
  else
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    echo "  FAILED seed=$SEED (exit code $?, ${ELAPSED}s)"
    FAILED=$((FAILED + 1))
  fi
  echo ""
done

# ----- Summary -----
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
