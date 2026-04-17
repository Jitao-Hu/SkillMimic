#!/usr/bin/env bash
# Inference for the SkillMimicMultiPlayer scene: N role-less humanoids,
# one shared basketball, and one static basket, all driven by the existing
# SkillMimic LLC checkpoint replaying the same motion.
#
# Run from repository root. Requires GPU + Isaac Gym.
#
# You can override paths/params via env vars:
#   CKPT=<path>  NUM_ENVS=16  MOTION=<motion_dir>
#   CUDA_VISIBLE_DEVICES=0  HEADLESS=0
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

GPU="${CUDA_VISIBLE_DEVICES:-0}"
NUM_ENVS="${NUM_ENVS:-16}"
MOTION="${MOTION:-skillmimic/data/motions/BallPlay-M/layup}"
CKPT="${CKPT:-skillmimic/data/models/mixedskills/nn/skillmimic_llc.pth}"
HEADLESS="${HEADLESS:-0}"   # 0=render, 1=headless

TASK="SkillMimicMultiPlayer"
CFG_ENV="skillmimic/data/cfg/skillmimic_multiplayer.yaml"
CFG_TRAIN="skillmimic/data/cfg/train/rlg/skillmimic.yaml"
STATE_INIT="${STATE_INIT:-20}"
EPISODE_LENGTH="${EPISODE_LENGTH:-140}"

if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: checkpoint not found: $CKPT" >&2
  echo "Set CKPT=<path/to/skillmimic_llc.pth> and re-run." >&2
  exit 1
fi

HEADLESS_ARG=""
if [[ "$HEADLESS" == "1" ]]; then
  HEADLESS_ARG="--headless"
fi

echo "============================================================"
echo "  Multiplayer inference scene"
echo "  Task       : $TASK"
echo "  GPU        : $GPU"
echo "  Num envs   : $NUM_ENVS"
echo "  Motion     : $MOTION"
echo "  Checkpoint : $CKPT"
echo "  Headless   : $HEADLESS"
echo "============================================================"

CUDA_VISIBLE_DEVICES=$GPU python skillmimic/run.py --test \
  --task $TASK --num_envs $NUM_ENVS \
  --cfg_env $CFG_ENV \
  --cfg_train $CFG_TRAIN \
  --motion_file "$MOTION" \
  --checkpoint "$CKPT" \
  --state_init $STATE_INIT \
  --episode_length $EPISODE_LENGTH \
  $HEADLESS_ARG
