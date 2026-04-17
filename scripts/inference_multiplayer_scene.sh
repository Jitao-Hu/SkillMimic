#!/usr/bin/env bash
# Inference for the SkillMimicMultiPlayer scene.
#
#   N role-less humanoids share one basketball and one drawn scoring goal
#   (the "hoop" rendered as line segments by HRLScoringLayup._draw_task).
#   Every humanoid runs the HRL+LLC policy independently: each picks its own
#   discrete skill, the LLC produces its own joint targets, so the humanoids
#   actually contest the ball instead of moving in lockstep.
#
#   --num_envs on the CLI is the number of *physical* scenes; the number of
#   virtual envs rl_games sees is num_envs * numPlayers (configured in
#   skillmimic/data/cfg/skillmimic_multiplayer.yaml).
#
# Run from repository root. Requires GPU + Isaac Gym.
#
# Common overrides via env vars:
#   NUM_ENVS=1                            (physical scenes)
#   NUM_PLAYERS=3                         (edit yaml or pass via python)
#   CKPT=skillmimic/data/models/hlc_scoring/nn/SkillMimic.pth   (HRL/HLC)
#   LLC_CKPT=skillmimic/data/models/mixedskills/nn/skillmimic_llc.pth
#   MOTION=skillmimic/data/motions/BallPlay-M/run
#   PROJTYPE=Mouse                        (viewer mouse picks the scoring target)
#   HEADLESS=0                            (1 = no viewer)
#   CUDA_VISIBLE_DEVICES=0
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

GPU="${CUDA_VISIBLE_DEVICES:-0}"
NUM_ENVS="${NUM_ENVS:-1}"
MOTION="${MOTION:-skillmimic/data/motions/BallPlay-M/run}"
CKPT="${CKPT:-skillmimic/data/models/hlc_scoring/nn/SkillMimic.pth}"
LLC_CKPT="${LLC_CKPT:-skillmimic/data/models/mixedskills/nn/skillmimic_llc.pth}"
PROJTYPE="${PROJTYPE:-Mouse}"
HEADLESS="${HEADLESS:-0}"

TASK="SkillMimicMultiPlayer"
CFG_ENV="skillmimic/data/cfg/skillmimic_multiplayer.yaml"
CFG_TRAIN="skillmimic/data/cfg/train/rlg/hrl_humanoid_discrete_layupscore.yaml"

for F in "$CKPT" "$LLC_CKPT"; do
  if [[ ! -f "$F" ]]; then
    echo "ERROR: checkpoint not found: $F" >&2
    exit 1
  fi
done

if [[ ! -d "$MOTION" && ! -f "$MOTION" ]]; then
  echo "ERROR: motion file / dir not found: $MOTION" >&2
  exit 1
fi

HEADLESS_ARG=""
if [[ "$HEADLESS" == "1" ]]; then
  HEADLESS_ARG="--headless"
  # Mouse projectiles need a viewer; drop projtype when headless.
  PROJTYPE="None"
fi

echo "============================================================"
echo "  SkillMimicMultiPlayer inference"
echo "  Task           : $TASK"
echo "  GPU            : $GPU"
echo "  Physical envs  : $NUM_ENVS"
echo "  Motion         : $MOTION"
echo "  HRL ckpt       : $CKPT"
echo "  LLC ckpt       : $LLC_CKPT"
echo "  Projtype       : $PROJTYPE"
echo "  Headless       : $HEADLESS"
echo "============================================================"

CUDA_VISIBLE_DEVICES=$GPU python skillmimic/run.py --test \
  --task $TASK \
  --num_envs $NUM_ENVS \
  --projtype "$PROJTYPE" \
  --cfg_env "$CFG_ENV" \
  --cfg_train "$CFG_TRAIN" \
  --motion_file "$MOTION" \
  --checkpoint "$CKPT" \
  --llc_checkpoint "$LLC_CKPT" \
  $HEADLESS_ARG
