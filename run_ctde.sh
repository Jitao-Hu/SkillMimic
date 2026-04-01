mkdir -p inference_log
LOG_FILE="inference_log/inference_ctde__$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG_FILE") 2>&1
echo "Logging to: $LOG_FILE"

# Checkpoint selection:
# - If CKPT is provided, use it (most reproducible).
# - Otherwise fall back to latest checkpoint by mtime.
if [ -n "${CKPT:-}" ]; then
  SELECTED_CKPT="$CKPT"
else
  SELECTED_CKPT=$(ls -t output/SkillMimicCTDE_*/nn/SkillMimicCTDE.pth 2>/dev/null | head -n 1)
fi

if [ -z "${SELECTED_CKPT:-}" ]; then
  echo "ERROR: No CTDE checkpoint found at output/SkillMimicCTDE_*/nn/SkillMimicCTDE.pth"
  exit 1
fi
echo "Using checkpoint: $SELECTED_CKPT"

# Seed control:
# If SEED is provided, pass it through (run.py supports --seed).
SEED_ARGS=()
if [ -n "${SEED:-}" ]; then
  SEED_ARGS+=(--seed "$SEED")
fi

CUDA_VISIBLE_DEVICES=1 python skillmimic/run.py --test --task HRLCTDEHumanoid --num_envs 1 \
  --cfg_env skillmimic/data/cfg/hrl_ctde_humanoid.yaml \
  --cfg_train skillmimic/data/cfg/train/rlg/hrl_ctde_humanoid.yaml \
  --motion_file skillmimic/data/motions/BallPlay-M/pass \
  --checkpoint "$SELECTED_CKPT" \
  "${SEED_ARGS[@]}" \
  --test_episodes 500 \
  # --headless
  # --save_images
  
  