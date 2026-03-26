mkdir -p inference_log
LOG_FILE="inference_log/inference_ctde__$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG_FILE") 2>&1
echo "Logging to: $LOG_FILE"

# Use latest CTDE checkpoint for headless inference.
LATEST_CKPT=$(ls -t output/SkillMimicCTDE_*/nn/SkillMimicCTDE.pth 2>/dev/null | head -n 1)
if [ -z "$LATEST_CKPT" ]; then
  echo "ERROR: No CTDE checkpoint found at output/SkillMimicCTDE_*/nn/SkillMimicCTDE.pth"
  exit 1
fi
echo "Using checkpoint: $LATEST_CKPT"

CUDA_VISIBLE_DEVICES=0 python skillmimic/run.py --test --task HRLCTDEHumanoid --num_envs 1 \
  --cfg_env skillmimic/data/cfg/hrl_ctde_humanoid.yaml \
  --cfg_train skillmimic/data/cfg/train/rlg/hrl_ctde_humanoid.yaml \
  --motion_file skillmimic/data/motions/BallPlay-M/pass \
  --checkpoint "$LATEST_CKPT" \
  --test_episodes 50 \
  --save_images
  # --headless
  