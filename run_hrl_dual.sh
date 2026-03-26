mkdir -p inference_log
LOG_FILE="inference_log/inference_hrl_dual__$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG_FILE") 2>&1
echo "Logging to: $LOG_FILE"

# Pre-CTDE (Dual HRL) for comparison.
# Use latest Dual HRL checkpoint automatically.
LATEST_CKPT=$(ls -t output/SkillMimicDualHRL_*/nn/SkillMimicDualHRL.pth 2>/dev/null | head -n 1)
if [ -z "$LATEST_CKPT" ]; then
  echo "ERROR: No Dual HRL checkpoint found at output/SkillMimicDualHRL_*/nn/SkillMimicDualHRL.pth"
  exit 1
fi
echo "Using checkpoint: $LATEST_CKPT"
CUDA_VISIBLE_DEVICES=1 python skillmimic/run.py --test --task HRLDualHumanoid --num_envs 1 \
  --cfg_env skillmimic/data/cfg/hrl_dual_humanoid.yaml \
  --cfg_train skillmimic/data/cfg/train/rlg/hrl_dual_humanoid.yaml \
  --motion_file skillmimic/data/motions/BallPlay-M/pass \
  --checkpoint "$LATEST_CKPT" \
  --test_episodes 500 \
  --headless