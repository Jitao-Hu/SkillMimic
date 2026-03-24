mkdir -p inference_log
LOG_FILE="inference_log/inference_hrl_dual__$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG_FILE") 2>&1
echo "Logging to: $LOG_FILE"

# Pre-CTDE (Dual HRL) for comparison: same 100 episodes, compare catch stats.
# Latest Dual HRL weights: SkillMimicDualHRL_20260319-15-53-47
CUDA_VISIBLE_DEVICES=1 python skillmimic/run.py --test --task HRLDualHumanoid --num_envs 1 \
  --cfg_env skillmimic/data/cfg/hrl_dual_humanoid.yaml \
  --cfg_train skillmimic/data/cfg/train/rlg/hrl_dual_humanoid.yaml \
  --motion_file skillmimic/data/motions/BallPlay-M/pass \
  --checkpoint output/SkillMimicDualHRL_20260319-15-53-47/nn/SkillMimicDualHRL.pth \
  --test_episodes 500 \
  --headless