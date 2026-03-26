mkdir -p inference_log
LOG_FILE="inference_log/inference_hrl_dual_no_bt_predict_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG_FILE") 2>&1
echo "Logging to: $LOG_FILE"

# Pre-bt-predict Dual HRL weights (before ball trajectory prediction reward)
# Chosen checkpoint: SkillMimicDualHRL_20260309-22-34-19
CUDA_VISIBLE_DEVICES=1 python skillmimic/run.py --test --task HRLDualHumanoid --num_envs 1 \
  --cfg_env skillmimic/data/cfg/hrl_dual_humanoid.yaml \
  --cfg_train skillmimic/data/cfg/train/rlg/hrl_dual_humanoid.yaml \
  --motion_file skillmimic/data/motions/BallPlay-M/pass \
  --checkpoint output/SkillMimicDualHRL_20260309-22-34-19/nn/SkillMimicDualHRL.pth \
  --test_episodes 500 \
  --headless