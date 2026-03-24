mkdir -p inference_log
LOG_FILE="inference_log/inference_ctde__$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG_FILE") 2>&1
echo "Logging to: $LOG_FILE"

# 使用最新 checkpoint 做 headless inference（无需显示屏，适合 SSH/后台跑）：
# Latest: SkillMimicCTDE_20260313-18-06-00 (HRL CTDE)
CUDA_VISIBLE_DEVICES=0 python skillmimic/run.py --test --task HRLCTDEHumanoid --num_envs 1 \
  --cfg_env skillmimic/data/cfg/hrl_ctde_humanoid.yaml \
  --cfg_train skillmimic/data/cfg/train/rlg/hrl_ctde_humanoid.yaml \
  --motion_file skillmimic/data/motions/BallPlay-M/pass \
  --checkpoint output/SkillMimicCTDE_20260313-18-06-00/nn/SkillMimicCTDE.pth \
  --test_episodes 500 \
  --headless
  # --save_images
  