mkdir -p inference_log
LOG_FILE="inference_log/debug_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG_FILE") 2>&1
echo "Logging to: $LOG_FILE"

# timeout 300s python skillmimic/run.py --test --task HRLDualHumanoid --num_envs 1 \
#   --cfg_env skillmimic/data/cfg/hrl_dual_humanoid.yaml \
#   --cfg_train skillmimic/data/cfg/train/rlg/hrl_dual_humanoid.yaml \
#   --motion_file skillmimic/data/motions/BallPlay-M/pass \
#   --checkpoint output/SkillMimicDualHRL_04-23-30-10/nn/SkillMimicDualHRL.pth \
#   --save_images

# CUDA_VISIBLE_DEVICES=1 timeout 30s python skillmimic/run.py --test --task HRLDualHumanoid --num_envs 1 \
#   --cfg_env skillmimic/data/cfg/hrl_dual_humanoid.yaml \
#   --cfg_train skillmimic/data/cfg/train/rlg/hrl_dual_humanoid.yaml \
#   --motion_file skillmimic/data/motions/BallPlay-M/pass \
#   --checkpoint output/SkillMimicDualHRL_09-14-40-17/nn/SkillMimicDualHRL.pth \
#   --headless

# 使用最新训练好的 HRL Dual checkpoint 做可视化 inference（需要显示屏）：
# CUDA_VISIBLE_DEVICES=1 python skillmimic/run.py --test --task HRLDualHumanoid --num_envs 1 \
#   --cfg_env skillmimic/data/cfg/hrl_dual_humanoid.yaml \
#   --cfg_train skillmimic/data/cfg/train/rlg/hrl_dual_humanoid.yaml \
#   --motion_file skillmimic/data/motions/BallPlay-M/pass \
#   --checkpoint output/SkillMimicDualHRL_08-21-42-38/nn/SkillMimicDualHRL.pth
#   # --save_images  # 需要导出图片/视频时可取消注释

# 使用最新 checkpoint 做 headless inference（无需显示屏，适合 SSH/后台跑）：
CUDA_VISIBLE_DEVICES=1 python skillmimic/run.py --test --task HRLDualHumanoid --num_envs 1 \
  --cfg_env skillmimic/data/cfg/hrl_dual_humanoid.yaml \
  --cfg_train skillmimic/data/cfg/train/rlg/hrl_dual_humanoid.yaml \
  --motion_file skillmimic/data/motions/BallPlay-M/pass \
  --checkpoint output/SkillMimicDualHRL_08-21-42-38/nn/SkillMimicDualHRL.pth \
  --save_images
  # --headless

# CUDA_VISIBLE_DEVICES=1 timeout 60s python skillmimic/run.py --test --task HRLDualHumanoid --num_envs 1 \
#   --cfg_env skillmimic/data/cfg/hrl_dual_humanoid.yaml \
#   --cfg_train skillmimic/data/cfg/train/rlg/hrl_dual_humanoid.yaml \
#   --motion_file skillmimic/data/motions/BallPlay-M/pass \
#   --checkpoint output/SkillMimicDualHRL_09-14-40-17/nn/SkillMimicDualHRL.pth \
#   --save_images