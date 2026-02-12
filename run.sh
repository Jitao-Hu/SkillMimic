LOG_FILE="debug_$(date +%Y%m%d_%H%M%S).log"
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

# 把 run.sh 中的 --headless 去掉，或者直接运行：
CUDA_VISIBLE_DEVICES=1 python skillmimic/run.py --test --task HRLDualHumanoid --num_envs 1 \
  --cfg_env skillmimic/data/cfg/hrl_dual_humanoid.yaml \
  --cfg_train skillmimic/data/cfg/train/rlg/hrl_dual_humanoid.yaml \
  --motion_file skillmimic/data/motions/BallPlay-M/pass \
  --checkpoint output/SkillMimicDualHRL_09-14-40-17/nn/SkillMimicDualHRL.pth

# CUDA_VISIBLE_DEVICES=1 timeout 60s python skillmimic/run.py --test --task HRLDualHumanoid --num_envs 1 \
#   --cfg_env skillmimic/data/cfg/hrl_dual_humanoid.yaml \
#   --cfg_train skillmimic/data/cfg/train/rlg/hrl_dual_humanoid.yaml \
#   --motion_file skillmimic/data/motions/BallPlay-M/pass \
#   --checkpoint output/SkillMimicDualHRL_09-14-40-17/nn/SkillMimicDualHRL.pth \
#   --save_images