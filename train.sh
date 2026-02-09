time CUDA_VISIBLE_DEVICES=1 python skillmimic/run.py --task HRLDualHumanoid \
  --cfg_env skillmimic/data/cfg/hrl_dual_humanoid.yaml \
  --cfg_train skillmimic/data/cfg/train/rlg/hrl_dual_humanoid.yaml \
  --motion_file skillmimic/data/motions/BallPlay-M/pass \
  --llc_checkpoint skillmimic/data/models/mixedskills/nn/skillmimic_llc.pth \
  --headless --num_envs 256 2>&1 | tee output/train_$(date +%Y%m%d_%H%M%S).log