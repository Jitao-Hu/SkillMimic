time CUDA_VISIBLE_DEVICES=0 python skillmimic/run.py \
  --task HRLCTDEHumanoid \
  --cfg_env skillmimic/data/cfg/hrl_ctde_humanoid.yaml \
  --cfg_train skillmimic/data/cfg/train/rlg/hrl_ctde_humanoid.yaml \
  --motion_file skillmimic/data/motions/BallPlay-M/pass \
  --checkpoint skillmimic/data/models/mixedskills/nn/skillmimic_llc.pth \
  --resume_from output/CTDE_5000_20260401-13-07-32/nn/CTDE_5000.pth \
  --max_iterations 8000 \
  --experiment CTDE_8000 \
  --headless --num_envs 512 2>&1 | tee output/train_$(date +%Y%m%d_%H%M%S)_ctde_8000.log

# Notes:
# - If you want to resume training WITHOUT overwriting the original experiment directory,
#   pass a custom --experiment name (run.py will then create a fresh timestamped output dir).
#   Example:
#     --experiment SkillMimicCTDE_resume_11000