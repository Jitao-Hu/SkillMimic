March 19

在observation里加入了ball trajectory，还在训练，wip
    - uses physics formula to calculate
        - PREDICT_DT=0.5
    - [optional] learned trajectory predictor
        - pred_type = gru or mlp
        - input ball_history
        - output prediction of ball position
        - position and velocity will then be converted into agent's heading frame and saved in their buffer
        - ball history -> traj_predictor -> prediction
    - training
        - 收集 rollout 数据, 计算 ground-truth future targets
        - 

debug why no improvements, find out if it is actually in the code

what went wrong
- maybe from the log, make sure that the trajectory prediction is working

make sure that the ball trajectory is actually in the code

Hypothesis that maybe HLC is choosing the right skill, but LLC is too weak to perform a successful catch, i will need to double check and verify
- need to verify this

todo：
- 理解ball_trajectory是通过什么实现的
    - 

总结上一周做的更改，组会可以拿来报告
    - RL训练结构讲解
    - rewards 对比
    - Next step

- 新训练
    - HRL-DUAL + Ball Trajectory，看看训练效果会不会好更好一些呢

In progress:
- small card training: 看看多一些epoch训练会不会让inference变得更好
    - 2000 epochs -> 5000 epochs

For March 20
Todo:
1. debug why no improvements, find out why other papers use it and it works, but it does not work on this project
    - find out if it is actually in the code
    - try to log it to prove that it works
        - make sure that the trajectory prediction is working

2. Hypothesis that maybe HLC is choosing the right skill, but LLC is too weak to perform a successful catch, i will need to double check and verify
    - need to verify this

March 24
1. The ctde trained for 5000 epochs
    - inference for 100 epochs and got 31.6 avg rewards
    - In comparison， the ctde trained for 2000 epochs, has an inference avg reward of 14.75371979266405 for 500 epochs
    - Next step, for the current 5000-epoch-training- weight, run also 500 epochs and see how it goes

Arch, Training Epochs, Inference Epochs, Avg Reward
HRL_DUAL, 2000, 500, 16.9
CTDE, 5000, 100, 31.6
CTDE, 2000, 500, 14.7