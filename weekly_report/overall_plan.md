March 19

在observation里加入了ball trajectory，还在训练，wip
    - uses physics formula to calculate
        - PREDICT_DT=0.5
    - [optional] learned trajectory predictor
        - pred_type = gru or mlp
        - input ball_history
        - output prediction of ball position

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

