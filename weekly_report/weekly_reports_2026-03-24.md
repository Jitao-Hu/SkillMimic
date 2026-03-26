1. The ctde trained for 5000 epochs
    - inference for 100 epochs and got 31.6 avg rewards
    - In comparison， the ctde trained for 2000 epochs, has an inference avg reward of 14.75371979266405 for 500 epochs
    - Next step, for the current 5000-epoch-training-weight, run also 500 epochs and see how it goes

Arch, Training Epochs, Inference Epochs, Avg Reward
CTDE, 5000, 100, 31.6
CTDE, 5000, 500, 25.9
CTDE, 2000, 500, 14.7
HRL_DUAL, 2000, 500, 16.9

TODO:
Refine the experiment plans, keep working
1. firgure out the difference between two arch of RL
2. How many epochs the skillmimic paper used during their training

3. In the inference log, how many epochs the weights were trained on should also be logged
    - Done

4. Inference for HRL_DUAL and CTDE without ball trajectory prediction to collect the old results
    - already go these from the logs

I have two architechtures for RL training, 

The following experiments are running or planned:

Experiment 1 HRL_DUAL_5000_epochs
Observation:
At 2000 epochs, HRL_DUAL performs slightly better than CTDE (16.9 vs 14.7) 

Hypothesis:
At 5000 epochs, HRL_DUAL still performs better than CTDE

Experiment Running:
1. HRL_DUAL training with 5000 epochs

Experiment 2 CTDE_8000_epochs

Observation: 
when CTDE's training epoch goes from 2000 to 5000, 
avg reward improves from 14.7 -> 25.9

Hypothesis:
7000/8000 epochs can further improve the average reward, but the 5000-8000 improvement rate could be smaller than 2000-5000

Experiment planned:
1. train CTDE with 3000 more epochs

Experiment 3
Observation:
With the current ball trajectory prediction in reward function, no significant improvements observed
CTDE: nan vs 14.7
with:
inference_log/inference_ctde__20260315_210141.log

HRL_DUAL: 15.2 vs 15.6/16.9
no bt_predict: /pub0/jerryhu/SkillMimicNew/inference_log/inference_old__20260315_210219.log
with:
inference_log/inference_hrl_dual__20260324_164621.log 15.6
inference_log/inference_hrl_dual__20260324_144905.log 16.9

Hypothesis
Why?

Experiment Planned:

Also trying to actually find out how the ball trajectory prediction is affcein the avg reward


Video:
1s: passer choose to not pass and ctacher decided to move towards the ball to get the reward

3s: successful catch

8s: after pass the ball, the passer decide to fight for the ball too

Two major things todo:
1. Ablation study - I have made many changes, which one actually helps the improvement 

2. How the success rate is measured.