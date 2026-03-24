I would like to have this auto logging systems for the current run python file

1. Whenever a training has been run
    - like either train_xyz.sh, where xyz is the task / architechture name
    - or python skillmimic/run.py --task HRLCTDEHumanoid etc.
    - then the information about this run should be stored into a .csv file
    - the csv file should have the following columns, you add can more as long as those are reasonable and helpful
        - task name: like CTDE_DUAL, HRL_DUAL, etc.
        - max_epochs: the maximum epochs for training
        - average reward: the average reward throughout the whole training (I am not sure if this makes sense)
        - hyperparameters from .yaml file
        - motion file
        - othe parameters from the command running, like num_envs, etc.
        - running time for the whole training
        - training start time
        - training finish time
        - path to log files and also checkpoint weights
        - Add more reasonable columns based on the info from log files and the command used
2. For inference, same thing should be done
    - save information of the run into a .csv and it should contain important information
        - epochs/iterations
        - running time duration
        - start, end time
        - avg rewards
        - etc.
        