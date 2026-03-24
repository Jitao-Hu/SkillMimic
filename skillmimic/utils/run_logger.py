import csv
import json
import os
import sys
import time
import uuid
from datetime import datetime

try:
    import fcntl
except ImportError:
    fcntl = None


class RunInterrupted(Exception):
    """Raised from signal handlers so finally blocks still run (SIGINT / SIGTERM)."""

    def __init__(self, signum):
        self.signum = signum
        super().__init__(f'interrupted by signal {signum}')


def _format_duration(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h {m}m {s}s"


def _to_serializable(val):
    """Best-effort conversion of tensors/arrays to plain Python types."""
    if hasattr(val, 'item'):
        return val.item()
    if hasattr(val, 'tolist'):
        val = val.tolist()
    if isinstance(val, list):
        return val[0] if len(val) == 1 else val
    return val


def exit_category_for_status(exit_status):
    if exit_status == 'completed':
        return 'normal'
    if exit_status in ('keyboard_interrupt', 'sigterm', 'system_exit'):
        return 'interrupted'
    if exit_status.startswith('signal_'):
        return 'interrupted'
    return 'failed'


TRAINING_COLUMNS = [
    'run_id', 'pid', 'exit_status', 'exit_category', 'error_detail',
    'timestamp', 'task', 'algo', 'experiment_name',
    'max_epochs', 'final_epoch', 'final_mean_reward',
    'num_envs', 'motion_file',
    'learning_rate', 'gamma', 'horizon_length', 'minibatch_size',
    'mini_epochs', 'entropy_coef', 'llc_steps', 'save_frequency',
    'checkpoint_path', 'resume_from', 'seed', 'headless',
    'start_time', 'end_time', 'duration_seconds', 'duration_human',
    'command', 'cfg_env_path', 'cfg_train_path', 'reward_weights',
]

INFERENCE_COLUMNS = [
    'run_id', 'pid', 'exit_status', 'exit_category', 'error_detail',
    'timestamp', 'task', 'algo', 'checkpoint',
    'test_episodes', 'num_envs', 'motion_file',
    'avg_reward', 'avg_steps', 'total_episodes',
    'seed', 'headless',
    'start_time', 'end_time', 'duration_seconds', 'duration_human',
    'command',
]


class RunLogger:
    def __init__(self, args, cfg, cfg_train, log_dir='logs'):
        self.args = args
        self.cfg = cfg
        self.cfg_train = cfg_train
        self.log_dir = log_dir
        self.is_training = getattr(args, 'train', True)
        self._start_time = None
        self._start_dt = None
        self.run_id = None

    def start_run(self):
        self._start_time = time.time()
        self._start_dt = datetime.now()
        self.run_id = uuid.uuid4().hex

    def finish_run(self, runner, exit_status='completed', error_detail=''):
        if self._start_time is None:
            self.start_run()
        end_time = time.time()
        end_dt = datetime.now()
        duration = end_time - self._start_time

        os.makedirs(self.log_dir, exist_ok=True)

        category = exit_category_for_status(exit_status)
        err = (error_detail or '').replace('\n', ' ').strip()
        if len(err) > 2000:
            err = err[:1997] + '...'

        base_meta = {
            'run_id': self.run_id or '',
            'pid': os.getpid(),
            'exit_status': exit_status,
            'exit_category': category,
            'error_detail': err,
        }

        if self.is_training:
            row = {**base_meta, **self._build_training_row(runner, end_dt, duration)}
            self._append_csv(
                os.path.join(self.log_dir, 'training_runs.csv'),
                TRAINING_COLUMNS,
                row,
            )
        else:
            row = {**base_meta, **self._build_inference_row(runner, end_dt, duration)}
            self._append_csv(
                os.path.join(self.log_dir, 'inference_runs.csv'),
                INFERENCE_COLUMNS,
                row,
            )

    def _build_training_row(self, runner, end_dt, duration):
        args = self.args
        cfg = self.cfg
        config = self.cfg_train['params']['config']
        algo_name = self.cfg_train['params']['algo']['name']

        final_mean_reward = ''
        final_epoch = ''

        agent = self._get_agent(runner)
        if agent is not None and hasattr(agent, '_run_results'):
            res = agent._run_results
            final_mean_reward = _to_serializable(res.get('last_mean_rewards', ''))
            final_epoch = _to_serializable(res.get('epoch_num', ''))

        exp_name = config.get('full_experiment_name', '')
        train_dir = config.get('train_dir', 'output/')
        ckpt_name = config.get('name', '')
        checkpoint_path = os.path.join(train_dir, exp_name, 'nn', f'{ckpt_name}.pth')

        reward_weights = cfg.get('env', {}).get('rewardWeights', {})
        coop_weights = cfg.get('env', {}).get('coopRewardWeights', {})
        all_weights = {**reward_weights, **coop_weights} if coop_weights else reward_weights

        return {
            'timestamp': end_dt.isoformat(timespec='seconds'),
            'task': args.task,
            'algo': algo_name,
            'experiment_name': exp_name,
            'max_epochs': config.get('max_epochs', ''),
            'final_epoch': final_epoch,
            'final_mean_reward': final_mean_reward,
            'num_envs': cfg.get('env', {}).get('numEnvs', ''),
            'motion_file': cfg.get('env', {}).get('motion_file', ''),
            'learning_rate': config.get('learning_rate', ''),
            'gamma': config.get('gamma', ''),
            'horizon_length': config.get('horizon_length', ''),
            'minibatch_size': config.get('minibatch_size', ''),
            'mini_epochs': config.get('mini_epochs', ''),
            'entropy_coef': config.get('entropy_coef', ''),
            'llc_steps': config.get('llc_steps', ''),
            'save_frequency': config.get('save_frequency', ''),
            'checkpoint_path': checkpoint_path,
            'resume_from': getattr(args, 'resume_from', ''),
            'seed': self.cfg_train['params'].get('seed', ''),
            'headless': getattr(args, 'headless', ''),
            'start_time': self._start_dt.isoformat(timespec='seconds'),
            'end_time': end_dt.isoformat(timespec='seconds'),
            'duration_seconds': round(duration, 1),
            'duration_human': _format_duration(duration),
            'command': ' '.join(sys.argv),
            'cfg_env_path': getattr(args, 'cfg_env', ''),
            'cfg_train_path': getattr(args, 'cfg_train', ''),
            'reward_weights': json.dumps(all_weights) if all_weights else '',
        }

    def _build_inference_row(self, runner, end_dt, duration):
        args = self.args
        cfg = self.cfg
        algo_name = self.cfg_train['params']['algo']['name']

        avg_reward = ''
        avg_steps = ''
        total_episodes = ''

        player = self._get_player(runner)
        if player is not None and hasattr(player, '_run_results'):
            res = player._run_results
            avg_reward = _to_serializable(res.get('avg_reward', ''))
            avg_steps = _to_serializable(res.get('avg_steps', ''))
            total_episodes = _to_serializable(res.get('games_played', ''))

        return {
            'timestamp': end_dt.isoformat(timespec='seconds'),
            'task': args.task,
            'algo': algo_name,
            'checkpoint': getattr(args, 'checkpoint', ''),
            'test_episodes': getattr(args, 'test_episodes', ''),
            'num_envs': cfg.get('env', {}).get('numEnvs', ''),
            'motion_file': cfg.get('env', {}).get('motion_file', ''),
            'avg_reward': avg_reward,
            'avg_steps': avg_steps,
            'total_episodes': total_episodes,
            'seed': self.cfg_train['params'].get('seed', ''),
            'headless': getattr(args, 'headless', ''),
            'start_time': self._start_dt.isoformat(timespec='seconds'),
            'end_time': end_dt.isoformat(timespec='seconds'),
            'duration_seconds': round(duration, 1),
            'duration_human': _format_duration(duration),
            'command': ' '.join(sys.argv),
        }

    def _get_agent(self, runner):
        if runner is None:
            return None
        for attr in ('_last_agent', 'agent', 'algo'):
            if hasattr(runner, attr):
                return getattr(runner, attr)
        return None

    def _get_player(self, runner):
        if runner is None:
            return None
        for attr in ('_last_player', 'player'):
            if hasattr(runner, attr):
                return getattr(runner, attr)
        return None

    @staticmethod
    def _append_csv(path, columns, row):
        dir_name = os.path.dirname(os.path.abspath(path))
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        with open(path, 'a+', newline='') as f:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.seek(0, os.SEEK_END)
                need_header = f.tell() == 0
                writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
                if need_header:
                    writer.writeheader()
                writer.writerow(row)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
            finally:
                if fcntl is not None:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UNLOCK)
        print(f'[RunLogger] Results saved to {path} (exit_status={row.get("exit_status", "")})')
