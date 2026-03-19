# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# HRL Agent for Dual Humanoid Pass-and-Catch Task
# Trains HLC while using frozen LLC to execute skills for both humanoids.

import copy
from datetime import datetime
import random
from gym import spaces
import numpy as np
import os
import time
import yaml

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import central_value
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common import a2c_common
from rl_games.common import datasets
from rl_games.common import schedulers
from rl_games.common import vecenv

import torch
from torch import optim

import learning.common_agent_discrete as common_agent_discrete
import learning.skillmimic_models as skillmimic_models
import learning.skillmimic_network_builder as skillmimic_network_builder
import learning.skillmimic_agent as skillmimic_agent
from learning.hrl_agent_discrete import HRLAgentDiscrete
from utils import torch_utils
from isaacgym.torch_utils import quat_rotate

from tensorboardX import SummaryWriter


class HRLDualAgent(HRLAgentDiscrete):
    """
    HRL Agent for dual humanoid environment.
    
    Extends HRLAgentDiscrete to:
    - Compute LLC actions for both humanoids
    - Handle paired skill selection (A:pass + B:catch)
    """
    
    def __init__(self, base_name, config):
        # Number of actions per humanoid (156 DOFs) - set before super().__init__
        # because _build_llc_agent_config is called during parent init
        self._single_action_size = 156  # Will be verified after env init
        
        # Phase 3: Guidance penalty (skill label alignment)
        # Read optional config values with sensible defaults
        self._guidance_penalty_weight = float(config.get('guidance_penalty_weight', 1.0))
        self._holding_dist_thresh = float(config.get('holding_dist_thresh', 0.2))
        self._holding_contact_thresh = float(config.get('holding_contact_thresh', 1.0))
        self._ball_to_b_speed_thresh = float(config.get('ball_to_b_speed_thresh', 0.5))

        # Trajectory predictor config (before super for _build_net_config)
        self._traj_pred_history = int(config.get("traj_pred_history", 0))
        self._traj_pred_horizons = config.get("traj_pred_horizons", [])
        self._traj_pred_hidden = int(config.get("traj_pred_hidden", 64))
        self._traj_pred_type = config.get("traj_pred_type", "gru")
        self._traj_pred_loss_weight = float(config.get("traj_pred_loss_weight", 0.0))
        self._use_traj_pred = (
            self._traj_pred_history > 0 and len(self._traj_pred_horizons) > 0
        )
        self._collecting_traj_data = False

        super().__init__(base_name, config)
        
        # Verify and update from actual env
        actual_size = self.vec_env.env.task._num_actions
        if self._single_action_size != actual_size:
            print(f"[HRLDualAgent] WARNING: Updated single action size from {self._single_action_size} to {actual_size}")
            self._single_action_size = actual_size
        
        print(f"[HRLDualAgent] Initialized for dual humanoid")
        print(f"[HRLDualAgent] Single humanoid action size: {self._single_action_size}")
        print(f"[HRLDualAgent] Control mapping: {self._control_mapping}")
        print(f"[HRLDualAgent] Latent dim: {self._latent_dim}")
        print(f"[HRLDualAgent] Guidance penalty weight: {self._guidance_penalty_weight}")
        print(f"[HRLDualAgent] traj_pred={self._use_traj_pred}  history={self._traj_pred_history}")
        print(f"[HRLDualAgent] traj_pred_horizons={self._traj_pred_horizons}  loss_w={self._traj_pred_loss_weight}")

        # Skill monitor (lightweight telemetry for HLC skill choices).
        # Logs aggregated stats every N HLC steps to TensorBoard (and optionally prints).
        self._skill_monitor_interval = int(config.get('skill_monitor_interval', 200))
        self._skill_monitor_print = self._as_bool(config.get('skill_monitor_print', True), True)
        self._skill_monitor_tb = self._as_bool(config.get('skill_monitor_tb', True), True)
        self._skill_monitor_hlc_steps_total = 0
        self._skill_monitor_reset_accumulators()
        
        return

    @staticmethod
    def _as_bool(v, default: bool) -> bool:
        if v is None:
            return default
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        if isinstance(v, str):
            return v.strip().lower() in ('1', 'true', 'yes', 'y', 'on')
        return default

    def _skill_monitor_reset_accumulators(self):
        self._skill_monitor_acc_total = 0.0
        self._skill_monitor_acc_a_pass = 0.0
        self._skill_monitor_acc_a_run = 0.0
        self._skill_monitor_acc_a_idle = 0.0
        self._skill_monitor_acc_b_catch = 0.0
        self._skill_monitor_acc_b_run = 0.0
        self._skill_monitor_acc_b_idle = 0.0
        self._skill_monitor_acc_ball_to_b = 0.0
        self._skill_monitor_acc_ball_to_b_b_catch = 0.0
        self._skill_monitor_acc_a_holding = 0.0
        self._skill_monitor_acc_a_holding_a_pass = 0.0
        return

    def _apply_guidance_penalty(self, rewards, actions):
        """
        Apply skill label guidance penalty on top of environment rewards.
        
        Penalties (per env step at HLC timescale):
        - If A is holding the ball but selected skill for A is not PASS (4): subtract weight
        - If ball is flying toward B but selected skill for B is not CATCH (3) or RUN (13): subtract weight
        """
        # Only apply when we have tensor observations / rewards
        if not self.is_tensor_obses:
            return rewards

        # Ensure actions is 1D long tensor on same device as rewards
        device = rewards.device
        actions_tensor = actions
        if actions_tensor.dim() > 1:
            actions_tensor = actions_tensor.squeeze(-1)
        actions_tensor = actions_tensor.long().to(device)

        # Map discrete HLC action to (skill_a, skill_b) using same control mapping as _compute_llc_action
        controlmapping = torch.tensor(self._control_mapping, device=device, dtype=torch.long)
        num_skills_per_humanoid = controlmapping.shape[0] // 2
        skill_idx = actions_tensor % num_skills_per_humanoid
        skill_a = controlmapping[skill_idx]
        skill_b = controlmapping[num_skills_per_humanoid + skill_idx]

        # Access underlying dual-humanoid task
        task = self.vec_env.env.task

        # Current sim state (per env)
        ball_pos = task._target_states[:, 0:3]
        ball_vel = task._target_states[:, 7:10]
        root_pos_a = task._humanoid_root_states[:, 0:3]
        root_pos_b = task._humanoid_b_root_states[:, 0:3]
        ball_contact_force = task._tar_contact_forces  # [N, 3]

        # Distances from ball to hands
        dist_ball_to_hand_a = task._get_closest_hand_distance(ball_pos, 'a')
        dist_ball_to_hand_b = task._get_closest_hand_distance(ball_pos, 'b')

        # ---------- State 1: A_holding (A holds the ball) ----------
        ball_has_contact = (torch.norm(ball_contact_force, dim=-1) > self._holding_contact_thresh)
        a_holding = (dist_ball_to_hand_a < self._holding_dist_thresh) & ball_has_contact

        # ---------- State 2: Ball_to_B (ball flying toward B) ----------
        ball_speed = torch.norm(ball_vel, dim=-1)
        to_b = torch.sum(ball_vel * (root_pos_b - ball_pos), dim=-1) > 0.0
        away_from_a = torch.sum(ball_vel * (ball_pos - root_pos_a), dim=-1) > 0.0
        ball_to_b = (ball_speed > self._ball_to_b_speed_thresh) & to_b & away_from_a & (dist_ball_to_hand_b < dist_ball_to_hand_a)

        # ---------- Penalties ----------
        penalty = torch.zeros_like(rewards)

        # 1) Pass guidance: A is holding but did not choose PASS (4)
        penalty += (a_holding & (skill_a != 4)).float() * self._guidance_penalty_weight

        # 2) Catch / run guidance: ball flying to B but B did not choose CATCH (3) or RUN (13)
        bad_b_skill = ~((skill_b == 3) | (skill_b == 13))
        penalty += (ball_to_b & bad_b_skill).float() * self._guidance_penalty_weight

        return rewards - penalty

    def _skill_monitor_update(self, actions):
        if self._skill_monitor_interval <= 0:
            return
        if getattr(self, 'rank', 0) != 0:
            return
        if not getattr(self, 'is_tensor_obses', False):
            return
        if not hasattr(self, 'vec_env') or self.vec_env is None:
            return

        with torch.no_grad():
            task = self.vec_env.env.task

            actions_tensor = actions
            if isinstance(actions_tensor, (list, tuple)):
                return
            if actions_tensor.dim() > 1:
                actions_tensor = actions_tensor.squeeze(-1)
            actions_tensor = actions_tensor.long().to(task._target_states.device)

            controlmapping = torch.tensor(self._control_mapping, device=actions_tensor.device, dtype=torch.long)
            num_skills_per_humanoid = controlmapping.shape[0] // 2
            skill_idx = actions_tensor % num_skills_per_humanoid
            skill_a = controlmapping[skill_idx]
            skill_b = controlmapping[num_skills_per_humanoid + skill_idx]

            # Current sim state (per env)
            ball_pos = task._target_states[:, 0:3]
            ball_vel = task._target_states[:, 7:10]
            root_pos_a = task._humanoid_root_states[:, 0:3]
            root_pos_b = task._humanoid_b_root_states[:, 0:3]
            ball_contact_force = task._tar_contact_forces  # [N, 3]

            dist_ball_to_hand_a = task._get_closest_hand_distance(ball_pos, 'a')
            dist_ball_to_hand_b = task._get_closest_hand_distance(ball_pos, 'b')

            ball_has_contact = (torch.norm(ball_contact_force, dim=-1) > self._holding_contact_thresh)
            a_holding = (dist_ball_to_hand_a < self._holding_dist_thresh) & ball_has_contact

            ball_speed = torch.norm(ball_vel, dim=-1)
            to_b = torch.sum(ball_vel * (root_pos_b - ball_pos), dim=-1) > 0.0
            away_from_a = torch.sum(ball_vel * (ball_pos - root_pos_a), dim=-1) > 0.0
            ball_to_b = (ball_speed > self._ball_to_b_speed_thresh) & to_b & away_from_a & (dist_ball_to_hand_b < dist_ball_to_hand_a)

            batch = float(skill_a.numel())
            self._skill_monitor_acc_total += batch
            self._skill_monitor_acc_a_pass += float((skill_a == 4).sum().item())
            self._skill_monitor_acc_a_run += float((skill_a == 13).sum().item())
            self._skill_monitor_acc_a_idle += float((skill_a == 31).sum().item())
            self._skill_monitor_acc_b_catch += float((skill_b == 3).sum().item())
            self._skill_monitor_acc_b_run += float((skill_b == 13).sum().item())
            self._skill_monitor_acc_b_idle += float((skill_b == 31).sum().item())

            self._skill_monitor_acc_ball_to_b += float(ball_to_b.sum().item())
            self._skill_monitor_acc_ball_to_b_b_catch += float((ball_to_b & (skill_b == 3)).sum().item())
            self._skill_monitor_acc_a_holding += float(a_holding.sum().item())
            self._skill_monitor_acc_a_holding_a_pass += float((a_holding & (skill_a == 4)).sum().item())

            self._skill_monitor_hlc_steps_total += 1
            if (self._skill_monitor_hlc_steps_total % self._skill_monitor_interval) != 0:
                return

            eps = 1e-6
            total = max(self._skill_monitor_acc_total, eps)
            a_pass = self._skill_monitor_acc_a_pass / total
            a_run = self._skill_monitor_acc_a_run / total
            a_idle = self._skill_monitor_acc_a_idle / total
            b_catch = self._skill_monitor_acc_b_catch / total
            b_run = self._skill_monitor_acc_b_run / total
            b_idle = self._skill_monitor_acc_b_idle / total

            ball_to_b_total = max(self._skill_monitor_acc_ball_to_b, 0.0)
            a_holding_total = max(self._skill_monitor_acc_a_holding, 0.0)
            ball_to_b_frac = ball_to_b_total / total
            p_b_catch_given_ball_to_b = (self._skill_monitor_acc_ball_to_b_b_catch / max(ball_to_b_total, eps)) if ball_to_b_total > 0 else 0.0
            p_a_pass_given_a_holding = (self._skill_monitor_acc_a_holding_a_pass / max(a_holding_total, eps)) if a_holding_total > 0 else 0.0

            step = int(self._skill_monitor_hlc_steps_total)
            if self._skill_monitor_tb and hasattr(self, 'writer') and self.writer is not None:
                self.writer.add_scalar('skills/A_pass_frac', a_pass, step)
                self.writer.add_scalar('skills/A_run_frac', a_run, step)
                self.writer.add_scalar('skills/A_idle_frac', a_idle, step)
                self.writer.add_scalar('skills/B_catch_frac', b_catch, step)
                self.writer.add_scalar('skills/B_run_frac', b_run, step)
                self.writer.add_scalar('skills/B_idle_frac', b_idle, step)
                self.writer.add_scalar('skills/ball_to_b_frac', ball_to_b_frac, step)
                self.writer.add_scalar('skills/p_B_catch_given_ball_to_B', p_b_catch_given_ball_to_b, step)
                self.writer.add_scalar('skills/p_A_pass_given_A_holding', p_a_pass_given_a_holding, step)

            if self._skill_monitor_print:
                print(
                    "[SkillMonitor]"
                    f" hlc_step={step}"
                    f" A(pass/run/idle)={a_pass:.3f}/{a_run:.3f}/{a_idle:.3f}"
                    f" B(catch/run/idle)={b_catch:.3f}/{b_run:.3f}/{b_idle:.3f}"
                    f" ball_to_B={ball_to_b_frac:.3f}"
                    f" P(B_catch|ball_to_B)={p_b_catch_given_ball_to_b:.3f}"
                    f" P(A_pass|A_holding)={p_a_pass_given_a_holding:.3f}"
                )

            self._skill_monitor_reset_accumulators()
            return

    def _catch_stats_log(self):
        """Periodically read per-attempt catch stats from env and log to tensorboard/wandb."""
        if not hasattr(self, '_cs_initialized'):
            task = self.vec_env.env.task
            self._cs_enabled = getattr(task, '_catch_stats_enabled', False)
            self._cs_interval = getattr(task, '_catch_stats_log_interval', 50)
            self._cs_print = getattr(task, '_catch_stats_print', False)
            self._cs_hlc_step = 0
            self._cs_initialized = True

        if not self._cs_enabled:
            return

        self._cs_hlc_step += 1
        if (self._cs_hlc_step % self._cs_interval) != 0:
            return

        task = self.vec_env.env.task
        stats = task.get_catch_stats()
        if not stats or stats['catch_attempts'] == 0:
            return

        step = self._cs_hlc_step
        if hasattr(self, 'writer') and self.writer is not None:
            self.writer.add_scalar('catch_stats/pass_success_rate', stats['pass_success_rate'], step)
            self.writer.add_scalar('catch_stats/catch_success_rate', stats['catch_success_rate'], step)
            self.writer.add_scalar('catch_stats/catch_fail_rate', stats['catch_fail_rate'], step)
            self.writer.add_scalar('catch_stats/alive_rate', stats['alive_rate'], step)
            self.writer.add_scalar('catch_stats/standing_rate', stats['standing_rate'], step)
            self.writer.add_scalar('catch_stats/upright_rate', stats['upright_rate'], step)
            self.writer.add_scalar('catch_stats/ground_contact_rate', stats['ground_contact_rate'], step)
            self.writer.add_scalar('catch_stats/pass_attempts', stats['pass_attempts'], step)
            self.writer.add_scalar('catch_stats/catch_successes', stats['catch_successes'], step)

        if self._cs_print:
            print(
                f"[CatchStats] hlc_step={step}"
                f" pass={stats['pass_successes']}/{stats['pass_attempts']}"
                f"({stats['pass_success_rate']:.3f})"
                f" catch={stats['catch_successes']}/{stats['catch_attempts']}"
                f"({stats['catch_success_rate']:.3f})"
                f" fails={stats['catch_fails']}"
                f" alive={stats['alive_rate']:.3f}"
                f" standing={stats['standing_rate']:.3f}"
                f" upright={stats['upright_rate']:.3f}"
                f" gnd_contact={stats['ground_contact_rate']:.3f}"
            )

        task.reset_catch_stats()

    def env_step(self, actions):
        """
        Override env_step to insert guidance penalty at HLC timescale.
        
        Logic mirrors HRLAgentDiscrete.env_step, with an additional
        call to _apply_guidance_penalty after averaging LLC rewards.
        """
        obs = self.obs['obs']

        rewards = 0.0
        done_count = 0.0
        terminate_count = 0.0
        for _ in range(self._llc_steps):
            llc_actions = self._compute_llc_action(obs, actions)

            obs, curr_rewards, curr_dones, infos = self.vec_env.step(llc_actions)

            rewards += curr_rewards
            done_count += curr_dones
            terminate_count += infos['terminate']

        # Average rewards over LLC steps
        rewards /= self._llc_steps

        # Apply guidance penalty on top of averaged rewards (Phase 3)
        if self.is_tensor_obses:
            rewards = self._apply_guidance_penalty(rewards, actions)
            self._skill_monitor_update(actions)
            self._catch_stats_log()

        dones = torch.zeros_like(done_count)
        dones[done_count > 0] = 1.0
        terminate = torch.zeros_like(terminate_count)
        terminate[terminate_count > 0] = 1.0
        infos['terminate'] = terminate

        # Collect ball/root data for trajectory prediction auxiliary loss
        if self._collecting_traj_data and self.is_tensor_obses:
            task = self.vec_env.env.task
            self._tp_buf_ball.append(task._target_states[:, 0:3].clone())
            self._tp_buf_pos_a.append(task._humanoid_root_states[:, 0:3].clone())
            self._tp_buf_rot_a.append(task._humanoid_root_states[:, 3:7].clone())
            self._tp_buf_pos_b.append(task._humanoid_b_root_states[:, 0:3].clone())
            self._tp_buf_rot_b.append(task._humanoid_b_root_states[:, 3:7].clone())
            self._tp_buf_dones.append(dones.clone())

        if self.is_tensor_obses:
            if self.value_size == 1:
                rewards = rewards.unsqueeze(1)
            return self.obs_to_tensors(obs), rewards.to(self.ppo_device), dones.to(self.ppo_device), infos
        else:
            # Non-tensor path kept consistent with base implementation
            rewards_np = rewards
            if self.value_size == 1:
                rewards_np = np.expand_dims(rewards_np, axis=1)
            return (
                self.obs_to_tensors(obs),
                torch.from_numpy(rewards_np).to(self.ppo_device).float(),
                torch.from_numpy(dones).to(self.ppo_device),
                infos,
            )

    def _compute_llc_action(self, obs, actions):
        """
        Compute LLC actions for BOTH humanoids.
        
        Maps HLC action to (skill_A, skill_B) pair:
        - Action 0: A=pass(4), B=catch(3)   -> passing mode
        - Action 1: A=run(13), B=run(13)    -> both running
        - Action 2: A=idle(31), B=idle(31)  -> both idle
        
        Args:
            obs: Observations [batch, obs_dim]
            actions: HLC actions [batch] - discrete skill pair selection
            
        Returns:
            llc_actions: [batch, 312] - concatenated actions for both humanoids
        """
        batch_size = obs.size(0)
        controlmapping = torch.tensor(self._control_mapping).to(self.device)
        
        # Map action to skill pair
        # First 3 entries are for humanoid A: [pass(4), run(13), idle(31)]
        # Last 3 entries are for humanoid B: [catch(3), run(13), idle(31)]
        num_skills_per_humanoid = len(controlmapping) // 2
        
        skill_idx = actions % num_skills_per_humanoid
        skill_a = controlmapping[skill_idx]
        skill_b = controlmapping[num_skills_per_humanoid + skill_idx]
        
        # Extract base observations for LLC (without task obs) for BOTH humanoids
        # Shape: [num_envs, 838] each (humanoid_obs + obj_obs)
        llc_obs_a_base, llc_obs_b_base = self.vec_env.env.task.get_llc_obs_pair()
        llc_obs_a = llc_obs_a_base
        llc_obs_b = llc_obs_b_base
        
        # Compute LLC action for humanoid A
        control_signal_a = torch.zeros((batch_size, 64), device=llc_obs_a.device)
        control_signal_a[torch.arange(batch_size), -64 + skill_a] = 1.0
        llc_obs_a = torch.cat((llc_obs_a, control_signal_a), dim=-1)
        
        processed_obs_a = self._llc_agent._preproc_obs(llc_obs_a)
        mu_a, _ = self._llc_agent.model.a2c_network.eval_actor(obs=processed_obs_a)
        llc_action_a = self._llc_agent.preprocess_actions(mu_a)
        
        # Compute LLC action for humanoid B (using B's own observations)
        control_signal_b = torch.zeros((batch_size, 64), device=llc_obs_b.device)
        control_signal_b[torch.arange(batch_size), -64 + skill_b] = 1.0
        llc_obs_b = torch.cat((llc_obs_b, control_signal_b), dim=-1)
        
        processed_obs_b = self._llc_agent._preproc_obs(llc_obs_b)
        mu_b, _ = self._llc_agent.model.a2c_network.eval_actor(obs=processed_obs_b)
        llc_action_b = self._llc_agent.preprocess_actions(mu_b)
        
        # Concatenate actions for both humanoids [batch, 312]
        llc_actions = torch.cat([llc_action_a, llc_action_b], dim=-1)
        
        return llc_actions

    def _build_llc_agent_config(self, config_params, network):
        """Build LLC agent config - LLC outputs single humanoid actions."""
        llc_env_info = copy.deepcopy(self.env_info)
        
        # LLC observation size: humanoid_obs + obj_obs + skill_embedding = 838 + 64 = 902
        llc_obs_size = 902
        
        obs_space = llc_env_info['observation_space']
        llc_env_info['observation_space'] = spaces.Box(
            obs_space.low[0], obs_space.high[0], shape=(llc_obs_size,)
        )
        
        # LLC outputs single humanoid actions (156)
        llc_env_info['action_space'] = spaces.Box(-1.0, 1.0, shape=(self._single_action_size,))

        config = config_params['config']
        config['network'] = network
        config['num_actors'] = self.num_actors
        config['features'] = {'observer': self.algo_observer}
        config['env_info'] = llc_env_info

        return config

    # ------------------------------------------------------------------
    # Override: inject trajectory predictor config into model builder
    # ------------------------------------------------------------------

    def _build_net_config(self):
        config = super()._build_net_config()
        if self._use_traj_pred:
            task_obs_size = self.vec_env.env.task.get_task_obs_size()
            ball_history_offset_a = 823 + 15 + 15 + task_obs_size
            ball_history_offset_b = ball_history_offset_a + self._traj_pred_history * 6
            config["traj_pred_history"] = self._traj_pred_history
            config["traj_pred_horizons"] = self._traj_pred_horizons
            config["traj_pred_hidden"] = self._traj_pred_hidden
            config["traj_pred_type"] = self._traj_pred_type
            config["ball_history_offset_a"] = ball_history_offset_a
            config["ball_history_offset_b"] = ball_history_offset_b
        return config

    # ------------------------------------------------------------------
    # play_steps / prepare_dataset / calc_gradients overrides for aux loss
    # ------------------------------------------------------------------

    def play_steps(self):
        if self._use_traj_pred and self._traj_pred_loss_weight > 0:
            self._collecting_traj_data = True
            self._tp_buf_ball = []
            self._tp_buf_pos_a = []
            self._tp_buf_rot_a = []
            self._tp_buf_pos_b = []
            self._tp_buf_rot_b = []
            self._tp_buf_dones = []

        batch_dict = super().play_steps()

        if self._use_traj_pred and self._traj_pred_loss_weight > 0:
            self._collecting_traj_data = False
            targets_a, targets_b, valid = self._compute_traj_targets()
            batch_dict["traj_targets_a"] = a2c_common.swap_and_flatten01(targets_a)
            batch_dict["traj_targets_b"] = a2c_common.swap_and_flatten01(targets_b)
            batch_dict["traj_valid"] = a2c_common.swap_and_flatten01(valid)

        return batch_dict

    def _compute_traj_targets(self):
        """Build ground-truth future ball position targets from the rollout buffer."""
        H_count = len(self._traj_pred_horizons)
        T = len(self._tp_buf_ball)  # horizon_length
        N = self._tp_buf_ball[0].size(0)  # num_envs
        device = self._tp_buf_ball[0].device

        horizons_hlc = [h // max(self._llc_steps, 1) for h in self._traj_pred_horizons]
        max_h = max(horizons_hlc)

        ball_pos = torch.stack(self._tp_buf_ball, dim=0)   # [T, N, 3]
        pos_a = torch.stack(self._tp_buf_pos_a, dim=0)     # [T, N, 3]
        rot_a = torch.stack(self._tp_buf_rot_a, dim=0)     # [T, N, 4]
        pos_b = torch.stack(self._tp_buf_pos_b, dim=0)
        rot_b = torch.stack(self._tp_buf_rot_b, dim=0)
        dones = torch.stack(self._tp_buf_dones, dim=0)     # [T, N]

        targets_a = torch.zeros(T, N, H_count * 3, device=device)
        targets_b = torch.zeros(T, N, H_count * 3, device=device)
        valid = torch.ones(T, N, 1, device=device)

        for t in range(T):
            if t + max_h >= T:
                valid[t] = 0.0
                continue

            done_in_range = dones[t + 1 : t + max_h + 1].any(dim=0)  # [N]
            valid[t, done_in_range] = 0.0

            hinv_a = torch_utils.calc_heading_quat_inv(rot_a[t])
            hinv_b = torch_utils.calc_heading_quat_inv(rot_b[t])

            for i, h in enumerate(horizons_hlc):
                fut_ball = ball_pos[t + h]  # [N, 3]
                targets_a[t, :, i * 3 : (i + 1) * 3] = quat_rotate(
                    hinv_a, fut_ball - pos_a[t]
                )
                targets_b[t, :, i * 3 : (i + 1) * 3] = quat_rotate(
                    hinv_b, fut_ball - pos_b[t]
                )

        return targets_a, targets_b, valid

    def prepare_dataset(self, batch_dict):
        super().prepare_dataset(batch_dict)
        if self._use_traj_pred and self._traj_pred_loss_weight > 0:
            tgt_a = batch_dict.get("traj_targets_a")
            tgt_b = batch_dict.get("traj_targets_b")
            tgt_v = batch_dict.get("traj_valid")
            if tgt_a is not None:
                self.dataset.values_dict["traj_targets_a"] = tgt_a
                self.dataset.values_dict["traj_targets_b"] = tgt_b
                self.dataset.values_dict["traj_valid"] = tgt_v

    def calc_gradients(self, input_dict):
        self.set_train()

        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        return_batch = input_dict['returns']
        logits_batch = input_dict['logits']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)

        lr = self.last_lr
        kl = 1.0
        lr_mul = 1.0
        curr_e_clip = lr_mul * self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch,
            'obs': obs_batch,
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_len

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            now_logits = res_dict['logits']

            a_info = self._actor_loss(
                old_action_log_probs_batch, action_log_probs, advantage, curr_e_clip
            )
            a_loss = a_info['actor_loss']

            c_info = self._critic_loss(
                value_preds_batch, values, curr_e_clip, return_batch, self.clip_value
            )
            c_loss = c_info['critic_loss']

            a_loss = torch.mean(a_loss)
            c_loss = torch.mean(c_loss)
            entropy = torch.mean(entropy)

            loss = (
                a_loss
                + self.critic_coef * c_loss
                - self.entropy_coef * entropy
            )

            # Auxiliary trajectory prediction loss
            traj_loss = torch.tensor(0.0, device=loss.device)
            if (
                self._use_traj_pred
                and self._traj_pred_loss_weight > 0
                and "traj_targets_a" in input_dict
            ):
                net = self.model.a2c_network
                pred_a = getattr(net, "_last_pred_a", None)
                pred_b = getattr(net, "_last_pred_b", None)
                tgt_a = input_dict["traj_targets_a"]
                tgt_b = input_dict["traj_targets_b"]
                tgt_valid = input_dict["traj_valid"].squeeze(-1)  # [B]

                if pred_a is not None and pred_b is not None:
                    mse_a = ((pred_a - tgt_a) ** 2).mean(dim=-1)
                    mse_b = ((pred_b - tgt_b) ** 2).mean(dim=-1)
                    per_sample = (mse_a + mse_b) * 0.5 * tgt_valid
                    valid_count = tgt_valid.sum().clamp(min=1.0)
                    traj_loss = per_sample.sum() / valid_count
                    loss = loss + self._traj_pred_loss_weight * traj_loss

            a_clip_frac = torch.mean(a_info['actor_clipped'].float())
            a_info['actor_loss'] = a_loss
            a_info['actor_clip_frac'] = a_clip_frac

            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        with torch.no_grad():
            reduce_kl = not self.is_rnn
            dist_before = torch.distributions.Categorical(logits=logits_batch)
            dist_now = torch.distributions.Categorical(logits=now_logits)
            kl_dist = torch.distributions.kl_divergence(dist_before, dist_now).mean()
            if reduce_kl:
                kl_dist = kl_dist.mean()

        self.train_result = {
            'entropy': entropy,
            'kl': kl_dist,
            'last_lr': self.last_lr,
            'lr_mul': lr_mul,
        }
        self.train_result.update(a_info)
        self.train_result.update(c_info)

        if self._use_traj_pred and self._traj_pred_loss_weight > 0:
            self.train_result["traj_pred_loss"] = traj_loss.detach()
            if hasattr(self, "writer") and self.writer is not None:
                self.writer.add_scalar(
                    "losses/traj_pred",
                    traj_loss.item(),
                    self.epoch_num,
                )

        return

    # ------------------------------------------------------------------
    # LLC observation extraction
    # ------------------------------------------------------------------

    def _extract_llc_obs(self, obs):
        """
        Extract LLC-compatible observations from HRL dual humanoid observations.
        
        LLC needs (838):
        - humanoid_obs: 823 dims
        - obj_obs: 15 dims
        """
        humanoid_obj_size = 838  # humanoid_obs + obj_obs (unchanged by goalSize)
        llc_obs = obs[..., :humanoid_obj_size]
        return llc_obs
