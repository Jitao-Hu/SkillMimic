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

    def _extract_llc_obs(self, obs):
        """
        Extract LLC-compatible observations from HRL dual humanoid observations.
        
        HRL dual obs structure (929 when goalSize=12):
        - humanoid_obs: 823 dims
        - obj_obs: 15 dims
        - other_humanoid_obs: 15 dims (NOT needed by LLC)
        - task_obs: goalSize dims (NOT needed by LLC, default 12)
        - condition: 64 dims
        
        LLC needs (838):
        - humanoid_obs: 823 dims
        - obj_obs: 15 dims
        """
        humanoid_obj_size = 838  # humanoid_obs + obj_obs (unchanged by goalSize)
        llc_obs = obs[..., :humanoid_obj_size]
        return llc_obs
