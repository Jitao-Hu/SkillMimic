# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# HRL Player for Dual Humanoid Pass-and-Catch Task
# Runs trained HLC while using frozen LLC to execute skills for both humanoids.

import copy
from gym import spaces
import numpy as np
import os
import torch 
import yaml

from rl_games.algos_torch import players
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common.player import BasePlayer

import learning.common_player_discrete as common_player_discrete
import learning.skillmimic_models as skillmimic_models
import learning.skillmimic_network_builder as skillmimic_network_builder
import learning.skillmimic_players as skillmimic_players
from learning.hrl_players_discrete import HRLPlayerDiscrete


class HRLDualPlayer(HRLPlayerDiscrete):
    """
    HRL Player for dual humanoid environment.
    
    Extends HRLPlayerDiscrete to compute LLC actions for both humanoids
    during inference/testing.
    """
    
    def __init__(self, config):
        # Number of actions per humanoid (156 DOFs) - set before super().__init__
        # because _build_llc_agent_config is called during parent init
        self._single_action_size = 156  # Will be verified after env init
        
        super().__init__(config)
        
        # Verify and update from actual env
        actual_size = self.env.task._num_actions
        if self._single_action_size != actual_size:
            print(f"[HRLDualPlayer] WARNING: Updated single action size from {self._single_action_size} to {actual_size}")
            self._single_action_size = actual_size
        
        print(f"[HRLDualPlayer] Initialized for dual humanoid")
        print(f"[HRLDualPlayer] Single humanoid action size: {self._single_action_size}")
        print(f"[HRLDualPlayer] Control mapping: {self._control_mapping}")
        
        return

    def _compute_llc_action(self, obs, actions):
        """
        Compute LLC actions for BOTH humanoids during inference.
        
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
        
        # Extract base observations for LLC (without task obs)
        llc_obs = self._extract_llc_obs(obs)
        
        # Compute LLC action for humanoid A
        control_signal_a = torch.zeros((batch_size, 64), device=llc_obs.device)
        control_signal_a[torch.arange(batch_size), -64 + skill_a] = 1.0
        llc_obs_a = torch.cat((llc_obs, control_signal_a), dim=-1)
        
        processed_obs_a = self._llc_agent._preproc_obs(llc_obs_a)
        mu_a, _ = self._llc_agent.model.a2c_network.eval_actor(obs=processed_obs_a)
        llc_action_a = players.rescale_actions(
            self.actions_low, self.actions_high, torch.clamp(mu_a, -1.0, 1.0)
        )
        
        # Compute LLC action for humanoid B
        # Note: Using same observation base (could be improved with B's own observations)
        control_signal_b = torch.zeros((batch_size, 64), device=llc_obs.device)
        control_signal_b[torch.arange(batch_size), -64 + skill_b] = 1.0
        llc_obs_b = torch.cat((llc_obs, control_signal_b), dim=-1)
        
        processed_obs_b = self._llc_agent._preproc_obs(llc_obs_b)
        mu_b, _ = self._llc_agent.model.a2c_network.eval_actor(obs=processed_obs_b)
        llc_action_b = players.rescale_actions(
            self.actions_low, self.actions_high, torch.clamp(mu_b, -1.0, 1.0)
        )
        
        # Concatenate actions for both humanoids [batch, 312]
        llc_actions = torch.cat([llc_action_a, llc_action_b], dim=-1)
        
        return llc_actions

    def _build_llc_agent_config(self, config_params, network):
        """Build LLC agent config - LLC outputs single humanoid actions."""
        llc_env_info = copy.deepcopy(self.env_info)
        
        # Calculate LLC observation size:
        # LLC expects: humanoid_obs + obj_obs + condition + skill_embedding
        # Dual env has: humanoid_obs + obj_obs + other_humanoid_obs + task_obs + condition
        # Need to remove other_humanoid_obs (15) and task_obs (task_size)
        obs_space = llc_env_info['observation_space']
        total_obs_size = obs_space.shape[0]  # 922 for HRL dual
        
        # Remove: other_humanoid_obs (15) + task_obs (task_size)
        # Add: skill_embedding (64)
        llc_obs_size = total_obs_size - 15 - self._task_size + 64  # 922 - 15 - 5 + 64 = 966
        # But LLC actually needs: original LLC obs (838) + condition (64) + skill (64) = 966? No...
        # LLC checkpoint was trained with: humanoid_obs + obj_obs + condition + skill = 902
        # So: 838 + 64 = 902 (without skill embedding in original)
        # With skill embedding: 838 + 64 = 902
        
        # Actually the LLC obs should be: 902 (838 base + 64 condition + 64 skill - 64 skill added later)
        # Let me recalculate:
        # - Original skillmimic obs: humanoid_obs + obj_obs + condition = 838 + 64 = 902 (but that's wrong too)
        # - The 902 from error is the checkpoint's expected input
        # - 902 - 64 (skill) = 838 = humanoid_obs + obj_obs
        # - With skill: 838 + 64 = 902
        
        # HRL dual total = 922
        # HRL dual without condition = 922 - 64 = 858
        # HRL dual without task_obs = 858 - 5 = 853
        # HRL dual without other_obs = 853 - 15 = 838 = humanoid_obs + obj_obs ✓
        
        # So LLC needs: humanoid_obs + obj_obs + skill = 838 + 64 = 902 ✓
        llc_obs_size = 902  # Fixed size for LLC
        
        llc_env_info['observation_space'] = spaces.Box(
            obs_space.low[0], obs_space.high[0], shape=(llc_obs_size,)
        )
        llc_env_info['amp_observation_space'] = self.env.amp_observation_space.shape
        llc_env_info['num_envs'] = self.env.task.num_envs
        
        # LLC outputs single humanoid actions (156)
        llc_env_info['action_space'] = spaces.Box(-1.0, 1.0, shape=(self._single_action_size,))

        config = config_params['config']
        config['network'] = network
        config['env_info'] = llc_env_info

        return config

    def _extract_llc_obs(self, obs):
        """
        Extract LLC-compatible observations from HRL dual humanoid observations.
        
        HRL dual obs structure (922):
        - humanoid_obs: 823 dims
        - obj_obs: 15 dims
        - other_humanoid_obs: 15 dims (NOT needed by LLC)
        - task_obs: 5 dims (NOT needed by LLC)
        - condition: 64 dims
        
        LLC needs (838):
        - humanoid_obs: 823 dims
        - obj_obs: 15 dims
        """
        # Structure: [humanoid_obs(823), obj_obs(15), other_obs(15), task_obs(5), condition(64)]
        # Total: 823 + 15 + 15 + 5 + 64 = 922
        
        humanoid_obj_size = 838  # humanoid_obs + obj_obs
        condition_size = 64
        
        # Extract humanoid_obs + obj_obs (first 838 dims)
        llc_obs = obs[..., :humanoid_obj_size]
        
        return llc_obs
