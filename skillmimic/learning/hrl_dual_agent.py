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
        
        return

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
        llc_obs_a_base, llc_obs_b_base = self.env.task.get_llc_obs_pair()
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
        humanoid_obj_size = 838  # humanoid_obs + obj_obs
        llc_obs = obs[..., :humanoid_obj_size]
        return llc_obs
