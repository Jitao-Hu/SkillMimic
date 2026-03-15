# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# HRL-CTDE Player for Dual Humanoid Pass-and-Catch (inference / testing).

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


class HRLCTDEPlayer(HRLPlayerDiscrete):
    """
    CTDE inference player for dual humanoid pass-and-catch.

    Decodes joint action (0-8) into independent per-agent skill selections,
    then runs the frozen LLC for both humanoids.
    """

    def __init__(self, config):
        self._single_action_size = 156

        self._control_mapping_a = config.get("control_mapping_a", [4, 13, 31])
        self._control_mapping_b = config.get("control_mapping_b", [3, 13, 31])
        self._skills_per_agent = len(self._control_mapping_a)
        self._per_agent_obs_size = config.get("per_agent_obs_size", 934)

        # Trajectory predictor config
        self._traj_pred_history = int(config.get("traj_pred_history", 0))
        self._traj_pred_horizons = config.get("traj_pred_horizons", [])
        self._traj_pred_hidden = int(config.get("traj_pred_hidden", 64))
        self._traj_pred_type = config.get("traj_pred_type", "gru")
        self._ball_history_offset = 823 + 15 + 15 + 15

        super().__init__(config)

        max_test_episodes = config.get("max_test_episodes", 0)
        if isinstance(max_test_episodes, int) and max_test_episodes > 0:
            print(f"[HRLCTDEPlayer] max_test_episodes={max_test_episodes}")
            self.games_num = max_test_episodes

        actual_size = self.env.task._num_actions
        if self._single_action_size != actual_size:
            print(
                f"[HRLCTDEPlayer] WARNING: single action size updated "
                f"{self._single_action_size} → {actual_size}"
            )
            self._single_action_size = actual_size

        print(f"[HRLCTDEPlayer] Initialized")
        print(f"  control_mapping_a={self._control_mapping_a}")
        print(f"  control_mapping_b={self._control_mapping_b}")

    # ------------------------------------------------------------------
    # Inject per_agent_obs_size into network config
    # ------------------------------------------------------------------

    def _build_net_config(self):
        config = super()._build_net_config()
        config["per_agent_obs_size"] = self._per_agent_obs_size
        config["skills_per_agent"] = self._skills_per_agent
        config["traj_pred_history"] = self._traj_pred_history
        config["traj_pred_horizons"] = self._traj_pred_horizons
        config["traj_pred_hidden"] = self._traj_pred_hidden
        config["traj_pred_type"] = self._traj_pred_type
        config["ball_history_offset"] = self._ball_history_offset
        return config

    # ------------------------------------------------------------------
    # LLC action: decode joint → (skill_A, skill_B) → dual LLC
    # ------------------------------------------------------------------

    def _compute_llc_action(self, obs, actions):
        batch_size = obs.size(0)

        mapping_a = torch.tensor(self._control_mapping_a, device=self.device)
        mapping_b = torch.tensor(self._control_mapping_b, device=self.device)
        K = self._skills_per_agent

        actions_1d = actions
        if actions_1d.dim() > 1:
            actions_1d = actions_1d.squeeze(-1)
        actions_1d = actions_1d.long()

        idx_a = actions_1d // K
        idx_b = actions_1d % K
        skill_a = mapping_a[idx_a]
        skill_b = mapping_b[idx_b]

        llc_obs_a, llc_obs_b = self.env.task.get_llc_obs_pair()

        # Agent A
        cs_a = torch.zeros((batch_size, 64), device=llc_obs_a.device)
        cs_a[torch.arange(batch_size), -64 + skill_a] = 1.0
        inp_a = torch.cat((llc_obs_a, cs_a), dim=-1)
        pa = self._llc_agent._preproc_obs(inp_a)
        mu_a, _ = self._llc_agent.model.a2c_network.eval_actor(obs=pa)
        act_a = players.rescale_actions(
            self.actions_low, self.actions_high, torch.clamp(mu_a, -1.0, 1.0)
        )

        # Agent B
        cs_b = torch.zeros((batch_size, 64), device=llc_obs_b.device)
        cs_b[torch.arange(batch_size), -64 + skill_b] = 1.0
        inp_b = torch.cat((llc_obs_b, cs_b), dim=-1)
        pb = self._llc_agent._preproc_obs(inp_b)
        mu_b, _ = self._llc_agent.model.a2c_network.eval_actor(obs=pb)
        act_b = players.rescale_actions(
            self.actions_low, self.actions_high, torch.clamp(mu_b, -1.0, 1.0)
        )

        return torch.cat([act_a, act_b], dim=-1)

    # ------------------------------------------------------------------
    # LLC config
    # ------------------------------------------------------------------

    def _build_llc_agent_config(self, config_params, network):
        llc_env_info = copy.deepcopy(self.env_info)

        obs_space = llc_env_info["observation_space"]
        llc_obs_size = 902
        llc_env_info["observation_space"] = spaces.Box(
            obs_space.low[0], obs_space.high[0], shape=(llc_obs_size,)
        )
        llc_env_info["amp_observation_space"] = self.env.amp_observation_space.shape
        llc_env_info["num_envs"] = self.env.task.num_envs
        llc_env_info["action_space"] = spaces.Box(
            -1.0, 1.0, shape=(self._single_action_size,)
        )

        config = config_params["config"]
        config["network"] = network
        config["env_info"] = llc_env_info
        return config

    def _extract_llc_obs(self, obs):
        return obs[..., :838]
