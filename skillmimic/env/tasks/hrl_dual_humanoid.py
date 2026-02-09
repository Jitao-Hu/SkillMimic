# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# HRL Dual Humanoid Environment for Pass-and-Catch Task
# Uses pre-trained LLC to execute skills, HLC selects skills for each humanoid.
#
# Architecture:
#   HLC: Outputs 2 discrete actions (skill selection for A and B)
#   LLC: Shared low-level controller executes selected skills

import os
import torch
from torch import Tensor
from typing import Tuple

from utils import torch_utils
from utils.motion_data_handler import MotionDataHandler

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *

from env.tasks.skillmimic_dual import SkillMimicDualHumanoid

# Skill IDs for pass-and-catch task
SKILL_PASS = 4    # 004: pass
SKILL_CATCH = 3   # 003: catch  
SKILL_RUN = 13    # 013: run/dribble
SKILL_IDLE = 31   # 031: layup (used as idle/stand)
SKILL_PICK = 1    # 001: pick up ball


class HRLDualHumanoid(SkillMimicDualHumanoid):
    """
    HRL environment for dual humanoid pass-and-catch task.
    
    HLC outputs discrete skill selections for each humanoid.
    LLC executes the selected skills.
    
    Action space: 2 discrete actions (one per humanoid)
    - Humanoid A: [pass, run, idle]
    - Humanoid B: [catch, run, idle]
    """
    
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        # HRL-specific config
        self._enable_task_obs = cfg["env"].get("enableTaskObs", True)
        self.goal_size = cfg["env"].get("goalSize", 5)
        
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
        self._termination_heights = torch.tensor(
            self.cfg["env"]["terminationHeight"], 
            device=self.device, dtype=torch.float
        )
        
        print(f"[HRLDualHumanoid] HRL environment initialized")
        print(f"[HRLDualHumanoid] Task obs enabled: {self._enable_task_obs}")
        
        return

    def get_action_size(self):
        """
        HLC action size - will be overridden by HRL agent.
        Returns single humanoid action size for compatibility.
        The actual discrete action handling is done by HRL agent.
        """
        return self._num_actions  # 156, LLC handles the actual actions

    def get_task_obs_size(self):
        """Size of task-specific observations."""
        if self._enable_task_obs:
            return self.goal_size
        return 0

    def _compute_observations(self, env_ids=None):
        """
        Compute observations for HRL.
        Includes humanoid state, ball state, other humanoid state, and task obs.
        """
        # Compute base observations (humanoid A state + ball + other humanoid)
        humanoid_a_obs = self._compute_humanoid_obs(env_ids)
        obj_obs = self._compute_obj_obs(env_ids)
        other_obs_for_a, _ = self._compute_other_humanoid_obs(env_ids)
        
        # Combine observations
        obs = torch.cat([humanoid_a_obs, obj_obs, other_obs_for_a], dim=-1)
        
        # Add task observations if enabled
        if self._enable_task_obs:
            task_obs = self._compute_task_obs(env_ids)
            obs = torch.cat([obs, task_obs], dim=-1)
        
        # Add condition embedding
        if env_ids is None:
            textemb_batch = self.hoi_data_label_batch
            obs = torch.cat((obs, textemb_batch), dim=-1)
            self.obs_buf[:] = obs
        else:
            textemb_batch = self.hoi_data_label_batch[env_ids]
            obs = torch.cat((obs, textemb_batch), dim=-1)
            self.obs_buf[env_ids] = obs

        return

    def _compute_task_obs(self, env_ids=None):
        """
        Compute task-specific observations for pass-and-catch.
        
        Returns:
            - Relative position of ball to each humanoid
            - Ball velocity direction
        """
        if env_ids is None:
            root_pos_a = self._humanoid_root_states[:, 0:3]
            root_pos_b = self._humanoid_b_root_states[:, 0:3]
            ball_pos = self._target_states[:, 0:3]
            ball_vel = self._target_states[:, 7:10]
            root_rot_a = self._humanoid_root_states[:, 3:7]
        else:
            root_pos_a = self._humanoid_root_states[env_ids, 0:3]
            root_pos_b = self._humanoid_b_root_states[env_ids, 0:3]
            ball_pos = self._target_states[env_ids, 0:3]
            ball_vel = self._target_states[env_ids, 7:10]
            root_rot_a = self._humanoid_root_states[env_ids, 3:7]

        # Compute local observations relative to humanoid A
        heading_rot = torch_utils.calc_heading_quat_inv(root_rot_a)
        
        # Ball position relative to A
        local_ball_pos = ball_pos - root_pos_a
        local_ball_pos = quat_rotate(heading_rot, local_ball_pos)
        
        # B position relative to A  
        local_b_pos = root_pos_b - root_pos_a
        local_b_pos = quat_rotate(heading_rot, local_b_pos)
        
        # Ball velocity direction
        ball_speed = torch.norm(ball_vel, dim=-1, keepdim=True)
        ball_vel_dir = ball_vel / (ball_speed + 1e-8)
        local_ball_vel = quat_rotate(heading_rot, ball_vel_dir)
        
        # Concatenate task observations [5 dims]
        task_obs = torch.cat([
            local_ball_pos[:, :2],      # 2: ball xy relative to A
            local_b_pos[:, :2],         # 2: B xy relative to A
            ball_speed,                 # 1: ball speed
        ], dim=-1)
        
        return task_obs

    def _compute_reward(self, actions):
        """
        Compute cooperative reward for pass-and-catch.
        
        Rewards:
        - Ball successfully passed (A releases, moves toward B)
        - Ball successfully caught (B catches)
        - Both humanoids standing
        - NEW: Standing posture rewards
        - NEW: Upright body orientation rewards
        - NEW: Ground contact penalties
        """
        ball_pos = self._target_states[:, 0:3]
        ball_vel = self._target_states[:, 7:10]
        
        root_pos_a = self._humanoid_root_states[:, 0:3]
        root_rot_a = self._humanoid_root_states[:, 3:7]
        root_pos_b = self._humanoid_b_root_states[:, 0:3]
        root_rot_b = self._humanoid_b_root_states[:, 3:7]
        
        height_a = self._rigid_body_pos[:, 0, 2]
        height_b = self._rigid_body_pos_b[:, 0, 2]
        
        dist_ball_to_hand_a = self._get_closest_hand_distance(ball_pos, 'a')
        dist_ball_to_hand_b = self._get_closest_hand_distance(ball_pos, 'b')
        
        ball_contact_force = self._tar_contact_forces
        contact_forces_a = self._contact_forces
        contact_forces_b = self._contact_forces_b
        
        # Use the cooperative reward function from parent
        from env.tasks.skillmimic_dual import compute_coop_reward
        
        self.rew_buf[:] = compute_coop_reward(
            ball_pos, ball_vel,
            root_pos_a, root_pos_b,
            root_rot_a, root_rot_b,
            height_a, height_b,
            dist_ball_to_hand_a, dist_ball_to_hand_b,
            ball_contact_force,
            contact_forces_a, contact_forces_b,
            self._non_foot_body_ids,
            self._reward_w_alive,
            self._reward_w_ball_to_hand,
            self._reward_w_pass_direction,
            self._reward_w_catch_success,
            self._reward_w_ball_height,
            self._reward_w_standing,
            self._reward_w_upright,
            self._reward_w_ground_contact_penalty,
            self._termination_heights
        )
        return

    def get_num_amp_obs(self):
        """Return AMP observation size (required by wrapper)."""
        return self.ref_hoi_obs_size
