# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# HRL-CTDE Humanoid Environment for Dual Pass-and-Catch Task
# CTDE = Centralized Training, Decentralized Execution
#
# Key differences from HRLDualHumanoid:
#   - Symmetric per-agent observations (obs_A and obs_B have identical structure)
#   - Role encoding differentiates passer (A) and catcher (B)
#   - Combined obs_buf = [obs_A || obs_B] for centralized critic
#   - Factorized action space: 3 skills per agent, 9 joint combinations

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
from env.tasks.humanoid_object_task import compute_obj_observations

SKILL_PASS = 4
SKILL_CATCH = 3
SKILL_RUN = 13
SKILL_IDLE = 31
SKILL_PICK = 1


class HRLCTDEHumanoid(SkillMimicDualHumanoid):
    """
    CTDE environment for dual humanoid pass-and-catch.

    Observation: [obs_A(934) || obs_B(934)] = 1868 total.
    Each per-agent obs = self_body(823) + ball(15) + partner(15)
                       + task_obs(15) + role(2) + condition(64).
    Action: single discrete in [0..8], decoded as (action_A, action_B).
    """

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self._enable_task_obs = cfg["env"].get("enableTaskObs", True)
        self.goal_size = cfg["env"].get("goalSize", 15)
        self._role_size = 2
        self._ball_history_len = cfg["env"].get("ballHistoryLength", 0)

        super().__init__(
            cfg=cfg,
            sim_params=sim_params,
            physics_engine=physics_engine,
            device_type=device_type,
            device_id=device_id,
            headless=headless,
        )

        humanoid_obj_size = 823 + 15
        self._llc_obs_a = torch.zeros(
            (self.num_envs, humanoid_obj_size), device=self.device, dtype=torch.float
        )
        self._llc_obs_b = torch.zeros(
            (self.num_envs, humanoid_obj_size), device=self.device, dtype=torch.float
        )

        self._termination_heights = torch.tensor(
            self.cfg["env"]["terminationHeight"],
            device=self.device,
            dtype=torch.float,
        )

        # Pre-compute static role tensors
        self._role_a = torch.zeros((self.num_envs, self._role_size), device=self.device)
        self._role_a[:, 0] = 1.0  # passer
        self._role_b = torch.zeros((self.num_envs, self._role_size), device=self.device)
        self._role_b[:, 1] = 1.0  # catcher

        # Ball history buffers: [num_envs, T, 6] (pos + vel in agent's heading frame)
        T = self._ball_history_len
        if T > 0:
            self._ball_history_a = torch.zeros(
                (self.num_envs, T, 6), device=self.device, dtype=torch.float
            )
            self._ball_history_b = torch.zeros(
                (self.num_envs, T, 6), device=self.device, dtype=torch.float
            )

        per_agent = self._get_per_agent_obs_size()
        print(f"[HRLCTDEHumanoid] Initialized  per_agent_obs={per_agent}  total_obs={per_agent*2}")
        print(f"[HRLCTDEHumanoid] task_obs={self.goal_size}  role={self._role_size}  ball_history={T}")

    # ------------------------------------------------------------------
    # Observation sizes
    # ------------------------------------------------------------------

    def _get_per_agent_obs_size(self):
        size = self._num_obs + 15 + 15  # self_body + ball + partner
        if self._enable_task_obs:
            size += self.goal_size
        size += self._ball_history_len * 6
        size += self._role_size
        size += self.condition_size
        return size

    def get_obs_size(self):
        return self._get_per_agent_obs_size() * 2

    def get_task_obs_size(self):
        if self._enable_task_obs:
            return self.goal_size
        return 0

    def get_action_size(self):
        return self._num_actions  # 156, LLC handles actual joint actions

    # ------------------------------------------------------------------
    # Compute ball obs from B's frame (same as HRLDualHumanoid)
    # ------------------------------------------------------------------

    def _compute_obj_obs_b(self, env_ids=None):
        if env_ids is None:
            root_states_b = self._humanoid_b_root_states
            tar_states = self._target_states
        else:
            root_states_b = self._humanoid_b_root_states[env_ids]
            tar_states = self._target_states[env_ids]
        return compute_obj_observations(root_states_b, tar_states)

    # ------------------------------------------------------------------
    # Core observation computation
    # ------------------------------------------------------------------

    def _compute_observations(self, env_ids=None):
        humanoid_a_obs = self._compute_humanoid_obs(env_ids)
        obj_obs_a = self._compute_obj_obs(env_ids)
        obj_obs_b = self._compute_obj_obs_b(env_ids)
        other_obs_for_a, other_obs_for_b = self._compute_other_humanoid_obs(env_ids)

        humanoid_b_obs = self._compute_humanoid_b_obs(env_ids)

        # Cache LLC observations
        llc_a = torch.cat([humanoid_a_obs, obj_obs_a], dim=-1)
        llc_b = torch.cat([humanoid_b_obs, obj_obs_b], dim=-1)
        if env_ids is None:
            self._llc_obs_a[:] = llc_a
            self._llc_obs_b[:] = llc_b
        else:
            self._llc_obs_a[env_ids] = llc_a
            self._llc_obs_b[env_ids] = llc_b

        # Task obs (symmetric, per-agent)
        if self._enable_task_obs:
            task_obs_a, task_obs_b = self._compute_task_obs_pair(env_ids)

        # Ball history update and retrieval
        if self._ball_history_len > 0:
            self._update_ball_history(env_ids)
            if env_ids is None:
                bh_a = self._ball_history_a.reshape(self.num_envs, -1)
                bh_b = self._ball_history_b.reshape(self.num_envs, -1)
            else:
                bh_a = self._ball_history_a[env_ids].reshape(len(env_ids), -1)
                bh_b = self._ball_history_b[env_ids].reshape(len(env_ids), -1)

        # Condition embedding
        if env_ids is None:
            cond = self.hoi_data_label_batch
            role_a = self._role_a
            role_b = self._role_b
        else:
            cond = self.hoi_data_label_batch[env_ids]
            role_a = self._role_a[env_ids]
            role_b = self._role_b[env_ids]

        # Build per-agent obs
        parts_a = [humanoid_a_obs, obj_obs_a, other_obs_for_a]
        parts_b = [humanoid_b_obs, obj_obs_b, other_obs_for_b]
        if self._enable_task_obs:
            parts_a.append(task_obs_a)
            parts_b.append(task_obs_b)
        if self._ball_history_len > 0:
            parts_a.append(bh_a)
            parts_b.append(bh_b)
        parts_a += [role_a, cond]
        parts_b += [role_b, cond]

        obs_a = torch.cat(parts_a, dim=-1)
        obs_b = torch.cat(parts_b, dim=-1)
        combined = torch.cat([obs_a, obs_b], dim=-1)

        if env_ids is None:
            self.obs_buf[:] = combined
        else:
            self.obs_buf[env_ids] = combined

    # ------------------------------------------------------------------
    # Per-agent task observations (symmetric 15-dim)
    # ------------------------------------------------------------------

    def _compute_task_obs_pair(self, env_ids=None):
        """Compute symmetric 15-dim task obs for both agents."""
        if env_ids is None:
            root_pos_a = self._humanoid_root_states[:, 0:3]
            root_rot_a = self._humanoid_root_states[:, 3:7]
            root_vel_a = self._humanoid_root_states[:, 7:10]
            root_pos_b = self._humanoid_b_root_states[:, 0:3]
            root_rot_b = self._humanoid_b_root_states[:, 3:7]
            root_vel_b = self._humanoid_b_root_states[:, 7:10]
            ball_pos = self._target_states[:, 0:3]
            ball_vel = self._target_states[:, 7:10]
            rb_pos_a = self._rigid_body_pos
            rb_vel_a = self._rigid_body_vel
            rb_pos_b = self._rigid_body_pos_b
            rb_vel_b = self._rigid_body_vel_b
        else:
            root_pos_a = self._humanoid_root_states[env_ids, 0:3]
            root_rot_a = self._humanoid_root_states[env_ids, 3:7]
            root_vel_a = self._humanoid_root_states[env_ids, 7:10]
            root_pos_b = self._humanoid_b_root_states[env_ids, 0:3]
            root_rot_b = self._humanoid_b_root_states[env_ids, 3:7]
            root_vel_b = self._humanoid_b_root_states[env_ids, 7:10]
            ball_pos = self._target_states[env_ids, 0:3]
            ball_vel = self._target_states[env_ids, 7:10]
            rb_pos_a = self._rigid_body_pos[env_ids]
            rb_vel_a = self._rigid_body_vel[env_ids]
            rb_pos_b = self._rigid_body_pos_b[env_ids]
            rb_vel_b = self._rigid_body_vel_b[env_ids]

        task_obs_a = self._agent_task_obs(
            root_pos_a, root_rot_a, root_vel_a, rb_pos_a, rb_vel_a,
            root_pos_b, root_rot_b, root_vel_b, rb_pos_b, rb_vel_b,
            ball_pos, ball_vel,
        )
        task_obs_b = self._agent_task_obs(
            root_pos_b, root_rot_b, root_vel_b, rb_pos_b, rb_vel_b,
            root_pos_a, root_rot_a, root_vel_a, rb_pos_a, rb_vel_a,
            ball_pos, ball_vel,
        )
        return task_obs_a, task_obs_b

    def _agent_task_obs(
        self,
        self_root_pos, self_root_rot, self_root_vel,
        self_rb_pos, self_rb_vel,
        partner_root_pos, partner_root_rot, partner_root_vel,
        partner_rb_pos, partner_rb_vel,
        ball_pos, ball_vel,
    ):
        """
        Symmetric 15-dim task obs from one agent's perspective.

        1. ball → self hand center (3)
        2. ball velocity (3)
        3. ball → partner hand center (3)
        4. partner velocity (3)
        5. predicted future ball position relative to self root (3)
        All in self's heading frame.
        """
        heading_inv = torch_utils.calc_heading_quat_inv(self_root_rot)

        self_hand = self_rb_pos[:, self._hand_body_ids, :].mean(dim=1)
        partner_hand = partner_rb_pos[:, self._hand_body_ids, :].mean(dim=1)

        ball_to_self_hand = quat_rotate(heading_inv, ball_pos - self_hand)
        ball_vel_local = quat_rotate(heading_inv, ball_vel)
        ball_to_partner_hand = quat_rotate(heading_inv, ball_pos - partner_hand)
        partner_vel_local = quat_rotate(heading_inv, partner_root_vel)

        PREDICT_DT = 0.5
        GRAVITY = torch.tensor([0.0, 0.0, -9.81], device=ball_pos.device)
        future_ball = ball_pos + ball_vel * PREDICT_DT + 0.5 * GRAVITY * (PREDICT_DT ** 2)
        future_ball_rel = quat_rotate(heading_inv, future_ball - self_root_pos)

        return torch.cat([
            ball_to_self_hand,
            ball_vel_local,
            ball_to_partner_hand,
            partner_vel_local,
            future_ball_rel,
        ], dim=-1)

    # ------------------------------------------------------------------
    # Ball history buffer
    # ------------------------------------------------------------------

    def _update_ball_history(self, env_ids=None):
        """Shift history left and append current ball state in each agent's heading frame."""
        if env_ids is None:
            root_rot_a = self._humanoid_root_states[:, 3:7]
            root_rot_b = self._humanoid_b_root_states[:, 3:7]
            ball_pos = self._target_states[:, 0:3]
            ball_vel = self._target_states[:, 7:10]
        else:
            root_rot_a = self._humanoid_root_states[env_ids, 3:7]
            root_rot_b = self._humanoid_b_root_states[env_ids, 3:7]
            ball_pos = self._target_states[env_ids, 0:3]
            ball_vel = self._target_states[env_ids, 7:10]

        heading_inv_a = torch_utils.calc_heading_quat_inv(root_rot_a)
        heading_inv_b = torch_utils.calc_heading_quat_inv(root_rot_b)

        bp_a = quat_rotate(heading_inv_a, ball_pos)
        bv_a = quat_rotate(heading_inv_a, ball_vel)
        bp_b = quat_rotate(heading_inv_b, ball_pos)
        bv_b = quat_rotate(heading_inv_b, ball_vel)

        frame_a = torch.cat([bp_a, bv_a], dim=-1)  # [N, 6]
        frame_b = torch.cat([bp_b, bv_b], dim=-1)

        if env_ids is None:
            self._ball_history_a[:, :-1] = self._ball_history_a[:, 1:].clone()
            self._ball_history_a[:, -1] = frame_a
            self._ball_history_b[:, :-1] = self._ball_history_b[:, 1:].clone()
            self._ball_history_b[:, -1] = frame_b
        else:
            self._ball_history_a[env_ids, :-1] = self._ball_history_a[env_ids, 1:].clone()
            self._ball_history_a[env_ids, -1] = frame_a
            self._ball_history_b[env_ids, :-1] = self._ball_history_b[env_ids, 1:].clone()
            self._ball_history_b[env_ids, -1] = frame_b

    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)
        if self._ball_history_len > 0:
            self._ball_history_a[env_ids] = 0.0
            self._ball_history_b[env_ids] = 0.0

    # ------------------------------------------------------------------
    # LLC observation pair
    # ------------------------------------------------------------------

    def get_llc_obs_pair(self):
        return self._llc_obs_a, self._llc_obs_b

    # ------------------------------------------------------------------
    # Reward (reuses cooperative reward from parent / HRLDualHumanoid)
    # ------------------------------------------------------------------

    def _compute_reward(self, actions):
        ball_pos = self._target_states[:, 0:3]
        ball_vel = self._target_states[:, 7:10]
        root_pos_a = self._humanoid_root_states[:, 0:3]
        root_rot_a = self._humanoid_root_states[:, 3:7]
        root_pos_b = self._humanoid_b_root_states[:, 0:3]
        root_rot_b = self._humanoid_b_root_states[:, 3:7]
        height_a = self._rigid_body_pos[:, 0, 2]
        height_b = self._rigid_body_pos_b[:, 0, 2]
        dist_a = self._get_closest_hand_distance(ball_pos, "a")
        dist_b = self._get_closest_hand_distance(ball_pos, "b")
        ball_cf = self._tar_contact_forces
        cf_a = self._contact_forces
        cf_b = self._contact_forces_b

        from env.tasks.skillmimic_dual import compute_coop_reward

        self.rew_buf[:] = compute_coop_reward(
            ball_pos, ball_vel,
            root_pos_a, root_pos_b,
            root_rot_a, root_rot_b,
            height_a, height_b,
            dist_a, dist_b,
            ball_cf, cf_a, cf_b,
            self._non_foot_body_ids,
            self._reward_w_alive,
            self._reward_w_ball_to_hand,
            self._reward_w_pass_direction,
            self._reward_w_catch_success,
            self._reward_w_ball_height,
            self._reward_w_standing,
            self._reward_w_upright,
            self._reward_w_ground_contact_penalty,
            self._reward_w_catch_fail,
            self._termination_heights,
        )

        self._store_reward_components(
            ball_pos, ball_vel, root_pos_a, root_pos_b,
            root_rot_a, root_rot_b, height_a, height_b,
            dist_a, dist_b, ball_cf, cf_a, cf_b,
        )

    def _store_reward_components(
        self, ball_pos, ball_vel, root_pos_a, root_pos_b,
        root_rot_a, root_rot_b, height_a, height_b,
        dist_a, dist_b, ball_cf, cf_a, cf_b,
    ):
        """Compute individual reward components and store in self.extras."""
        device = ball_pos.device
        MIN_STANDING_HEIGHT = 0.8
        K_PASS = 2.0
        BALL_IN_FLIGHT_SPEED = 0.5
        BALL_LEFT_HAND_DIST = 0.2
        K_CATCH_SOFT = 2.0
        CATCH_HAND_DIST = 0.15
        CG2_CONTACT_THRESH = 1.0
        BALL_GROUND_Z = 0.3
        CATCH_FAIL_RADIUS = 2.0

        alive_a = (height_a > self._termination_heights).float()
        alive_b = (height_b > self._termination_heights).float()
        r_alive = (alive_a * alive_b * self._reward_w_alive).mean()

        ball_to_a = torch.norm(ball_pos - root_pos_a, dim=-1)
        ball_to_b = torch.norm(ball_pos - root_pos_b, dim=-1)
        b_is_catcher = (ball_to_a < ball_to_b).float()
        catcher_hand_dist = dist_b * b_is_catcher + dist_a * (1.0 - b_is_catcher)
        ball_speed = torch.norm(ball_vel, dim=-1)

        ball_in_flight = (ball_speed > BALL_IN_FLIGHT_SPEED) & (dist_a > BALL_LEFT_HAND_DIST)
        dist_to_catcher = dist_b * b_is_catcher + dist_a * (1.0 - b_is_catcher)
        R_pass = torch.where(
            ball_in_flight,
            torch.exp(-K_PASS * dist_to_catcher),
            torch.ones_like(ball_speed),
        )
        r_pass_mean = R_pass.mean()

        ball_has_contact = (torch.norm(ball_cf, dim=-1) > CG2_CONTACT_THRESH).float()
        hard_catch_dist = (catcher_hand_dist < CATCH_HAND_DIST).float()
        catcher_height = height_b * b_is_catcher + height_a * (1.0 - b_is_catcher)
        catcher_standing = (catcher_height > MIN_STANDING_HEIGHT).float()
        hard_catch = ball_has_contact * hard_catch_dist * catcher_standing
        soft_catch = torch.exp(-K_CATCH_SOFT * catcher_hand_dist)
        R_catch = (1.0 - hard_catch) * soft_catch + hard_catch
        r_coop = (R_pass * R_catch * self._reward_w_catch_success).mean()

        standing_a = torch.clamp((height_a - MIN_STANDING_HEIGHT) / 0.5, 0.0, 1.0)
        standing_b = torch.clamp((height_b - MIN_STANDING_HEIGHT) / 0.5, 0.0, 1.0)
        r_standing = ((standing_a + standing_b) * 0.5 * self._reward_w_standing).mean()

        x_a, y_a = root_rot_a[:, 0], root_rot_a[:, 1]
        x_b, y_b = root_rot_b[:, 0], root_rot_b[:, 1]
        up_z_a = 1.0 - 2.0 * (x_a * x_a + y_a * y_a)
        up_z_b = 1.0 - 2.0 * (x_b * x_b + y_b * y_b)
        r_upright = ((torch.clamp(up_z_a, 0, 1) + torch.clamp(up_z_b, 0, 1))
                     * 0.5 * self._reward_w_upright).mean()

        contact_thresh = 10.0
        nf = self._non_foot_body_ids
        gc_a = torch.clamp(
            torch.sum(torch.clamp(torch.norm(cf_a[:, nf, :], dim=-1) - contact_thresh, min=0.0), dim=-1) / 100.0,
            max=2.0,
        )
        gc_b = torch.clamp(
            torch.sum(torch.clamp(torch.norm(cf_b[:, nf, :], dim=-1) - contact_thresh, min=0.0), dim=-1) / 100.0,
            max=2.0,
        )
        r_ground_contact = ((gc_a + gc_b) * self._reward_w_ground_contact_penalty).mean()

        ball_near_ground = (ball_pos[:, 2] < BALL_GROUND_Z).float()
        ball_near_b = (ball_to_b < CATCH_FAIL_RADIUS).float()
        r_catch_fail = (ball_near_ground * ball_near_b * (1.0 - hard_catch)
                        * self._reward_w_catch_fail).mean()

        self.extras["reward_components"] = {
            "alive": r_alive.item(),
            "coop": r_coop.item(),
            "pass": r_pass_mean.item(),
            "catch": R_catch.mean().item(),
            "standing": r_standing.item(),
            "upright": r_upright.item(),
            "ground_contact": r_ground_contact.item(),
            "catch_fail": r_catch_fail.item(),
        }

    def get_num_amp_obs(self):
        return self.ref_hoi_obs_size
