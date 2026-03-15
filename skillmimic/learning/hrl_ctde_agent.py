# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# HRL-CTDE Agent for Dual Humanoid Pass-and-Catch.
#
# Key differences from HRLDualAgent:
#   - Decodes joint action (0-8) into independent (action_A, action_B)
#   - Separate control_mapping per agent (3 skills each)
#   - Passes per_agent_obs_size to the network builder

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
from env.tasks.hrl_ctde_humanoid import SKILL_PASS, SKILL_CATCH, SKILL_RUN, SKILL_IDLE
from utils import torch_utils
from isaacgym.torch_utils import quat_rotate

from tensorboardX import SummaryWriter


class HRLCTDEAgent(HRLAgentDiscrete):
    """
    CTDE training agent for dual humanoid pass-and-catch.

    Joint action space: 9 discrete (3 skills_A × 3 skills_B).
    Uses factorized policy network (HRLCTDEBuilder).
    """

    def __init__(self, base_name, config):
        self._single_action_size = 156

        # Per-agent control mappings (set before super so they survive)
        self._control_mapping_a = config.get("control_mapping_a", [4, 13, 31])
        self._control_mapping_b = config.get("control_mapping_b", [3, 13, 31])
        self._skills_per_agent = len(self._control_mapping_a)

        # Per-agent obs size (will be passed to network builder)
        self._per_agent_obs_size = config.get("per_agent_obs_size", 934)

        # Guidance penalty
        self._guidance_penalty_weight = float(
            config.get("guidance_penalty_weight", 1.0)
        )
        self._holding_dist_thresh = float(config.get("holding_dist_thresh", 0.2))
        self._holding_contact_thresh = float(
            config.get("holding_contact_thresh", 1.0)
        )
        self._ball_to_b_speed_thresh = float(
            config.get("ball_to_b_speed_thresh", 0.5)
        )

        # Trajectory predictor config (before super for _build_net_config)
        self._traj_pred_history = int(config.get("traj_pred_history", 0))
        self._traj_pred_horizons = config.get("traj_pred_horizons", [])
        self._traj_pred_hidden = int(config.get("traj_pred_hidden", 64))
        self._traj_pred_type = config.get("traj_pred_type", "gru")
        self._traj_pred_loss_weight = float(config.get("traj_pred_loss_weight", 0.0))
        self._use_traj_pred = (
            self._traj_pred_history > 0 and len(self._traj_pred_horizons) > 0
        )

        # Compute ball_history_offset within per-agent obs:
        # layout: self_body(823) + ball(15) + partner(15) + task_obs(15) = 868
        self._ball_history_offset = 823 + 15 + 15 + 15

        self._collecting_traj_data = False

        super().__init__(base_name, config)

        actual_size = self.vec_env.env.task._num_actions
        if self._single_action_size != actual_size:
            print(
                f"[HRLCTDEAgent] WARNING: single action size updated "
                f"{self._single_action_size} → {actual_size}"
            )
            self._single_action_size = actual_size

        # Skill monitor
        self._skill_monitor_interval = int(config.get("skill_monitor_interval", 200))
        self._skill_monitor_print = _as_bool(config.get("skill_monitor_print", True), True)
        self._skill_monitor_tb = _as_bool(config.get("skill_monitor_tb", True), True)
        self._skill_monitor_hlc_steps_total = 0
        self._skill_monitor_reset_accumulators()

        print(f"[HRLCTDEAgent] Initialized")
        print(f"  control_mapping_a={self._control_mapping_a}")
        print(f"  control_mapping_b={self._control_mapping_b}")
        print(f"  per_agent_obs_size={self._per_agent_obs_size}")
        print(f"  skills_per_agent={self._skills_per_agent}")
        print(f"  traj_pred={self._use_traj_pred}  history={self._traj_pred_history}")
        print(f"  traj_pred_horizons={self._traj_pred_horizons}  loss_w={self._traj_pred_loss_weight}")

    # ------------------------------------------------------------------
    # Override: inject per_agent_obs_size into model builder config
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
    # Action space: 9 joint actions  (K_a * K_b)
    # ------------------------------------------------------------------

    def _setup_action_space(self):
        super()._setup_action_space()
        self.actions_num = self._latent_dim
        return

    # ------------------------------------------------------------------
    # env_step: LLC loop + guidance penalty + monitoring
    # ------------------------------------------------------------------

    def env_step(self, actions):
        obs = self.obs["obs"]

        rewards = 0.0
        done_count = 0.0
        terminate_count = 0.0
        for _ in range(self._llc_steps):
            llc_actions = self._compute_llc_action(obs, actions)
            obs, curr_rewards, curr_dones, infos = self.vec_env.step(llc_actions)
            rewards += curr_rewards
            done_count += curr_dones
            terminate_count += infos["terminate"]

        rewards /= self._llc_steps

        if self.is_tensor_obses:
            rewards = self._apply_guidance_penalty(rewards, actions)
            self._skill_monitor_update(actions)
            self._catch_stats_log()
            self._log_reward_components(infos)
            self._log_per_agent_entropy()

        dones = torch.zeros_like(done_count)
        dones[done_count > 0] = 1.0
        terminate = torch.zeros_like(terminate_count)
        terminate[terminate_count > 0] = 1.0
        infos["terminate"] = terminate

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
            return (
                self.obs_to_tensors(obs),
                rewards.to(self.ppo_device),
                dones.to(self.ppo_device),
                infos,
            )
        else:
            rewards_np = rewards
            if self.value_size == 1:
                rewards_np = np.expand_dims(rewards_np, axis=1)
            return (
                self.obs_to_tensors(obs),
                torch.from_numpy(rewards_np).to(self.ppo_device).float(),
                torch.from_numpy(dones).to(self.ppo_device),
                infos,
            )

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

            # Per-env: invalidate if any done in (t, t+max_h]
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

        value_preds_batch = input_dict["old_values"]
        old_action_log_probs_batch = input_dict["old_logp_actions"]
        advantage = input_dict["advantages"]
        return_batch = input_dict["returns"]
        logits_batch = input_dict["logits"]
        actions_batch = input_dict["actions"]
        obs_batch = input_dict["obs"]
        obs_batch = self._preproc_obs(obs_batch)

        lr = self.last_lr
        kl = 1.0
        lr_mul = 1.0
        curr_e_clip = lr_mul * self.e_clip

        batch_dict = {
            "is_train": True,
            "prev_actions": actions_batch,
            "obs": obs_batch,
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict["rnn_masks"]
            batch_dict["rnn_states"] = input_dict["rnn_states"]
            batch_dict["seq_length"] = self.seq_len

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict["prev_neglogp"]
            values = res_dict["values"]
            entropy = res_dict["entropy"]
            now_logits = res_dict["logits"]

            a_info = self._actor_loss(
                old_action_log_probs_batch, action_log_probs, advantage, curr_e_clip
            )
            a_loss = a_info["actor_loss"]

            c_info = self._critic_loss(
                value_preds_batch, values, curr_e_clip, return_batch, self.clip_value
            )
            c_loss = c_info["critic_loss"]

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

            a_clip_frac = torch.mean(a_info["actor_clipped"].float())
            a_info["actor_loss"] = a_loss
            a_info["actor_clip_frac"] = a_clip_frac

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
            "entropy": entropy,
            "kl": kl_dist,
            "last_lr": self.last_lr,
            "lr_mul": lr_mul,
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

    # ------------------------------------------------------------------
    # LLC action: decode joint action → (skill_A, skill_B) → dual LLC
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

        llc_obs_a, llc_obs_b = self.vec_env.env.task.get_llc_obs_pair()

        # Agent A
        cs_a = torch.zeros((batch_size, 64), device=llc_obs_a.device)
        cs_a[torch.arange(batch_size), -64 + skill_a] = 1.0
        inp_a = torch.cat((llc_obs_a, cs_a), dim=-1)
        pa = self._llc_agent._preproc_obs(inp_a)
        mu_a, _ = self._llc_agent.model.a2c_network.eval_actor(obs=pa)
        act_a = self._llc_agent.preprocess_actions(mu_a)

        # Agent B
        cs_b = torch.zeros((batch_size, 64), device=llc_obs_b.device)
        cs_b[torch.arange(batch_size), -64 + skill_b] = 1.0
        inp_b = torch.cat((llc_obs_b, cs_b), dim=-1)
        pb = self._llc_agent._preproc_obs(inp_b)
        mu_b, _ = self._llc_agent.model.a2c_network.eval_actor(obs=pb)
        act_b = self._llc_agent.preprocess_actions(mu_b)

        return torch.cat([act_a, act_b], dim=-1)

    # ------------------------------------------------------------------
    # LLC config: 902 = 838 + 64
    # ------------------------------------------------------------------

    def _build_llc_agent_config(self, config_params, network):
        llc_env_info = copy.deepcopy(self.env_info)

        llc_obs_size = 902
        obs_space = llc_env_info["observation_space"]
        llc_env_info["observation_space"] = spaces.Box(
            obs_space.low[0], obs_space.high[0], shape=(llc_obs_size,)
        )
        llc_env_info["action_space"] = spaces.Box(
            -1.0, 1.0, shape=(self._single_action_size,)
        )

        config = config_params["config"]
        config["network"] = network
        config["num_actors"] = self.num_actors
        config["features"] = {"observer": self.algo_observer}
        config["env_info"] = llc_env_info
        return config

    # ------------------------------------------------------------------
    # _extract_llc_obs  (not used directly, but kept for compatibility)
    # ------------------------------------------------------------------

    def _extract_llc_obs(self, obs):
        return obs[..., :838]

    # ------------------------------------------------------------------
    # Guidance penalty  (same logic as HRLDualAgent but with CTDE decoding)
    # ------------------------------------------------------------------

    def _apply_guidance_penalty(self, rewards, actions):
        if not self.is_tensor_obses:
            return rewards

        device = rewards.device
        at = actions if actions.dim() == 1 else actions.squeeze(-1)
        at = at.long().to(device)

        mapping_a = torch.tensor(self._control_mapping_a, device=device, dtype=torch.long)
        mapping_b = torch.tensor(self._control_mapping_b, device=device, dtype=torch.long)
        K = self._skills_per_agent
        skill_a = mapping_a[at // K]
        skill_b = mapping_b[at % K]

        task = self.vec_env.env.task

        ball_pos = task._target_states[:, 0:3]
        ball_vel = task._target_states[:, 7:10]
        root_pos_a = task._humanoid_root_states[:, 0:3]
        root_pos_b = task._humanoid_b_root_states[:, 0:3]
        ball_cf = task._tar_contact_forces

        dist_a = task._get_closest_hand_distance(ball_pos, "a")
        dist_b = task._get_closest_hand_distance(ball_pos, "b")

        ball_has_contact = torch.norm(ball_cf, dim=-1) > self._holding_contact_thresh
        a_holding = (dist_a < self._holding_dist_thresh) & ball_has_contact

        ball_speed = torch.norm(ball_vel, dim=-1)
        to_b = torch.sum(ball_vel * (root_pos_b - ball_pos), dim=-1) > 0.0
        away_a = torch.sum(ball_vel * (ball_pos - root_pos_a), dim=-1) > 0.0
        ball_to_b = (
            (ball_speed > self._ball_to_b_speed_thresh) & to_b & away_a & (dist_b < dist_a)
        )

        penalty = torch.zeros_like(rewards)
        penalty += (a_holding & (skill_a != SKILL_PASS)).float() * self._guidance_penalty_weight
        bad_b = ~((skill_b == SKILL_CATCH) | (skill_b == SKILL_RUN))
        penalty += (ball_to_b & bad_b).float() * self._guidance_penalty_weight

        return rewards - penalty

    # ------------------------------------------------------------------
    # Skill monitor
    # ------------------------------------------------------------------

    def _skill_monitor_reset_accumulators(self):
        self._sm_total = 0.0
        self._sm_a = {4: 0.0, 13: 0.0, 31: 0.0}
        self._sm_b = {3: 0.0, 13: 0.0, 31: 0.0}
        K = getattr(self, "_skills_per_agent", 3)
        self._sm_joint = [0.0] * (K * K)
        self._sm_ball_to_b = 0.0
        self._sm_ball_to_b_catch = 0.0
        self._sm_a_hold = 0.0
        self._sm_a_hold_pass = 0.0

    def _skill_monitor_update(self, actions):
        if self._skill_monitor_interval <= 0:
            return
        if getattr(self, "rank", 0) != 0:
            return
        if not self.is_tensor_obses:
            return

        with torch.no_grad():
            task = self.vec_env.env.task
            at = actions if actions.dim() == 1 else actions.squeeze(-1)
            at = at.long().to(task._target_states.device)

            mapping_a = torch.tensor(self._control_mapping_a, device=at.device, dtype=torch.long)
            mapping_b = torch.tensor(self._control_mapping_b, device=at.device, dtype=torch.long)
            K = self._skills_per_agent
            skill_a = mapping_a[at // K]
            skill_b = mapping_b[at % K]

            batch = float(skill_a.numel())
            self._sm_total += batch
            for sid in self._sm_a:
                self._sm_a[sid] += float((skill_a == sid).sum().item())
            for sid in self._sm_b:
                self._sm_b[sid] += float((skill_b == sid).sum().item())
            for j in range(K * K):
                self._sm_joint[j] += float((at == j).sum().item())

            ball_pos = task._target_states[:, 0:3]
            ball_vel = task._target_states[:, 7:10]
            root_pos_a = task._humanoid_root_states[:, 0:3]
            root_pos_b = task._humanoid_b_root_states[:, 0:3]
            ball_cf = task._tar_contact_forces
            dist_a = task._get_closest_hand_distance(ball_pos, "a")
            dist_b = task._get_closest_hand_distance(ball_pos, "b")

            ball_has_contact = torch.norm(ball_cf, dim=-1) > self._holding_contact_thresh
            a_holding = (dist_a < self._holding_dist_thresh) & ball_has_contact
            ball_speed = torch.norm(ball_vel, dim=-1)
            to_b = torch.sum(ball_vel * (root_pos_b - ball_pos), dim=-1) > 0.0
            away_a = torch.sum(ball_vel * (ball_pos - root_pos_a), dim=-1) > 0.0
            ball_to_b = (
                (ball_speed > self._ball_to_b_speed_thresh) & to_b & away_a & (dist_b < dist_a)
            )

            self._sm_ball_to_b += float(ball_to_b.sum().item())
            self._sm_ball_to_b_catch += float((ball_to_b & (skill_b == 3)).sum().item())
            self._sm_a_hold += float(a_holding.sum().item())
            self._sm_a_hold_pass += float((a_holding & (skill_a == 4)).sum().item())

            self._skill_monitor_hlc_steps_total += 1
            if (self._skill_monitor_hlc_steps_total % self._skill_monitor_interval) != 0:
                return

            eps = 1e-6
            total = max(self._sm_total, eps)
            step = int(self._skill_monitor_hlc_steps_total)

            a_pass = self._sm_a[4] / total
            a_run = self._sm_a[13] / total
            a_idle = self._sm_a[31] / total
            b_catch = self._sm_b[3] / total
            b_run = self._sm_b[13] / total
            b_idle = self._sm_b[31] / total

            ball_to_b_frac = self._sm_ball_to_b / total
            p_b_catch = (
                self._sm_ball_to_b_catch / max(self._sm_ball_to_b, eps)
                if self._sm_ball_to_b > 0
                else 0.0
            )
            p_a_pass = (
                self._sm_a_hold_pass / max(self._sm_a_hold, eps)
                if self._sm_a_hold > 0
                else 0.0
            )

            if self._skill_monitor_tb and hasattr(self, "writer") and self.writer is not None:
                self.writer.add_scalar("skills/A_pass_frac", a_pass, step)
                self.writer.add_scalar("skills/A_run_frac", a_run, step)
                self.writer.add_scalar("skills/A_idle_frac", a_idle, step)
                self.writer.add_scalar("skills/B_catch_frac", b_catch, step)
                self.writer.add_scalar("skills/B_run_frac", b_run, step)
                self.writer.add_scalar("skills/B_idle_frac", b_idle, step)
                self.writer.add_scalar("skills/ball_to_b_frac", ball_to_b_frac, step)
                self.writer.add_scalar("skills/p_B_catch_given_ball_to_B", p_b_catch, step)
                self.writer.add_scalar("skills/p_A_pass_given_A_holding", p_a_pass, step)
                for j in range(K * K):
                    self.writer.add_scalar(
                        f"skills/joint_{j}", self._sm_joint[j] / total, step
                    )

            if self._skill_monitor_print:
                print(
                    f"[SkillMonitor] hlc_step={step}"
                    f" A(pass/run/idle)={a_pass:.3f}/{a_run:.3f}/{a_idle:.3f}"
                    f" B(catch/run/idle)={b_catch:.3f}/{b_run:.3f}/{b_idle:.3f}"
                    f" ball_to_B={ball_to_b_frac:.3f}"
                    f" P(B_catch|ball_to_B)={p_b_catch:.3f}"
                    f" P(A_pass|A_holding)={p_a_pass:.3f}"
                )

            self._skill_monitor_reset_accumulators()

    # ------------------------------------------------------------------
    # Per-agent entropy logging
    # ------------------------------------------------------------------

    def _log_per_agent_entropy(self):
        if not hasattr(self, "_entropy_step"):
            self._entropy_step = 0
            self._entropy_interval = self._skill_monitor_interval
        self._entropy_step += 1
        if (self._entropy_step % self._entropy_interval) != 0:
            return
        net = self.model.a2c_network
        logits_a = getattr(net, "_last_logits_a", None)
        logits_b = getattr(net, "_last_logits_b", None)
        if logits_a is None or logits_b is None:
            return
        with torch.no_grad():
            probs_a = torch.softmax(logits_a, dim=-1)
            probs_b = torch.softmax(logits_b, dim=-1)
            ent_a = -(probs_a * torch.log(probs_a + 1e-8)).sum(dim=-1).mean()
            ent_b = -(probs_b * torch.log(probs_b + 1e-8)).sum(dim=-1).mean()
        if hasattr(self, "writer") and self.writer is not None:
            self.writer.add_scalar("entropy/agent_a", ent_a.item(), self._entropy_step)
            self.writer.add_scalar("entropy/agent_b", ent_b.item(), self._entropy_step)

    # ------------------------------------------------------------------
    # Reward component logging
    # ------------------------------------------------------------------

    def _log_reward_components(self, infos):
        if not hasattr(self, "_rc_step"):
            self._rc_step = 0
            self._rc_interval = self._skill_monitor_interval
        self._rc_step += 1
        if (self._rc_step % self._rc_interval) != 0:
            return
        components = infos.get("reward_components", None)
        if components is None:
            return
        if hasattr(self, "writer") and self.writer is not None:
            for name, val in components.items():
                self.writer.add_scalar(f"rewards/{name}", val, self._rc_step)

    # ------------------------------------------------------------------
    # Catch stats logging (same as HRLDualAgent)
    # ------------------------------------------------------------------

    def _catch_stats_log(self):
        if not hasattr(self, "_cs_initialized"):
            task = self.vec_env.env.task
            self._cs_enabled = getattr(task, "_catch_stats_enabled", False)
            self._cs_interval = getattr(task, "_catch_stats_log_interval", 50)
            self._cs_print = getattr(task, "_catch_stats_print", False)
            self._cs_hlc_step = 0
            self._cs_initialized = True

        if not self._cs_enabled:
            return

        self._cs_hlc_step += 1
        if (self._cs_hlc_step % self._cs_interval) != 0:
            return

        task = self.vec_env.env.task
        stats = task.get_catch_stats()
        if not stats or stats["catch_attempts"] == 0:
            return

        step = self._cs_hlc_step
        if hasattr(self, "writer") and self.writer is not None:
            for key, val in stats.items():
                self.writer.add_scalar(f"catch_stats/{key}", val, step)

        if self._cs_print:
            print(
                f"[CatchStats] hlc_step={step}"
                f" pass={stats['pass_successes']}/{stats['pass_attempts']}"
                f"({stats['pass_success_rate']:.3f})"
                f" catch={stats['catch_successes']}/{stats['catch_attempts']}"
                f"({stats['catch_success_rate']:.3f})"
            )

        task.reset_catch_stats()


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------

def _as_bool(v, default: bool) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y", "on")
    return default
