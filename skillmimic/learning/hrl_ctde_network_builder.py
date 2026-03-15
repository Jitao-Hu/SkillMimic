# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Factorized policy network for HRL-CTDE dual humanoid.
#
# Actor:  splits combined obs → shared MLP on each agent's obs → per-agent
#         logits → joint logits via outer sum.
# Critic: standard MLP on the full combined obs (centralized).
# Optional: learned trajectory predictor augments per-agent obs.

from rl_games.algos_torch import network_builder

import torch
import torch.nn as nn

from learning.hrl_ctde_trajectory_predictor import TrajectoryPredictor


class HRLCTDEBuilder(network_builder.A2CBuilder):
    """Builder for the CTDE factorized policy network."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            # Pop CTDE-specific params before parent consumes kwargs
            self._per_agent_obs_size = kwargs.pop("per_agent_obs_size")
            self._skills_per_agent = kwargs.pop("skills_per_agent", 3)

            # Trajectory predictor config (optional)
            self._traj_pred_history = kwargs.pop("traj_pred_history", 0)
            self._traj_pred_horizons = kwargs.pop("traj_pred_horizons", [])
            self._traj_pred_hidden = kwargs.pop("traj_pred_hidden", 64)
            self._traj_pred_type = kwargs.pop("traj_pred_type", "gru")
            self._ball_history_offset = kwargs.pop("ball_history_offset", 0)

            self._use_traj_pred = (
                self._traj_pred_history > 0 and len(self._traj_pred_horizons) > 0
            )
            self._traj_pred_out_dim = len(self._traj_pred_horizons) * 3 if self._use_traj_pred else 0

            # Parent builds actor/critic MLPs sized for the FULL combined obs.
            # We only use the parent's critic; the actor is rebuilt below.
            super().__init__(params, **kwargs)

            # --- Trajectory predictor sub-module ---
            self.traj_predictor = None
            if self._use_traj_pred:
                self.traj_predictor = TrajectoryPredictor(
                    history_len=self._traj_pred_history,
                    num_horizons=len(self._traj_pred_horizons),
                    hidden_size=self._traj_pred_hidden,
                    pred_type=self._traj_pred_type,
                )

            # --- Logit initializer from config (same as HRLBuilder) ---
            if self.is_discrete and "logit_init" in self.space_config:
                logit_init_fn = self.init_factory.create(
                    **self.space_config["logit_init"]
                )
            else:
                logit_init_fn = None

            # --- Build factored actor MLP (per-agent obs [+ pred] → hidden) ---
            mlp_units = params["mlp"]["units"]
            mlp_act = params["mlp"].get("activation", "relu")
            act_cls = {"relu": nn.ReLU, "elu": nn.ELU, "tanh": nn.Tanh}.get(
                mlp_act, nn.ReLU
            )

            layers = []
            in_size = self._per_agent_obs_size + self._traj_pred_out_dim
            for idx, out_size in enumerate(mlp_units):
                layers.append(nn.Linear(in_size, out_size))
                layers.append(act_cls())
                in_size = out_size
            self.factored_actor_mlp = nn.Sequential(*layers)

            self.factored_logits = nn.Linear(in_size, self._skills_per_agent)
            if logit_init_fn is not None:
                logit_init_fn(self.factored_logits.weight)
            nn.init.zeros_(self.factored_logits.bias)

        # ---------- predictor helper ----------

        def _extract_and_predict(self, obs_agent):
            """Extract ball history from per-agent obs and run predictor."""
            T = self._traj_pred_history
            off = self._ball_history_offset
            bh = obs_agent[..., off : off + T * 6]
            return self.traj_predictor(bh)

        # ---------- forward ----------

        def forward(self, obs_dict):
            obs = obs_dict["obs"]
            seq_length = obs_dict.get("seq_length", 1)
            states = obs_dict.get("rnn_states", None)

            d = self._per_agent_obs_size
            obs_a = obs[..., :d]
            obs_b = obs[..., d : 2 * d]

            # ---- Trajectory predictor (if enabled) ----
            if self._use_traj_pred:
                pred_a = self._extract_and_predict(obs_a)
                pred_b = self._extract_and_predict(obs_b)
                self._last_pred_a = pred_a
                self._last_pred_b = pred_b
                actor_in_a = torch.cat([obs_a, pred_a], dim=-1)
                actor_in_b = torch.cat([obs_b, pred_b], dim=-1)
            else:
                actor_in_a = obs_a
                actor_in_b = obs_b

            # ---- Factored actor (shared weights) ----
            h_a = self.factored_actor_mlp(actor_in_a)
            h_b = self.factored_actor_mlp(actor_in_b)
            logits_a = self.factored_logits(h_a)  # [N, K]
            logits_b = self.factored_logits(h_b)  # [N, K]

            # Stash per-agent logits for entropy diagnostics
            self._last_logits_a = logits_a.detach()
            self._last_logits_b = logits_b.detach()

            # Outer sum → joint logits  [N, K*K]
            # P(joint) = softmax(logits_a[i] + logits_b[j])
            joint = logits_a.unsqueeze(2) + logits_b.unsqueeze(1)  # [N, K, K]
            joint_logits = joint.reshape(obs.size(0), -1)  # [N, K*K]

            # ---- Centralized critic (full obs) ----
            if self.separate:
                c_out = self.critic_cnn(obs)
                c_out = c_out.contiguous().view(c_out.size(0), -1)
                c_out = self.critic_mlp(c_out)
            else:
                c_out = torch.cat([h_a, h_b], dim=-1)
            value = self.value_act(self.value(c_out))

            return joint_logits, value, states

        # ---------- eval helpers (used by some rl_games code paths) ----------

        def eval_critic(self, obs):
            c_out = self.critic_cnn(obs)
            c_out = c_out.contiguous().view(c_out.size(0), -1)
            c_out = self.critic_mlp(c_out)
            value = self.value_act(self.value(c_out))
            return value

    # ---- builder entry-point ----

    def build(self, name, **kwargs):
        net = HRLCTDEBuilder.Network(self.params, **kwargs)
        return net
