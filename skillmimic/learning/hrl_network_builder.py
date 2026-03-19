# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from rl_games.algos_torch import network_builder

import torch
import torch.nn as nn


class HRLBuilder(network_builder.A2CBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            # Pop trajectory predictor config before parent consumes kwargs
            self._traj_pred_history = kwargs.pop("traj_pred_history", 0)
            self._traj_pred_horizons = kwargs.pop("traj_pred_horizons", [])
            self._traj_pred_hidden = kwargs.pop("traj_pred_hidden", 64)
            self._traj_pred_type = kwargs.pop("traj_pred_type", "gru")
            self._ball_history_offset_a = kwargs.pop("ball_history_offset_a", 0)
            self._ball_history_offset_b = kwargs.pop("ball_history_offset_b", 0)

            self._use_traj_pred = (
                self._traj_pred_history > 0 and len(self._traj_pred_horizons) > 0
            )
            self._traj_pred_out_dim = (
                len(self._traj_pred_horizons) * 3 if self._use_traj_pred else 0
            )

            super().__init__(params, **kwargs)

            if self.is_continuous:
                if (not self.space_config['learn_sigma']):
                    actions_num = kwargs.get('actions_num')
                    sigma_init = self.init_factory.create(**self.space_config['sigma_init'])
                    self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=False, dtype=torch.float32), requires_grad=False)
                    sigma_init(self.sigma)
            elif self.is_discrete:
                logits_init = self.init_factory.create(**self.space_config['logit_init'])
                logits_init(self.logits.weight)
                torch.nn.init.zeros_(self.logits.bias)

            # --- Trajectory predictor sub-module ---
            self.traj_predictor = None
            if self._use_traj_pred:
                from learning.hrl_ctde_trajectory_predictor import TrajectoryPredictor

                self.traj_predictor = TrajectoryPredictor(
                    history_len=self._traj_pred_history,
                    num_horizons=len(self._traj_pred_horizons),
                    hidden_size=self._traj_pred_hidden,
                    pred_type=self._traj_pred_type,
                )

                # Build augmented actor MLP: obs + pred_a + pred_b → hidden → logits
                mlp_units = params["mlp"]["units"]
                mlp_act = params["mlp"].get("activation", "relu")
                act_cls = {"relu": nn.ReLU, "elu": nn.ELU, "tanh": nn.Tanh}.get(
                    mlp_act, nn.ReLU
                )

                obs_size = self.actor_mlp[0].in_features
                augmented_in = obs_size + 2 * self._traj_pred_out_dim

                layers = []
                in_size = augmented_in
                for out_size in mlp_units:
                    layers.append(nn.Linear(in_size, out_size))
                    layers.append(act_cls())
                    in_size = out_size
                self.traj_actor_mlp = nn.Sequential(*layers)

                self.traj_logits = nn.Linear(in_size, self.logits.out_features)
                if self.is_discrete and "logit_init" in self.space_config:
                    traj_logit_init = self.init_factory.create(
                        **self.space_config["logit_init"]
                    )
                    traj_logit_init(self.traj_logits.weight)
                nn.init.zeros_(self.traj_logits.bias)

                print(
                    f"[HRLBuilder] TrajectoryPredictor enabled: "
                    f"history={self._traj_pred_history} horizons={self._traj_pred_horizons} "
                    f"pred_out={self._traj_pred_out_dim} augmented_actor_in={augmented_in}"
                )

            return

        def _extract_and_predict(self, obs, offset):
            """Extract ball history at given offset and run through predictor."""
            T = self._traj_pred_history
            bh = obs[..., offset : offset + T * 6]
            return self.traj_predictor(bh)

        def forward(self, obs_dict):
            if self._use_traj_pred and self.is_discrete:
                obs = obs_dict["obs"]
                states = obs_dict.get("rnn_states", None)

                pred_a = self._extract_and_predict(obs, self._ball_history_offset_a)
                pred_b = self._extract_and_predict(obs, self._ball_history_offset_b)
                self._last_pred_a = pred_a
                self._last_pred_b = pred_b

                actor_input = torch.cat([obs, pred_a, pred_b], dim=-1)
                a_out = self.traj_actor_mlp(actor_input)
                logits = self.traj_logits(a_out)

                c_out = self.critic_cnn(obs)
                c_out = c_out.contiguous().view(c_out.size(0), -1)
                c_out = self.critic_mlp(c_out)
                value = self.value_act(self.value(c_out))

                return logits, value, states

            if self.is_continuous:
                mu, sigma, value, states = super().forward(obs_dict)
                norm_mu = torch.tanh(mu)
                return norm_mu, sigma, value, states
            elif self.is_discrete:
                logits, value, states = super().forward(obs_dict)
                return logits, value, states

        def eval_critic(self, obs):
            c_out = self.critic_cnn(obs)
            c_out = c_out.contiguous().view(c_out.size(0), -1)
            c_out = self.critic_mlp(c_out)              
            value = self.value_act(self.value(c_out))
            return value

    def build(self, name, **kwargs):
        net = HRLBuilder.Network(self.params, **kwargs)
        return net