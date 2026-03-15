# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Trajectory Predictor sub-module for HRL-CTDE.
#
# Takes a ball history buffer [N, T, 6] (pos + vel in agent heading frame)
# and predicts future ball positions at H horizons → [N, H*3].

import torch
import torch.nn as nn


class TrajectoryPredictor(nn.Module):
    """
    Predicts future ball positions from a history of ball states.

    Supports two architecture types:
      - 'gru': 1-layer GRU over time steps, then linear head
      - 'mlp': flattened history through a 2-layer MLP
    """

    def __init__(self, history_len, num_horizons, hidden_size=64,
                 pred_type="gru", input_dim=6):
        super().__init__()
        self.history_len = history_len
        self.num_horizons = num_horizons
        self.hidden_size = hidden_size
        self.pred_type = pred_type
        self.input_dim = input_dim
        self.output_dim = num_horizons * 3

        if pred_type == "gru":
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
            )
            self.head = nn.Linear(hidden_size, self.output_dim)
        elif pred_type == "mlp":
            flat_dim = history_len * input_dim
            self.mlp = nn.Sequential(
                nn.Linear(flat_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, self.output_dim),
            )
        else:
            raise ValueError(f"Unknown predictor type: {pred_type}")

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, ball_history):
        """
        Args:
            ball_history: [N, T, 6] or [N, T*6] flattened

        Returns:
            pred_positions: [N, H*3] predicted future ball positions
        """
        if ball_history.dim() == 2:
            ball_history = ball_history.reshape(
                ball_history.size(0), self.history_len, self.input_dim
            )

        if self.pred_type == "gru":
            _, h_n = self.rnn(ball_history)  # h_n: [1, N, hidden]
            out = self.head(h_n.squeeze(0))  # [N, H*3]
        else:
            flat = ball_history.reshape(ball_history.size(0), -1)
            out = self.mlp(flat)

        return out
