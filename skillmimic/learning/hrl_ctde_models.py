# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Model wrapper for HRL-CTDE factorized discrete policy.

import torch.nn as nn
from rl_games.algos_torch.models import ModelA2C


class ModelHRLCTDE(ModelA2C):
    def __init__(self, network):
        super().__init__(network)

    def build(self, config):
        net = self.network_builder.build("hrl_ctde", **config)
        for name, _ in net.named_parameters():
            print(name)
        return ModelHRLCTDE.Network(net)

    class Network(ModelA2C.Network):
        def __init__(self, a2c_network):
            super().__init__(a2c_network)
