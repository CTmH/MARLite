import torch.nn as nn
from copy import deepcopy
from marlite.algorithm.model.permute import Permute
from marlite.algorithm.model.channel_selector import ChannelSelector
from marlite.algorithm.model.self_attention import SelfAttention, SelfAttentionFixedPE, SelfAttentionLearnablePE

REGISTERED_MODULES = {
    "Permute": Permute,
    "ChannelSelector": ChannelSelector,
    "SelfAttention": SelfAttention,
    "SelfAttentionFixedPE": SelfAttentionFixedPE,
    "SelfAttentionLearnablePE": SelfAttentionLearnablePE
}

class CustomModel(nn.Module):
    def __init__(self, **kwargs):
        config = kwargs
        super(CustomModel, self).__init__()
        layers = []
        for conf in config['layers']:
            layer_config = deepcopy(conf)
            layer_type = layer_config.pop('type')
            if layer_type in REGISTERED_MODULES:
                layer_class = REGISTERED_MODULES[layer_type]
                layers.append(layer_class(**layer_config))
            else:
                layer_class = getattr(nn, layer_type)
                layers.append(layer_class(**layer_config))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)