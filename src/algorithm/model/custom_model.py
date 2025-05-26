import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from .permute import Permute
from .channel_selector import ChannelSelector

class CustomModel(nn.Module):
    def __init__(self, **kwargs):
        config = kwargs
        super(CustomModel, self).__init__()
        layers = []
        for layer_config in config['layers']:
            layer_type = layer_config.pop('type')
            if layer_type == 'Permute':
                layers.append(Permute(layer_config['dims']))
            elif layer_type == 'ChannelSelector':
                num_channels = layer_config['num_channels']
                layers.append(ChannelSelector(num_channels))
            else:
                layer_class = getattr(nn, layer_type)
                layers.append(layer_class(**layer_config))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)