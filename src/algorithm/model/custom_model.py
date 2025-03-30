import torch.nn as nn
import torch.nn.functional as F
from .permute import Permute
from .channel_selector import ChannelSelector

class CustomModel(nn.Module):
    def __init__(self, **kwargs):
        config = kwargs
        super(CustomModel, self).__init__()
        layers = []
        for layer_config in config['layers']:
            layer_type = layer_config['type']
            if layer_type == 'Flatten':
                layers.append(nn.Flatten())
            elif layer_type == 'Permute':
                layers.append(Permute(layer_config['dims']))
            elif layer_type == 'ChannelSelector':
                num_channels = layer_config['num_channels']
                layers.append(ChannelSelector(num_channels))
            elif layer_type == 'Linear':
                in_features = layer_config['in_features']
                out_features = layer_config['out_features']
                bias = layer_config.get('bias', True)
                layers.append(nn.Linear(in_features, out_features, bias))
            elif layer_type == 'Conv2d':
                in_channels = layer_config['in_channels']
                out_channels = layer_config['out_channels']
                kernel_size = layer_config['kernel_size']
                stride = layer_config.get('stride', 1)
                padding = layer_config.get('padding', 0)
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding))
            elif layer_type == 'MaxPool2d':
                kernel_size = layer_config['kernel_size']
                stride = layer_config.get('stride', kernel_size)
                padding = layer_config.get('padding', 0)
                layers.append(nn.MaxPool2d(kernel_size, stride=stride, padding=padding))
            elif layer_type == 'AvgPool2d':
                kernel_size = layer_config['kernel_size']
                stride = layer_config.get('stride', kernel_size)
                padding = layer_config.get('padding', 0)
                layers.append(nn.AvgPool2d(kernel_size, stride=stride, padding=padding))
            elif layer_type == 'AdaptiveAvgPool2d':
                output_size = layer_config['output_size']
                layers.append(nn.AdaptiveAvgPool2d(output_size=output_size))
            elif layer_type == 'ReLU':
                layers.append(nn.ReLU())
            elif layer_type == 'Sigmoid':
                layers.append(nn.Sigmoid())
            elif layer_type == 'Tanh':
                layers.append(nn.Tanh())
            elif layer_type == 'Softmax':
                dim = layer_config['dim']
                layers.append(nn.Softmax(dim=dim))
            elif layer_type == 'LogSoftmax':
                dim = layer_config['dim']
                layers.append(nn.LogSoftmax(dim=dim))
            elif layer_type == 'Dropout':
                p = layer_config.get('p', 0.5)
                layers.append(nn.Dropout(p=p))
            elif layer_type == 'BatchNorm1d':
                num_features = layer_config['num_features']
                layers.append(nn.BatchNorm1d(num_features=num_features))
            elif layer_type == 'BatchNorm2d':
                num_features = layer_config['num_features']
                layers.append(nn.BatchNorm2d(num_features=num_features))
            elif layer_type == 'LayerNorm':
                normalized_shape = layer_config['normalized_shape']
                layers.append(nn.LayerNorm(normalized_shape=normalized_shape))
            elif layer_type == 'LeakyReLU':
                negative_slope = layer_config.get('negative_slope', 0.1)
                layers.append(nn.LeakyReLU(negative_slope=negative_slope))
            elif layer_type == 'ReLU':
                layers.append(nn.ReLU())
            elif layer_type == 'Softmax':
                layers.append(nn.Softmax(dim=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)