import torch.nn as nn

class ChannelSelector(nn.Module):
    def __init__(self, num_channels: list):
        super(ChannelSelector, self).__init__()
        self.num_channels = num_channels
    def forward(self, x):
        return x[..., self.num_channels]