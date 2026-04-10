from typing import List

import torch
import torch.nn as nn


class ChannelSelector(nn.Module):
    def __init__(self, num_channels: List[int], dim: int = -1):
        super(ChannelSelector, self).__init__()
        self.num_channels = num_channels
        self.dim = dim

    def forward(self, x):
        indices = torch.tensor(self.num_channels, dtype=torch.int64, device=x.device)
        return x.index_select(self.dim, indices)
