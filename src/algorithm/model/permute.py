import torch
import torch.nn as nn

class Permute(nn.Module):
    def __init__(self, order):
        super(Permute, self).__init__()
        self.order = order

    def forward(self, x: torch.Tensor):
        # x is expected to be a 4D tensor with shape (batch_size, channels, height, width)
        return x.permute(*self.order)