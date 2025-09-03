import torch
import torch.nn as nn

class Permute(nn.Module):
    def __init__(self, dims):
        super(Permute, self).__init__()
        self.order = dims

    def forward(self, x: torch.Tensor):
        # Check if in DataParallel
        if x.dim() == len(self.order) + 1:
            adjusted_order = (0,) + tuple(i + 1 for i in self.order)
            return x.permute(*adjusted_order)
        else:
            return x.permute(*self.order)