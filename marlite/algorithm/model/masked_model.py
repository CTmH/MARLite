import torch
import torch.nn as nn

class MaskedModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super(MaskedModel, self).__init__()

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        raise NotImplementedError
