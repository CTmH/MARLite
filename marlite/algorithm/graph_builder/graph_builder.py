from typing import Tuple, List
import numpy as np
import torch
from torch import nn


class GraphBuilder(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, List[np.ndarray]]:
        raise NotImplementedError

    def reset(self) -> nn.Module:
        return self