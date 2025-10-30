from torch import nn
from typing import Tuple, List
from numpy import ndarray
class GraphBuilder(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, states) -> Tuple[ndarray, List[ndarray]]:
        raise NotImplementedError

    def reset(self) -> nn.Module:
        return self