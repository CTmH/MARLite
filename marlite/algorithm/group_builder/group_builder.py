import numpy as np
from torch import nn
from typing import List


class GroupBuilder(nn.Module):
    def __init__(self, dtype=np.int16):
        super().__init__()
        self.dtype = dtype

    def forward(self, states: np.ndarray) -> np.ndarray:
        """
        Process states and return zone indices for each agent.

        Args:
            states: numpy array of shape (batch_size, ...) representing environment states

        Returns:
            group_indices: numpy array of shape (batch_size, n_agents) with dtype
                          determined by self.dtype (default int16),
                          each element is the zone/group ID for that agent
        """
        raise NotImplementedError

    def reset(self) -> nn.Module:
        return self
