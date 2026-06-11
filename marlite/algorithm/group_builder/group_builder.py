import numpy as np
import torch
from torch import nn
from typing import Dict


class GroupBuilder(nn.Module):
    _TORCH_DTYPE_MAP: Dict[str, torch.dtype] = {
        'int8': torch.int8,
        'int16': torch.int16,
        'int32': torch.int32,
        'int64': torch.int64,
        'float32': torch.float32,
        'float64': torch.float64,
    }

    def __init__(self, dtype: str = 'int16'):
        super().__init__()
        self.dtype = dtype
        self._torch_dtype = self._TORCH_DTYPE_MAP.get(dtype, torch.int16)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Process states and return zone indices for each agent.

        Args:
            states: tensor of shape (batch_size, ...) representing environment states

        Returns:
            group_indices: tensor of shape (batch_size, n_agents)
                          each element is the zone/group ID for that agent
        """
        raise NotImplementedError

    def reset(self) -> nn.Module:
        return self
