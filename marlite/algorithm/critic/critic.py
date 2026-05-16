"""Base class for critic networks in MARLite.

A critic estimates the value of a state (or state-action pair) and serves
as a building block for both value-based and actor-critic algorithms.
"""

import torch
import torch.nn as nn
from typing import Dict


class Critic(nn.Module):
    """Abstract base class for all critic networks.

    Subclasses must implement ``forward()``, which processes state
    information and returns a value estimate.  The exact forward signature
    varies depending on whether the critic is a state-value function (V),
    a Q-value mixer, or another variant.
    """

    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
