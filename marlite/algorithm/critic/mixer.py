"""Mixer base class for Q-value mixing networks.

A mixer combines per-agent Q-values with global state information to
produce a centralised Q-tot estimate, as used in QMIX and related
value-based multi-agent algorithms.
"""

import torch
import torch.nn as nn
from typing import Dict

from marlite.algorithm.critic.critic import Critic


class Mixer(Critic):
    """Base class for Q-value mixer networks.

    Subclasses implement ``forward(q_value_from_agents, states,
    alive_mask, padding_mask)`` and return a dict containing at least
    ``"q_tot"``.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        q_value_from_agents: torch.Tensor,
        states: torch.Tensor,
        alive_mask: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
