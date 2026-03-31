# src/marlite/algorithm/agents/agent_group.py
from typing import Dict, List, Any, Optional
import numpy as np
import torch
import torch.nn as nn


class AgentGroup(nn.Module):
    """Base class for managing a group of agents in a multi-agent reinforcement learning system.

    This class inherits from nn.Module, enabling direct use of nn.Module's parameter
    management, device handling, and state_dict/load_state_dict functionality.
    Optimizer management is handled by the Trainer class instead of within AgentGroup.
    """

    def __init__(self):
        super().__init__()

    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def forward(
        self,
        observations: torch.Tensor,
        traj_padding_mask: torch.Tensor,
        alive_mask: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Forward pass for agent group.

        Args:
            observations: Dictionary of agent observations
            traj_padding_mask: Padding mask for trajectory processing
            alive_mask: Mask indicating which agents are alive

        Returns:
            Dictionary containing forward pass results
        """
        raise NotImplementedError

    def act(
        self,
        observations: Dict[str, np.ndarray],
        state: np.ndarray,
        avail_actions: Dict[str, Any],
        traj_padding_mask: np.ndarray,
        alive_agents: List[str],
        epsilon: float,
    ) -> Dict[str, Any]:
        """
        Generate actions for the agent group.

        Args:
            observations: Dictionary of agent observations
            state: Global state information for generating communication graph.
            avail_actions: Available actions for each agent
            traj_padding_mask: Padding mask for trajectory processing
            alive_agents: List indicating which agents are alive
            epsilon: Exploration rate

        Returns:
            Dictionary containing actions and other relevant information
        """
        raise NotImplementedError

    def reset(self) -> "AgentGroup":
        """
        Reset the agent group state.

        Returns:
            Self reference for method chaining
        """
        raise NotImplementedError
