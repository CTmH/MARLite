# src/marlite/algorithm/agents/agent_group.py
from typing import Dict, List, Any, Optional
import numpy as np
import torch


class AgentGroup(object):
    """Base class for managing a group of agents in a multi-agent reinforcement learning system."""

    def forward(self, observations: Dict[str, np.ndarray], traj_padding_mask: torch.Tensor,
                alive_mask: torch.Tensor) -> Dict[str, Any]:
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

    def act(self, observations: Dict[str, np.ndarray], state: np.ndarray, avail_actions: Dict[str, Any],
            traj_padding_mask: np.ndarray, alive_agents: List[str], epsilon: float) -> Dict[str, Any]:
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

    def set_agent_group_params(self, model_params: Dict[str, dict],
                              feature_extractor_params: Dict[str, dict]) -> None:
        """
        Set parameters for the agent group.

        Args:
            model_params: Parameters for the main model
            feature_extractor_params: Parameters for the feature extractor
        """
        raise NotImplementedError

    def get_agent_group_params(self) -> Dict[str, dict]:
        """
        Get current parameters of the agent group.

        Returns:
            Dictionary containing model and feature extractor parameters
        """
        raise NotImplementedError

    def zero_grad(self) -> 'AgentGroup':
        """
        Zero gradients for all parameters in the agent group.

        Returns:
            Self reference for method chaining
        """
        raise NotImplementedError

    def step(self) -> 'AgentGroup':
        """
        Perform one training step.

        Returns:
            Self reference for method chaining
        """
        raise NotImplementedError

    def lr_scheduler_step(self, reward) -> 'AgentGroup':
        """
        Perform a learning rate scheduler step based on epoch and reward.

        This method is typically called at the end of each training epoch to adjust
        the learning rate according to a predefined schedule or performance metric.

        Args:
            epoch: Current training epoch number
            reward: Reward signal used for adjusting the learning rate, often representing
                   the performance of the agent group in the current epoch

        """
        raise NotImplementedError

    def to_device(self, device) -> 'AgentGroup':
        """
        Move all tensors to specified device.

        Args:
            device: Target device (e.g., 'cuda', 'cpu')

        Returns:
            Self reference for method chaining
        """
        raise NotImplementedError

    def eval(self) -> 'AgentGroup':
        """
        Set the agent group to evaluation mode.

        Returns:
            Self reference for method chaining
        """
        raise NotImplementedError

    def train(self) -> 'AgentGroup':
        """
        Set the agent group to training mode.

        Returns:
            Self reference for method chaining
        """
        raise NotImplementedError

    def share_memory(self) -> 'AgentGroup':
        """
        Share memory between processes.

        Returns:
            Self reference for method chaining
        """
        raise NotImplementedError

    def wrap_data_parallel(self) -> 'AgentGroup':
        """
        Wrap the agent group in data parallelism.

        Returns:
            Self reference for method chaining
        """
        raise NotImplementedError

    def unwrap_data_parallel(self) -> 'AgentGroup':
        """
        Unwrap the agent group from data parallelism.

        Returns:
            Self reference for method chaining
        """
        raise NotImplementedError

    def save_params(self, path: str) -> 'AgentGroup':
        """
        Save agent group parameters to disk.

        Args:
            path: Path to save parameters

        Returns:
            Self reference for method chaining
        """
        raise NotImplementedError

    def load_params(self, path: str) -> 'AgentGroup':
        """
        Load agent group parameters from disk.

        Args:
            path: Path to load parameters from

        Returns:
            Self reference for method chaining
        """
        raise NotImplementedError

    def compile_models(self) -> 'AgentGroup':
        """
        Compile models for improved performance.

        Returns:
            Self reference for method chaining
        """
        raise NotImplementedError

    def reset(self) -> 'AgentGroup':
        """
        Reset the agent group state.

        Returns:
            Self reference for method chaining
        """
        raise NotImplementedError