"""
Graph worker group implementation.

This module provides GraphWorkerGroup that manages Graph workers for multi-GPU training.
"""

from typing import Any, Dict
from marlite.trainer.trainer_worker_group.base_worker_group import BaseWorkerGroup
from marlite.trainer.trainer_worker.graph_worker import GraphWorker


class GraphWorkerGroup(BaseWorkerGroup):
    """
    Worker group for GraphQMIX algorithm multi-GPU training.

    This group manages GraphWorker instances, each holding copies of:
    - eval_agent_group (with graph processing)
    - target_agent_group
    - eval_critic
    - target_critic
    """

    def __init__(
        self,
        device_ids: list,
        agent_group_config,
        critic_config,
        critic_optimizer_config,
        agent_optimizer_config,
        gamma: float = 0.9,
        max_grad_norm: float = 5.0,
        init_method: str = None,
    ):
        """
        Initialize Graph worker group.

        Args:
            device_ids: List of CUDA device IDs
            agent_group_config: Configuration for agent group (GraphAgentGroup)
            critic_config: Configuration for critic
            critic_optimizer_config: Configuration for critic optimizer
            agent_optimizer_config: Configuration for agent group optimizer
            gamma: Discount factor
            init_method: URL for distributed initialization
        """
        self.agent_group_config = agent_group_config
        self.critic_config = critic_config
        self.critic_optimizer_config = critic_optimizer_config
        self.agent_optimizer_config = agent_optimizer_config
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm

        super().__init__(
            device_ids=device_ids,
            world_size=len(device_ids),
            init_method=init_method,
        )

    def _get_worker_class(self):
        """Return GraphWorker class."""
        return GraphWorker

    def _create_worker_kwargs(self) -> Dict[str, Any]:
        """Create kwargs for GraphWorker initialization."""
        kwargs = super()._create_worker_kwargs()
        kwargs["gamma"] = self.gamma
        kwargs["agent_group_config"] = self.agent_group_config
        kwargs["critic_config"] = self.critic_config
        kwargs["critic_optimizer_config"] = self.critic_optimizer_config
        kwargs["agent_optimizer_config"] = self.agent_optimizer_config
        return kwargs
