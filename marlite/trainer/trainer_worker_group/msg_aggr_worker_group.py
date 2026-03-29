"""
Message aggregation worker group implementation.

This module provides MsgAggrWorkerGroup that manages message aggregation workers
for multi-GPU training.
"""

from typing import Any, Dict
from marlite.trainer.trainer_worker_group.base_worker_group import BaseWorkerGroup
from marlite.trainer.trainer_worker.msg_aggr_worker import (
    MsgAggrWorker,
    ProbMsgAggrWorker,
)


class MsgAggrWorkerGroup(BaseWorkerGroup):
    """
    Worker group for MsgAggrQMIX algorithm multi-GPU training.

    This group manages MsgAggrWorker instances, each holding copies of:
    - eval_agent_group (with message aggregation)
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
        agent_group_optimizer_config,
        gamma: float = 0.9,
        warmup_epochs: int = 0,
        msg_aggr_weight: float = 1.0,
        is_probabilistic: bool = False,
        init_method: str = "tcp://localhost:29500",
    ):
        """
        Initialize MsgAggr worker group.

        Args:
            device_ids: List of CUDA device IDs
            agent_group_config: Configuration for agent group
            critic_config: Configuration for critic
            critic_optimizer_config: Configuration for critic optimizer
            agent_group_optimizer_config: Configuration for agent group optimizer
            gamma: Discount factor
            warmup_epochs: Number of warmup epochs before message aggregation loss
            msg_aggr_weight: Weight for message aggregation loss
            is_probabilistic: Whether to use probabilistic message aggregation
            init_method: URL for distributed initialization
        """
        self.agent_group_config = agent_group_config
        self.critic_config = critic_config
        self.critic_optimizer_config = critic_optimizer_config
        self.agent_group_optimizer_config = agent_group_optimizer_config
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        self.msg_aggr_weight = msg_aggr_weight
        self.is_probabilistic = is_probabilistic

        super().__init__(
            device_ids=device_ids,
            world_size=len(device_ids),
            init_method=init_method,
        )

    def _get_worker_class(self):
        """Return MsgAggrWorker or ProbMsgAggrWorker class based on configuration."""
        return ProbMsgAggrWorker if self.is_probabilistic else MsgAggrWorker

    def _create_worker_kwargs(self) -> Dict[str, Any]:
        """Create kwargs for MsgAggrWorker initialization."""
        kwargs = super()._create_worker_kwargs()
        kwargs["gamma"] = self.gamma
        kwargs["warmup_epochs"] = self.warmup_epochs
        kwargs["msg_aggr_weight"] = self.msg_aggr_weight
        kwargs["agent_group_config"] = self.agent_group_config
        kwargs["critic_config"] = self.critic_config
        kwargs["critic_optimizer_config"] = self.critic_optimizer_config
        kwargs["agent_group_optimizer_config"] = self.agent_group_optimizer_config
        return kwargs
