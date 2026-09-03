"""
MAPPO worker group implementation.

This module provides MAPPOWorkerGroup that manages MAPPOWorker instances
for multi-GPU training with the MAPPO algorithm.
"""

from typing import Any, Dict
from marlite.trainer.trainer_worker_group.base_worker_group import OnPolicyWorkerGroup
from marlite.trainer.trainer_worker.mappo_worker import MAPPOWorker


class MAPPOWorkerGroup(OnPolicyWorkerGroup):
    """
    Worker group for MAPPO algorithm multi-GPU training.

    This group manages MAPPOWorker instances, each holding copies of:
    - eval_agent_group
    - eval_critic

    Workers compute PPO losses on data slices and synchronize gradients
    via all_reduce. Unlike QMIX, MAPPO does not use target networks.
    """

    def __init__(
        self,
        device_ids: list,
        agent_group_config,
        critic_config,
        critic_optimizer_config,
        agent_optimizer_config,
        gamma: float = 0.99,
        clip_epsilon: float = 0.2,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 5.0,
        init_method: str = None,
    ):
        """
        Initialize MAPPO worker group.

        Args:
            device_ids: List of CUDA device IDs.
            agent_group_config: Configuration for agent group.
            critic_config: Configuration for critic.
            critic_optimizer_config: Configuration for critic optimizer.
            agent_optimizer_config: Optimizer config for agent group.
            gamma: Discount factor.
            clip_epsilon: PPO clip range.
            gae_lambda: Reserved for future GAE support; GAE is not
                implemented and this parameter currently has no effect.
            entropy_coef: Entropy bonus coefficient.
            vf_coef: Value function loss coefficient.
            max_grad_norm: Maximum gradient norm for clipping.
            init_method: URL for distributed initialization.
        """
        self.agent_group_config = agent_group_config
        self.critic_config = critic_config
        self.critic_optimizer_config = critic_optimizer_config
        self.agent_optimizer_config = agent_optimizer_config
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        # Reserved for future GAE support; current MAPPO uses one-step TD
        # advantages and does not read this value during loss computation.
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        super().__init__(
            device_ids=device_ids,
            world_size=len(device_ids),
            init_method=init_method,
        )

    def _get_worker_class(self):
        """Return MAPPOWorker class."""
        return MAPPOWorker

    def _create_worker_kwargs(self) -> Dict[str, Any]:
        """Create kwargs for MAPPOWorker initialization."""
        kwargs = super()._create_worker_kwargs()
        kwargs["gamma"] = self.gamma
        kwargs["clip_epsilon"] = self.clip_epsilon
        kwargs["gae_lambda"] = self.gae_lambda
        kwargs["entropy_coef"] = self.entropy_coef
        kwargs["vf_coef"] = self.vf_coef
        kwargs["max_grad_norm"] = self.max_grad_norm
        kwargs["agent_group_config"] = self.agent_group_config
        kwargs["critic_config"] = self.critic_config
        kwargs["critic_optimizer_config"] = self.critic_optimizer_config
        kwargs["agent_optimizer_config"] = self.agent_optimizer_config
        return kwargs
