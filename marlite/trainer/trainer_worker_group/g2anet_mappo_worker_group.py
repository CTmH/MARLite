"""
G2ANet MAPPO worker group for multi-GPU training.

Factory that spawns :class:`G2ANetMAPPOWorker` instances, each holding
copies of the eval agent group and critic models for graph-based PPO.
"""

from typing import Any, Dict
from marlite.trainer.trainer_worker_group.base_worker_group import OnPolicyWorkerGroup, _slice_batch


class G2ANetMAPPOWorkerGroup(OnPolicyWorkerGroup):
    """Worker group for G2ANet MAPPO multi-GPU training."""

    def __init__(
        self,
        device_ids: list,
        agent_group_config,
        critic_config,
        critic_optimizer_config,
        agent_optimizer_config,
        gamma: float = 0.99,
        clip_epsilon: float = 0.2,
        # Reserved for future GAE support; GAE is not implemented.
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 5.0,
        init_method: str = None,
    ):
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
        from marlite.trainer.trainer_worker.g2anet_mappo_worker import (
            G2ANetMAPPOWorker,
        )
        return G2ANetMAPPOWorker

    def _create_worker_kwargs(self) -> Dict[str, Any]:
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
