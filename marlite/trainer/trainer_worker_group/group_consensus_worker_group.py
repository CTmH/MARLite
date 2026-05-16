from typing import Any, Dict
from marlite.trainer.trainer_worker_group.base_worker_group import BaseWorkerGroup
from marlite.trainer.trainer_worker.group_consensus_worker import GroupConsensusWorker


class GroupConsensusWorkerGroup(BaseWorkerGroup):
    def __init__(
        self,
        device_ids: list,
        agent_group_config,
        critic_config,
        critic_optimizer_config,
        agent_optimizer_config,
        gamma: float = 0.9,
        max_grad_norm: float = 5.0,
        kl_divergence_weight: float = 0.005,
        warmup_epochs: int = 0,
        init_method: str = None,
    ):
        self.agent_group_config = agent_group_config
        self.critic_config = critic_config
        self.critic_optimizer_config = critic_optimizer_config
        self.agent_optimizer_config = agent_optimizer_config
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.kl_divergence_weight = kl_divergence_weight
        self.warmup_epochs = warmup_epochs

        super().__init__(
            device_ids=device_ids,
            world_size=len(device_ids),
            init_method=init_method,
        )

    def _get_worker_class(self):
        return GroupConsensusWorker

    def _create_worker_kwargs(self) -> Dict[str, Any]:
        kwargs = super()._create_worker_kwargs()
        kwargs["gamma"] = self.gamma
        kwargs["max_grad_norm"] = self.max_grad_norm
        kwargs["agent_group_config"] = self.agent_group_config
        kwargs["critic_config"] = self.critic_config
        kwargs["critic_optimizer_config"] = self.critic_optimizer_config
        kwargs["agent_optimizer_config"] = self.agent_optimizer_config
        kwargs["kl_divergence_weight"] = self.kl_divergence_weight
        kwargs["warmup_epochs"] = self.warmup_epochs
        return kwargs
