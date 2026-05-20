from typing import Any, Dict
from marlite.trainer.trainer_worker_group.base_worker_group import (
    BaseWorkerGroup,
    _slice_batch,
)


class VAEGroupConsensusWorkerGroup(BaseWorkerGroup):
    def __init__(
        self,
        device_ids: list,
        agent_group_config,
        critic_config,
        critic_optimizer_config,
        agent_optimizer_config,
        gamma: float = 0.9,
        max_grad_norm: float = 5.0,
        ssl_model_config=None,
        ssl_optimizer_config=None,
        reconstruction_loss=None,
        data_constructor=None,
        kl_divergence_weight: float = 0.005,
        self_supervised_learning_loss_weight: float = 1.0,
        loss_combination_method: str = "weighted_sum",
        pit_loss_alpha: float = 0.9,
        warmup_epochs: int = 0,
        recon_mode: str = "per_agent",
        kl_on_group: bool = False,
        kl_on_agent: bool = True,
        init_method: str = None,
    ):
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.agent_group_config = agent_group_config
        self.critic_config = critic_config
        self.critic_optimizer_config = critic_optimizer_config
        self.agent_optimizer_config = agent_optimizer_config
        self.ssl_model_config = ssl_model_config
        self.ssl_optimizer_config = ssl_optimizer_config
        self.reconstruction_loss = reconstruction_loss
        self.data_constructor = data_constructor
        self.kl_divergence_weight = kl_divergence_weight
        self.self_supervised_learning_loss_weight = self_supervised_learning_loss_weight
        self.loss_combination_method = loss_combination_method
        self.pit_loss_alpha = pit_loss_alpha
        self.warmup_epochs = warmup_epochs
        self.recon_mode = recon_mode
        self.kl_on_group = kl_on_group
        self.kl_on_agent = kl_on_agent

        super().__init__(
            device_ids=device_ids,
            world_size=len(device_ids),
            init_method=init_method,
        )

    def _get_worker_class(self):
        from marlite.trainer.trainer_worker.vae_group_consensus_worker import (
            VAEGroupConsensusWorker,
        )

        return VAEGroupConsensusWorker

    def _create_worker_kwargs(self) -> Dict[str, Any]:
        kwargs = {}
        kwargs["agent_group_config"] = self.agent_group_config
        kwargs["critic_config"] = self.critic_config
        kwargs["critic_optimizer_config"] = self.critic_optimizer_config
        kwargs["agent_optimizer_config"] = self.agent_optimizer_config
        kwargs["gamma"] = self.gamma
        kwargs["max_grad_norm"] = self.max_grad_norm
        kwargs["ssl_model_config"] = self.ssl_model_config
        kwargs["ssl_optimizer_config"] = self.ssl_optimizer_config
        kwargs["reconstruction_loss"] = self.reconstruction_loss
        kwargs["data_constructor"] = self.data_constructor
        kwargs["kl_divergence_weight"] = self.kl_divergence_weight
        kwargs["self_supervised_learning_loss_weight"] = (
            self.self_supervised_learning_loss_weight
        )
        kwargs["loss_combination_method"] = self.loss_combination_method
        kwargs["pit_loss_alpha"] = self.pit_loss_alpha
        kwargs["warmup_epochs"] = self.warmup_epochs
        kwargs["recon_mode"] = self.recon_mode
        kwargs["kl_on_group"] = self.kl_on_group
        kwargs["kl_on_agent"] = self.kl_on_agent
        return kwargs

    def train_step(self, batch: Dict[str, Any]) -> tuple:
        batch_slices = _slice_batch(batch, self.world_size)
        for i in range(self.world_size):
            self.cmd_queues[i].put("TRAIN_STEP")
            self.data_queues[i].put(batch_slices[i])

        combined_losses = []
        critic_losses = []
        vae_losses = []
        for _ in range(self.world_size):
            combined, critic, vae = self.loss_queue.get()
            combined_losses.append(combined)
            critic_losses.append(critic)
            vae_losses.append(vae)

        return (
            sum(combined_losses) / len(combined_losses),
            sum(critic_losses) / len(critic_losses),
            sum(vae_losses) / len(vae_losses),
        )
