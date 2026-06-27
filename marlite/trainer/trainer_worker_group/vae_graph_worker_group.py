"""
VAEGraphQMIXWorkerGroup — worker group for VAE-based joint RL+SSL multi-GPU training.

This module provides VAEGraphQMIXWorkerGroup that manages VAEGraphQMIXWorker
instances for joint RL+SSL training with VAEGraphQMIXTrainer.
"""

from typing import Any, Dict
from marlite.trainer.trainer_worker_group.base_worker_group import (
    OffPolicyWorkerGroup,
    _slice_batch,
)


class VAEGraphQMIXWorkerGroup(OffPolicyWorkerGroup):
    """
    Worker group for VAE-based joint RL+SSL multi-GPU training.

    This group manages VAEGraphQMIXWorker instances that support joint training where:
    - eval_agent_group, target_agent_group for RL
    - eval_critic, target_critic for RL
    - ssl_model for SSL (VAE decoder)

    Workers execute train_step() that computes:
    combined_loss = critic_loss + self_supervised_learning_loss_weight * vae_loss

    Returns (combined_loss, critic_loss, vae_loss) aggregated across workers.
    """

    def __init__(
        self,
        device_ids: list,
        agent_group_config,
        critic_config,
        critic_optimizer_config,
        agent_optimizer_config,
        ssl_model_config: ModelConfig,
        ssl_optimizer_config: OptimizerConfig,
        reconstruction_loss,
        gamma: float = 0.9,
        max_grad_norm: float = 5.0,
        kl_divergence_weight: float = 1.0,
        self_supervised_learning_loss_weight: float = 1.0,
        loss_combination_method: str = "weighted_sum",
        pit_loss_alpha: float = 0.9,
        data_constructor=None,
        warmup_epochs: int = 0,
        init_method: str = None,
    ):
        """
        Initialize VAE Graph worker group.

        Args:
            device_ids: List of CUDA device IDs
            agent_group_config: Configuration for agent group
            critic_config: Configuration for critic
            critic_optimizer_config: Configuration for critic optimizer
            agent_optimizer_config: Configuration for agent group optimizer
            gamma: Discount factor
            ssl_model_config: Configuration for SSL model (VAE decoder)
            ssl_optimizer_config: Configuration for SSL optimizer
            reconstruction_loss: Loss function for reconstruction
            kl_divergence_weight: Weight for KL divergence loss
            self_supervised_learning_loss_weight: Weight for VAE loss in combined loss
            loss_combination_method: Method to combine RL and SSL losses
                - "weighted_sum": combined_loss = critic_loss + weight * vae_loss
                - "pit_loss": use PITLoss to combine critic_loss and vae_loss
            pit_loss_alpha: Alpha parameter for PITLoss (exponential decay rate)
            data_constructor: Data constructor for SSL preprocessing
            warmup_epochs: Number of epochs to train with RL only before enabling SSL
            init_method: URL for distributed initialization
        """
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.agent_group_config = agent_group_config
        self.critic_config = critic_config
        self.critic_optimizer_config = critic_optimizer_config
        self.agent_optimizer_config = agent_optimizer_config
        self.ssl_model_config = ssl_model_config
        self.ssl_optimizer_config = ssl_optimizer_config
        self.reconstruction_loss = reconstruction_loss
        self.kl_divergence_weight = kl_divergence_weight
        self.self_supervised_learning_loss_weight = self_supervised_learning_loss_weight
        self.loss_combination_method = loss_combination_method
        self.pit_loss_alpha = pit_loss_alpha
        self.data_constructor = data_constructor
        self.warmup_epochs = warmup_epochs

        super().__init__(
            device_ids=device_ids,
            world_size=len(device_ids),
            init_method=init_method,
        )

    def _get_worker_class(self):
        """Return VAEGraphQMIXWorker class."""
        from marlite.trainer.trainer_worker.vae_graph_worker import VAEGraphQMIXWorker

        return VAEGraphQMIXWorker

    def _create_worker_kwargs(self) -> Dict[str, Any]:
        """Create kwargs for VAEGraphQMIXWorker initialization."""
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
        kwargs["kl_divergence_weight"] = self.kl_divergence_weight
        kwargs["self_supervised_learning_loss_weight"] = (
            self.self_supervised_learning_loss_weight
        )
        kwargs["loss_combination_method"] = self.loss_combination_method
        kwargs["pit_loss_alpha"] = self.pit_loss_alpha
        kwargs["data_constructor"] = self.data_constructor
        kwargs["warmup_epochs"] = self.warmup_epochs
        return kwargs

    def train_step(self, batch: Dict[str, Any]) -> tuple:
        """
        Execute one training step across all workers.

        Distributes the batch slices to workers, each computes gradients on
        its data slice, then synchronizes via all_reduce.

        Args:
            batch: Full batch from DataLoader

        Returns:
            Tuple of (avg_combined_loss, avg_critic_loss, avg_vae_loss)
        """
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
