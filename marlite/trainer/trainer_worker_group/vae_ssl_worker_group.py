"""
VAE SSL (Self-Supervised Learning) worker group implementation.

This module provides VAESSLWorkerGroup that manages VAE-based SSL workers
for multi-GPU training with VAEGraphQMIXTrainer.
"""

from typing import Any, Dict
from marlite.trainer.trainer_worker_group.ssl_worker_group import SSLWorkerGroup
from marlite.trainer.trainer_worker.ssl_worker import VAESSLWorker


class VAESSLWorkerGroup(SSLWorkerGroup):
    """
    Worker group for VAE-based Self-Supervised Learning multi-GPU training.

    This group manages VAESSLWorker instances, each holding copies of:
    - ssl_model (VAE decoder)
    - eval_agent_group (for representation learning)

    Extends SSLWorkerGroup with VAE-specific training logic.
    """

    def __init__(
        self,
        device_ids: list,
        ssl_model_config,
        agent_group_config,
        ssl_optimizer_config,
        agent_group_optimizer_config,
        reconstruction_loss=None,
        kl_divergence_weight: float = 1.0,
        data_constructor=None,
        init_method: str = "tcp://localhost:29500",
    ):
        """
        Initialize VAE SSL worker group.

        Args:
            device_ids: List of CUDA device IDs
            ssl_model_config: Configuration for VAE SSL model
            agent_group_config: Configuration for agent group
            ssl_optimizer_config: Configuration for SSL optimizer
            agent_group_optimizer_config: Configuration for agent group optimizer
            reconstruction_loss: Loss function for reconstruction
            kl_divergence_weight: Weight for KL divergence loss
            data_constructor: Data constructor for processing observations
            init_method: URL for distributed initialization
        """
        self.data_constructor = data_constructor

        super().__init__(
            device_ids=device_ids,
            ssl_model_config=ssl_model_config,
            agent_group_config=agent_group_config,
            ssl_optimizer_config=ssl_optimizer_config,
            agent_group_optimizer_config=agent_group_optimizer_config,
            reconstruction_loss=reconstruction_loss,
            kl_divergence_weight=kl_divergence_weight,
            init_method=init_method,
        )

    def _get_worker_class(self):
        """Return VAESSLWorker class."""
        return VAESSLWorker

    def _create_worker_kwargs(self) -> Dict[str, Any]:
        """Create kwargs for VAESSLWorker initialization."""
        kwargs = super()._create_worker_kwargs()
        kwargs["data_constructor"] = self.data_constructor
        return kwargs
