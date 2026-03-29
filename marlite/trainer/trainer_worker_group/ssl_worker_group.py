"""
SSL (Self-Supervised Learning) worker group implementation.

This module provides SSLWorkerGroup that manages SSL workers for multi-GPU training.
"""

from typing import Any, Dict
from marlite.trainer.trainer_worker_group.base_worker_group import BaseWorkerGroup
from marlite.trainer.trainer_worker.ssl_worker import SSLWorker


class SSLWorkerGroup(BaseWorkerGroup):
    """
    Worker group for Self-Supervised Learning multi-GPU training.

    This group manages SSLWorker instances, each holding copies of:
    - ssl_model (for self-supervised learning)
    - eval_agent_group (for representation learning)

    Workers in this group execute ssl_train_step() instead of train_step().
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
        init_method: str = None,
    ):
        """
        Initialize SSL worker group.

        Args:
            device_ids: List of CUDA device IDs
            ssl_model_config: Configuration for SSL model
            agent_group_config: Configuration for agent group
            ssl_optimizer_config: Configuration for SSL optimizer
            agent_group_optimizer_config: Configuration for agent group optimizer
            reconstruction_loss: Loss function for reconstruction
            kl_divergence_weight: Weight for KL divergence loss
            init_method: URL for distributed initialization
        """
        self.ssl_model_config = ssl_model_config
        self.agent_group_config = agent_group_config
        self.ssl_optimizer_config = ssl_optimizer_config
        self.agent_group_optimizer_config = agent_group_optimizer_config
        self.reconstruction_loss = reconstruction_loss
        self.kl_divergence_weight = kl_divergence_weight

        super().__init__(
            device_ids=device_ids,
            world_size=len(device_ids),
            init_method=init_method,
        )

    def _get_worker_class(self):
        """Return SSLWorker class."""
        return SSLWorker

    def _create_worker_kwargs(self) -> Dict[str, Any]:
        """Create kwargs for SSLWorker initialization."""
        kwargs = super()._create_worker_kwargs()
        kwargs["ssl_model_config"] = self.ssl_model_config
        kwargs["agent_group_config"] = self.agent_group_config
        kwargs["ssl_optimizer_config"] = self.ssl_optimizer_config
        kwargs["agent_group_optimizer_config"] = self.agent_group_optimizer_config
        kwargs["reconstruction_loss"] = self.reconstruction_loss
        kwargs["kl_divergence_weight"] = self.kl_divergence_weight
        return kwargs

    def write_params_to_workers(
        self,
        ssl_model_params: Dict[str, Any],
        agent_group_params: Dict[str, Any],
        blocking: bool = True,
    ):
        """
        Write SSL model and agent group parameters to all workers.

        Args:
            ssl_model_params: SSL model state dict
            agent_group_params: Agent group parameters dict
            blocking: Whether to wait for workers to acknowledge
        """
        trainable_params = {
            "ssl_model": ssl_model_params,
            "eval_agent_group": agent_group_params,
        }

        for i in range(self.world_size):
            self.cmd_queue.put("SYNC_FROM_MAIN")
            self.param_queues[i].put(trainable_params.copy())

        if blocking:
            for i in range(self.world_size):
                ack = self.param_queues[i].get()
                if ack != "ACK":
                    raise RuntimeError(f"Worker {i}: Expected ACK, got {ack}")

    def broadcast_params(self):
        """Broadcast current SSL and agent group parameters to all workers."""
        self.cmd_queue.put("SYNC_TO_MAIN")
        latest_params = self.param_queues[0].get()

        for i in range(self.world_size):
            self.cmd_queue.put("BROADCAST")
            self.param_queues[i].put(latest_params.copy())

    def read_params_from_worker0(self) -> tuple:
        """
        Read latest SSL model and agent group parameters from Worker 0.

        Returns:
            Tuple of (ssl_model_params, agent_group_params)
        """
        self.cmd_queue.put("SYNC_TO_MAIN")
        params = self.param_queues[0].get()

        return params.get("ssl_model"), params.get("eval_agent_group")

    def ssl_train_step(self, batch: Dict[str, Any]) -> float:
        """
        Execute one SSL training step across all workers.

        Args:
            batch: Full batch from DataLoader

        Returns:
            Average SSL loss across all workers
        """
        # Send train command to all workers
        for _ in range(self.world_size):
            self.cmd_queue.put("SSL_TRAIN_STEP")
            self.data_queue.put(batch)

        # Collect losses from all workers
        losses = []
        for _ in range(self.world_size):
            loss = self.loss_queue.get()
            losses.append(loss)

        return sum(losses) / len(losses)
