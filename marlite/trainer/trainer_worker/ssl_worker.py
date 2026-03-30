"""
SSL (Self-Supervised Learning) worker implementation for multi-GPU training.

This module provides the SSLWorker class that implements the training logic
for self-supervised learning in a multi-GPU setting.
"""

import torch
import torch.distributed as dist
from typing import Any, Dict
from marlite.trainer.trainer_worker.base_worker import BaseWorker


class SSLWorker(BaseWorker):
    """
    Worker for Self-Supervised Learning multi-GPU training.

    This worker holds ssl_model and eval_agent_group for self-supervised learning.
    The ssl_learn method trains the ssl_model to reconstruct observations.
    """

    def __init__(
        self,
        worker_id: int,
        device_id: int,
        rank: int,
        world_size: int,
        init_method: str,
        ssl_model_config=None,
        agent_group_config=None,
        ssl_optimizer_config=None,
        agent_optimizer_config=None,
        reconstruction_loss=None,
        kl_divergence_weight: float = 1.0,
        **kwargs,
    ):
        """
        Initialize SSL worker.

        Args:
            worker_id: Unique worker identifier
            device_id: CUDA device ID
            rank: Global rank in distributed training
            world_size: Total number of processes
            init_method: URL for distributed initialization
            ssl_model_config: Configuration for SSL model
            agent_group_config: Configuration for agent group
            ssl_optimizer_config: Configuration for SSL optimizer
            agent_optimizer_config: Configuration for agent group optimizer
            reconstruction_loss: Loss function for reconstruction
            kl_divergence_weight: Weight for KL divergence loss
        """
        super().__init__(worker_id, device_id, rank, world_size, init_method)
        self.reconstruction_loss = reconstruction_loss
        self.kl_divergence_weight = kl_divergence_weight

        if ssl_model_config is not None:
            self.ssl_model = ssl_model_config.get_model().to(self.device)
        else:
            self.ssl_model = None

        if agent_group_config is not None:
            self.eval_agent_group = agent_group_config.get_agent_group().to(self.device)
        else:
            self.eval_agent_group = None

        if ssl_optimizer_config is not None and self.ssl_model is not None:
            if hasattr(ssl_optimizer_config, "get_optimizer"):
                self.ssl_optimizer = ssl_optimizer_config.get_optimizer(
                    self.ssl_model.parameters()
                )
            else:
                self.ssl_optimizer = ssl_optimizer_config
        else:
            self.ssl_optimizer = None

        if agent_optimizer_config is not None and self.eval_agent_group is not None:
            if hasattr(agent_optimizer_config, "get_optimizer"):
                self.agent_optimizer = agent_optimizer_config.get_optimizer(
                    self.eval_agent_group.parameters()
                )
            else:
                self.agent_optimizer = agent_optimizer_config
        else:
            self.agent_optimizer = None

    def set_ssl_models(
        self,
        ssl_model,
        eval_agent_group,
        ssl_optimizer,
        agent_group_optimizer,
    ):
        """
        Set SSL model and agent group for this worker.

        Args:
            ssl_model: Self-supervised model
            eval_agent_group: Evaluation agent group
            ssl_optimizer: Optimizer for ssl_model
            agent_group_optimizer: Optimizer for agent group
        """
        self.ssl_model = ssl_model.to(self.device)
        self.eval_agent_group = eval_agent_group.to(self.device)
        self.ssl_optimizer = ssl_optimizer
        self.agent_group_optimizer = agent_group_optimizer

        self.eval_agent_group.train()

    def reduce_gradients(self):
        """
        Reduce (average) gradients across all workers for both ssl_model and agent_group.
        """
        # SSL model gradients
        for param in self.ssl_model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= self.world_size

        # Agent group gradients
        for param in self.eval_agent_group.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= self.world_size

    def write_params_to_shared_memory(self, shared_memory: Dict[str, Any]):
        """Write current parameters to shared memory including ssl_model."""
        shared_memory["eval_agent_group"] = (
            self.eval_agent_group.get_agent_group_params()
        )
        shared_memory["ssl_model"] = self.ssl_model.state_dict()

    def sync_params_from_shared_memory(self, shared_memory: Dict[str, Any]):
        """Synchronize parameters from shared memory."""
        if "eval_agent_group" in shared_memory:
            self.eval_agent_group.set_agent_group_params(
                shared_memory["eval_agent_group"]
            )
        if "ssl_model" in shared_memory:
            self.ssl_model.load_state_dict(shared_memory["ssl_model"])


class VAESSLWorker(SSLWorker):
    """
    Worker for VAE-based Self-Supervised Learning multi-GPU training.

    Implements ssl_train_step() for VAE training with reconstruction and KL losses.
    Used by VAEGraphQMIXTrainer.
    """

    def __init__(
        self,
        worker_id: int,
        device_id: int,
        rank: int,
        world_size: int,
        init_method: str,
        ssl_model_config=None,
        agent_group_config=None,
        ssl_optimizer_config=None,
        agent_optimizer_config=None,
        reconstruction_loss=None,
        kl_divergence_weight: float = 1.0,
        data_constructor=None,
        **kwargs,
    ):
        """
        Initialize VAE SSL worker.

        Args:
            worker_id: Unique worker identifier
            device_id: CUDA device ID
            rank: Global rank in distributed training
            world_size: Total number of processes
            init_method: URL for distributed initialization
            ssl_model_config: Configuration for VAE SSL model
            agent_group_config: Configuration for agent group
            ssl_optimizer_config: Configuration for SSL optimizer
            agent_optimizer_config: Configuration for agent group optimizer
            reconstruction_loss: Loss function for reconstruction
            kl_divergence_weight: Weight for KL divergence loss
            data_constructor: Data constructor for processing observations
        """
        super().__init__(
            worker_id,
            device_id,
            rank,
            world_size,
            init_method,
            ssl_model_config,
            agent_group_config,
            ssl_optimizer_config,
            agent_optimizer_config,
            reconstruction_loss,
            kl_divergence_weight,
            **kwargs,
        )
        self.data_constructor = data_constructor

    def ssl_train_step(self, batch: Dict[str, Any]) -> float:
        """
        Execute one SSL training step (VAE forward + backward).

        Implements VAE training with:
        - Reconstruction loss
        - KL divergence loss

        Args:
            batch: Dictionary containing:
                - observations: Raw observations
                - alive_mask: Agent alive masks
                - timestep_padding_mask: Padding masks
                - states: State sequences
                - edge_indices: Graph edge indices

        Returns:
            loss: Computed VAE loss value
        """
        # Unpack batch data
        obs = batch["observations"].to(dtype=torch.float32)
        obs_mask = batch["timestep_padding_mask"].to(dtype=torch.bool)
        formatted = batch["formatted_obs"]
        construct_mask = batch["construct_padding_mask"]

        bs = obs.shape[0]
        n_agents = batch.get("n_agents", obs.shape[2] if len(obs.shape) > 2 else 1)

        # Process edge indices - edge_indices is already sliced per batch
        edge_indices = batch.get("edge_indices", None)
        if edge_indices is not None:
            last_ts_edges = [edges[-1] for edges in edge_indices]
        else:
            last_ts_edges = None

        # Move to device
        obs = obs.to(self.device, dtype=torch.float32)
        obs_mask = obs_mask.to(dtype=torch.bool)
        obs_mask = torch.stack([obs_mask] * n_agents, dim=1).to(self.device)
        formatted = formatted.to(self.device, dtype=torch.float32)
        construct_mask = construct_mask.to(self.device, dtype=torch.bool)

        # Transpose observations: (B, T, N, O) -> (B, N, T, O)
        obs = obs.transpose(1, 2)

        # VAE forward pass through agent group
        self.eval_agent_group.train()
        msg, _ = self.eval_agent_group._process_observations(obs, obs_mask)
        estimates, _, mu, _, log_var = (
            self.eval_agent_group._compute_local_state_estimates(msg, last_ts_edges)
        )

        # VAE decoder forward
        reconstructed_obs = self.ssl_model(estimates)
        reconstructed_obs = torch.reshape(reconstructed_obs, formatted.shape)

        # Compute reconstruction loss
        reconstruction_loss = self._compute_ssl_loss(
            reconstructed_obs.view(-1, *reconstructed_obs.shape[2:]),
            formatted.view(-1, *formatted.shape[2:]),
            construct_mask.view(-1, *construct_mask.shape[2:]),
        )

        # Compute KL divergence loss
        # KL(q(z|x) || p(z)) = 0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        kl_divergence = -0.5 * torch.sum(
            1 + log_var - mu.pow(2) - torch.exp(log_var), dim=-1
        )
        kl_divergence = torch.mean(kl_divergence)

        # Total VAE loss
        vae_loss = reconstruction_loss + self.kl_divergence_weight * kl_divergence

        # Backward pass
        self.eval_agent_group.zero_grad()
        self.ssl_model.zero_grad()
        vae_loss.backward()

        # Synchronize gradients
        self.reduce_gradients()

        # Clip and optimize
        torch.nn.utils.clip_grad_norm_(list(self.ssl_model.parameters()), max_norm=5.0)
        self.ssl_optimizer.step()
        self.eval_agent_group.step()

        return vae_loss.detach().cpu().item()

    def _compute_ssl_loss(self, pred_set, target_set, mask=None):
        """
        Compute SSL reconstruction loss.

        Args:
            pred_set: Predicted values
            target_set: Target values
            mask: Optional mask for valid elements

        Returns:
            loss: Computed loss value
        """
        if hasattr(self.reconstruction_loss, "reconstruction_loss"):
            return self.reconstruction_loss.reconstruction_loss(
                pred_set, target_set, mask
            )
        else:
            return self.reconstruction_loss(pred_set, target_set)

    def handle_command(self, cmd, param_queue, data_queue, loss_queue):
        if cmd == "SSL_TRAIN_STEP":
            batch = data_queue.get()
            loss = self.ssl_train_step(batch)
            loss_queue.put(loss)
            return True
        return super().handle_command(cmd, param_queue, data_queue, loss_queue)
