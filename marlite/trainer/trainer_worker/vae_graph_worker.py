"""
VAE Graph worker implementation for joint RL+SSL multi-GPU training.

This module provides the VAEGraphQMIXWorker class that implements the training logic
for VAE-based GraphQMIX algorithm in a multi-GPU setting.

Training computes combined_loss = td_error + self_supervised_learning_loss_weight * vae_loss
in a single forward pass, where vae_loss is computed using local_state_estimates,
mu, and log_var returned directly from eval_agent_group.forward().
"""

import io
import torch
import torch.distributed as dist
from typing import Any, Dict, List, Optional

from marlite.algorithm.agents import AgentGroupConfig
from marlite.algorithm.critic import CriticConfig
from marlite.algorithm.model import ModelConfig
from marlite.util.optimizer_config import OptimizerConfig
from marlite.util.loss_func import PITLoss, ReconstructionLoss
from marlite.trainer.trainer_worker.offpolicy_worker import OffPolicyWorker


class VAEGraphQMIXWorker(OffPolicyWorker):
    """
    Worker for VAE-based GraphQMIX algorithm multi-GPU training.

    Implements train_step() method that executes one batch of joint RL+SSL training.
    The eval_agent_group.forward() returns local_state_estimates, mu, log_var which
    are used for both RL Q-value computation and VAE reconstruction.

    Training flow:
        combined_loss = critic_loss + weight * vae_loss
        - Single backward pass
        - Synchronized gradients for critic, agent, and ssl_model
        - Separate optimizer steps
    """

    critic_optimizer: torch.optim.Optimizer
    agent_optimizer: torch.optim.Optimizer
    ssl_optimizer: torch.optim.Optimizer

    def __init__(
        self,
        worker_id: int,
        device_id: int,
        rank: int,
        world_size: int,
        init_method: str,
        agent_group_config: AgentGroupConfig,
        critic_config: CriticConfig,
        critic_optimizer_config: OptimizerConfig,
        agent_optimizer_config: OptimizerConfig,
        ssl_model_config: ModelConfig,
        ssl_optimizer_config: OptimizerConfig,
        reconstruction_loss,
        data_constructor,
        gamma: float,
        max_grad_norm: float,
        kl_divergence_weight: float,
        self_supervised_learning_loss_weight: float,
        loss_combination_method: str,
        pit_loss_alpha: float,
        warmup_epochs: int,
        **kwargs,
    ):
        """
        Initialize VAE Graph worker.

        Args:
            worker_id: Unique worker identifier
            device_id: CUDA device ID
            rank: Global rank in distributed training
            world_size: Total number of processes
            init_method: URL for distributed initialization
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
        """
        super().__init__(worker_id, device_id, rank, world_size, init_method)
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.loss_combination_method = loss_combination_method
        self.pit_loss_alpha = pit_loss_alpha

        # Initialize RL models
        self.eval_agent_group = agent_group_config.get_agent_group()
        self.target_agent_group = agent_group_config.get_agent_group()
        self.eval_critic = critic_config.get_critic()
        self.target_critic = critic_config.get_critic()

        self.eval_agent_group.train()
        self.target_agent_group.eval()
        self.eval_critic.train()
        self.target_critic.eval()

        # Initialize RL optimizers
        self.critic_optimizer = critic_optimizer_config.get_optimizer(
            self.eval_critic.parameters()
        )
        self.agent_optimizer = agent_optimizer_config.get_optimizer(
            self.eval_agent_group.parameters()
        )

        # Initialize SSL components
        self.ssl_model = ssl_model_config.get_model()
        self.reconstruction_loss = reconstruction_loss
        if not isinstance(self.reconstruction_loss, ReconstructionLoss):
            raise TypeError(
                f"reconstruction_loss must be a ReconstructionLoss subclass, "
                f"got {type(self.reconstruction_loss).__name__}"
            )
        self.ssl_optimizer = ssl_optimizer_config.get_optimizer(
            self.ssl_model.parameters()
        )
        self.pit_loss = PITLoss(
            num_tasks=2,
            alpha=self.pit_loss_alpha,
            reduction="mean",
        )
        self.kl_divergence_weight = kl_divergence_weight
        self.self_supervised_learning_loss_weight = self_supervised_learning_loss_weight
        self.data_constructor = data_constructor
        self.warmup_epochs = warmup_epochs

    def move_to_device(self, device: str):
        """
        Move all models to the specified device.

        Args:
            device: Target device string (e.g., 'cuda:0' or 'cpu')
        """
        if self.eval_agent_group is not None:
            self.eval_agent_group.to(device)
        if self.target_agent_group is not None:
            self.target_agent_group.to(device)
        if self.eval_critic is not None:
            self.eval_critic.to(device)
        if self.target_critic is not None:
            self.target_critic.to(device)
        self.ssl_model.to(device)
        self.device = device

    def reduce_gradients(self):
        super().reduce_gradients()
        for param in self.ssl_model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= self.world_size

    def synchronize_eval_params(self):
        super().synchronize_eval_params()
        for param in self.ssl_model.parameters():
            dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
            param.data /= self.world_size

    def get_params_for_main(self) -> Dict[str, Any]:
        """
        Get current parameters to send back to main process.

        Returns:
            Dictionary containing cloned parameter data
        """
        params = {
            "eval_agent_group": {
                k: v.clone().cpu()
                for k, v in self.eval_agent_group.state_dict().items()
            },
            "target_agent_group": {
                k: v.clone().cpu()
                for k, v in self.target_agent_group.state_dict().items()
            },
            "eval_critic": {
                k: v.clone().cpu() for k, v in self.eval_critic.state_dict().items()
            },
            "target_critic": {
                k: v.clone().cpu() for k, v in self.target_critic.state_dict().items()
            },
        }
        params["ssl_model"] = {
            k: v.clone().cpu() for k, v in self.ssl_model.state_dict().items()
        }
        return params

    def sync_params_from_main(self, params):
        """
        Synchronize parameters received from main process.

        Delegates bytes-deserialisation, eval/target agent_group +
        critic, and target_update_* handling to
        :class:`OffPolicyWorker`.  Only the SSL auxiliary model
        (VAE decoder) is added here.
        """
        params = super().sync_params_from_main(params)
        if "ssl_model" in params and self.ssl_model is not None:
            self.ssl_model.load_state_dict(
                {k: v.clone() for k, v in params["ssl_model"].items()}
            )

    def train_step(self, batch: Dict[str, Any]) -> tuple:
        """
        Execute one training step on the given batch.

        When SSL is enabled, computes combined_loss = critic_loss + weight * vae_loss
        using local_state_estimates, mu, log_var from forward pass.
        Returns (combined_loss, critic_loss, vae_loss).

        Args:
            batch: Dictionary containing:
                - alive_mask: Agent alive masks
                - observations: Observation sequences (B, T, N, O)
                - timestep_padding_mask: Padding masks
                - states: State sequences (B, T, S)
                - edge_indices: Graph edge indices (list of arrays)
                - actions: Action sequences
                - rewards: Reward sequences
                - next_states: Next state sequences
                - next_observations: Next observation sequences
                - next_edge_indices: Edge indices for next states
                - next_timestep_padding_mask: Padding masks for next states
                - next_avail_actions: Available actions for next states
                - next_alive_mask: Alive masks for next states
                - terminations: Termination flags
                - formatted_obs: Processed observations for SSL (B, T, N, E)
                - construct_padding_mask: Padding mask for SSL (B, T, N)

        Returns:
            Tuple of (combined_loss, critic_loss, vae_loss) or single critic_loss if SSL disabled
        """
        # Get current epoch from batch to determine warmup status
        current_epoch = batch.get("epoch", 0)
        is_warmup = current_epoch < self.warmup_epochs

        # Extract batch data
        alive_mask = batch["alive_mask"].to(dtype=torch.bool)
        observations = batch["observations"].to(dtype=torch.float32)  # (B, T, N, O)
        timestep_padding_mask = batch["timestep_padding_mask"].to(dtype=torch.bool)
        states = batch["states"].to(dtype=torch.float32)  # (B, T, S)
        edge_indices = batch["edge_indices"]
        actions = batch["actions"].to(dtype=torch.int)
        rewards = batch["rewards"].to(dtype=torch.float32)
        next_states = batch["next_states"].to(dtype=torch.float32)
        next_observations = batch["next_observations"].to(dtype=torch.float32)
        next_edge_indices = batch["next_edge_indices"]
        next_timestep_padding_mask = batch["next_timestep_padding_mask"].to(
            dtype=torch.bool
        )
        next_avail_actions = batch["next_avail_actions"]
        next_alive_mask = batch["next_alive_mask"].to(dtype=torch.bool)
        terminations = batch["terminations"].to(dtype=torch.bool)
        formatted = batch["formatted_obs"].to(dtype=torch.float32)  # (B, T, N, E)
        construct_mask = batch["construct_padding_mask"].to(dtype=torch.bool)

        bs = states.shape[0]  # Batch size
        n_agents = rewards.shape[2]  # Number of agents

        # Prepare masks and move to device
        next_alive_mask = next_alive_mask.to(self.device)
        alive_mask = alive_mask.to(self.device)

        # Handle action mask
        if isinstance(next_avail_actions, torch.Tensor):
            use_action_mask = True
            next_avail_actions = next_avail_actions[:, -1, :, :]
            next_avail_actions = next_avail_actions.to(
                dtype=torch.bool, device=self.device
            )
        else:
            use_action_mask = False

        # Process rewards and terminations
        r_last = self._aggregate_rewards(rewards[:, -1]).to(self.device)  # (B, N) -> (B)
        termination_last = terminations[:, -1].prod(dim=-1).to(self.device)  # (B, N) -> (B)

        # Process padding masks - expand to (B, N, T)
        timestep_padding_mask = torch.stack(
            [timestep_padding_mask] * n_agents, dim=1
        ).to(self.device)
        next_timestep_padding_mask = torch.stack(
            [next_timestep_padding_mask] * n_agents, dim=1
        ).to(self.device)

        # Extract last edge indices for current and next states
        last_edge_indices = [edge_indices[i][-1] for i in range(bs)]
        last_next_edge_indices = [next_edge_indices[i][-1] for i in range(bs)]

        # === RL Forward Pass (also gets SSL data from same forward) ===
        self.eval_agent_group.reset().train()
        # Transpose: (B, T, N, O) -> (B, N, T, O)
        observations_transposed = torch.transpose(observations, 1, 2).to(self.device)
        states = states.to(self.device)

        # Forward returns: q_val, edge_indices, local_state_estimates, mu, std, log_var
        ret = self.eval_agent_group(
            observations_transposed,
            states,
            timestep_padding_mask,
            alive_mask[:, -1, :],
            last_edge_indices,
        )
        q_val = ret["q_val"]
        # Get SSL data from forward return (for VAE reconstruction)
        estimates = ret["local_state_estimates"]  # (B, N, T, E)
        mu = ret["mu"]
        log_var = ret["log_var"]

        # Get actions at last timestep: (B, T, N, A) -> (B, N, A)
        actions_last = actions[:, -1].to(device=self.device, dtype=torch.int64)
        q_val = torch.gather(q_val, dim=-1, index=actions_last.unsqueeze(-1)).squeeze(
            -1
        )
        # q_val: (B, N)

        self.eval_critic.train()
        ret_critic = self.eval_critic(
            q_val, states, alive_mask, timestep_padding_mask[:, 0, :]
        )
        q_tot = ret_critic["q_tot"]
        # q_tot: (B,)

        # === Compute TD Targets (Double Q-learning) ===
        with torch.no_grad():
            # Double Q: eval agent group selects best actions
            self.eval_agent_group.reset().eval()
            next_observations_transposed = torch.transpose(next_observations, 1, 2).to(
                self.device
            )
            next_states = next_states.to(self.device)
            ret_next_eval = self.eval_agent_group(
                next_observations_transposed,
                next_states,
                next_timestep_padding_mask,
                next_alive_mask[:, -1, :],
                last_next_edge_indices,
            )
            q_val_next_eval = ret_next_eval["q_val"]
            if use_action_mask:
                q_val_next_eval = torch.masked_fill(
                    q_val_next_eval, ~next_avail_actions, -torch.inf
                )
            best_actions = q_val_next_eval.argmax(dim=-1)
            # best_actions: (B, N)

            # Double Q: target agent group evaluates chosen actions
            self.target_agent_group.reset().eval()
            ret_next_target = self.target_agent_group(
                next_observations_transposed,
                next_states,
                next_timestep_padding_mask,
                next_alive_mask[:, -1, :],
                last_next_edge_indices,
            )
            q_val_next = ret_next_target["q_val"].gather(
                dim=-1, index=best_actions.unsqueeze(-1)
            ).squeeze(-1)
            # q_val_next: (B, N)

            self.target_critic.eval()
            ret_next_critic = self.target_critic(
                q_val_next,
                next_states,
                next_alive_mask,
                next_timestep_padding_mask[:, 0, :],
            )
            q_tot_next = ret_next_critic["q_tot"]
            # q_tot_next: (B,)

        # Compute TD target: y_tot = r + gamma * (1 - terminations) * q_tot_next
        y_tot = r_last + (1 - termination_last) * self.gamma * q_tot_next

        # Compute critic loss (TD error)
        critic_loss = torch.nn.functional.mse_loss(q_tot, y_tot.detach())

        # === Compute VAE Loss (if not in warmup) ===
        if not is_warmup:
            # VAE decoder forward pass
            # estimates: (B, N, T, E), formatted: (B, T, N, E)
            formatted_device = formatted.to(self.device)
            construct_mask_device = construct_mask.to(self.device)
            reconstructed_obs = self.ssl_model(estimates)
            reconstructed_obs = torch.reshape(reconstructed_obs, formatted_device.shape)

            # Compute reconstruction loss
            reconstruction_loss = self._compute_ssl_loss(
                reconstructed_obs.view(-1, *reconstructed_obs.shape[2:]),
                formatted_device.view(-1, *formatted_device.shape[2:]),
                construct_mask_device.view(-1, *construct_mask_device.shape[2:]),
            )

            # Compute KL divergence loss
            # KL(q(z|x) || p(z)) = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
            # mu/log_var: (B, N, T, E), alive_mask: (B, T+1, N)
            kl_per_dim = 1 + log_var - mu.pow(2) - torch.exp(log_var)
            kl_per_agent_t = -0.5 * kl_per_dim.sum(dim=-1)  # (B, N, T)
            mask = alive_mask[:, :mu.shape[2], :].transpose(1, 2)  # (B, N, T)
            kl_divergence = (kl_per_agent_t * mask).sum() / mask.sum().clamp(min=1)

            vae_loss = reconstruction_loss + self.kl_divergence_weight * kl_divergence
            combined_loss = self._combine_rl_ssl_loss(critic_loss, vae_loss)
        else:
            vae_loss = 0.0
            combined_loss = critic_loss

        # === Backward Pass ===
        self.critic_optimizer.zero_grad()
        self.agent_optimizer.zero_grad()
        self.ssl_optimizer.zero_grad()

        combined_loss.backward()

        # Synchronize gradients across all workers
        self.reduce_gradients()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.eval_critic.parameters(), max_norm=self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.eval_agent_group.parameters(), max_norm=self.max_grad_norm)
        if not is_warmup:
            torch.nn.utils.clip_grad_norm_(self.ssl_model.parameters(), max_norm=self.max_grad_norm)

        # Optimizer steps
        self.critic_optimizer.step()
        self.agent_optimizer.step()
        if not is_warmup:
            self.ssl_optimizer.step()

        # Per-batch target update (hard / ema / polyak)
        self._update_target_after_batch()

        # Return losses
        vae_loss_value = (
            vae_loss.detach().cpu().item()
            if isinstance(vae_loss, torch.Tensor)
            else vae_loss
        )
        return (
            combined_loss.detach().cpu().item(),
            critic_loss.detach().cpu().item(),
            vae_loss_value,
        )

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
        return self.reconstruction_loss(pred_set, target_set, mask)

    def _combine_rl_ssl_loss(self, critic_loss, vae_loss):
        """
        Combine RL (critic) loss and SSL (VAE) loss using the specified method.

        Args:
            critic_loss: RL critic loss tensor (scalar)
            vae_loss: SSL VAE loss tensor (scalar)

        Returns:
            combined_loss: Combined loss tensor (scalar)
        """
        if self.loss_combination_method == "pit_loss":
            losses = torch.stack([critic_loss, vae_loss])  # (2,)
            combined_loss = self.pit_loss(losses)
        else:
            combined_loss = (
                critic_loss + self.self_supervised_learning_loss_weight * vae_loss
            )
        return combined_loss

    def handle_command(
        self,
        cmd: str,
        param_queue,
        data_queue,
        loss_queue,
        ack_queue=None,
    ) -> bool:
        """
        Handle a command from the main process.

        Args:
            cmd: Command string
            param_queue: Queue for parameter exchange
            data_queue: Queue for receiving training data
            loss_queue: Queue for returning loss values
            ack_queue: Queue for sending ACK signals back to main process

        Returns:
            True if should continue, False if should stop
        """
        if cmd == "TRAIN_STEP":
            batch = data_queue.get()
            result = self.train_step(batch)
            del batch
            combined, critic, vae = result
            loss_queue.put((combined, critic, vae))
            return True

        if cmd == "SYNC_LR":
            lr_data = param_queue.get()
            if "critic_lr" in lr_data and self.critic_optimizer is not None:
                for param_group in self.critic_optimizer.param_groups:
                    param_group["lr"] = lr_data["critic_lr"]
            if "agent_lr" in lr_data and self.agent_optimizer is not None:
                for param_group in self.agent_optimizer.param_groups:
                    param_group["lr"] = lr_data["agent_lr"]
            if "ssl_lr" in lr_data and self.ssl_optimizer is not None:
                for param_group in self.ssl_optimizer.param_groups:
                    param_group["lr"] = lr_data["ssl_lr"]
            if ack_queue:
                ack_queue.put("ACK")
            return True

        return super().handle_command(
            cmd, param_queue, data_queue, loss_queue, ack_queue
        )

    def cleanup(self):
        """Clean up distributed process group."""
        if dist.is_initialized():
            dist.destroy_process_group()
