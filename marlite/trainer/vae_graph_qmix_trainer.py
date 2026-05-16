import torch
import numpy as np
import time
import absl.logging as logging
from tqdm import tqdm

from marlite.trainer.self_supervised_qmix_trainer import SelfSupervisedQMIXTrainer
from marlite.trainer.trainer_worker_group import VAEGraphWorkerGroup
from marlite.util.trajectory_dataset import (
    SSLEnrichedTrajectoryDataset,
    TrajectoryDataLoader,
)


class VAEGraphQMIXTrainer(SelfSupervisedQMIXTrainer):
    """
    A GraphQMIXTrainer subclass that works with probabilistic agent groups.

    This trainer handles ProbObsGNNCommAgentGroup, ProbSeqGNNCommAgentGroup,
    DualPathObsGNNCommAgentGroup, and DualPathProbObsGNNCommAgentGroup.

    It optimizes the parameters of the independent multivariate Gaussian distributions
    output by probabilistic agent groups using a VAE decoder.

    Joint RL+SSL Training:
    - During warmup_epochs: only RL loss is used
    - After warmup: combined loss is computed using the specified combination method
        - "weighted_sum": combined_loss = critic_loss + weight * vae_loss
        - "pit_loss": PITLoss-based combination

    SSL and RL loss combination is controlled by:
        - loss_combination_method: "weighted_sum" or "pit_loss"
        - self_supervised_learning_loss_weight: weight for VAE loss (used in weighted_sum mode)
        - pit_loss_alpha: alpha for PITLoss (used in pit_loss mode)
    """

    def __init__(
        self,
        kl_divergence_weight=0.005,
        warmup_epochs=0,
        loss_combination_method="weighted_sum",
        pit_loss_alpha=0.9,
        **kwargs,
    ):
        """
        Initialize VAEGraphQMIXTrainer.

        Args:
            kl_divergence_weight: Weight for KL divergence in VAE loss
            warmup_epochs: Number of epochs to train with RL only before enabling SSL
            loss_combination_method: Method to combine RL and SSL losses
                - "weighted_sum": combined_loss = critic_loss + weight * vae_loss
                - "pit_loss": use PITLoss to combine critic_loss and vae_loss
            pit_loss_alpha: Alpha parameter for PITLoss (exponential decay rate)
            **kwargs: Additional arguments passed to parent class
        """
        self.kl_divergence_weight = kl_divergence_weight
        self.warmup_epochs = warmup_epochs
        super().__init__(
            loss_combination_method=loss_combination_method,
            pit_loss_alpha=pit_loss_alpha,
            **kwargs,
        )

    def _create_worker_group(self):
        """Create VAEGraphWorkerGroup for multi-GPU joint RL+SSL training."""
        if not self.use_multi_gpu:
            return None

        return VAEGraphWorkerGroup(
            device_ids=list(range(len(self.device_list))),
            agent_group_config=self.agent_group_config,
            critic_config=self.critic_config,
            critic_optimizer_config=self.critic_optimizer_config,
            agent_optimizer_config=self.agent_optimizer_config,
            gamma=self.gamma,
            max_grad_norm=self.max_grad_norm,
            ssl_model_config=self.ssl_model_config,
            ssl_optimizer_config=self.ssl_optimizer_config,
            reconstruction_loss=self.reconstruction_loss,
            kl_divergence_weight=self.kl_divergence_weight,
            self_supervised_learning_loss_weight=self.self_supervised_learning_loss_weight,
            loss_combination_method=self.loss_combination_method,
            pit_loss_alpha=self.pit_loss_alpha,
            data_constructor=self.data_constructor,
            warmup_epochs=self.warmup_epochs,
        )

    def learn(self, sample_size, batch_size: int, times: int = 1):
        """
        Joint RL+SSL learning.

        Args:
            sample_size: Number of samples to draw from replay buffer
            batch_size: Batch size for training
            times: Number of times to iterate over the sampled data

        Returns:
            Combined loss (avg across batches)
        """
        if not self.use_multi_gpu:
            return self._joint_learn_single_gpu(sample_size, batch_size, times)
        return self._joint_learn_multi_gpu(sample_size, batch_size, times)

    def _joint_learn_single_gpu(self, sample_size, batch_size: int, times: int = 1):
        """
        Joint RL+SSL learning on single GPU.

        Training flow per batch:
        1. Compute RL td_error via _compute_rl_loss
        2. Compute VAE loss via _compute_vae_loss
        3. During warmup: combined_loss = critic_loss
        4. After warmup: combined_loss = critic_loss + weight * vae_loss
        5. Single backward pass
        6. Separate optimizer steps for critic, agent, and ssl_model

        Args:
            sample_size: Number of samples to draw from replay buffer
            batch_size: Batch size for training
            times: Number of times to iterate over the sampled data

        Returns:
            Combined loss (avg across batches)
        """
        total_combined = 0.0
        total_critic = 0.0
        total_vae = 0.0
        total_batches = 0

        # Move models to device
        self.eval_agent_group.to(self.train_device)
        self.eval_critic.to(self.train_device)
        self.target_agent_group.to(self.train_device)
        self.target_critic.to(self.train_device)
        self.ssl_model.to(self.train_device)

        is_warmup = self.current_epoch < self.warmup_epochs

        for t in range(times):
            dataset = self.replaybuffer.sample(sample_size)
            ssl_start = time.time()
            ssl_dataset = SSLEnrichedTrajectoryDataset(dataset, self.data_constructor)
            dataloader = TrajectoryDataLoader(
                ssl_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=self.n_workers,
            )
            ssl_time = time.time() - ssl_start
            logging.info(
                f"  Self-supervised learning data construction: {ssl_time:.4f}s"
            )
            with tqdm(
                total=sample_size, desc=f"Times {t + 1}/{times}", unit="batch"
            ) as pbar:
                for batch in dataloader:
                    # Compute combined RL+SSL loss in single forward pass
                    combined_loss, critic_loss, vae_loss = self._compute_loss(
                        batch, is_warmup
                    )

                    # === Backward Pass ===
                    self.critic_optimizer.zero_grad()
                    self.agent_optimizer.zero_grad()
                    self.ssl_optimizer.zero_grad()

                    combined_loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.eval_critic.parameters(), max_norm=self.max_grad_norm
                    )
                    torch.nn.utils.clip_grad_norm_(
                        self.eval_agent_group.parameters(), max_norm=self.max_grad_norm
                    )
                    torch.nn.utils.clip_grad_norm_(
                        self.ssl_model.parameters(), max_norm=self.max_grad_norm
                    )

                    # Optimizer steps
                    self.critic_optimizer.step()
                    self.agent_optimizer.step()
                    self.ssl_optimizer.step()

                    # Accumulate losses
                    total_combined += combined_loss.detach().cpu().item()
                    total_critic += critic_loss.detach().cpu().item()
                    if isinstance(vae_loss, torch.Tensor):
                        total_vae += vae_loss.detach().cpu().item()
                    total_batches += 1

                    bs = batch["states"].shape[0]
                    pbar.update(bs)

        # Move models back to CPU
        self.eval_agent_group.to("cpu")
        self.eval_critic.to("cpu")
        self.target_agent_group.to("cpu")
        self.target_critic.to("cpu")
        self.ssl_model.to("cpu")

        torch.cuda.empty_cache()

        avg_combined = total_combined / total_batches
        avg_critic = total_critic / total_batches
        avg_vae = total_vae / total_batches
        logging.info(
            f"  Combined Loss: {avg_combined:.4f}, RL Loss: {avg_critic:.4f}, VAE Loss: {avg_vae:.4f}"
        )

        return avg_combined

    def _joint_learn_multi_gpu(self, sample_size, batch_size: int, times: int = 1):
        """
        Joint RL+SSL learning on multiple GPUs.

        Delegates to VAEGraphWorkerGroup which handles:
        - Batch distribution across workers
        - Gradient synchronization
        - Loss aggregation

        Args:
            sample_size: Number of samples to draw from replay buffer
            batch_size: Batch size for training
            times: Number of times to iterate over the sampled data

        Returns:
            Combined loss (avg across batches)
        """
        self.worker_group.move_models_to_gpu()

        total_combined = 0.0
        total_critic = 0.0
        total_vae = 0.0
        total_batches = 0

        for t in range(times):
            dataset = self.replaybuffer.sample(sample_size)
            ssl_start = time.time()
            ssl_dataset = SSLEnrichedTrajectoryDataset(dataset, self.data_constructor)
            dataloader = TrajectoryDataLoader(
                ssl_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=self.n_workers,
            )
            ssl_time = time.time() - ssl_start
            logging.info(
                f"  Self-supervised learning data construction: {ssl_time:.4f}s"
            )
            with tqdm(
                total=sample_size, desc=f"Times {t + 1}/{times}", unit="batch"
            ) as pbar:
                for batch in dataloader:
                    batch["epoch"] = self.current_epoch
                    combined, critic, vae = self.worker_group.train_step(batch)

                    total_combined += combined
                    total_critic += critic
                    total_vae += vae
                    total_batches += 1

                    bs = batch["states"].shape[0]
                    pbar.update(bs)

        self.worker_group.move_models_to_cpu()
        torch.cuda.empty_cache()

        avg_combined = total_combined / total_batches
        avg_critic = total_critic / total_batches
        avg_vae = total_vae / total_batches
        logging.info(
            f"  Combined Loss: {avg_combined:.4f}, Critic Loss: {avg_critic:.4f}, VAE Loss: {avg_vae:.4f}"
        )

        return avg_combined

    def _compute_loss(self, batch, is_warmup: bool):
        """
        Compute combined RL+SSL loss in a single forward pass.

        The eval_agent_group.forward() returns local_state_estimates, mu, log_var
        which are used for both RL Q-value computation and VAE reconstruction.

        Args:
            batch: Dictionary containing RL and SSL data
            is_warmup: If True, only compute RL loss (SSL disabled during warmup)

        Returns:
            Tuple of (combined_loss, critic_loss, vae_loss)
        """
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
        next_alive_mask = next_alive_mask.to(self.train_device)
        alive_mask = alive_mask.to(self.train_device)

        # Handle action mask
        if isinstance(next_avail_actions, torch.Tensor):
            use_action_mask = True
            next_avail_actions = next_avail_actions[:, -1, :, :]
            next_avail_actions = next_avail_actions.to(
                dtype=torch.bool, device=self.train_device
            )
        else:
            use_action_mask = False

        # Process rewards and terminations
        rewards = rewards[:, -1]  # (B, T, N) -> (B, N)
        rewards = rewards.sum(dim=1).to(self.train_device)  # (B, N) -> (B)
        terminations = terminations[:, -1]  # (B, T, N) -> (B, N)
        terminations = terminations.prod(dim=1).to(self.train_device)  # (B, N) -> (B)

        # Process padding masks - expand to (B, N, T)
        timestep_padding_mask = torch.stack(
            [timestep_padding_mask] * n_agents, dim=1
        ).to(self.train_device)
        next_timestep_padding_mask = torch.stack(
            [next_timestep_padding_mask] * n_agents, dim=1
        ).to(self.train_device)

        # Extract last edge indices for current and next states
        last_edge_indices = [edge_indices[i][-1] for i in range(bs)]
        last_next_edge_indices = [next_edge_indices[i][-1] for i in range(bs)]

        # === RL Forward Pass (also gets SSL data from same forward) ===
        self.eval_agent_group.reset().train()
        # Transpose: (B, T, N, O) -> (B, N, T, O)
        observations_transposed = torch.transpose(observations, 1, 2).to(
            self.train_device
        )
        states = states.to(self.train_device)

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
        actions_last = actions[:, -1].to(device=self.train_device, dtype=torch.int64)
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

        # === Compute TD Targets ===
        with torch.no_grad():
            self.target_agent_group.reset().eval()
            next_observations_transposed = torch.transpose(next_observations, 1, 2).to(
                self.train_device
            )
            next_states = next_states.to(self.train_device)
            ret_next = self.target_agent_group(
                next_observations_transposed,
                next_states,
                next_timestep_padding_mask,
                next_alive_mask[:, -1, :],
                last_next_edge_indices,
            )
            q_val_next = ret_next["q_val"]
            if use_action_mask:
                q_val_next = torch.masked_fill(
                    q_val_next, ~next_avail_actions, -torch.inf
                )
            q_val_next = q_val_next.max(dim=-1).values
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
        y_tot = rewards + (1 - terminations) * self.gamma * q_tot_next

        # Compute critic loss (TD error)
        critic_loss = torch.nn.functional.mse_loss(q_tot, y_tot.detach())

        # === Compute VAE Loss (if not warmup) ===
        if is_warmup:
            vae_loss = torch.tensor(0.0, device=self.train_device)
        else:
            # VAE decoder forward pass
            # estimates: (B, N, T, E), formatted: (B, T, N, E)
            formatted_device = formatted.to(self.train_device)
            construct_mask_device = construct_mask.to(self.train_device)
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
            kl_divergence = -0.5 * torch.sum(
                1 + log_var - mu.pow(2) - torch.exp(log_var), dim=-1
            )
            kl_divergence = torch.mean(kl_divergence)

            vae_loss = reconstruction_loss + self.kl_divergence_weight * kl_divergence

        # Combined loss using the specified combination method
        if is_warmup:
            combined_loss = critic_loss
        else:
            combined_loss = self._combine_rl_ssl_loss(critic_loss, vae_loss)

        return combined_loss, critic_loss, vae_loss
