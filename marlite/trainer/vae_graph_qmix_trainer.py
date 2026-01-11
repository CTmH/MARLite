import torch
import numpy as np
from typing import List
from torch.nn import DataParallel
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from copy import deepcopy

from marlite.trainer.self_supervised_qmix_trainer import SelfSupervisedQMIXTrainer
from marlite.util.trajectory_dataset import TrajectoryDataLoader
from marlite.algorithm.model.gather_layer import GatherLayer

class VAEGraphQMIXTrainer(SelfSupervisedQMIXTrainer):
    """
    A GraphQMIXTrainer subclass that works with probabilistic agent groups.
    This trainer handles ProbObsGNNCommAgentGroup, ProbSeqGNNCommAgentGroup,
    DualPathObsGNNCommAgentGroup, and DualPathProbObsGNNCommAgentGroup.
    It optimizes the parameters of the independent multivariate Gaussian distributions
    output by probabilistic agent groups using a VAE decoder.
    """

    def __init__(self, kl_divergence_weight=1.0, **kwargs):

        super().__init__(**kwargs)
        self.kl_divergence_weight = kl_divergence_weight

        # Create data constructor from config
        self.data_constructor = self.data_constructor_config.get_data_constructor()

        self.eval_decoder = self.decoder_config.get_model()
        self.target_decoder = self.decoder_config.get_model()
        self.target_decoder.load_state_dict(self.eval_decoder.state_dict())
        self.best_decoder_params = deepcopy(self.eval_decoder.state_dict())
        self._cached_decoder_params = deepcopy(self.eval_decoder.state_dict())

        params_need_optim = [
            {'params': self.eval_critic.parameters()},
            {'params': self.eval_decoder.parameters()}
        ]
        self.optimizer = self.critic_optimizer_config.get_optimizer(params_need_optim)
        if self.lr_scheduler_conf:
            self.lr_scheduler = self.lr_scheduler_conf.get_lr_scheduler(self.optimizer)
        else:
            self.lr_scheduler = None
    '''
    def _compute_vae_loss(self,
                         mu: torch.Tensor,
                         std: torch.Tensor,
                         log_var: torch.Tensor,
                         observations: torch.Tensor,
                         states: torch.Tensor,
                         edge_indices: List[List[np.ndarray]],
                         alive_mask: torch.Tensor):
        """
        Internal function to compute VAE loss for probabilistic agent groups.

        Args:
            mu: Mean of the Gaussian distribution (batch_size, n_agents, feature_dim)
            std: Standard deviation of the Gaussian distribution (batch_size, n_agents, feature_dim)
            observations: Actual observations from the environment
            states: Environment states
            edge_indices: Communication graph edge indices
            alive_mask: Mask indicating which agents are alive
            log_var: Log variance of the Gaussian distribution (batch_size, n_agents, feature_dim), optional

        Returns:
            VAE total loss (reconstruction loss + KL divergence loss)
        """
        # Sample from the Gaussian distribution to get latent representations
        eps = torch.randn_like(std)
        z = mu + eps * std  # Reparameterization trick (batch_size, n_agents, feature_dim)

        # Decode the latent representations to reconstruct observations
        reconstructed_obs = self.eval_decoder(z)  # (batch_size, n_agents, feature_dim)

        # Format the original observations for comparison
        observations_np = observations.detach().cpu().numpy()
        states_np = states.detach().cpu().numpy()
        alive_mask_np = alive_mask.detach().cpu().numpy()
        formatted_obs, padding_mask = self.data_constructor.process(observations_np, states_np, edge_indices, alive_mask_np)  # (batch_size, n_agents, feature_dim)

        shape = (-1,) + formatted_obs.shape[-2:]
        reconstructed_obs = torch.reshape(reconstructed_obs, shape)
        formatted_obs = formatted_obs.reshape(shape)
        padding_mask = padding_mask.reshape(-1, padding_mask.shape[-1])
        formatted_obs = torch.tensor(formatted_obs, device=reconstructed_obs.device)
        padding_mask = torch.tensor(padding_mask, device=reconstructed_obs.device)

        # Calculate reconstruction loss
        reconstruction_loss = self.reconstruction_loss(reconstructed_obs, formatted_obs, padding_mask)

        # Calculate KL divergence loss
        # KL divergence between the learned distribution q(z|x) and prior p(z)
        # For Gaussian prior N(0, I) and posterior N(mu, std^2), KL divergence is:
        # If log_var is provided, use it directly for better numerical stability
        # KL(q(z|x) || p(z)) = 0.5 * sum(1 + log_var - mu^2 - exp(log_var))

        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - torch.exp(log_var), dim=-1)  # Sum over feature dimension

        kl_divergence = torch.mean(kl_divergence)  # Average over batch and agents

        # Handle DataParallel case: gather losses from all GPUs
        if self.use_data_parallel:
            # Gather losses from all GPUs
            gathered_recon_losses = GatherLayer.apply(reconstruction_loss)
            reconstruction_loss = torch.stack(gathered_recon_losses).mean() if gathered_recon_losses else reconstruction_loss

        # Total VAE loss: reconstruction loss + KL divergence loss
        total_vae_loss = reconstruction_loss + self.kl_divergence_weight * kl_divergence

        return total_vae_loss
    '''
    def self_supervised_learn(self, sample_size, batch_size: int, times: int = 1):
        total_loss = 0.0
        total_batches = 0

        self.eval_agent_group.to(self.train_device)
        self.eval_decoder.to(self.train_device)

        if self.use_data_parallel:
            self.eval_agent_group.wrap_data_parallel()
            self.eval_decoder = DataParallel(self.eval_decoder)

        # VAE self supervised learning
        self.eval_agent_group.train()
        for t in range(times):
            with tqdm(total=sample_size, desc=f'Times {t+1}/{times}', unit='batch') as pbar:
                # Implement the learning logic for QMix
                # Get a batch of data from the replay buffer
                data = self.replaybuffer.sample(sample_size)
                data = list(data)
                alive_mask = np.array([e['alive_mask'] for e in data])
                obs_padding_mask = np.array([e['obs_padding_mask'] for e in data])
                observations = np.array([e['observations'] for e in data])
                states = np.array([e['states'] for e in data])
                edge_indices = [e['edge_indices'] for e in data]
                alive_mask = np.array(alive_mask)
                observations = np.array(observations)
                formatted_obs, construct_padding_mask = self.data_constructor.process(observations, states, edge_indices, alive_mask)
                edge_indices = torch.tensor(edge_indices)
                observations = torch.tensor(observations)
                formatted_obs = torch.tensor(formatted_obs)
                construct_padding_mask = torch.tensor(construct_padding_mask)
                obs_padding_mask = torch.tensor(obs_padding_mask)
                dataset = TensorDataset(observations, obs_padding_mask, formatted_obs, edge_indices, construct_padding_mask)
                dataloader = DataLoader(dataset,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=self.n_workers)
                bs = states.shape[0]

                for obs, obs_mask, formatted, edge, construct_mask in dataloader:
                    msg, _ = self.eval_agent_group._process_observations(obs, obs_mask)
                    estimates, _, mu, std, log_var = self.eval_agent_group._compute_local_state_estimates(msg, edge)
                    reconstructed_obs = self.eval_decoder(estimates)
                    reconstructed_obs = torch.reshape(reconstructed_obs, formatted.shape)
                    reconstruction_loss = self.reconstruction_loss(reconstructed_obs, formatted_obs, construct_mask)
                    # Calculate KL divergence loss
                    # KL divergence between the learned distribution q(z|x) and prior p(z)
                    # For Gaussian prior N(0, I) and posterior N(mu, std^2), KL divergence is:
                    # If log_var is provided, use it directly for better numerical stability
                    # KL(q(z|x) || p(z)) = 0.5 * sum(1 + log_var - mu^2 - exp(log_var))

                    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - torch.exp(log_var), dim=-1)  # Sum over feature dimension

                    kl_divergence = torch.mean(kl_divergence)  # Average over batch and agents

                    # Total VAE loss: reconstruction loss + KL divergence loss
                    vae_loss = reconstruction_loss + self.kl_divergence_weight * kl_divergence

                    # Compute VAE loss
                    vae_loss = self._compute_vae_loss(
                        mu, std, log_var, observations, states, edge_indices, alive_mask
                    )

                    # Optimize the networks - only use the main optimizer since it includes both critic and decoder params
                    self.eval_agent_group.zero_grad()
                    self.eval_decoder.zero_grad()

                    vae_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self.eval_decoder.parameters()),
                        max_norm=5.0
                    )
                    self.optimizer.step()
                    self.eval_agent_group.step()

                    total_loss += vae_loss.detach().cpu().item()
                    total_batches += 1

                    pbar.update(bs)

        if self.use_data_parallel:
            self.eval_agent_group.unwrap_data_parallel()
            self.eval_decoder = self.eval_decoder.module

        self.eval_agent_group.to("cpu")
        self.eval_decoder.to("cpu")
        torch.cuda.empty_cache()

        return total_loss / total_batches

    def learn(self, sample_size, batch_size: int, times: int = 1):
        total_loss = 0.0
        total_batches = 0

        # Move models to the appropriate device before wrapping with DataParallel
        self.eval_agent_group.to(self.train_device)
        self.eval_critic.to(self.train_device)
        self.target_agent_group.to(self.train_device)
        self.target_critic.to(self.train_device)

        if self.use_data_parallel:
            self.eval_agent_group.wrap_data_parallel()
            self.eval_critic = DataParallel(self.eval_critic)
            self.target_agent_group.wrap_data_parallel()
            self.target_critic = DataParallel(self.target_critic)

        for t in range(times):
            with tqdm(total=sample_size, desc=f'Times {t+1}/{times}', unit='batch') as pbar:
                # Implement the learning logic for QMix
                # Get a batch of data from the replay buffer
                dataset = self.replaybuffer.sample(sample_size)
                dataloader = TrajectoryDataLoader(dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=self.n_workers)
                for batch in dataloader:
                    # Extract batch data
                    alive_mask = batch['alive_mask']
                    observations = batch['observations']
                    obs_padding_mask = batch['obs_padding_mask']
                    states = batch['states']
                    edge_indices = batch['edge_indices']
                    actions = batch['actions']
                    rewards = batch['rewards']
                    next_states = batch['next_states']
                    next_observations = batch['next_observations']
                    next_obs_padding_mask = batch['next_obs_padding_mask']
                    next_avail_actions = batch['next_avail_actions']
                    terminations = batch['terminations']
                    truncations = batch['truncations']
                    bs = states.shape[0]  # Actual batch size
                    n_agents = rewards.shape[1]

                    # Create alive_mask_next from terminations and truncations
                    alive_mask = torch.tensor(alive_mask).to(dtype=torch.bool) # (B, N, T)
                    alive_mask = alive_mask.permute(0, 2, 1).to(self.train_device) # (B, N, T) -> (B, T, N)
                    terminations = torch.tensor(terminations[:,:,-1]).to(self.train_device) # (B, N, T) -> (B, N)
                    truncations = torch.tensor(truncations[:,:,-1]).to(self.train_device) # (B, N, T) -> (B, N)
                    next_alive_mask = ~(terminations | truncations)
                    next_alive_mask = next_alive_mask.unsqueeze(dim=1)
                    next_alive_mask = torch.cat([alive_mask[:,1:,:], next_alive_mask], dim=1)
                    # Action mask: (B, N, T, Actions) -> (B, N, Actions)
                    if np.issubdtype(next_avail_actions.dtype, np.number):
                        use_action_mask = True
                        next_avail_actions = torch.tensor(next_avail_actions[:,:,-1,:])
                        next_avail_actions = next_avail_actions.to(dtype=torch.bool, device=self.train_device)
                    else:
                        use_action_mask = False

                    rewards = torch.Tensor(rewards[:,:,-1]).to(self.train_device) # (B, N, T) -> (B, N)
                    rewards = rewards.sum(dim=1) # (B, N) -> (B) Sum over all agents rewards
                    terminations = terminations.prod(dim=1) # (B, N) -> (B) if all agents are terminated then game over

                    obs_padding_mask = torch.tensor(obs_padding_mask, dtype=torch.bool) # (B, T)
                    obs_padding_mask = torch.stack([obs_padding_mask] * n_agents, dim=1).to(self.train_device) # (B, N, T)
                    next_obs_padding_mask = torch.tensor(next_obs_padding_mask, dtype=torch.bool)
                    next_obs_padding_mask = torch.stack([next_obs_padding_mask] * n_agents, dim=1).to(self.train_device)

                    # Compute the Q-tot
                    last_edge_indices = [edge_indices[i][-1] for i in range(bs)] # (B, T, 2, N) -> (B, 2, N) Take only the last edge indices
                    observations = torch.tensor(observations, dtype=torch.float, device=self.train_device)
                    self.eval_agent_group.reset().train() # Reset Graph Builder intervals
                    ret = self.eval_agent_group.forward(observations, states, obs_padding_mask, alive_mask[:,-1,:], last_edge_indices) # obs.shape (B, N, T, F)
                    q_val = ret['q_val']
                    actions = torch.Tensor(actions[:,:,-1:]).to(device=self.train_device, dtype=torch.int64) # (B, N, T, A)
                    q_val = torch.gather(q_val, dim=-1, index=actions)
                    q_val = q_val.squeeze(-1) # (B, N, 1) -> (B, N)
                    states = torch.Tensor(states).to(self.train_device)
                    self.eval_critic.train()
                    ret = self.eval_critic(q_val, states, alive_mask, obs_padding_mask[:,0,:])
                    q_tot = ret['q_tot']
                    state_features = ret['state_features']

                    # Double Q-learning, we use eval agent group to choose actions,and use target critic to compute q_target
                    with torch.no_grad():
                        next_observations = torch.tensor(next_observations, dtype=torch.float, device=self.train_device)
                        self.target_agent_group.reset().eval() # Reset Graph Builder intervals
                        ret_next = self.eval_agent_group.forward(next_observations, next_states, next_obs_padding_mask, next_alive_mask[:,-1,:], last_edge_indices)
                        q_val_next = ret_next['q_val']
                        if use_action_mask:
                            q_val_next = torch.masked_fill(q_val_next, ~next_avail_actions, -torch.inf)
                        q_val_next = q_val_next.max(dim=-1).values
                        next_states = torch.Tensor(next_states).to(self.train_device) # (B, T, F) -> (B, F) Take only the last state in the sequence
                        self.target_critic.eval()
                        ret_next = self.target_critic(q_val_next, next_states, next_alive_mask, next_obs_padding_mask[:,0,:])
                        q_tot_next = ret_next['q_tot']

                    # Compute the TD target
                    y_tot = rewards + (1 - terminations) * self.gamma * q_tot_next

                    # Compute the critic loss
                    critic_loss = torch.nn.functional.mse_loss(q_tot, y_tot.detach())
                    if self.use_data_parallel:
                        critic_loss = critic_loss.mean() # Reduce across all GPUs

                    # Optimize the critic network
                    self.eval_agent_group.zero_grad()
                    self.eval_critic.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.eval_critic.parameters(),
                        max_norm=5.0
                    )
                    self.optimizer.step()
                    self.eval_agent_group.step()

                    total_loss += critic_loss.detach().cpu().item()
                    total_batches += 1

                    pbar.update(bs)

        if self.use_data_parallel:
            self.eval_agent_group.unwrap_data_parallel()
            self.eval_critic = self.eval_critic.module
            self.target_agent_group.unwrap_data_parallel()
            self.target_critic = self.target_critic.module

        self.eval_agent_group.to("cpu")
        self.eval_critic.to("cpu")
        self.target_agent_group.to("cpu")
        self.target_critic.to("cpu")
        torch.cuda.empty_cache()

        return total_loss / total_batches