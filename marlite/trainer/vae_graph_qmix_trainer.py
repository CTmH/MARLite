import torch
import numpy as np
from typing import Callable
from torch.nn import DataParallel
from tqdm import tqdm
from copy import deepcopy

from marlite.algorithm.model import ModelConfig
from marlite.trainer.semi_supervised_qmix_trainer import SemiSupervisedQMIXTrainer
from marlite.util.trajectory_dataset import TrajectoryDataLoader
from marlite.util.self_supervised_data_constructor.self_supervised_data_constructor_config import SelfSupervisedDataConstructorConfig


class VAEGraphQMIXTrainer(SemiSupervisedQMIXTrainer):
    """
    A GraphQMIXTrainer subclass that works with probabilistic agent groups.
    This trainer handles ProbObsGNNCommAgentGroup, ProbSeqGNNCommAgentGroup,
    DualPathObsGNNCommAgentGroup, and DualPathProbObsGNNCommAgentGroup.
    It optimizes the parameters of the independent multivariate Gaussian distributions
    output by probabilistic agent groups using a VAE decoder.
    """

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

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

    def _compute_vae_loss(self,
                         mu,
                         std,
                         observations,
                         states,
                         edge_indices,
                         alive_mask):
        """
        Internal function to compute VAE loss for probabilistic agent groups.

        Args:
            mu: Mean of the Gaussian distribution (batch_size, n_agents, feature_dim)
            std: Standard deviation of the Gaussian distribution (batch_size, n_agents, feature_dim)
            observations: Actual observations from the environment
            states: Environment states
            edge_indices: Communication graph edge indices
            batch_size: Size of the batch
            n_agents: Number of agents

        Returns:
            VAE reconstruction loss
        """
        if self.eval_decoder is None or self.data_constructor is None or self.reconstruction_loss_fn is None:
            # If decoder components are not provided, return 0 loss
            return torch.tensor(0.0, device=mu.device)

        # Sample from the Gaussian distribution to get latent representations
        eps = torch.randn_like(std)
        z = mu + eps * std  # Reparameterization trick (batch_size, n_agents, feature_dim)

        # Use edge_indices to find communication partners for each agent
        # Decode the latent representations to reconstruct observations
        reconstructed_obs = self.eval_decoder(z)  # (batch_size, n_agents, feature_dim)

        # Format the original observations for comparison
        formatted_obs = self.data_constructor.process(observations, states, edge_indices, alive_mask)  # (batch_size, n_agents, feature_dim)

        # Calculate reconstruction loss
        reconstruction_loss = self.reconstruction_loss_fn(reconstructed_obs, formatted_obs)

        return reconstruction_loss

    def learn(self, sample_size, batch_size: int, times: int = 1):
        total_loss = 0.0
        total_batches = 0

        # Move models to the appropriate device before wrapping with DataParallel
        self.eval_agent_group.to(self.train_device)
        self.eval_critic.to(self.train_device)
        self.target_agent_group.to(self.train_device)
        self.target_critic.to(self.train_device)

        # Also move decoder to the appropriate device if it exists
        if self.eval_decoder is not None:
            self.eval_decoder.to(self.train_device)

        if self.use_data_parallel:
            self.eval_agent_group.wrap_data_parallel()
            self.eval_critic = DataParallel(self.eval_critic)
            self.target_agent_group.wrap_data_parallel()
            self.target_critic = DataParallel(self.target_critic)
            if self.eval_decoder is not None:
                self.eval_decoder = DataParallel(self.eval_decoder)

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
                    edge_indices = [edge_indices[i][-1] for i in range(bs)] # (B, T, 2, N) -> (B, 2, N) Take only the last edge indices
                    observations = torch.tensor(observations, dtype=torch.float, device=self.train_device)
                    self.eval_agent_group.reset().train() # Reset Graph Builder intervals
                    ret = self.eval_agent_group.forward(observations, states, obs_padding_mask, alive_mask[:,-1,:], edge_indices) # obs.shape (B, N, T, F)
                    q_val = ret['q_val']
                    mu = ret['mu']
                    std = ret['std']

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
                        ret_next = self.eval_agent_group.forward(next_observations, next_states, next_obs_padding_mask, next_alive_mask[:,-1,:], edge_indices)
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

                    # Compute VAE loss
                    vae_loss = self._compute_vae_loss(
                        mu, std, observations, states, edge_indices, alive_mask
                    )

                    # Total loss is the sum of critic loss and VAE loss
                    total_batch_loss = critic_loss + self.self_supervised_learning_loss_weight * vae_loss

                    if self.use_data_parallel:
                        total_batch_loss = total_batch_loss.mean() # Reduce across all GPUs

                    # Optimize the networks - only use the main optimizer since it includes both critic and decoder params
                    self.eval_agent_group.zero_grad()
                    self.eval_critic.zero_grad()
                    self.eval_decoder.zero_grad()

                    total_batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self.eval_critic.parameters()) +
                        list(self.eval_agent_group.parameters()) +
                        (list(self.eval_decoder.parameters()) if self.eval_decoder is not None else []),
                        max_norm=5.0
                    )
                    self.optimizer.step()
                    self.eval_agent_group.step()

                    total_loss += total_batch_loss.detach().cpu().item()
                    total_batches += 1

                    pbar.update(bs)

        if self.use_data_parallel:
            self.eval_agent_group.unwrap_data_parallel()
            self.eval_critic = self.eval_critic.module
            self.target_agent_group.unwrap_data_parallel()
            self.target_critic = self.target_critic.module
            if self.eval_decoder is not None:
                self.eval_decoder = self.eval_decoder.module

        self.eval_agent_group.to("cpu")
        self.eval_critic.to("cpu")
        self.target_agent_group.to("cpu")
        self.target_critic.to("cpu")
        if self.eval_decoder is not None:
            self.eval_decoder.to("cpu")
        torch.cuda.empty_cache()

        return total_loss / total_batches