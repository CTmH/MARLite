import numpy as np
import torch
from torch.nn import DataParallel
from tqdm import tqdm

from marlite.trainer.trainer import Trainer
from marlite.util.trajectory_dataset import TrajectoryDataLoader

class QMIXTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
                    alive_mask = batch['alive_mask'].to(dtype=torch.bool)
                    observations = batch['observations'].to(dtype=torch.float32)
                    obs_padding_mask = batch['obs_padding_mask'].to(dtype=torch.bool)
                    states = batch['states'].to(dtype=torch.float32)
                    actions = batch['actions'].to(dtype=torch.int)
                    rewards = batch['rewards'].to(dtype=torch.float32)
                    next_states = batch['next_states'].to(dtype=torch.float32)
                    next_observations = batch['next_observations'].to(dtype=torch.float32)
                    next_obs_padding_mask = batch['next_obs_padding_mask'].to(dtype=torch.bool)
                    next_avail_actions = batch['next_avail_actions'] # Numpy array
                    next_alive_mask = batch['next_alive_mask'].to(dtype=torch.bool)
                    terminations = batch['terminations'].to(dtype=torch.bool)
                    bs = states.shape[0]  # Actual batch size
                    n_agents = rewards.shape[2]

                    # Create alive_mask_next from terminations and truncations
                    next_alive_mask = next_alive_mask.to(self.train_device)
                    alive_mask = alive_mask.to(self.train_device)
                    # Action mask: (B, N, T, Actions) -> (B, N, Actions)
                    if isinstance(next_avail_actions, torch.Tensor):
                        use_action_mask = True
                        next_avail_actions = next_avail_actions[:,-1,:,:] # (B, T, N, Actions) -> (B, N, Actions)
                        next_avail_actions = next_avail_actions.to(dtype=torch.bool, device=self.train_device)
                    else:
                        use_action_mask = False

                    rewards = rewards[:,-1] # (B, T, N) -> (B, N)
                    rewards = rewards.sum(dim=1).to(self.train_device) # (B, N) -> (B) Sum over all agents rewards
                    terminations = terminations[:,-1] # (B, T, N) -> (B, N)
                    terminations = terminations.prod(dim=1).to(self.train_device) # (B, N) -> (B) if all agents are terminated then game over

                    # obs_padding_mask = torch.tensor(obs_padding_mask, dtype=torch.bool) # (B, T) # REMOVED: already converted above
                    obs_padding_mask = torch.stack([obs_padding_mask] * n_agents, dim=1).to(self.train_device)  # (B, N, T)
                    # next_obs_padding_mask = torch.tensor(next_obs_padding_mask, dtype=torch.bool) # REMOVED: already converted above
                    next_obs_padding_mask = torch.stack([next_obs_padding_mask] * n_agents, dim=1).to(self.train_device)

                    # Compute the Q-tot
                    self.eval_agent_group.train()
                    observations = torch.transpose(observations, 1, 2).to(self.train_device) # obs.shape (B, T, N, F) -> (B, N, T, F)
                    ret = self.eval_agent_group.forward(observations, obs_padding_mask, alive_mask[:,-1,:])
                    q_val = ret['q_val']
                    actions = actions[:,-1].to(device=self.train_device, dtype=torch.int64) # (B, T, N) -> (B, N) # REMOVED torch.Tensor wrapper
                    q_val = torch.gather(q_val, dim=-1, index=actions.unsqueeze(-1)) # (B, N) -> (B, N, 1)
                    q_val = q_val.squeeze(-1) # (B, N, 1) -> (B, N)
                    states = states.to(self.train_device)
                    self.eval_critic.train()
                    ret = self.eval_critic(q_val, states, alive_mask, obs_padding_mask[:,0,:])
                    q_tot = ret['q_tot']

                    # Compute TD targets
                    with torch.no_grad():
                        self.target_agent_group.eval()
                        next_observations = torch.transpose(next_observations, 1, 2).to(self.train_device) # obs.shape (B, T, N, F) -> (B, N, T, F)
                        ret_next = self.target_agent_group.forward(next_observations, next_obs_padding_mask, next_alive_mask[:,-1,:])
                        q_val_next = ret_next['q_val']
                        if use_action_mask:
                            q_val_next = torch.masked_fill(q_val_next, ~next_avail_actions, -torch.inf)
                        q_val_next = q_val_next.max(dim=-1).values
                        next_states = next_states.to(self.train_device)
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