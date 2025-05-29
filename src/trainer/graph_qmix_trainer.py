import os
import logging
import torch
from copy import deepcopy
from torch.utils.data import DataLoader
from tqdm import tqdm

from .trainer import Trainer
from ..algorithm.model import GRUModel
from ..algorithm.agents import QMIXAgentGroup
from ..algorithm.critic.qmix_critic_model import QMIXCriticModel
from ..util.trajectory_dataset import TrajectoryDataLoader
from ..util.scheduler import Scheduler

class GraphQMIXTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Agent group

    def learn(self, sample_size, batch_size: int, times: int = 1):
        total_loss = 0.0
        total_batches = 0

        for t in range(times):
            with tqdm(total=sample_size, desc=f'Times {t+1}/{times}', unit='batch') as pbar:
                # Implement the learning logic for QMix
                # Get a batch of data from the replay buffer
                dataset = self.replaybuffer.sample(sample_size)
                dataloader = TrajectoryDataLoader(dataset, batch_size=batch_size, shuffle=True)
                for batch in dataloader:
                    # Extract batch data
                    observations, states, actions, rewards, next_states, next_observations, terminations = batch
                    bs = states.shape[0]  # Actual batch size

                    # Build the graph for the current state
                    states = states[:,-1,:] # (B, T, F) -> (B, F) Take only the last state in the sequence
                    _, edge_index_batch = self.env.build_my_team_graph_batch(states)

                    # Compute the Q-tot
                    self.eval_agent_group.train().to(self.train_device)
                    q_val = self.eval_agent_group.forward(observations, edge_index_batch)

                    states = torch.Tensor(states).to(self.train_device)
                    self.eval_critic.train().to(self.train_device)
                    q_tot = self.eval_critic(q_val, states)

                    # Build the graph for the next state
                    next_states = next_states[:,-1,:] # (B, T, F) -> (B, F) Take only the last state in the sequence
                    _, edge_index_batch_next = self.env.build_my_team_graph_batch(next_states)

                    # Double Q-learning, we use eval agent group to choose actions,and use target critic to compute q_target
                    with torch.no_grad():
                        self.target_agent_group.eval().to(self.train_device)
                        q_val_next = self.eval_agent_group.forward(next_observations, edge_index_batch_next)
                        next_states = torch.Tensor(next_states).to(self.train_device) # (B, T, F) -> (B, F) Take only the last state in the sequence
                        self.target_critic.eval().to(self.train_device) 
                        q_tot_next = self.target_critic(q_val_next, next_states)

                    # Compute the TD target
                    rewards = torch.Tensor(rewards[:,:,-1]).to(self.train_device) # (B, N, T) -> (B, N)
                    rewards = rewards.sum(dim=1) # (B, N) -> (B) Sum over all agents rewards
                    terminations = torch.Tensor(terminations[:,:,-1]).to(self.train_device) # (B, N, T) -> (B, N)
                    terminations = terminations.prod(dim=1) # (B, N) -> (B) if all agents are terminated then game over

                    y_tot = rewards + (1 - terminations) * self.gamma * q_tot_next

                    # Compute the critic loss
                    critic_loss = torch.nn.functional.mse_loss(q_tot, y_tot.detach())
                        
                    # Optimize the critic network
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.eval_critic.parameters(), 
                        max_norm=5.0
                    )
                    self.optimizer.step()
                    self.eval_agent_group.step()

                    self.eval_agent_group.zero_grad()
                    self.eval_critic.zero_grad()
                    
                    total_loss += critic_loss.detach().cpu().item()
                    total_batches += 1

                    pbar.update(bs)
            
        return total_loss / total_batches