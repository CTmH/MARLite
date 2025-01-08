import torch
from copy import deepcopy
from torch.utils.data import DataLoader
from tqdm import tqdm

from .trainer import Trainer
from ..algorithm.model import RNNModel
from ..algorithm.agents import QMIXAgentGroup
from ..algorithm.critic.qmix_critic_model import QMIXCriticModel
from ..util.trajectory_dataset import TrajectoryDataLoader
from ..util.scheduler import Scheduler

class QMIXTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
                    # Compute the Q-tot
                    q_val = [None for _ in range(len(self.agents))]
                    self.eval_agent_group.train().to(self.train_device)
                    for (model_name, model), (_, fe) in zip(self.eval_agent_group.models.items(), 
                                                            self.eval_agent_group.feature_extractors.items()):
                        selected_agents = self.eval_agent_group.model_to_agents[model_name]
                        idx = self.eval_agent_group.model_to_agent_indices[model_name]
                        # observation shape: (Batch Size, Agent Number, Time Step, Feature Dimensions) (B, N, T, F)
                        obs = observations[:,idx]
                        obs = torch.Tensor(obs)
                        # (B, N, T, (obs_shape)) -> (B*N*T, (obs_shape))
                        bs = obs.shape[0]
                        n_agents = len(selected_agents)
                        ts = obs.shape[2]
                        obs_shape = list(obs.shape[3:])
                        obs = obs.reshape(bs*n_agents*ts, *obs_shape)
                        obs = obs.to(self.train_device)  # Convert to tensor and move to device
                        obs_vectorized = fe(obs) # (B*N*T, (obs_shape)) -> (B*N*T, F)
                        obs_vectorized = obs_vectorized.reshape(bs*n_agents, ts, -1) # (B*N*T, F) -> (B*N, T, F)
                        if isinstance(model, RNNModel):
                            h = [model.init_hidden() for _ in range(obs_vectorized.shape[0])]
                            h = torch.stack(h).to(self.train_device)
                            h = h.permute(1, 0, 2)
                            q_selected, _ = model(obs_vectorized, h)
                            q_selected = q_selected[:,-1,:] # get the last output 
                        # TODO: Add code for handling other types of models (e.g., CNNs)
                        q_selected = q_selected.reshape(bs, n_agents, -1) # (B, N, Action Space)
                        q_selected = q_selected.permute(1, 0, 2)  # (N, B, Action Space)
                        for i, q in zip(idx, q_selected):
                            q_val[i] = q
                    
                    q_val = torch.stack(q_val).to(self.train_device) # (N, B, Action Space)
                    q_val = q_val.permute(1, 0, 2)  # (B, N, Action Space)

                    states = torch.Tensor(states[:,-1,:]).to(self.train_device) # (B, T, F) -> (B, F) Take only the last state in the sequence
                    self.eval_critic.train().to(self.train_device)
                    q_tot = self.eval_critic(q_val, states)

                    # Compute TD targets
                    q_val = [None for _ in range(len(self.agents))]
                    self.target_agent_group.train().to(self.train_device) # cudnn RNN backward can only be called in training mode
                    for (model_name, model), (_, fe) in zip(self.target_agent_group.models.items(), 
                                                            self.target_agent_group.feature_extractors.items()):
                        selected_agents = self.target_agent_group.model_to_agents[model_name]
                        idx = self.target_agent_group.model_to_agent_indices[model_name]
                        # observation shape: (Batch Size, Agent Number, Time Step, Feature Dimensions) (B, N, T, F)
                        obs = next_observations[:,idx]
                        obs = torch.Tensor(obs)
                        # (B, N, T, (obs_shape)) -> (B*N*T, (obs_shape))
                        bs = obs.shape[0]
                        n_agents = len(selected_agents)
                        ts = obs.shape[2]
                        obs_shape = list(obs.shape[3:])
                        obs = obs.reshape(bs*n_agents*ts, *obs_shape)
                        obs = obs.to(self.train_device)  # Convert to tensor and move to device
                        obs_vectorized = fe(obs) # (B*N*T, (obs_shape)) -> (B*N*T, F)
                        obs_vectorized = obs_vectorized.reshape(bs*n_agents, ts, -1) # (B*N*T, F) -> (B*N, T, F)
                        if isinstance(model, RNNModel):
                            h = [model.init_hidden() for _ in range(obs_vectorized.shape[0])]
                            h = torch.stack(h).to(self.train_device)
                            h = h.permute(1, 0, 2)
                            q_selected, _ = model(obs_vectorized, h)
                            q_selected = q_selected[:,-1,:] # get the last output 
                        # TODO: Add code for handling other types of models (e.g., CNNs)
                        q_selected = q_selected.reshape(bs, len(selected_agents), -1) # (B, N, Action Space)
                        q_selected = q_selected.permute(1, 0, 2)  # (N, B, Action Space)
                        for i, q in zip(idx, q_selected):
                            q_val[i] = q

                    q_val = torch.stack(q_val).to(self.train_device) # (N, B, Action Space)
                    q_val = q_val.permute(1, 0, 2)  # (B, N, Action Space)

                    next_states = torch.Tensor(next_states[:,-1,:]).to(self.train_device) # (B, T, F) -> (B, F) Take only the last state in the sequence
                    self.target_critic.train().to(self.train_device) # cudnn RNN backward can only be called in training mode
                    q_tot_next = self.target_critic(q_val, next_states)

                    # Compute the TD target
                    rewards = torch.Tensor(rewards[:,:,-1]).to(self.train_device) # (B, N, T) -> (B, N)
                    rewards = rewards.sum(dim=1) # (B, N) -> (B) Sum over all agents rewards
                    terminations = torch.Tensor(terminations[:,:,-1]).to(self.train_device) # (B, N, T) -> (B, N)
                    terminations = terminations.prod(dim=1) # (B, N) -> (B) if all agents are terminated then game over

                    y_tot = rewards + (1 - terminations) * self.gamma * q_tot_next

                    # Compute the critic loss
                    critic_loss = torch.nn.functional.mse_loss(q_tot, y_tot)
                        
                    # Optimize the critic network
                    self.eval_agent_group.zero_grad()
                    self.eval_critic.zero_grad()
                    critic_loss.backward()
                    self.optimizer.step()
                    self.eval_agent_group.step()

                    total_loss += critic_loss.item()
                    total_batches += 1

                    pbar.update(bs)
            
        return total_loss / total_batches