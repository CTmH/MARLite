import torch
from copy import deepcopy
from torch.utils.data import DataLoader

from .learner import Learner
from ..algorithm.model import RNNModel
from ..algorithm.agents import QMIXAgentGroup
from ..algorithm.critic.qmix_critic import QMIXCritic

class QMIXLearner(Learner):
    def __init__(self, 
                 agents, 
                 env_config, 
                 model_configs,
                 critic_config,
                 traj_len: int, 
                 num_workers: int, 
                 buffer_capacity: int = 50000,
                 episode_limit: int = 500,
                 device: str = 'cpu'):
        super().__init__(agents, env_config, model_configs, critic_config, traj_len, num_workers, buffer_capacity, episode_limit, device)
        self.target_agent_group = QMIXAgentGroup(agents=self.agents, model_configs=self.model_configs, device=self.device)
        self.eval_agent_group = deepcopy(self.target_agent_group)
        self.target_critic = QMIXCritic(critic_config['state_shape'], critic_config['n_agents'], critic_config['qmix_hidden_dim'], critic_config['hyper_hidden_dim'])
        self.eval_critic = deepcopy(self.target_critic)
        self.optimizer = torch.optim.Adam(self.target_critic.parameters(), lr=0.001)

    def learn(self, sample_size, batch_size: int, epochs: int):
        for epoch in range(epochs):
            # Implement the learning logic for QMix
            # Get a batch of data from the replay buffer
            dataset = self.replay_buffer.sample(sample_size)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for batch in dataloader:
                observations, states, actions, rewards, next_state, next_observations, terminations = self.extract_batch(batch)
                # Compute the Q-tot
                q_val = [None for _ in range(len(self.agents))]
                for model_name, model in self.target_agent_group.models.items():
                    selected_agents = self.target_agent_group.model_to_agents[model_name]
                    idx = self.target_agent_group.model_to_agent_indices[model_name]
                    # observation shape: (Batch Size, Agent Number, Time Step, Feature Dimensions) (B, N, T, F)
                    obs = observations[:,idx]
                    obs = torch.Tensor(obs)
                    # (B, N, T, F) -> (B*N, T, F)
                    obs = obs.reshape(obs.shape[0] * obs.shape[1], obs.shape[2], obs.shape[3])
                    obs = obs.to(self.device)  # Convert to tensor and move to device
                    model.train().to(self.device)
                    if isinstance(model, RNNModel):
                        h = [model.init_hidden() for _ in range(obs.shape[0])]
                        h = torch.stack(h).to(self.device)
                        h = h.permute(1, 0, 2)
                        model.train().to(self.device)
                        q_selected, _ = model(obs, h)
                        q_selected = q_selected[:,-1,:] # get the last output 
                    # TODO: Add code for handling other types of models (e.g., CNNs)
                    q_selected = q_selected.reshape(batch_size, len(selected_agents), -1) # (B, N, Action Space)
                    q_selected = q_selected.permute(1, 0, 2)  # (N, B, Action Space)
                    for i, q in zip(idx, q_selected):
                        q_val[i] = q
                
                q_val = torch.stack(q_val) # (N, B, Action Space)
                q_val = q_val.permute(1, 0, 2)  # (B, N, Action Space)

                states = torch.Tensor(states[:,-1,:]) # (B, T, F) -> (B, F) Take only the last state in the sequence
                self.target_critic.train().to(self.device)
                q_tot = self.target_critic(q_val, states)

                # Compute TD targets
                q_val = [None for _ in range(len(self.agents))]
                for model_name, model in self.target_agent_group.models.items():
                    selected_agents = self.target_agent_group.model_to_agents[model_name]
                    idx = self.target_agent_group.model_to_agent_indices[model_name]
                    # observation shape: (Batch Size, Agent Number, Time Step, Feature Dimensions) (B, N, T, F)
                    obs = next_observations[:,idx]
                    obs = torch.Tensor(obs)
                    # (B, N, T, F) -> (B*N, T, F)
                    obs = obs.reshape(obs.shape[0] * obs.shape[1], obs.shape[2], obs.shape[3])
                    obs = obs.to(self.device)  # Convert to tensor and move to device
                    model.train().to(self.device)
                    if isinstance(model, RNNModel):
                        h = [model.init_hidden() for _ in range(obs.shape[0])]
                        h = torch.stack(h).to(self.device)
                        h = h.permute(1, 0, 2)
                        model.eval().to(self.device)
                        q_selected, _ = model(obs, h)
                        q_selected = q_selected[:,-1,:] # get the last output 
                    # TODO: Add code for handling other types of models (e.g., CNNs)
                    q_selected = q_selected.reshape(batch_size, len(self.agent_config), -1) # (B, N, Action Space)
                    q_selected = q_selected.permute(1, 0, 2)  # (N, B, Action Space)
                    for i, q in zip(idx, q_selected):
                        q_val[i] = q

                q_val = torch.stack(q_val) # (N, B, Action Space)
                q_val = q_val.permute(1, 0, 2)  # (B, N, Action Space)

                next_state = torch.Tensor(next_state[:,-1,:]) # (B, T, F) -> (B, F) Take only the last state in the sequence
                self.eval_critic.eval().to(self.device)
                q_tot_next = self.eval_critic(q_val, next_state)

                # Compute the TD target
                rewards = torch.Tensor(rewards[:,:,-1]) # (B, N, T) -> (B, N)
                rewards = rewards.sum(dim=1) # (B, N) -> (B) Sum over all agents rewards
                terminations = torch.Tensor(terminations[:,:,-1]) # (B, N, T) -> (B, N)
                terminations = terminations.prod(dim=1) # (B, N) -> (B) if all agents are terminated then game over

                q_tot_next = rewards + (1 - terminations) * self.gamma * q_tot_next

                # Compute the critic loss
                critic_loss = torch.nn.functional.mse_loss(q_tot, q_tot_next)
                    
                # Optimize the critic network
                self.target_agent_group.zero_grad()
                self.target_critic.zero_grad()
                critic_loss.backward()
                self.target_agent_group.step()
                self.optimizer.step()
            
        return self