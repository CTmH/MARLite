from .learner import Learner
import torch

from ..algorithm.model import RNNModel

class QMixLearner(Learner):
    def __init__(self, 
                 agents, 
                 env_config, 
                 model_configs, 
                 traj_len: int, 
                 num_workers: int, 
                 buffer_capacity: int = 50000,
                 episode_limit: int = 500,
                 device: str = 'cpu'):
        super().__init__(agents, env_config, model_configs, traj_len, num_workers, buffer_capacity, episode_limit, device)

    def learn(self, batch_size: int, epochs: int):
        # Implement the learning logic for QMix
            
        # Get a batch of data from the replay buffer
        dataloader = self.replay_buffer.get_dataloader(batch_size=batch_size)
        for batch in dataloader:
            observations, states, actions, rewards, next_state, next_observations, terminations = self.__extract_batch(batch)
            # Compute the Q-tot
            q_val = [None for _ in range(len(self.agents))]
            for model_name, model in self.target_agent_group.models.items():
                selected_agents = self.target_agent_group.model_to_agents[model_name]
                idx = self.target_agent_group.model_to_agent_indices[model_name]
                # observation shape: (Batch Size, Time Step, Agent Number, Feature Dimensions) (B, T, N, F)
                obs = observations[:,:,idx]
                obs = torch.Tensor(obs)
                # (B, N, T, F) -> (B*N, T, F)
                obs = obs.reshape(obs.shape[0] * obs.shape[1], obs.shape[2], obs.shape[3])
                obs = obs.to(self.device)  # Convert to tensor and move to device
                model.train().to(self.device)
                if isinstance(model, RNNModel):
                    h = [model.init_hidden() for _ in range(obs.shape[0])]
                    h = torch.stack(h).to(self.device)
                    q_selected, _ = model(obs, h)
                # TODO: Add code for handling other types of models (e.g., CNNs)
                q_selected = q_selected.reshape(batch_size, len(self.agents), -1) # (B, N, Action Space)
                q_selected = q_selected.permute(1, 0, 2)  # (N, B, Action Space)
                for i, q in zip(idx, q_selected):
                    q_val[i] = q
            
            q_val = torch.stack(q_val) # (N, B, Action Space)
            q_val = q_val.permute(1, 0, 2)  # (B, N, Action Space)

            states = torch.Tensor(states[:,-1,:]) # (B, T, F) -> (B, F) Take only the last state in the sequence
            q_tot = self.target_critic(states, q_val)

            # Compute TD targets
            q_val = [None for _ in range(len(self.agents))]
            for model_name, model in self.target_agent_group.models.items():
                selected_agents = self.target_agent_group.model_to_agents[model_name]
                idx = self.target_agent_group.model_to_agent_indices[model_name]
                # observation shape: (Batch Size, Time Step, Agent Number, Feature Dimensions) (B, T, N, F)
                obs = next_observations[:,:,idx]
                obs = torch.Tensor(obs)
                # (B, N, T, F) -> (B*N, T, F)
                obs = obs.reshape(obs.shape[0] * obs.shape[1], obs.shape[2], obs.shape[3])
                obs = obs.to(self.device)  # Convert to tensor and move to device
                model.train().to(self.device)
                if isinstance(model, RNNModel):
                    h = [model.init_hidden() for _ in range(obs.shape[0])]
                    h = torch.stack(h).to(self.device)
                    q_selected, _ = model(obs, h)
                # TODO: Add code for handling other types of models (e.g., CNNs)
                q_selected = q_selected.reshape(batch_size, len(self.agents), -1) # (B, N, Action Space)
                q_selected = q_selected.permute(1, 0, 2)  # (N, B, Action Space)
                for i, q in zip(idx, q_selected):
                    q_val[i] = q

            # Compute the target Q-values using the critic network
            next_action = self.agents.select_action(next_state, explore=False)
            q_tot_next = self.critic(next_state, next_action)
            q_tot_next = reward + (1 - terminations) * self.gamma * q_tot_next

            # Compute the critic loss
            critic_loss = torch.nn.functional.mse_loss(current_q_values, q_tot_next)
                
            # Optimize the critic network
            self.optimizer.zero_grad()
            critic_loss.backward()
            self.optimizer.step()

            # Update the agents' policies using the actor network
            for agent in self.agents:
                agent.update_policy(state)