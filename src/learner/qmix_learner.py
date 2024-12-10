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
            q_val = [None for _ in range(len(self.agents))]
            for model_name, model in self.target_agent_group.models.items():
                selected_agents = self.target_agent_group.model_to_agents[model_name]
                idx = self.target_agent_group.model_to_agent_indices[model_name]
                # observation shape: (Batch Size, Time Step, Agent Number, Feature Dimensions) (B, T, N, F)
                obs = observations[:,:,idx]
                # (B, N, T, F) -> (B*N, T, F)
                obs = obs.reshape(obs.shape[0] * obs.shape[1], obs.shape[2], obs.shape[3])
                obs = torch.Tensor(obs).to(self.device)  # Convert to tensor and move to device
                model.train().to(self.device)
                if isinstance(model, RNNModel):
                    h = [model.init_hidden() for _ in range(obs.shape[0])]
                    h = torch.stack(h).to(self.device)
                    q_selected = model(obs, h)
                # TODO: Add code for handling other types of models (e.g., CNNs)
                for i, q in zip(idx, q_selected):
                    q_val[i] = q


                next_obs = next_observations[:,:,idx]
                act = actions[:,:,idx]
                r = rewards[:,:,idx]

                # Reshape observations and actions to match the model's input shape
                obs = torch.Tensor(obs).to(self.device)
                obs = obs.reshape()
                q = model(obs)
                
            
                # Convert to tensors and move to device

            # Compute the target Q-values using the critic network
            with torch.no_grad():
                next_action = self.agents.select_action(next_state, explore=False)
                target_q_values = self.critic(next_state, next_action)
                target_q_values = reward + (1 - terminations) * self.gamma * target_q_values

            # Compute the current Q-values using the critic network
            current_q_values = self.critic(state, action)

            # Compute the critic loss
            critic_loss = torch.nn.functional.mse_loss(current_q_values, target_q_values)
                
            # Optimize the critic network
            self.optimizer.zero_grad()
            critic_loss.backward()
            self.optimizer.step()

            # Update the agents' policies using the actor network
            for agent in self.agents:
                agent.update_policy(state)