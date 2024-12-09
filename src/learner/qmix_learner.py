from .learner import Learner
import torch

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
            # Extract necessary components from the trajectory
            observations = [traj['observations'] for traj in batch]
            states = [traj['states'] for traj in batch]
            actions = [traj['actions'] for traj in batch]
            rewards = [traj['rewards'] for traj in batch]
            next_state = [traj['next_states'][-1] for traj in batch] # Only need the next state from the last step of each trajectory
            terminations = [traj['terminations'] for traj in batch]

            
            # Convert to tensors and move to device
            state = torch.tensor(state).to(self.device)
            action = torch.tensor(action).to(self.device)
            reward = torch.tensor(reward).to(self.device)
            next_state = torch.tensor(next_state).to(self.device)
            terminations = torch.tensor(terminations).to(self.device)

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