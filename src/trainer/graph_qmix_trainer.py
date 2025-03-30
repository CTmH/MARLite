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
        self.best_agent_comm_model_params = self.eval_agent_group.get_comm_model_params()
        self.target_agent_group.set_comm_model_params(self.best_agent_comm_model_params)


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

                    # Compute TD targets
                    self.target_agent_group.train().to(self.train_device) # cudnn RNN backward can only be called in training mode
                    q_val_next = self.target_agent_group.forward(next_observations, edge_index_batch_next)

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
    
    def save_current_model(self, checkpoint: str):
        agent_model_params, agent_feature_extractor_params = self.eval_agent_group.get_model_params()
        agent_comm_model_params = self.eval_agent_group.get_comm_model_params()
        critic_params = self.eval_critic.state_dict()
        self.save_params(checkpoint,
                         agent_model_params,
                         agent_feature_extractor_params,
                         agent_comm_model_params,
                         critic_params)
        return self
    
    def save_best_model(self):
        self.save_params('best_model',
                         self.best_agent_model_params,
                         self.best_agent_fe_params,
                         self.best_agent_comm_model_params,
                         self.best_critic_params)
        return self

    def save_params(self,
                    checkpoint: str,
                    agent_model_params: dict,
                    agent_feature_extractor_params: dict,
                    agent_comm_model_params: dict,
                    critic_params: dict):
        # Agent models and feature extractors
        for model_name, params in agent_model_params.items():
            path = os.path.join(self.agentsdir, model_name, 'model')
            os.makedirs(path, exist_ok=True)
            model_path = os.path.join(path, f'{checkpoint}.pth')
            torch.save(params, model_path)
            logging.info(f"Actor {model_name} saved to {model_path}")

        for model_name, params in agent_feature_extractor_params.items():
            path = os.path.join(self.agentsdir, model_name, 'feature_extractor')
            os.makedirs(path, exist_ok=True)
            model_path = os.path.join(path, f'{checkpoint}.pth')
            torch.save(params, model_path)
            logging.info(f"{model_name}'s feature extractor saved to {model_path}")

        # Communication model params
        path = os.path.join(self.agentsdir, 'comm_model')
        os.makedirs(path, exist_ok=True)
        comm_model_path = os.path.join(path, f'{checkpoint}.pth')
        torch.save(agent_comm_model_params, comm_model_path)
        logging.info(f"Communication model saved to {comm_model_path}")

        # Critic params
        path = os.path.join(self.criticdir, 'model')
        os.makedirs(path, exist_ok=True)
        critic_path = os.path.join(path, f'{checkpoint}.pth')
        torch.save(critic_params, critic_path)
        logging.info(f"Critic model saved to {critic_path}")

    def load_model(self, checkpoint: str):
        # Agent models and feature extractors
        agent_model_params = {}
        agent_feature_extractor_params = {}
        for model_name in self.eval_agent_group.models.keys():
            model_path = os.path.join(self.agentsdir, model_name, 'model', f'{checkpoint}.pth')
            if os.path.exists(model_path):
                params = torch.load(model_path, weights_only=True)
                agent_model_params[model_name] = params
                logging.info(f"Actor {model_name} loaded from {model_path}")
            else:
                logging.warning(f"Model path for actor {model_name} does not exist: {model_path}")
                raise FileNotFoundError(f"Model path for actor {model_name} does not exist: {model_path}")

            fe_path = os.path.join(self.agentsdir, model_name, 'feature_extractor', f'{checkpoint}.pth')
            if os.path.exists(fe_path):
                params = torch.load(fe_path, weights_only=True)
                agent_feature_extractor_params[model_name] = params
                logging.info(f"{model_name}'s feature extractor loaded from {fe_path}")
            else:
                logging.warning(f"Feature extractor path for actor {model_name} does not exist: {fe_path}")
                raise FileNotFoundError(f"Feature extractor path for actor {model_name} does not exist: {fe_path}")
        self.eval_agent_group.set_model_params(agent_model_params, agent_feature_extractor_params)
        logging.info("All actor models and feature extractors loaded successfully.")

        # Communication model params
        comm_model_path = os.path.join(self.agentsdir, 'comm_model', f'{checkpoint}.pth')
        if os.path.exists(comm_model_path):
            params = torch.load(comm_model_path, weights_only=True)
            self.eval_agent_group.set_comm_model_params(params)
            logging.info(f"Communication model loaded from {comm_model_path}")
        else:
            logging.warning(f"Communication model path does not exist: {comm_model_path}")
            raise FileNotFoundError(f"Communication model path does not exist: {comm_model_path}")

        # Critic params
        critic_path = os.path.join(self.criticdir, 'model', f'{checkpoint}.pth')
        if os.path.exists(critic_path):
            self.eval_critic.load_state_dict(torch.load(critic_path, weights_only=True))
            logging.info(f"Critic model loaded from {critic_path}")
        else:
            logging.warning(f"Critic model path does not exist: {critic_path}")
            raise FileNotFoundError(f"Critic model path does not exist: {critic_path}")

        self.update_target_model_params()

        return self

    def update_target_model_params(self):
        # Update the evaluation models with the latest weights from the training models
        agent_model_params, agent_fe_params = self.eval_agent_group.get_model_params()
        self.target_agent_group.set_model_params(agent_model_params, agent_fe_params)
        agent_comm_model_params = self.eval_agent_group.get_comm_model_params()
        self.target_agent_group.set_comm_model_params(agent_comm_model_params)
        critic_params = deepcopy(self.eval_critic.state_dict())  # Update critic parameters
        self.target_critic.load_state_dict(critic_params)
        return self

    def update_best_params(self):
        self.best_agent_model_params, self.best_agent_fe_params = self.eval_agent_group.get_model_params()
        self.best_comm_model_params = self.eval_agent_group.get_comm_model_params()  # Update communication model parameters
        self.best_critic_params = deepcopy(self.eval_critic.state_dict())  # Update critic parameters
        return self