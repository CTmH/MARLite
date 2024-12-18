import os
import numpy as np
import torch
from typing import Dict
from copy import deepcopy
from multiprocessing import Process, Queue, Pool
import time
import logging
import csv

from ..algorithm.agents import AgentGroup
from ..environment.env_config import EnvConfig
from ..algorithm.model import ModelConfig
from ..rolloutworker.rolloutworker import RolloutWorker
from ..util.replay_buffer import ReplayBuffer
from ..util.scheduler import Scheduler

def get_episode(worker, episode_limit, epsilon):
    return worker.generate_episode(episode_limit, epsilon)

class Learner():
    def __init__(self, 
                 agents: Dict[str, str], 
                 env_config: EnvConfig, 
                 model_configs: ModelConfig,
                 epsilon_scheduler: Scheduler,
                 sample_ratio_scheduler: Scheduler,
                 critic_config: Dict[str, any],
                 traj_len: int, 
                 n_workers: int, 
                 epochs = 10000,
                 buffer_capacity: int = 50000,
                 episode_limit: int = 500,
                 n_episodes: int = 1000,
                 gamma: float = 0.9,
                 critic_lr: float = 0.01,
                 critic_optimizer = torch.optim.Adam,
                 workdir: str = "",
                 device: str = 'cpu'):
        
        self.env_config = env_config
        self.model_configs = model_configs
        self.traj_len = traj_len
        self.n_workers = n_workers
        self.episode_limit = episode_limit
        self.epochs = epochs
        self.sample_ratio = sample_ratio_scheduler
        self.device = device
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity, traj_len=self.traj_len)
        self.agents = agents
        self.target_agent_group = AgentGroup(agents=self.agents, model_configs=self.model_configs, device=self.device)
        # Set the same parameters for evaluation agents as target agents.
        self.target_models_params = self.target_agent_group.get_model_params()
        self.eval_agent_group = deepcopy(self.target_agent_group)  # Deep copy the target agent group to create evaluation agents.
        # Critic
        self.target_critic = torch.nn.Module()
        self.eval_critic = torch.nn.Module()
        self.critic_params = deepcopy(self.target_critic.state_dict())
        self.optimizer = None
        self.epsilon = epsilon_scheduler
        self.gamma = gamma
        self.n_episodes = n_episodes
        # Work directory
        self.workdir = workdir
        os.makedirs(self.workdir, exist_ok=True)
        self.logdir = os.path.join(workdir, 'logs')
        self.modeldir = os.path.join(workdir, 'models')
        self.results = {}  # Dictionary to save intermediate results
        os.makedirs(self.logdir, exist_ok=True)
        os.makedirs(self.modeldir, exist_ok=True)
        self.log = logging.basicConfig(level=logging.INFO,
                                       format='%(asctime)s - %(levelname)s - %(message)s',
                                       handlers=[
                                                 logging.StreamHandler(),
                                                 #logging.FileHandler(os.path.join(self.logdir, "training.log"))
                                                 ])

    def learn(self, sample_size, batch_size: int, times: int):
        raise NotImplementedError
    
    def save_model(self, filename: str):
        for model_name, params in self.target_agent_group.get_model_params().items():
            actor_path = os.path.join(self.modeldir, f'actor_{model_name}_{filename}.pth')
            torch.save(params, actor_path)
            logging.info(f"Actor model {model_name} saved to {actor_path}")

        critic_path = os.path.join(self.modeldir, f'critic_{filename}.pth')
        torch.save(self.target_critic.state_dict(), critic_path)
        logging.info(f"Critic model saved to {critic_path}")
    
    def load_model(self, filename: str):
        actor_params = {}
        for model_name in self.target_agent_group.models.keys():
            actor_path = os.path.join(self.modeldir, f'actor_{model_name}_{filename}.pth')
            params = torch.load(actor_path)
            actor_params[model_name] = params
            logging.info(f"Actor model {model_name} loaded from {actor_path}")
        
        self.target_agent_group.set_model_params(actor_params)
        self.eval_agent_group.set_model_params(actor_params)
        logging.info(f"All actor model set")

        critic_path = os.path.join(self.modeldir, f'critic_{filename}.pth')
        self.target_critic.load_state_dict(torch.load(critic_path))
        self.eval_critic.load_state_dict(torch.load(critic_path))
        logging.info(f"Critic model loaded from {critic_path}")

    def collect_experience(self, n_episodes: int, episode_limit: int, epsilon: int):
        """
        Collect experiences using multiple rollout workers.
        """
        job = RolloutWorker(env_config=self.env_config,
                            agent_group=self.eval_agent_group,
                            rnn_traj_len=self.traj_len)
        
        logging.info(f"Collecting {n_episodes} episodes with limit {episode_limit} and epsilon {epsilon}")
        episodes = [job.generate_episode(episode_limit, epsilon) for _ in range(n_episodes)]

        for episode in episodes:
            self.replay_buffer.add_episode(episode)
        
        return self

    def update_params(self):
        # Update the evaluation models with the latest weights from the training models
        self.target_models_params = self.target_agent_group.get_model_params()
        self.eval_agent_group.set_model_params(self.target_models_params)
        self.critic_params = deepcopy(self.target_critic.state_dict())  # Update critic parameters
        self.eval_critic.load_state_dict(self.critic_params)

    def evaluate(self, times=100):
        job = RolloutWorker(env_config=self.env_config,
                                 agent_group=self.eval_agent_group,
                                 rnn_traj_len=self.traj_len)
        
        logging.info(f"Evaluating for {times} episodes")
        episodes = [job.generate_episode(self.episode_limit, epsilon=0.0) for _ in range(times)]
        rewards = np.array([episode['episode_reward'] for episode in episodes])
        
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        logging.info(f"Evaluation results: Mean reward {mean_reward}, Std reward {std_reward}")
        
        return mean_reward, std_reward
    
    def train(self, target_reward, eval_interval=500, batch_size=64, learning_times_per_epoch=1):
        best_mean_reward = -np.inf
        best_reward_std = np.inf
        n_episodes = self.n_episodes

        # Training loop
        for epoch in range(self.epochs):

            logging.info(f"Epoch {epoch}: Collecting experiences")
            self.collect_experience(n_episodes=n_episodes,
                                    episode_limit=self.episode_limit,
                                    epsilon=self.epsilon.get_value(epoch))
            sample_ratio = self.sample_ratio.get_value(epoch)
            sample_size = len(self.replay_buffer.buffer) * sample_ratio
            sample_size = round(sample_size)
            
            logging.info(f"Epoch {epoch}: Learning with batch size {batch_size} and times {learning_times_per_epoch}")
            self.learn(sample_size=sample_size, batch_size=batch_size, times=learning_times_per_epoch)
            
            mean_reward, reward_std = self.evaluate()

            self.save_intermediate_results(epoch, mean_reward, reward_std)
            
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                best_reward_std = reward_std
                logging.info(f"Epoch {epoch}: New best mean reward {best_mean_reward}")
                self.update_params()

            if mean_reward >= target_reward:
                logging.info(f"Target reward reached: {mean_reward} >= {target_reward}")
                break
        
        self.save_intermediate_results('best', best_mean_reward, best_reward_std)
        self.save_results_to_csv()
        return best_mean_reward, best_reward_std
    
    def save_intermediate_results(self, epoch, mean_reward, reward_std):
        self.results[epoch] = {
            'mean_reward': mean_reward,
            'reward_std': reward_std
        }
        logging.info(f"Intermediate results saved for epoch {epoch}: Mean reward {mean_reward}, Std reward {reward_std}")

    def save_results_to_csv(self):
        csv_path = os.path.join(self.logdir, 'results.csv')
        with open(csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'Mean Reward', 'Reward Std'])
            for epoch, result in self.results.items():
                writer.writerow([epoch, result['mean_reward'], result['reward_std']])
        logging.info(f"Results saved to {csv_path}")