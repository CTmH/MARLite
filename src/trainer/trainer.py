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
from ..rollout.rolloutmanager_config import RolloutManagerConfig
from ..rollout.multiprocess_rolloutworker import MultiProcessRolloutWorker
from ..replaybuffer.replaybuffer_config import ReplayBufferConfig
from ..replaybuffer.normal_replaybuffer import NormalReplayBuffer
from ..util.scheduler import Scheduler
from ..algorithm.agents.agent_group_config import AgentGroupConfig
from ..algorithm.critic.critic_config import CriticConfig
from ..util.optimizer_config import OptimizerConfig

def get_episode(worker, episode_limit, epsilon):
    return worker.generate_episode(episode_limit, epsilon)

class Trainer():
    def __init__(self, 
                 env_config: EnvConfig, 
                 agent_group_config: AgentGroupConfig,
                 critic_config: CriticConfig,
                 epsilon_scheduler: Scheduler,
                 sample_ratio_scheduler: Scheduler,
                 critic_optimizer_config: OptimizerConfig,
                 rolloutworker_config: RolloutManagerConfig,
                 replaybuffer_config: ReplayBufferConfig,
                 traj_len: int,
                 n_workers: int,
                 epochs = 10000,
                 buffer_capacity: int = 50000,
                 episode_limit: int = 500,
                 n_episodes: int = 1000,
                 gamma: float = 0.9,
                 workdir: str = "",
                 train_device: str = 'cpu',
                 eval_device: str = 'cpu'):
        
        self.env_config = env_config
        self.critic_config = critic_config
        self.traj_len = traj_len
        self.n_workers = n_workers
        self.episode_limit = episode_limit
        self.epochs = epochs
        self.sample_ratio = sample_ratio_scheduler
        self.train_device = train_device
        self.eval_device = eval_device
        self.epsilon = epsilon_scheduler
        self.gamma = gamma
        self.n_episodes = n_episodes

        self.replaybuffer = replaybuffer_config.create_replaybuffer()
        self.rollout

        # Agent group
        self.target_agent_group = agent_group_config.get_agent_group()
        self.eval_agent_group = agent_group_config.get_agent_group()
        self.target_agent_model_params, self.target_agent_fe_params = self.target_agent_group.get_model_params()
        self.eval_agent_group.set_model_params(model_params=self.target_agent_model_params, fe_params=self.target_agent_fe_params)  # Load the model parameters to eval agent group
        self.agents = self.target_agent_group.agents
        
        # Critic
        self.target_critic = critic_config.get_critic()
        self.eval_critic = critic_config.get_critic()
        self.target_critic_params = deepcopy(self.target_critic.state_dict())
        self.eval_critic.load_state_dict(self.target_critic_params)
        self.optimizer = critic_optimizer_config.get_optimizer(self.target_critic.parameters())

        # Work directory
        self.workdir = workdir
        os.makedirs(self.workdir, exist_ok=True)
        self.logdir = os.path.join(workdir, 'logs')
        self.modeldir = os.path.join(workdir, 'models')
        self.agentsdir = os.path.join(self.modeldir, 'agents')
        self.criticdir = os.path.join(self.modeldir, 'critic')
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
    
    def save_model(self, checkpoint: str):
        agent_model_params, agent_feature_extractor_params = self.target_agent_group.get_model_params()
        for model_name, params in agent_model_params.items():
            model_path = os.path.join(self.agentsdir, model_name, 'model',f'{checkpoint}.pth')
            torch.save(params, model_path)
            logging.info(f"Actor {model_name} saved to {model_path}")

        for model_name, params in agent_feature_extractor_params.items():
            model_path = os.path.join(self.agentsdir, model_name, 'feature_extractor',f'{checkpoint}.pth')
            torch.save(params, model_path)
            logging.info(f"{model_name}'s feature extractor saved to {model_path}")

        critic_path = os.path.join(self.criticdir, 'model', f'{checkpoint}.pth')
        torch.save(self.target_critic.state_dict(), critic_path)
        logging.info(f"Critic model saved to {critic_path}")

    def load_model(self, checkpoint: str):
        agent_model_params = {}
        agent_feature_extractor_params = {}
        for model_name in self.target_agent_group.models.keys():
            model_path = os.path.join(self.agentsdir, model_name, 'model', f'{checkpoint}.pth')
            if os.path.exists(model_path):
                params = torch.load(model_path)
                agent_model_params[model_name] = params
                logging.info(f"Actor {model_name} loaded from {model_path}")
            else:
                logging.warning(f"Model path for actor {model_name} does not exist: {model_path}")

            fe_path = os.path.join(self.agentsdir, model_name, 'feature_extractor', f'{checkpoint}.pth')
            if os.path.exists(fe_path):
                params = torch.load(fe_path)
                agent_feature_extractor_params[model_name] = params
                logging.info(f"{model_name}'s feature extractor loaded from {fe_path}")
            else:
                logging.warning(f"Feature extractor path for actor {model_name} does not exist: {fe_path}")
        self.target_agent_group.set_model_params(agent_model_params, agent_feature_extractor_params)
        logging.info("All actor models and feature extractors loaded successfully.")

        critic_path = os.path.join(self.criticdir, 'model', f'{checkpoint}.pth')
        if os.path.exists(critic_path):
            self.target_critic.load_state_dict(torch.load(critic_path))
            logging.info(f"Critic model loaded from {critic_path}")
        else:
            logging.warning(f"Critic model path does not exist: {critic_path}")

        self.update_params()

        return self

    def collect_experience(self, n_episodes: int, episode_limit: int, epsilon: int):
        """
        Collect experiences using multiple rollout workers.
        """
        job = MultiProcessRolloutWorker(env_config=self.env_config,
                            agent_group=self.eval_agent_group,
                            rnn_traj_len=self.traj_len,
                            device=self.eval_device)
        
        logging.info(f"Collecting {n_episodes} episodes with limit {episode_limit} and epsilon {epsilon}")
        episodes = [job.rollout(episode_limit, epsilon) for _ in range(n_episodes)]

        for episode in episodes:
            self.replaybuffer.add_episode(episode)
        
        return self

    def update_params(self):
        # Update the evaluation models with the latest weights from the training models
        self.target_agent_model_params, self.target_agent_fe_params = self.target_agent_group.get_model_params()
        self.eval_agent_group.set_model_params(self.target_agent_model_params, self.target_agent_fe_params)
        self.target_critic_params = deepcopy(self.target_critic.state_dict())  # Update critic parameters
        self.eval_critic.load_state_dict(self.target_critic_params)

    def evaluate(self, times=100):
        job = MultiProcessRolloutWorker(env_config=self.env_config,
                            agent_group=self.eval_agent_group,
                            rnn_traj_len=self.traj_len,
                            device=self.eval_device)
        
        logging.info(f"Evaluating for {times} episodes")
        episodes = [job.rollout(self.episode_limit, epsilon=0.0) for _ in range(times)]
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

            self.target_agent_group.set_model_params(self.target_agent_model_params, self.target_agent_fe_params)
            self.target_critic.load_state_dict(self.target_critic_params)

            logging.info(f"Epoch {epoch}: Collecting experiences")
            self.collect_experience(n_episodes=n_episodes,
                                    episode_limit=self.episode_limit,
                                    epsilon=self.epsilon.get_value(epoch))
            sample_ratio = self.sample_ratio.get_value(epoch)
            sample_size = len(self.replaybuffer.buffer) * sample_ratio
            sample_size = round(sample_size)
            
            logging.info(f"Epoch {epoch}: Learning with batch size {batch_size} and learning {learning_times_per_epoch} times per epoch")
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