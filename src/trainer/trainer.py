import os
import numpy as np
import torch
from typing import Dict
from copy import deepcopy
from multiprocessing import Process, Queue, Pool
import datetime
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
from ..rollout.rolloutmanager_config import RolloutManagerConfig

class Trainer():
    def __init__(self, 
                 env_config: EnvConfig, 
                 agent_group_config: AgentGroupConfig,
                 critic_config: CriticConfig,
                 epsilon_scheduler: Scheduler,
                 sample_ratio_scheduler: Scheduler,
                 critic_optimizer_config: OptimizerConfig,
                 rolloutmanager_config: RolloutManagerConfig,
                 replaybuffer_config: ReplayBufferConfig,
                 gamma: float = 0.9,
                 eval_epsilon: float = 0.01,
                 workdir: str = "",
                 train_device: str = 'cpu',
                 eval_device: str = 'cpu'):
        
        self.env_config = env_config
        self.env = env_config.create_env()
        self.critic_config = critic_config
        self.sample_ratio = sample_ratio_scheduler
        self.train_device = train_device
        self.eval_device = eval_device
        self.epsilon = epsilon_scheduler
        self.eval_epsilon = eval_epsilon
        self.gamma = gamma

        self.replaybuffer = replaybuffer_config.create_replaybuffer()
        self.rolloutmanager_config = rolloutmanager_config

        # Agent group
        self.eval_agent_group = agent_group_config.get_agent_group()
        self.target_agent_group = agent_group_config.get_agent_group()
        self.best_agent_group_params = self.eval_agent_group.get_agent_group_params()
        self.target_agent_group.set_agent_group_params(self.best_agent_group_params)  # Load the model parameters to eval agent group
        
        # Critic
        self.eval_critic = critic_config.get_critic()
        self.target_critic = critic_config.get_critic()
        self.best_critic_params = deepcopy(self.eval_critic.state_dict())
        self.target_critic.load_state_dict(self.best_critic_params)
        self.optimizer = critic_optimizer_config.get_optimizer(self.eval_critic.parameters())

        # Work directory
        self.workdir = workdir
        self.logdir = os.path.join(workdir, 'logs')
        self.checkpointdir = os.path.join(workdir, 'checkpoints')

        self.results = {}
        self.log = logging.basicConfig(level=logging.INFO,
                                       format='%(asctime)s - %(levelname)s - %(message)s',
                                       handlers=[
                                                 logging.StreamHandler(),
                                                 #logging.FileHandler(os.path.join(self.logdir, "training.log"))
                                                 ])

    def learn(self, sample_size, batch_size: int, times: int):
        raise NotImplementedError
    
    def save_current_model(self, checkpoint: str):
        agent_path = os.path.join(self.checkpointdir, checkpoint, "agent")
        os.makedirs(agent_path, exist_ok=True)
        self.eval_agent_group.save_params(agent_path)

        critic_path = os.path.join(self.checkpointdir, checkpoint, "critic")
        os.makedirs(critic_path, exist_ok=True)
        critic_params = self.eval_critic.state_dict()
        torch.save(critic_params, os.path.join(critic_path, "critic.pth"))
        return self
    
    def load_model(self, checkpoint: str):
        agent_path = os.path.join(self.checkpointdir, checkpoint, "agent")
        self.eval_agent_group.load_params(agent_path)
        critic_path = os.path.join(self.checkpointdir, checkpoint, "critic", "critic.pth")
        self.eval_critic.load_state_dict(torch.load(critic_path, weights_only=True))
        self.update_target_model_params()
        return self
    
    def save_best_model(self):
        self.eval_agent_group.set_agent_group_params(self.best_agent_group_params)
        self.eval_critic.load_state_dict(self.best_critic_params)
        self.save_current_model(checkpoint = 'best')
        return self

    def collect_experience(self, epsilon: float):
        """
        Collect experiences using multiple rollout workers.
        """
        self.eval_agent_group.eval().to(self.eval_device)
        manager = self.rolloutmanager_config.create_manager(self.eval_agent_group,
                                                           self.env_config,
                                                           epsilon)
        episodes = manager.generate_episodes()
        manager.cleanup()

        for episode in episodes:
            self.replaybuffer.add_episode(episode)
        
        return self

    def update_target_model_params(self):
        # Update the evaluation models with the latest weights from the training models
        agent_group_params = self.eval_agent_group.get_agent_group_params()
        self.target_agent_group.set_agent_group_params(agent_group_params)
        critic_params = deepcopy(self.eval_critic.state_dict())  # Update critic parameters
        self.target_critic.load_state_dict(critic_params)
        return self
    
    def update_best_params(self):
        self.best_agent_group_params = self.eval_agent_group.get_agent_group_params()
        self.best_critic_params = deepcopy(self.eval_critic.state_dict())  # Update critic parameters
        return self

    def evaluate(self):
        self.eval_agent_group.eval().to(self.eval_device)
        manager = self.rolloutmanager_config.create_eval_manager(self.eval_agent_group,
                                                           self.env_config,
                                                           self.eval_epsilon)
        
        logging.info(f"Evaluating model...")
        episodes = manager.generate_episodes()
        manager.cleanup()
        rewards = np.array([episode['episode_reward'] for episode in episodes])
        
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        logging.info(f"Evaluation results: Mean reward {mean_reward}, Std reward {std_reward}")
        
        return mean_reward, std_reward
    
    def train(self, epochs, target_reward, eval_interval=1, batch_size=64, learning_times_per_epoch=1):
        best_mean_reward = -np.inf
        best_reward_std = np.inf
        best_loss = np.inf

        # Training loop
        for epoch in range(epochs):

            logging.info(f"Epoch {epoch}: Collecting experiences")
            self.collect_experience(epsilon=self.epsilon.get_value(epoch))
            sample_ratio = self.sample_ratio.get_value(epoch)
            sample_size = len(self.replaybuffer.buffer) * sample_ratio
            sample_size = round(sample_size)
            
            # Learn and update eval model
            logging.info(f"Epoch {epoch}: Learning with batch size {batch_size} and learning {learning_times_per_epoch} times per epoch")
            loss = self.learn(sample_size=sample_size, batch_size=batch_size, times=learning_times_per_epoch)
            logging.info(f"Epoch {epoch}: Loss {loss}")

            # Save checkpoint
            checkpoint_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_name = f"checkpoint_{checkpoint_time}_{epoch}"
            self.save_current_model(checkpoint_name)
            logging.info(f"Checkpoint saved at {checkpoint_name}")

            mean_reward, reward_std = self.evaluate()
            self.save_intermediate_results(epoch, loss, mean_reward, reward_std)

            if epoch % eval_interval == 0:
            
                self.update_target_model_params()
                logging.info(f"Epoch {epoch}: Target model updated with eval model params")
                
                if mean_reward > best_mean_reward:
                    best_mean_reward = mean_reward
                    best_reward_std = reward_std
                    best_loss = loss
                    self.update_best_params()
                    logging.info(f"Epoch {epoch}: New best mean reward {best_mean_reward}")
                    
                if mean_reward >= target_reward:
                    logging.info(f"Target reward reached: {mean_reward} >= {target_reward}")
                    break

        self.save_intermediate_results('best', best_loss, best_mean_reward, best_reward_std)
        self.save_results_to_csv()
        self.save_best_model()
        return best_mean_reward, best_reward_std
    
    def save_intermediate_results(self, epoch, loss, mean_reward, reward_std):
        self.results[epoch] = {
            'loss': loss,
            'mean_reward': mean_reward,
            'reward_std': reward_std
        }
        logging.info(f"Intermediate results saved for epoch {epoch}: Loss {loss}, Mean reward {mean_reward}, Std reward {reward_std}")

    def save_results_to_csv(self):
        os.makedirs(self.logdir, exist_ok=True)
        csv_path = os.path.join(self.logdir, 'results.csv')
        with open(csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'Mean Reward', 'Reward Std'])
            for epoch, result in self.results.items():
                writer.writerow([epoch, result['mean_reward'], result['reward_std']])
        logging.info(f"Results saved to {csv_path}")