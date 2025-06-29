import os
import numpy as np
import torch
from copy import deepcopy
import datetime
import logging
import csv

from ..environment.env_config import EnvConfig
from ..rollout.rolloutmanager_config import RolloutManagerConfig
from ..replaybuffer.replaybuffer_config import ReplayBufferConfig
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
                 eval_threshold: float = 0.03,
                 workdir: str = "",
                 train_device: str = 'cpu',
                 eval_device: str = 'cpu',
                 n_workers = 1,
                 use_data_parallel: bool = False):

        self.env_config = env_config
        self.env = env_config.create_env()
        self.critic_config = critic_config
        self.sample_ratio = sample_ratio_scheduler
        self.epsilon = epsilon_scheduler
        self.eval_epsilon = eval_epsilon
        self.eval_threshold = eval_threshold
        self.gamma = gamma
        self.n_workers = n_workers

        self.replaybuffer = replaybuffer_config.create_replaybuffer()
        self.rolloutmanager_config = rolloutmanager_config

        # Agent group
        self.eval_agent_group = agent_group_config.get_agent_group()
        self.target_agent_group = agent_group_config.get_agent_group()
        self.best_agent_group_params = self.eval_agent_group.get_agent_group_params()
        self.target_agent_group.set_agent_group_params(self.best_agent_group_params)  # Load the model parameters to eval agent group
        self._cached_agent_group_params = self.best_agent_group_params

        # Critic
        self.eval_critic = critic_config.get_critic()
        self.target_critic = critic_config.get_critic()
        self.best_critic_params = deepcopy(self.eval_critic.state_dict())
        self.target_critic.load_state_dict(self.best_critic_params)
        self._cached_critic_params = self.best_critic_params

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

        # Device
        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus > 1 and use_data_parallel:
            self.train_device = "cuda"
            self.use_data_parallel = use_data_parallel
        else:
            self.train_device = train_device
            self.use_data_parallel = False
        self.eval_device = eval_device

        # Metrics
        self.best_mean_reward = -np.inf
        self.best_reward_std = np.inf

    def learn(self, sample_size, batch_size: int, times: int):
        raise NotImplementedError

    def save_current_model(self, checkpoint: str):
        agent_path = os.path.join(self.checkpointdir, checkpoint, "agent")
        os.makedirs(agent_path, exist_ok=True)
        self.eval_agent_group.to("cpu")
        self.eval_agent_group.save_params(agent_path)

        critic_path = os.path.join(self.checkpointdir, checkpoint, "critic")
        os.makedirs(critic_path, exist_ok=True)
        self.eval_critic.to("cpu")
        critic_params = self.eval_critic.state_dict()
        torch.save(critic_params, os.path.join(critic_path, "critic.pth"))
        return self

    def load_checkpoint(self, checkpoint: str, checkpoint_mean_reward = -np.inf, checkpoint_reward_std = np.inf):
        self.best_mean_reward = checkpoint_mean_reward
        self.best_reward_std = checkpoint_reward_std
        agent_path = os.path.join(self.checkpointdir, checkpoint, "agent")
        self.eval_agent_group.to("cpu")
        self.eval_critic.to("cpu")
        self.eval_agent_group.load_params(agent_path)
        critic_path = os.path.join(self.checkpointdir, checkpoint, "critic", "critic.pth")
        self.eval_critic.load_state_dict(torch.load(critic_path, weights_only=True))
        self.best_agent_group_params = self.eval_agent_group.get_agent_group_params()
        self.best_critic_params = deepcopy(self.eval_critic.state_dict())
        self._cached_agent_group_params = self.eval_agent_group.get_agent_group_params()
        self._cached_critic_params = deepcopy(self.eval_critic.state_dict())
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

        self.eval_agent_group.to("cpu")
        torch.cuda.empty_cache()

        return self

    def update_target_model_params(self):
        agent_group_params = self.eval_agent_group.get_agent_group_params()
        self.target_agent_group.set_agent_group_params(agent_group_params)
        critic_params = deepcopy(self.eval_critic.state_dict())  # Update critic parameters
        self.target_critic.load_state_dict(critic_params)
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

        sum_total = rewards.sum()
        max_val = rewards.max()
        min_val = rewards.min()
        adjusted_sum = sum_total - max_val - min_val
        adjusted_count = len(rewards) - 2
        mean_reward = adjusted_sum / adjusted_count

        std_reward = np.std(rewards)
        logging.info(f"Evaluation results: Mean reward {mean_reward}, Std reward {std_reward}")

        self.eval_agent_group.to("cpu")
        torch.cuda.empty_cache()

        return mean_reward, std_reward

    def train(self, epochs, target_reward, eval_interval=1, batch_size=64, learning_times_per_epoch=1):

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

            if mean_reward >= self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.best_reward_std = reward_std
                best_loss = loss
                self.best_agent_group_params = self.eval_agent_group.get_agent_group_params()
                self.best_critic_params = deepcopy(self.eval_critic.state_dict())
                logging.info(f"Epoch {epoch}: New best mean reward {self.best_mean_reward}")

            if mean_reward >= self.best_mean_reward * (1 - self.eval_threshold):
                self._cached_agent_group_params = self.eval_agent_group.get_agent_group_params()
                self._cached_critic_params = deepcopy(self.eval_critic.state_dict())
                logging.info(f"Epoch {epoch}: Cached parameters updated with current parameters.")

            if mean_reward >= target_reward:
                logging.info(f"Epoch {epoch}: Target reward reached: {mean_reward} >= {target_reward}")
                break

            if epoch % eval_interval == 0:
                self.eval_agent_group.set_agent_group_params(self._cached_agent_group_params)
                self.eval_critic.load_state_dict(self._cached_critic_params)
                logging.info(f"Epoch {epoch}: Eval model updated with cached parameters.")
                self.update_target_model_params()
                logging.info(f"Epoch {epoch}: Target model updated with eval model parameters.")

        self.save_intermediate_results('best', best_loss, self.best_mean_reward, self.best_reward_std)
        self.save_results_to_csv()
        self.save_best_model()
        return self.best_mean_reward, self.best_reward_std

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