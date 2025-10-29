import os
import sys
import csv
import torch
import random
import datetime
from copy import deepcopy
from absl import logging
import numpy as np
from marlite.environment import EnvConfig
from marlite.rollout import RolloutManagerConfig
from marlite.replaybuffer import ReplayBufferConfig
from marlite.algorithm.agents import AgentGroupConfig
from marlite.algorithm.critic import CriticConfig
from marlite.rollout import RolloutManagerConfig
from marlite.util.optimizer_config import OptimizerConfig
from marlite.util.lr_scheduler_config import LRSchedulerConfig
from marlite.util.scheduler import Scheduler

class Trainer():
    def __init__(self,
                 env_config: EnvConfig,
                 agent_group_config: AgentGroupConfig,
                 critic_config: CriticConfig,
                 epsilon_scheduler: Scheduler,
                 sample_ratio_scheduler: Scheduler,
                 critic_optimizer_config: OptimizerConfig,
                 lr_scheduler_conf: LRSchedulerConfig,
                 rolloutmanager_config: RolloutManagerConfig,
                 replaybuffer_config: ReplayBufferConfig,
                 gamma: float = 0.9,
                 eval_epsilon: float = 0.01,
                 eval_threshold: float = 0.03,
                 eval_episodes_to_replay_ratio: float = 0.25,
                 workdir: str = "",
                 train_device: str = 'cpu',
                 n_workers = 1,
                 use_data_parallel: bool = False,
                 compile_models: bool = False,):

        self.env_config = env_config
        self.critic_config = critic_config
        self.sample_ratio = sample_ratio_scheduler
        self.epsilon = epsilon_scheduler
        self.eval_epsilon = eval_epsilon
        self.eval_threshold = eval_threshold
        self.eval_episodes_to_replay_ratio = eval_episodes_to_replay_ratio
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
        self.target_critic.load_state_dict(self.eval_critic.state_dict())
        self.best_critic_params = deepcopy(self.eval_critic.state_dict())
        self._cached_critic_params = deepcopy(self.eval_critic.state_dict())

        self.optimizer = critic_optimizer_config.get_optimizer(self.eval_critic.parameters())
        if lr_scheduler_conf:
            self.lr_scheduler = lr_scheduler_conf.get_lr_scheduler(self.optimizer)
        else:
            self.lr_scheduler = None

        # Work directory
        self.workdir = workdir
        self.logdir = os.path.join(workdir, 'logs')
        self.checkpointdir = os.path.join(workdir, 'checkpoints')

        self.training_history = {}

        # Configure absl logging
        os.makedirs(self.logdir, exist_ok=True)
        # Set the log file (absl will automatically add timestamps and process info)
        logging.get_absl_handler().use_absl_log_file('training', self.logdir)
        # Set log level
        logging.set_verbosity(logging.INFO)
        logging.get_absl_handler().python_handler.stream = sys.stdout  # Ensure output to console

        # Device
        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus > 1 and use_data_parallel:
            self.train_device = "cuda"
            self.use_data_parallel = use_data_parallel
        else:
            self.train_device = train_device
            self.use_data_parallel = False

        # torch.compile
        self.compile_models = compile_models
        if self.compile_models:
            logging.info(f"Compiling models...")
            self.eval_agent_group = self.eval_agent_group.to(self.train_device).compile_models().to('cpu')
            self.target_agent_group = self.target_agent_group.to(self.train_device).compile_models().to('cpu')
            self.eval_critic = torch.compile(self.eval_critic.to(self.train_device)).to('cpu')
            self.target_critic = torch.compile(self.target_critic.to(self.train_device)).to('cpu')

        # Metrics
        self.best_mean_reward = -np.inf
        self.best_reward_std = np.inf
        self.best_win_rate = .0

        self.current_epoch = 0

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

    def load_checkpoint(self, checkpoint: str, checkpoint_mean_reward = -np.inf, checkpoint_reward_std = np.inf, checkpoint_win_rate = .0):
        self.best_mean_reward = checkpoint_mean_reward
        self.best_reward_std = checkpoint_reward_std
        self.best_win_rate = checkpoint_win_rate
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
        self.eval_agent_group.eval()
        manager = self.rolloutmanager_config.create_manager(self.eval_agent_group,
                                                           self.env_config,
                                                           epsilon)
        episodes = manager.generate_episodes()

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
        self.eval_agent_group.eval()
        manager = self.rolloutmanager_config.create_eval_manager(self.eval_agent_group,
                                                           self.env_config,
                                                           self.eval_epsilon)

        logging.info(f"Evaluating model...")
        episodes = manager.generate_episodes()
        manager.cleanup()
        rewards = np.array([episode['episode_reward'] for episode in episodes])
        win_tags = np.array([episode['win_tag'] for episode in episodes])
        win_rate = win_tags.mean().item()

        sum_total = rewards.sum()
        max_val = rewards.max()
        min_val = rewards.min()
        adjusted_sum = sum_total - max_val - min_val
        adjusted_count = len(rewards) - 2
        mean_reward = adjusted_sum / adjusted_count
        mean_reward = mean_reward.item()
        std_reward = np.std(rewards).item()

        logging.info(f"Evaluation results: Mean reward {mean_reward:.4f}, Std reward {std_reward:.4f}, Win rate {win_rate:.4f}")

        self.eval_agent_group.to("cpu")
        torch.cuda.empty_cache()

        # Sample episodes based on eval_episodes_to_replay_ratio
        num_episodes_to_add = int(len(episodes) * self.eval_episodes_to_replay_ratio)
        if num_episodes_to_add > 0:
            # Randomly sample episodes
            sampled_indices = random.sample(range(len(episodes)), num_episodes_to_add)
            for i in sampled_indices:
                self.replaybuffer.add_episode(episodes[i])

        return mean_reward, std_reward, win_rate

    def train(self, epochs, target_reward, eval_interval=1, update_target_interval=1, batch_size=64, learning_times_per_epoch=1):

        best_loss = np.inf
        # Training loop
        for epoch in range(epochs):
            self.current_epoch = epoch

            logging.info(f"Epoch {epoch}: Collecting experiences")
            self.collect_experience(epsilon=self.epsilon.get_value(epoch))
            sample_ratio = self.sample_ratio.get_value(epoch)
            sample_size = len(self.replaybuffer.buffer) * sample_ratio
            sample_size = round(sample_size)

            # Learn and update eval model
            agent_group_lr = self.eval_agent_group.optimizer.param_groups[0]['lr']
            critic_lr = self.optimizer.param_groups[0]['lr']
            logging.info(f"Epoch {epoch}: Batch size: {batch_size}, Critic learning rate: {critic_lr:.8f}, Agent learning rate: {agent_group_lr:.8f}")
            logging.info(f"Epoch {epoch}: Learning {learning_times_per_epoch} times per epoch ...")
            loss = self.learn(sample_size=sample_size, batch_size=batch_size, times=learning_times_per_epoch)
            logging.info(f"Epoch {epoch}: Loss {loss:.4f}")

            # Save checkpoint
            checkpoint_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_name = f"checkpoint_{checkpoint_time}_{epoch}"
            self.save_current_model(checkpoint_name)
            logging.info(f"Checkpoint saved at {checkpoint_name}")

            mean_reward, reward_std, win_rate = self.evaluate()
            self.save_intermediate_results(epoch, loss, mean_reward, reward_std, win_rate)

            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(mean_reward)
            elif isinstance(self.lr_scheduler, torch.optim.lr_scheduler.LRScheduler):
                self.lr_scheduler.step()
            self.eval_agent_group.lr_scheduler_step(mean_reward)

            #if mean_reward >= self.best_mean_reward * (1 - self.eval_threshold):
            if (mean_reward - self.best_mean_reward) / max(abs(self.best_mean_reward), 1) >= -self.eval_threshold:
                self._cached_agent_group_params = self.eval_agent_group.get_agent_group_params()
                self._cached_critic_params = deepcopy(self.eval_critic.state_dict())
                logging.info(f"Epoch {epoch}: Cached parameters updated with current parameters.")

            if mean_reward >= self.best_mean_reward or win_rate > self.best_win_rate:
                self.best_mean_reward = mean_reward
                self.best_reward_std = reward_std
                self.best_win_rate = win_rate
                best_loss = loss
                self.best_agent_group_params = self.eval_agent_group.get_agent_group_params()
                self.best_critic_params = deepcopy(self.eval_critic.state_dict())
                logging.info(f"Epoch {epoch}: New best mean reward {self.best_mean_reward:.4f}")

            if mean_reward >= target_reward:
                logging.info(f"Epoch {epoch}: Target reward reached: {mean_reward:.4f} >= {target_reward:.4f}")
                break

            if epoch % eval_interval == 0:
                self.eval_agent_group.set_agent_group_params(self._cached_agent_group_params)
                self.eval_critic.load_state_dict(self._cached_critic_params)
                self.update_target_model_params()
                logging.info(f"Epoch {epoch}: Eval model and Target model updated with cached parameters.")

            if epoch % update_target_interval == 0:
                self.update_target_model_params()
                logging.info(f"Epoch {epoch}: Target model updated with eval model parameters.")

        self.save_intermediate_results('best', best_loss, self.best_mean_reward, self.best_reward_std, self.best_win_rate)
        self.save_results_to_csv()
        self.save_best_model()
        return self.best_mean_reward, self.best_reward_std

    def save_intermediate_results(self, epoch, loss, mean_reward, reward_std, win_rate):
        self.training_history[epoch] = {
            'loss': loss,
            'mean_reward': mean_reward,
            'reward_std': reward_std,
            'win_rate': win_rate,
        }
        logging.info(f"Intermediate results saved for epoch {epoch}: Loss {loss:.4f}, " +
                     f"Mean reward {mean_reward:.4f}, Std reward {reward_std:.4f}, Win rate {win_rate:.2f}")

    def save_results_to_csv(self):
        os.makedirs(self.logdir, exist_ok=True)
        csv_path = os.path.join(self.logdir, 'results.csv')
        with open(csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'Mean_Reward', 'Reward_Std', 'Win_Rate'])
            for epoch, result in self.training_history.items():
                writer.writerow([epoch, result['mean_reward'], result['reward_std'], result['win_rate']])
        logging.info(f"Results saved to {csv_path}")