import io
import os
import sys
import yaml
import torch
import random
import datetime
import numpy as np
from copy import deepcopy
from absl import logging
from typing import List, Union, Optional
from abc import abstractmethod

from marlite.environment import EnvConfig
from marlite.rollout import RolloutManagerConfig
from marlite.replaybuffer import ReplayBufferConfig
from marlite.algorithm.agents import AgentGroupConfig
from marlite.algorithm.critic import CriticConfig
from marlite.rollout import RolloutManagerConfig
from marlite.util.optimizer_config import OptimizerConfig
from marlite.util.lr_scheduler_config import LRSchedulerConfig
from marlite.util.scheduler import Scheduler
from marlite.analyzer import AnalyzerConfig


def _serialize_to_buffer(state_dict):
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    buffer.seek(0)
    return buffer


def _deserialize_from_buffer(buffer):
    buffer.seek(0)
    return torch.load(buffer, weights_only=True)


def get_device_list(train_device: Union[str, List[str]]) -> tuple:
    """
    Parse train_device configuration and return device list.

    Args:
        train_device: Either a string (single device) or a list of strings (multiple devices)
            - "cpu": CPU training
            - "cuda" or "cuda:0": Single GPU training
            - ["cuda:0", "cuda:1", ...]: Multi-GPU training with worker processes

    Returns:
        tuple of (device_list, use_multi_gpu):
            - device_list: List of device strings
            - use_multi_gpu: Whether to use multi-GPU training
    """
    if isinstance(train_device, str):
        if train_device == "cuda":
            return ["cuda:0"], False
        else:
            return [train_device], False
    elif isinstance(train_device, list):
        if len(train_device) == 0:
            raise ValueError("train_device list cannot be empty")
        return train_device, True
    else:
        raise ValueError(
            f"train_device must be a string or a list of strings, got {type(train_device)}"
        )


class Trainer:
    """
    Base Trainer class for multi-agent reinforcement learning.

    Supports two training modes:
    1. Single-GPU/CPU training: Models stay on main process
    2. Multi-GPU training: Uses worker processes for parallel gradient computation
    """

    def __init__(
        self,
        env_config: EnvConfig,
        agent_group_config: AgentGroupConfig,
        critic_config: CriticConfig,
        epsilon_scheduler: Scheduler,
        sample_ratio_scheduler: Scheduler,
        critic_optimizer_config: OptimizerConfig,
        agent_optimizer_config: OptimizerConfig,
        lr_scheduler_conf: LRSchedulerConfig,
        agent_lr_scheduler_conf: Optional[LRSchedulerConfig],
        rolloutmanager_config: RolloutManagerConfig,
        replaybuffer_config: ReplayBufferConfig,
        analyzer_config: AnalyzerConfig,
        eval_metric_list: List[str] = ["reward"],
        gamma: float = 0.9,
        eval_epsilon: float = 0.01,
        eval_threshold: float = 0.03,
        eval_episodes_to_replay_ratio: float = 0.25,
        workdir: str = "",
        train_device: Union[str, List[str]] = "cpu",
        n_workers: int = 1,
        compile_models: bool = False,
        sample_mode: str = "ratio",
    ):
        self.env_config = env_config
        self.critic_config = critic_config
        self.agent_optimizer_config = agent_optimizer_config
        self.sample_ratio = sample_ratio_scheduler
        self.epsilon = epsilon_scheduler
        self.eval_epsilon = eval_epsilon
        self.eval_threshold = eval_threshold
        self.eval_episodes_to_replay_ratio = eval_episodes_to_replay_ratio
        self.gamma = gamma
        self.n_workers = n_workers
        self.eval_metric_list = eval_metric_list
        self.sample_mode = sample_mode

        if self.sample_mode not in ["ratio", "direct"]:
            raise ValueError(
                f"Invalid sample_mode: {self.sample_mode}. Must be 'ratio' or 'direct'"
            )

        self.replaybuffer = replaybuffer_config.create_replaybuffer()
        self.rolloutmanager_config = rolloutmanager_config
        self.analyzer = analyzer_config.create_analyzer()

        # Agent group
        self.agent_group_config = agent_group_config
        self.eval_agent_group = agent_group_config.get_agent_group()
        self.target_agent_group = agent_group_config.get_agent_group()
        self.best_agent_group_params = _serialize_to_buffer(
            self.eval_agent_group.state_dict()
        )
        self.target_agent_group.load_state_dict(
            _deserialize_from_buffer(self.best_agent_group_params)
        )
        self._cached_agent_group_params = _serialize_to_buffer(
            self.eval_agent_group.state_dict()
        )

        # Critic
        self.eval_critic = critic_config.get_critic()
        self.target_critic = critic_config.get_critic()
        self.target_critic.load_state_dict(self.eval_critic.state_dict())
        self.best_critic_params = _serialize_to_buffer(self.eval_critic.state_dict())
        self._cached_critic_params = _serialize_to_buffer(self.eval_critic.state_dict())
        self.critic_optimizer_config = critic_optimizer_config
        self.lr_scheduler_conf = lr_scheduler_conf
        self.agent_lr_scheduler_conf = agent_lr_scheduler_conf

        self.optimizer = self.critic_optimizer_config.get_optimizer(
            self.eval_critic.parameters()
        )

        self.agent_optimizer = self.agent_optimizer_config.get_optimizer(
            self.eval_agent_group.parameters()
        )

        if isinstance(lr_scheduler_conf, LRSchedulerConfig):
            self.lr_scheduler = lr_scheduler_conf.get_lr_scheduler(self.optimizer)
        else:
            self.lr_scheduler = None

        if isinstance(agent_lr_scheduler_conf, LRSchedulerConfig):
            self.agent_lr_scheduler = agent_lr_scheduler_conf.get_lr_scheduler(
                self.agent_optimizer
            )
        else:
            self.agent_lr_scheduler = None

        # Work directory
        self.workdir = workdir
        self.logdir = os.path.join(workdir, "logs")
        self.checkpointdir = os.path.join(workdir, "checkpoints")

        self.training_history = {}

        # Configure absl logging
        os.makedirs(self.logdir, exist_ok=True)
        logging.get_absl_handler().use_absl_log_file("training", self.logdir)
        logging.set_verbosity(logging.INFO)
        logging.get_absl_handler().python_handler.stream = sys.stdout

        # Device configuration
        self.train_device_config = train_device
        self.device_list, self.use_multi_gpu = get_device_list(train_device)

        # Multi-GPU worker group (to be created by subclasses)
        self.worker_group = None

        if self.use_multi_gpu:
            self.train_device = self.device_list[0]
            self.worker_group = self._create_worker_group()
            if self.worker_group is not None:
                self.worker_group.start_workers()
                self._sync_params_to_workers()
            logging.info(
                f"Using multi-GPU training with {len(self.device_list)} devices: {self.device_list}"
            )
        else:
            self.train_device = self.device_list[0]
            logging.info(f"Using single device: {self.train_device}")

        # torch.compile
        self.compile_models = compile_models
        if self.compile_models:
            logging.info(f"Compiling models...")
            self.eval_agent_group = torch.compile(
                self.eval_agent_group.to(self.train_device)
            ).to("cpu")
            self.target_agent_group = torch.compile(
                self.target_agent_group.to(self.train_device)
            ).to("cpu")
            self.eval_critic = torch.compile(self.eval_critic.to(self.train_device)).to(
                "cpu"
            )
            self.target_critic = torch.compile(
                self.target_critic.to(self.train_device)
            ).to("cpu")

        # Metrics
        self.best_metrics = {key: -np.inf for key in self.eval_metric_list}

        self.current_epoch = 0

    @abstractmethod
    def _create_worker_group(self):
        """
        Create and return the appropriate worker group for this trainer.

        Subclasses should override this to create their specific worker group.

        Returns:
            WorkerGroup instance or None if not using multi-GPU
        """
        pass

    def _sync_params_to_workers(self):
        """Synchronize model parameters and learning rates to all workers."""
        if self.worker_group is None:
            return

        trainable_params = {
            "eval_agent_group": self.eval_agent_group.state_dict(),
            "target_agent_group": self.target_agent_group.state_dict(),
            "eval_critic": self.eval_critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
        }
        self.worker_group.broadcast_params(trainable_params)

        critic_lr = self.optimizer.param_groups[0]["lr"]
        agent_lr = self.agent_optimizer.param_groups[0]["lr"]
        self.worker_group.sync_lr_to_workers(critic_lr, agent_lr)

    def _sync_eval_params_from_workers(self):
        """Sync eval model parameters from workers to trainer before evaluation."""
        if self.worker_group is None:
            return
        eval_params = self.worker_group.read_params_from_worker0()
        self.eval_agent_group.load_state_dict(eval_params["eval_agent_group"])
        self.eval_critic.load_state_dict(eval_params["eval_critic"])

    def learn(self, sample_size, batch_size: int, times: int):
        raise NotImplementedError

    def save_current_model(self, checkpoint: str):
        agent_path = os.path.join(self.checkpointdir, checkpoint, "agent")
        os.makedirs(agent_path, exist_ok=True)
        self.eval_agent_group.to("cpu")
        agent_params = self.eval_agent_group.state_dict()
        torch.save(agent_params, os.path.join(agent_path, "agent.pth"))

        critic_path = os.path.join(self.checkpointdir, checkpoint, "critic")
        os.makedirs(critic_path, exist_ok=True)
        self.eval_critic.to("cpu")
        critic_params = self.eval_critic.state_dict()
        torch.save(critic_params, os.path.join(critic_path, "critic.pth"))
        return self

    def load_checkpoint(self, checkpoint: str):
        self.best_metrics = {key: -np.inf for key in self.eval_metric_list}
        agent_path = os.path.join(self.checkpointdir, checkpoint, "agent", "agent.pth")
        self.eval_agent_group.to("cpu")
        self.eval_critic.to("cpu")
        self.eval_agent_group.load_state_dict(torch.load(agent_path, weights_only=True))
        critic_path = os.path.join(
            self.checkpointdir, checkpoint, "critic", "critic.pth"
        )
        self.eval_critic.load_state_dict(torch.load(critic_path, weights_only=True))
        self.best_agent_group_params = _serialize_to_buffer(
            self.eval_agent_group.state_dict()
        )
        self.best_critic_params = _serialize_to_buffer(self.eval_critic.state_dict())
        self._cached_agent_group_params = _serialize_to_buffer(
            self.eval_agent_group.state_dict()
        )
        self._cached_critic_params = _serialize_to_buffer(self.eval_critic.state_dict())
        self.update_target_model_params()
        return self

    def save_best_model(self):
        self.eval_agent_group.load_state_dict(
            _deserialize_from_buffer(self.best_agent_group_params)
        )
        self.eval_critic.load_state_dict(
            _deserialize_from_buffer(self.best_critic_params)
        )
        self.save_current_model(checkpoint="best")
        return self

    def collect_experience(self, epsilon: float):
        """
        Collect experiences using multiple rollout workers.
        """
        self.eval_agent_group.eval().to("cpu")
        manager = self.rolloutmanager_config.create_manager(
            self.eval_agent_group, self.env_config, epsilon
        )
        episodes = manager.generate_episodes()

        for episode in episodes:
            self.replaybuffer.add_episode(episode)

        self.eval_agent_group.to("cpu")
        torch.cuda.empty_cache()

        return self

    def update_target_model_params(self):
        self.target_agent_group.load_state_dict(
            deepcopy(self.eval_agent_group.state_dict())
        )
        self.target_critic.load_state_dict(deepcopy(self.eval_critic.state_dict()))
        return self

    def evaluate(self):
        self.eval_agent_group.eval()
        manager = self.rolloutmanager_config.create_eval_manager(
            self.eval_agent_group, self.env_config, self.eval_epsilon
        )

        episodes = manager.generate_episodes()
        manager.cleanup()

        result = self.analyzer(episodes)

        logging.info(f"Evaluation results:")
        for key in self.eval_metric_list:
            logging.info(
                f"{key}: Mean:{result[key]['mean']:.4f} Std:{result[key].get('std', 0):.4f}"
            )

        self.eval_agent_group.to("cpu")
        torch.cuda.empty_cache()

        num_episodes_to_add = int(len(episodes) * self.eval_episodes_to_replay_ratio)
        if num_episodes_to_add > 0:
            sampled_indices = random.sample(range(len(episodes)), num_episodes_to_add)
            for i in sampled_indices:
                self.replaybuffer.add_episode(episodes[i])

        return result

    def train(
        self,
        epochs,
        target_first_metric,
        eval_interval=1,
        update_target_interval=1,
        batch_size=64,
        learning_times_per_epoch=1,
    ):
        for epoch in range(epochs):
            self.current_epoch = epoch

            logging.info(f"Epoch {epoch}: Collecting experiences")
            self.collect_experience(epsilon=self.epsilon.get_value(epoch))

            if self.sample_mode == "ratio":
                sample_ratio = self.sample_ratio.get_value(epoch)
                sample_size = len(self.replaybuffer.buffer) * sample_ratio
                sample_size = round(sample_size)
            else:
                sample_size = round(self.sample_ratio.get_value(epoch))
            sample_size = min(sample_size, len(self.replaybuffer.buffer))

            # Learn and update eval model
            agent_group_lr = self.agent_optimizer.param_groups[0]["lr"]
            critic_lr = self.optimizer.param_groups[0]["lr"]
            logging.info(
                f"Epoch {epoch}: Batch size: {batch_size}, Critic learning rate: {critic_lr:.8f}, Agent learning rate: {agent_group_lr:.8f}"
            )
            logging.info(
                f"Epoch {epoch}: Learning {learning_times_per_epoch} times per epoch ..."
            )

            # Sync params if using multi-GPU (e.g., after checkpoint load)
            self._sync_params_to_workers()

            loss = self.learn(
                sample_size=sample_size,
                batch_size=batch_size,
                times=learning_times_per_epoch,
            )
            logging.info(f"Epoch {epoch}: Loss {loss:.4f}")

            # Sync eval params from workers before evaluation
            self._sync_eval_params_from_workers()

            # Save checkpoint
            checkpoint_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_name = f"checkpoint_{checkpoint_time}_{epoch}"
            self.save_current_model(checkpoint_name)
            logging.info(f"Checkpoint saved at {checkpoint_name}")

            result = self.evaluate()
            metrics = {key: result[key]["mean"] for key in self.eval_metric_list}
            first_metric = next(iter(metrics.values()))
            first_metric_name = next(iter(metrics.keys()))
            self.save_intermediate_results(epoch, result)

            if isinstance(
                self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                self.lr_scheduler.step(first_metric)
            elif isinstance(self.lr_scheduler, torch.optim.lr_scheduler.LRScheduler):
                self.lr_scheduler.step()

            if isinstance(
                self.agent_lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                self.agent_lr_scheduler.step(first_metric)
            elif isinstance(
                self.agent_lr_scheduler, torch.optim.lr_scheduler.LRScheduler
            ):
                self.agent_lr_scheduler.step()

            cache_params = []
            update_best = []
            for metric_name in self.eval_metric_list:
                metric = metrics[metric_name]
                best_metric = self.best_metrics[metric_name]
                cache_params.append(
                    (metric - best_metric) / max(abs(best_metric), 1)
                    >= -self.eval_threshold
                )
                update_best.append(metric >= best_metric)
            cache_params = np.array(cache_params, dtype=np.bool_)
            update_best = np.array(update_best, dtype=np.bool_)

            if cache_params.any():
                self._cached_agent_group_params = _serialize_to_buffer(
                    self.eval_agent_group.state_dict()
                )
                self._cached_critic_params = _serialize_to_buffer(
                    self.eval_critic.state_dict()
                )
                logging.info(
                    f"Epoch {epoch}: Cached parameters updated with current parameters."
                )

            if update_best.any():
                self.best_metrics = metrics
                self.best_agent_group_params = _serialize_to_buffer(
                    self.eval_agent_group.state_dict()
                )
                self.best_critic_params = _serialize_to_buffer(
                    self.eval_critic.state_dict()
                )
                logging.info(
                    f"Epoch {epoch}: New best {first_metric_name}: {first_metric:.4f}"
                )

            if first_metric >= target_first_metric:
                logging.info(
                    f"Epoch {epoch}: {first_metric_name} reached: {first_metric:.4f} >= {target_first_metric:.4f}"
                )
                break

            if epoch % eval_interval == 0:
                self.eval_agent_group.load_state_dict(
                    _deserialize_from_buffer(self._cached_agent_group_params)
                )
                self.eval_critic.load_state_dict(
                    _deserialize_from_buffer(self._cached_critic_params)
                )
                self.update_target_model_params()
                logging.info(
                    f"Epoch {epoch}: Eval model and Target model updated with cached parameters."
                )

            if epoch % update_target_interval == 0:
                self.update_target_model_params()
                logging.info(
                    f"Epoch {epoch}: Target model updated with eval model parameters."
                )

        logging.info(
            f"Best strategy: {yaml.dump(self.best_metrics, default_flow_style=False, sort_keys=False)}"
        )
        self.save_best_model()
        return self.best_metrics

    def save_intermediate_results(self, epoch, metrics):
        self.training_history[epoch] = metrics
        os.makedirs(self.logdir, exist_ok=True)
        yaml_path = os.path.join(self.logdir, "results.yaml")
        with open(yaml_path, "w") as file:
            yaml.dump(self.training_history, file)
        logging.info(
            f"Intermediate results saved for epoch {epoch}. Results saved to {yaml_path}"
        )

    def __del__(self):
        if self.worker_group is not None:
            self.worker_group.shutdown()
