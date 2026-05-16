import os
import sys
import yaml
import torch
import random
import datetime
import numpy as np
from absl import logging
from typing import List, Union, Optional
from abc import abstractmethod

from marlite.environment import EnvConfig
from marlite.rollout import RolloutManagerConfig
from marlite.replaybuffer import ReplayBufferConfig
from marlite.algorithm.agents import AgentGroupConfig
from marlite.algorithm.critic import CriticConfig
from marlite.util.optimizer_config import OptimizerConfig
from marlite.util.lr_scheduler_config import LRSchedulerConfig
from marlite.util.scheduler import Scheduler
from marlite.analyzer import AnalyzerConfig
from marlite.util.serialization import (
    serialize_to_buffer,
    deserialize_from_buffer,
    get_state_dict,
    load_state_dict_into,
)


def get_device_list(train_device: Union[str, List[str]]) -> tuple:
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
        self.replaybuffer_config = replaybuffer_config
        self.rolloutmanager_config = rolloutmanager_config
        self.analyzer = analyzer_config.create_analyzer()

        self.agent_group_config = agent_group_config
        self.eval_agent_group = agent_group_config.get_agent_group()

        self.eval_critic = critic_config.get_critic()
        self.critic_optimizer_config = critic_optimizer_config

        self.critic_optimizer = self.critic_optimizer_config.get_optimizer(
            self.eval_critic.parameters()
        )
        self.agent_optimizer = self.agent_optimizer_config.get_optimizer(
            self.eval_agent_group.parameters()
        )

        self.lr_scheduler_conf = lr_scheduler_conf
        self.agent_lr_scheduler_conf = agent_lr_scheduler_conf

        if isinstance(lr_scheduler_conf, LRSchedulerConfig):
            self.lr_scheduler = lr_scheduler_conf.get_lr_scheduler(
                self.critic_optimizer
            )
        else:
            self.lr_scheduler = None

        if isinstance(agent_lr_scheduler_conf, LRSchedulerConfig):
            self.agent_lr_scheduler = agent_lr_scheduler_conf.get_lr_scheduler(
                self.agent_optimizer
            )
        else:
            self.agent_lr_scheduler = None

        self.workdir = workdir
        self.logdir = os.path.join(workdir, "logs")
        self.checkpointdir = os.path.join(workdir, "checkpoints")

        self.training_history = {}

        os.makedirs(self.logdir, exist_ok=True)
        logging.get_absl_handler().use_absl_log_file("training", self.logdir)
        logging.set_verbosity(logging.INFO)
        logging.get_absl_handler().python_handler.stream = sys.stdout

        self.train_device_config = train_device
        self.device_list, self.use_multi_gpu = get_device_list(train_device)
        self.worker_group = None

        if self.use_multi_gpu:
            self.train_device = self.device_list[0]
        else:
            self.train_device = self.device_list[0]
            logging.info(f"Using single device: {self.train_device}")

        self.compile_models = compile_models
        self.best_metrics = {key: -np.inf for key in self.eval_metric_list}
        self.current_epoch = 0

    def _setup_multi_gpu(self):
        if self.use_multi_gpu:
            self.train_device = self.device_list[0]
            self.worker_group = self._create_worker_group()
            if self.worker_group is not None:
                self.worker_group.start_workers()
                self._sync_params_to_workers()
            logging.info(
                f"Using multi-GPU training with {len(self.device_list)} devices: {self.device_list}"
            )

    def _compile_eval_models(self):
        if self.compile_models and not self.use_multi_gpu:
            logging.info("Compiling models...")
            self.eval_agent_group = torch.compile(
                self.eval_agent_group.to(self.train_device)
            ).to("cpu")
            self.eval_critic = torch.compile(
                self.eval_critic.to(self.train_device)
            ).to("cpu")

    @abstractmethod
    def _create_worker_group(self):
        pass

    def _sync_params_to_workers(self):
        if self.worker_group is None:
            return

        trainable_params = {
            "eval_agent_group": get_state_dict(self.eval_agent_group),
            "eval_critic": get_state_dict(self.eval_critic),
        }
        self._add_target_params_for_sync(trainable_params)
        self.worker_group.broadcast_params(trainable_params)

        critic_lr = self.critic_optimizer.param_groups[0]["lr"]
        agent_lr = self.agent_optimizer.param_groups[0]["lr"]
        self.worker_group.sync_lr_to_workers(critic_lr, agent_lr)

    def _add_target_params_for_sync(self, trainable_params):
        pass

    def _sync_eval_params_from_workers(self):
        if self.worker_group is None:
            return
        eval_params = self.worker_group.read_params_from_worker0()
        load_state_dict_into(
            self.eval_agent_group, eval_params["eval_agent_group"]
        )
        load_state_dict_into(self.eval_critic, eval_params["eval_critic"])

    @abstractmethod
    def learn(self, sample_size, batch_size: int, times: int):
        raise NotImplementedError

    def save_current_model(self, checkpoint: str):
        agent_path = os.path.join(self.checkpointdir, checkpoint, "agent")
        os.makedirs(agent_path, exist_ok=True)
        self.eval_agent_group.to("cpu")
        agent_params = get_state_dict(self.eval_agent_group)
        torch.save(agent_params, os.path.join(agent_path, "agent.pth"))

        critic_path = os.path.join(self.checkpointdir, checkpoint, "critic")
        os.makedirs(critic_path, exist_ok=True)
        self.eval_critic.to("cpu")
        critic_params = get_state_dict(self.eval_critic)
        torch.save(critic_params, os.path.join(critic_path, "critic.pth"))
        return self

    def load_checkpoint(self, checkpoint: str):
        self.best_metrics = {key: -np.inf for key in self.eval_metric_list}
        agent_path = os.path.join(self.checkpointdir, checkpoint, "agent", "agent.pth")
        self.eval_agent_group.to("cpu")
        self.eval_critic.to("cpu")
        load_state_dict_into(
            self.eval_agent_group, torch.load(agent_path, weights_only=True)
        )
        critic_path = os.path.join(
            self.checkpointdir, checkpoint, "critic", "critic.pth"
        )
        load_state_dict_into(
            self.eval_critic, torch.load(critic_path, weights_only=True)
        )
        return self

    def save_best_model(self):
        self.save_current_model(checkpoint="best")
        return self

    def collect_experience(self, epsilon: float):
        self.eval_agent_group.eval().to("cpu")
        serialized_params = serialize_to_buffer(
            get_state_dict(self.eval_agent_group)
        )
        manager = self.rolloutmanager_config.create_manager(
            self.agent_group_config, serialized_params, self.env_config, epsilon
        )
        episodes = manager.generate_episodes()

        for episode in episodes:
            self.replaybuffer.add_episode(episode)

        self.eval_agent_group.to("cpu")
        torch.cuda.empty_cache()
        return self

    def evaluate(self):
        self.eval_agent_group.eval().to("cpu")
        serialized_params = serialize_to_buffer(
            get_state_dict(self.eval_agent_group)
        )
        manager = self.rolloutmanager_config.create_eval_manager(
            self.agent_group_config,
            serialized_params,
            self.env_config,
            self.eval_epsilon,
        )

        episodes = manager.generate_episodes()

        result = self.analyzer(episodes)

        logging.info(f"Evaluation results:")
        for key in result.keys():
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

    @abstractmethod
    def train(self, **kwargs):
        raise NotImplementedError

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
        if hasattr(self, "worker_group") and self.worker_group is not None:
            self.worker_group.shutdown()
