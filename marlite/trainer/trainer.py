import os
import sys
import yaml
import torch
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


def _extract_cuda_ids(device_list: List[str]) -> List[int]:
    """Return integer GPU ids from a list of ``"cuda:N"`` strings."""
    return [int(d.split(":")[-1]) for d in device_list]


class Trainer:
    def __init__(
        self,
        env_config: EnvConfig,
        agent_group_config: AgentGroupConfig,
        critic_config: CriticConfig,
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
        workdir: str = "",
        train_device: Union[str, List[str]] = "cpu",
        n_workers: int = 1,
        compile_models: bool = False,
        sample_mode: str = "ratio",
        max_grad_norm: float = 5.0,
        reward_aggr_mode: str = "sum",
    ):
        self.env_config = env_config
        self.critic_config = critic_config
        self.agent_optimizer_config = agent_optimizer_config
        self.sample_ratio = sample_ratio_scheduler
        self.gamma = gamma
        self.n_workers = n_workers
        self.eval_metric_list = eval_metric_list
        self.sample_mode = sample_mode
        self.max_grad_norm = max_grad_norm
        self.reward_aggr_mode = reward_aggr_mode

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
                if self.train_device.startswith("cuda"):
                    self.worker_group.move_models_to_gpu()
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
            "reward_aggr_mode": self.reward_aggr_mode,
        }
        # Off-policy trainers (subclasses) override _add_target_params_for_sync
        # to also push target_agent_group, target_critic, target_update_mode,
        # target_update_tau, target_update_interval.
        self._add_target_params_for_sync(trainable_params)
        self.worker_group.broadcast_params(trainable_params)

        critic_lr = self.critic_optimizer.param_groups[0]["lr"]
        agent_lr = self.agent_optimizer.param_groups[0]["lr"]
        self.worker_group.sync_lr_to_workers(
            critic_lr, agent_lr, **self._extra_sync_kwargs()
        )

    def _extra_sync_kwargs(self) -> dict:
        """Return extra learning-rate kwargs forwarded to workers via SYNC_LR.

        Subclasses that maintain additional optimizers (e.g. ``v_optimizer`` for
        QTRAN, ``ssl_optimizer`` for self-supervised variants) override this
        to inject their LRs into the per-epoch sync payload.

        Returns:
            Mapping of additional kwarg names to values.  Default is empty.
        """
        return {}

    def _add_target_params_for_sync(self, trainable_params):
        pass

    def _get_device_ids(self) -> List[int]:
        """Return integer GPU ids from ``self.device_list``."""
        return _extract_cuda_ids(self.device_list)

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
        """Save model parameters to disk.

        Base implementation is abstract — subclasses must override to
        save the models they own (on-policy: eval only; off-policy:
        eval + target; SSL variants: eval + target + ssl_model).
        """
        raise NotImplementedError("subclass must implement save_current_model")

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
        raise NotImplementedError("subclass must implement save_best_model")

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

    def _aggregate_rewards(self, rewards: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Aggregate per-agent rewards along the agent dimension.

        At rollout time, dead / missing agents get reward zero-padded via
        ``ensure_all_agents_present`` in ``env_util.py``, so the batch tensor
        always has shape ``(..., n_agents)`` with zeros for dead agents.

        Modes:
            ``"sum"`` (default):  sum across all agents (dead agents contribute 0).
            ``"mean"``:           divide by ``n_agents`` (including dead ones).
        """
        if self.reward_aggr_mode == "sum":
            return rewards.sum(dim=dim)
        elif self.reward_aggr_mode == "mean":
            return rewards.mean(dim=dim)
        else:
            raise ValueError(
                f"Unknown reward_aggr_mode '{self.reward_aggr_mode}'. "
                "Expected 'sum' or 'mean'."
            )

    def __del__(self):
        if hasattr(self, "worker_group") and self.worker_group is not None:
            self.worker_group.shutdown()
