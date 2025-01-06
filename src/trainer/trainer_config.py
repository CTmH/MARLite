import torch
import numpy as np
from copy import deepcopy
from ..algorithm.model.model_config import ModelConfig
from ..algorithm.agents.agent_group_config import AgentGroupConfig
from ..algorithm.critic.critic_config import CriticConfig
from ..environment.env_config import EnvConfig
from ..util.scheduler import Scheduler
from ..util.optimizer_config import OptimizerConfig
from .trainer import Trainer
from .qmix_trainer import QMIXTrainer
from ..util.optimizer_config import OptimizerConfig
from ..rollout.rolloutmanager_config import RolloutManagerConfig
from ..replaybuffer.replaybuffer_config import ReplayBufferConfig

class TrainerConfig:
    def __init__(self, config_dict: dict):
        self.config = deepcopy(config_dict)
        self.agent_group_config = AgentGroupConfig(**self.config['agent_group_config'])
        self.env_config = EnvConfig(**self.config['env_config'])
        critic_conf = self.config['critic_config']
        critic_optimizer_conf = critic_conf.pop('optimizer')
        self.critic_config = CriticConfig(**critic_conf)
        self.critic_optimizer_config = OptimizerConfig(**critic_optimizer_conf)
        self.rolloutmanager_config = RolloutManagerConfig(**self.config['rollout_config'])
        self.replaybuffer_config = ReplayBufferConfig(**self.config['replaybuffer_config'])

        self.epsilon_scheduler = Scheduler(**self.config['epsilon_scheduler'])
        self.sample_ratio_scheduler = Scheduler(**self.config['sample_ratio_scheduler'])

        self.trainer_config = self.config['trainer_config']
        self.trainer_type = self.trainer_config.pop('type')
        self.train_args = self.trainer_config.pop('train_args')
        self.checkpoint = self.trainer_config.pop('checkpoint', None)
        self.trainer = None

        self.registered_trainers = {
            'QMIX': QMIXTrainer,
        }

    def create_trainer(self) -> Trainer:
        if self.trainer_type in self.registered_trainers:
            trainer_class = self.registered_trainers[self.trainer_type]
            self.trainer = trainer_class(
                env_config=self.env_config,
                agent_group_config = self.agent_group_config,
                critic_config = self.critic_config,
                epsilon_scheduler = self.epsilon_scheduler,
                sample_ratio_scheduler = self.sample_ratio_scheduler,
                critic_optimizer_config = self.critic_optimizer_config,
                rolloutmanager_config = self.rolloutmanager_config,
                replaybuffer_config = self.replaybuffer_config,
                **self.trainer_config
            )
            if self.checkpoint:
                self.trainer.load_model(self.checkpoint)
        else:
            raise ValueError(f"Unsupported algorithm: {self.trainer_type}")
        return self.trainer
    
    def run(self):
        self.create_trainer()
        if self.trainer:
            return self.trainer.train(**self.train_args)
        else:
            raise ValueError("Trainer not created. Please call create_learner() first.")