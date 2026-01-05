from copy import deepcopy
from typing import Dict, Tuple, Type
import numpy as np
from marlite.algorithm.agents import AgentGroupConfig
from marlite.algorithm.critic import CriticConfig
from marlite.environment import EnvConfig
from marlite.rollout import RolloutManagerConfig
from marlite.replaybuffer import ReplayBufferConfig
from marlite.util.scheduler import Scheduler
from marlite.util.optimizer_config import OptimizerConfig
from marlite.util.lr_scheduler_config import LRSchedulerConfig
from marlite.analyzer import AnalyzerConfig
from marlite.trainer.trainer import Trainer
from marlite.trainer.qmix_trainer import QMIXTrainer
from marlite.trainer.graph_qmix_trainer import GraphQMIXTrainer
from marlite.trainer.msg_aggr_qmix_trainer import MsgAggrQMIXTrainer, ProbMsgAggrQMIXTrainer
from marlite.trainer.semi_supervised_qmix_trainer import SemiSupervisedQMIXTrainer
from marlite.trainer.vae_graph_qmix_trainer import VAEGraphQMIXTrainer
from marlite.util.self_supervised_data_constructor.self_supervised_data_constructor_config import SelfSupervisedDataConstructorConfig

REGISTERED_TRAINERS = {
    'QMIX': QMIXTrainer,
    'GraphQMIX': GraphQMIXTrainer,
    'MsgAggr': MsgAggrQMIXTrainer,
    'ProbMsgAggr': ProbMsgAggrQMIXTrainer,
}

def process_config_dict(config_dict: Dict) -> Tuple[Dict, Type[Trainer], Dict, str, float]:
    # Create deep copy to avoid modifying original dict
    config = deepcopy(config_dict)

    # Process agent group config
    agent_group_config = AgentGroupConfig(**config['agent_group_config'])

    # Process environment config
    env_config = EnvConfig(**config['env_config'])

    # Process critic config
    critic_conf = config['critic_config']
    critic_optimizer_conf = critic_conf.pop('optimizer')
    critic_optimizer_config = OptimizerConfig(**critic_optimizer_conf)
    lr_scheduler_conf = LRSchedulerConfig(**critic_conf.pop('lr_scheduler')) if 'lr_scheduler' in critic_conf else None
    critic_config = CriticConfig(**critic_conf)

    # Process rollout and replay buffer configs
    rolloutmanager_config = RolloutManagerConfig(**config['rollout_config'])
    replaybuffer_config = ReplayBufferConfig(**config['replaybuffer_config'])

    # Process schedulers
    epsilon_scheduler = Scheduler(**config['epsilon_scheduler'])
    sample_ratio_scheduler = Scheduler(**config['sample_ratio_scheduler'])

    # Process analyzer config
    analyzer_config = AnalyzerConfig(**config['analyzer_config'])

    # Process trainer config
    trainer_config = config['trainer_config']
    trainer_type = trainer_config.pop('type')

    # Extract training arguments and checkpoint settings
    train_args = trainer_config.pop('train_args')
    checkpoint = trainer_config.pop('checkpoint', None)

    # Map trainer type to class
    trainer_class = REGISTERED_TRAINERS[trainer_type]

    # Build kwargs dictionary
    trainer_kwargs = {
        'env_config': env_config,
        'agent_group_config': agent_group_config,
        'critic_config': critic_config,
        'epsilon_scheduler': epsilon_scheduler,
        'sample_ratio_scheduler': sample_ratio_scheduler,
        'critic_optimizer_config': critic_optimizer_config,
        'lr_scheduler_conf': lr_scheduler_conf,
        'rolloutmanager_config': rolloutmanager_config,
        'replaybuffer_config': replaybuffer_config,
        'analyzer_config': analyzer_config,
        **trainer_config
    }

    return trainer_kwargs, trainer_class, train_args, checkpoint

class TrainerConfig:
    def __init__(self, config_dict: Dict[str, Dict]):
        self.config = deepcopy(config_dict)
        self.trainer_kwargs, self.trainer_class, self.train_args, self.checkpoint = process_config_dict(self.config)
        self.trainer = None

    def create_trainer(self) -> Trainer:
        self.trainer = self.trainer_class(**self.trainer_kwargs)
        if self.checkpoint:
            self.trainer.load_checkpoint(
                checkpoint=self.checkpoint,
                checkpoint_mean_reward=self.checkpoint_mean_reward
            )
        return self.trainer

    def run(self):
        self.create_trainer()
        if self.trainer:
            return self.trainer.train(**self.train_args)
        else:
            raise ValueError("Trainer not created. Please call create_learner() first.")
'''
class TrainerConfig:
    def __init__(self, config_dict: Dict[str, Dict]):
        self.config = deepcopy(config_dict)
        self.agent_group_config = AgentGroupConfig(**self.config['agent_group_config'])
        self.env_config = EnvConfig(**self.config['env_config'])
        critic_conf = self.config['critic_config']
        critic_optimizer_conf = critic_conf.pop('optimizer')
        self.critic_optimizer_config = OptimizerConfig(**critic_optimizer_conf)
        if 'lr_scheduler' in critic_conf.keys():
            lr_scheduler_conf = critic_conf.pop('lr_scheduler')
            self.lr_scheduler_conf = LRSchedulerConfig(**lr_scheduler_conf)
        else:
            self.lr_scheduler_conf = None
        self.critic_config = CriticConfig(**critic_conf)
        self.rolloutmanager_config = RolloutManagerConfig(**self.config['rollout_config'])
        self.replaybuffer_config = ReplayBufferConfig(**self.config['replaybuffer_config'])

        self.epsilon_scheduler = Scheduler(**self.config['epsilon_scheduler'])
        self.sample_ratio_scheduler = Scheduler(**self.config['sample_ratio_scheduler'])

        self.analyzer_config = AnalyzerConfig(**self.config['analyzer_config'])

        self.trainer_config = self.config['trainer_config']
        self.trainer_type = self.trainer_config.pop('type')
        self.train_args = self.trainer_config.pop('train_args')
        self.checkpoint = self.trainer_config.pop('checkpoint', None)
        self.checkpoint_mean_reward = self.trainer_config.pop('checkpoint_mean_reward', -np.inf)
        self.trainer = None

        self.registered_trainers = {
            'QMIX': QMIXTrainer,
            'GraphQMIX': GraphQMIXTrainer,
            'MsgAggr': MsgAggrQMIXTrainer,
            'ProbMsgAggr': ProbMsgAggrQMIXTrainer,
        }

    def create_trainer(self) -> Trainer:
        if self.trainer_type in self.registered_trainers:
            trainer_class = self.registered_trainers[self.trainer_type]
            self.trainer = trainer_class(
                env_config=self.env_config,
                agent_group_config = self.agent_group_config,
                critic_config = self.critic_config,
                epsilon_scheduler = deepcopy(self.epsilon_scheduler),
                sample_ratio_scheduler = deepcopy(self.sample_ratio_scheduler),
                critic_optimizer_config = self.critic_optimizer_config,
                lr_scheduler_conf = self.lr_scheduler_conf,
                rolloutmanager_config = self.rolloutmanager_config,
                replaybuffer_config = self.replaybuffer_config,
                analyzer_config = self.analyzer_config,
                **self.trainer_config
            )
            if self.checkpoint:
                self.trainer.load_checkpoint(checkpoint=self.checkpoint,
                                             checkpoint_mean_reward=self.checkpoint_mean_reward)
        else:
            raise ValueError(f"Unsupported algorithm: {self.trainer_type}")
        return self.trainer

    def run(self):
        self.create_trainer()
        if self.trainer:
            return self.trainer.train(**self.train_args)
        else:
            raise ValueError("Trainer not created. Please call create_learner() first.")
'''