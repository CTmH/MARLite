import yaml
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
from util.optimizer_config import OptimizerConfig

class TrainerConfig:
    def __init__(self, config_path):
        self.algorithm, self.config, self.train_args = load_config_from_yaml(config_path)
        self.trainer = None

    def create_trainer(self):
        if self.algorithm == 'QMIX':
            learner = QMIXTrainer(**self.config)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        self.trainer = learner
        return self
    
    def run(self):
        if self.trainer:
            return self.trainer.train(**self.train_args)
        else:
            raise ValueError("Trainer not created. Please call create_learner() first.")

def load_config_from_yaml(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Environment configuration
    # TODO: Implement Other environment configurations
    env_config = EnvConfig(**config['env_config'])
    env = env_config.create_env()
    env.reset()
    agent_list = env.agents
    key = env.agents[0]
    state_shape = env.state().shape
    n_agents = len(env.agents) # Number of agents in the environment
    action_space_shape = env.action_space(key).n.item()
    env.close()
    
    # Agent Group Configuration
    agent_group_config = deepcopy(config['agent_group_config'])
    agent_group_config = AgentGroupConfig(**agent_group_config)
    
    # Critic configuration
    critic_config = deepcopy(config['critic_config'])
    critic_optimizer_config = critic_config.pop('optimizer')
    critic_config = CriticConfig(**critic_config)
    critic_optimizer_config = OptimizerConfig(**critic_optimizer_config)

    # Scheduler
    scheduler_config = config['epsilon_scheduler']
    scheduler_config['decay_steps'] = config['trainer_config']['epochs']
    epsilon_scheduler = Scheduler(**scheduler_config)
    scheduler_config = config['sample_ratio_scheduler']
    scheduler_config['decay_steps'] = config['trainer_config']['epochs']
    sample_ratio_scheduler = Scheduler(**scheduler_config)

    return config['trainer_config']['algorithm'], {
        'env_config': env_config,
        'agent_group_config': agent_group_config,
        'critic_config': critic_config,
        'critic_optimizer_config': critic_optimizer_config,
        'epsilon_scheduler': epsilon_scheduler,
        'sample_ratio_scheduler': sample_ratio_scheduler,
        'traj_len': config['rolloutworker_config']['traj_len'],
        'n_workers': config['rolloutworker_config']['n_workers'],
        'buffer_capacity': config['buffer_config']['capacity'],
        'episode_limit': config['rolloutworker_config']['episode_limit'],
        'n_episodes': config['rolloutworker_config']['n_episodes'],
        'gamma': config['trainer_config']['gamma'],
        'epochs': config['trainer_config']['epochs'],
        'workdir': config['trainer_config']['workdir'],
        'device': config['trainer_config']['device']
    }, config['trainer_config']['train_args']

def get_optimizer(opt_name: str):
    if opt_name == "Adam":
        return torch.optim.Adam
    elif opt_name == "SGD":
        return torch.optim.SGD
    else:
        raise ValueError(f"Unknown optimizer name {opt_name}")