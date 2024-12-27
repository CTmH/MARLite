import yaml
import torch
import numpy as np
from copy import deepcopy
from ..algorithm.model.model_config import ModelConfig
from ..algorithm.critic.critic_config import CriticConfig
from ..environment.env_config import EnvConfig
from ..util.scheduler import Scheduler
from .trainer import Trainer
from .qmix_trainer import QMIXTrainer

class TrainerConfig:
    def __init__(self, config_path):
        self.algorithm, self.config, self.train_params = load_config_from_yaml(config_path)
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
            return self.trainer.train(**self.train_params)
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
    # TODO: Add CNN for observation and state
    state_shape = env.state().shape
    n_agents = len(env.agents) # Number of agents in the environment
    action_space_shape = env.action_space(key).n.item()
    env.close()
    
    # agents
    agent_list = {agent_id: model_id for agent_id, model_id in zip(agent_list, config['agent_list'])}
    model_configs = {}
    feature_extractor_configs = {}
    for model in config['model_configs']:
        conf = deepcopy(config['model_configs'][model])
        if 'feature_extractor' in conf:  # Check if feature extractor is defined in the model configuration. If not, use Identity as default.
            fe_conf = conf.pop('feature_extractor')
        else:
            fe_conf = {'model_type': 'Identity'}
        model_configs[model] = ModelConfig(**conf)
        feature_extractor_configs[model] = ModelConfig(**fe_conf)
    
    # Critic configuration
    critic_config = deepcopy(config['critic_config'])
    if 'feature_extractor' in critic_config:  # Check if feature extractor is defined in the critic configuration. If not, use Identity as default.
        fe_conf = critic_config.pop('feature_extractor')
    else:
        fe_conf = {'model_type': 'Identity'}
    critic = CriticConfig(**critic_config)
    critic = critic.get_critic()
    critic_feature_extractor = ModelConfig(**fe_conf)
    critic_feature_extractor = critic_feature_extractor.get_model()

    # Scheduler
    scheduler_config = config['epsilon_scheduler']
    scheduler_config['decay_steps'] = config['trainer_config']['epochs']
    epsilon_scheduler = Scheduler(**scheduler_config)
    scheduler_config = config['sample_ratio_scheduler']
    scheduler_config['decay_steps'] = config['trainer_config']['epochs']
    sample_ratio_scheduler = Scheduler(**scheduler_config)

    return config['trainer_config']['algorithm'], {
        'agents': agent_list,
        'env_config': env_config,
        'model_configs': model_configs,
        'epsilon_scheduler': epsilon_scheduler,
        'sample_ratio_scheduler': sample_ratio_scheduler,
        'critic_config': critic_config,
        'traj_len': config['rolloutworker_config']['traj_len'],
        'n_workers': config['rolloutworker_config']['n_workers'],
        'epochs': config['trainer_config']['epochs'],
        'buffer_capacity': config['buffer_config']['capacity'],
        'episode_limit': config['rolloutworker_config']['episode_limit'],
        'n_episodes': config['rolloutworker_config']['n_episodes'],
        'gamma': config['trainer_config']['gamma'],
        'critic_lr': config['trainer_config']['critic_lr'],
        'critic_optimizer': get_optimizer(config['trainer_config']['critic_optimizer']),
        'workdir': config['trainer_config']['workdir'],
        'device': config['trainer_config']['device']
    }, {
        'target_reward': config['trainer_config']['target_reward'],
        'eval_interval': config['trainer_config']['eval_interval'],
        'batch_size': config['trainer_config']['batch_size'],
        'learning_times_per_epoch': config['trainer_config']['learning_times_per_epoch'],
    }

def get_optimizer(opt_name: str):
    if opt_name == "Adam":
        return torch.optim.Adam
    elif opt_name == "SGD":
        return torch.optim.SGD
    else:
        raise ValueError(f"Unknown optimizer name {opt_name}")