from typing import Dict, Tuple, Type, Any
from copy import deepcopy

from marlite.algorithm.agents import AgentGroupConfig
from marlite.algorithm.critic import CriticConfig
from marlite.algorithm.model import ModelConfig
from marlite.environment import EnvConfig
from marlite.rollout import RolloutManagerConfig
from marlite.replaybuffer import ReplayBufferConfig
from marlite.util.scheduler import Scheduler
from marlite.util.optimizer_config import OptimizerConfig
from marlite.util.lr_scheduler_config import LRSchedulerConfig
from marlite.util.loss_func import REGISTERED_RECONSTRUCTION_LOSS
from marlite.analyzer import AnalyzerConfig
from marlite.util.self_supervised_data_constructor.self_supervised_data_constructor_config import SelfSupervisedDataConstructorConfig
from marlite.config_processor.config_processor import ConfigProcessor


class QMIXConfigProcessor(ConfigProcessor):
    def get_common_kwargs(self, config: Dict) -> Dict:
        """Process common configurations and return common kwargs for all trainers"""
        # Process common configurations
        agent_group_config = AgentGroupConfig(**config['agent_group_config'])
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

        # Build common kwargs dictionary
        common_kwargs = {
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
        }

        return common_kwargs

    def get_specific_kwargs(self, trainer_config: Dict) -> Dict:
        """Process the config and return specific kwargs for the trainer"""
        return {}

    def process(self, config: Dict) -> Tuple[Dict, Dict, str]:
        """Process the config and return trainer_kwargs, trainer_class, train_args, and checkpoint"""
        # Create deep copy to avoid modifying original dict
        config_copy = deepcopy(config)

        # Process trainer config to extract trainer_type
        trainer_config = config_copy['trainer_config']
        trainer_type = trainer_config.pop('type')

        # Extract training arguments and checkpoint settings
        train_args = trainer_config.pop('train_args')
        checkpoint = trainer_config.pop('checkpoint', None)

        # Process common and specific kwargs
        common_kwargs = self.get_common_kwargs(config_copy)
        specific_kwargs = self.get_specific_kwargs(trainer_config)

        # Combine common and specific kwargs
        trainer_kwargs = {**common_kwargs, **specific_kwargs}

        # Add remaining items from trainer_config to trainer_kwargs
        trainer_kwargs.update(trainer_config)

        return trainer_kwargs, train_args, checkpoint

class SemiSupervisedQMIXConfigProcessor(QMIXConfigProcessor):
    def get_specific_kwargs(self, trainer_config: Dict) -> Dict:
        # Build ModelConfig from the configuration
        decoder_config = ModelConfig(**trainer_config.pop('decoder_config'))
        data_constructor_config = SelfSupervisedDataConstructorConfig(**trainer_config.pop('data_constructor_config'))
        reconstruction_loss_config: Dict[str, Any] = trainer_config.pop('reconstruction_loss_config')
        reconstruction_loss_type = reconstruction_loss_config.pop('type')
        reconstruction_loss_class = REGISTERED_RECONSTRUCTION_LOSS[reconstruction_loss_type]
        reconstruction_loss = reconstruction_loss_class(**reconstruction_loss_config)
        return {'decoder_config': decoder_config, 'data_constructor_config': data_constructor_config, 'reconstruction_loss': reconstruction_loss}