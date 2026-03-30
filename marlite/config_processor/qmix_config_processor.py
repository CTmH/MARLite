from typing import Dict, Tuple, Type, Any, Optional
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
from marlite.util.self_supervised_data_constructor.self_supervised_data_constructor_config import (
    SelfSupervisedDataConstructorConfig,
)
from marlite.config_processor.config_processor import ConfigProcessor


class QMIXConfigProcessor(ConfigProcessor):
    def parse_agent_group_config(
        self, config: Dict[str, Dict]
    ) -> Tuple[AgentGroupConfig, OptimizerConfig, Optional[LRSchedulerConfig]]:
        """Parse agent group configuration and extract optimizer/lr_scheduler"""
        agent_group_conf = deepcopy(config["agent_group_config"])
        agent_optimizer_conf = agent_group_conf.pop("optimizer")
        agent_optimizer_config = OptimizerConfig(**agent_optimizer_conf)
        agent_lr_scheduler_conf = (
            LRSchedulerConfig(**agent_group_conf.pop("lr_scheduler"))
            if "lr_scheduler" in agent_group_conf
            else None
        )
        agent_group_config = AgentGroupConfig(**agent_group_conf)
        return agent_group_config, agent_optimizer_config, agent_lr_scheduler_conf

    def parse_env_config(self, config: Dict[str, Dict]) -> EnvConfig:
        """Parse environment configuration"""
        return EnvConfig(**config["env_config"])

    def parse_critic_config(
        self, config: Dict[str, Dict]
    ) -> Tuple[CriticConfig, OptimizerConfig, LRSchedulerConfig]:
        """Parse critic configuration"""
        critic_conf = config["critic_config"].copy()
        critic_optimizer_conf = critic_conf.pop("optimizer")
        critic_optimizer_config = OptimizerConfig(**critic_optimizer_conf)
        lr_scheduler_conf = (
            LRSchedulerConfig(**critic_conf.pop("lr_scheduler"))
            if "lr_scheduler" in critic_conf
            else None
        )
        critic_config = CriticConfig(**critic_conf)
        return critic_config, critic_optimizer_config, lr_scheduler_conf

    def parse_rollout_config(self, config: Dict[str, Dict]) -> RolloutManagerConfig:
        """Parse rollout configuration"""
        return RolloutManagerConfig(**config["rollout_config"])

    def parse_replaybuffer_config(self, config: Dict[str, Dict]) -> ReplayBufferConfig:
        """Parse replay buffer configuration"""
        return ReplayBufferConfig(**config["replaybuffer_config"])

    def parse_analyzer_config(self, config: Dict[str, Dict]) -> AnalyzerConfig:
        """Parse analyzer configuration"""
        return AnalyzerConfig(**config["analyzer_config"])

    def parse_trainer_config(
        self, config: Dict[str, Dict]
    ) -> Tuple[Scheduler, Scheduler, Dict, Dict, str, str]:
        """Parse trainer configuration and return trainer_config, train_args, checkpoint and trainer_type"""
        trainer_config = deepcopy(config["trainer_config"])

        # Extract training arguments and checkpoint settings
        train_args = trainer_config.pop("train_args")
        checkpoint = trainer_config.pop("checkpoint", None)
        trainer_type = trainer_config.pop("type")

        # Handle epsilon_scheduler and sample_ratio_scheduler as part of trainer_config
        epsilon_scheduler = Scheduler(**trainer_config.pop("epsilon_scheduler"))
        sample_ratio_scheduler = Scheduler(
            **trainer_config.pop("sample_ratio_scheduler")
        )

        return (
            epsilon_scheduler,
            sample_ratio_scheduler,
            trainer_config,
            train_args,
            checkpoint,
            trainer_type,
        )

    def process(self, config: Dict[str, Dict]) -> Tuple[Dict, Dict, str]:
        """Process the config and return trainer_kwargs, train_args, and checkpoint"""
        config = deepcopy(config)
        # Parse individual components
        agent_group_config, agent_optimizer_config, agent_lr_scheduler_conf = (
            self.parse_agent_group_config(config)
        )
        env_config = self.parse_env_config(config)
        critic_config, critic_optimizer_config, lr_scheduler_conf = (
            self.parse_critic_config(config)
        )
        rolloutmanager_config = self.parse_rollout_config(config)
        replaybuffer_config = self.parse_replaybuffer_config(config)
        analyzer_config = self.parse_analyzer_config(config)
        (
            epsilon_scheduler,
            sample_ratio_scheduler,
            trainer_config,
            train_args,
            checkpoint,
            trainer_type,
        ) = self.parse_trainer_config(config)

        # Build trainer kwargs dictionary
        trainer_kwargs = {
            "env_config": env_config,
            "agent_group_config": agent_group_config,
            "agent_optimizer_config": agent_optimizer_config,
            "agent_lr_scheduler_conf": agent_lr_scheduler_conf,
            "critic_config": critic_config,
            "critic_optimizer_config": critic_optimizer_config,
            "lr_scheduler_conf": lr_scheduler_conf,
            "rolloutmanager_config": rolloutmanager_config,
            "replaybuffer_config": replaybuffer_config,
            "analyzer_config": analyzer_config,
            "epsilon_scheduler": epsilon_scheduler,
            "sample_ratio_scheduler": sample_ratio_scheduler,
        }

        # Add remaining items from trainer_config to trainer_kwargs
        trainer_kwargs.update(trainer_config)

        return trainer_kwargs, train_args, checkpoint


class SemiSupervisedQMIXConfigProcessor(QMIXConfigProcessor):
    def parse_self_supervised_learning_config(
        self, config: Dict[str, Dict]
    ) -> Tuple[ModelConfig, SelfSupervisedDataConstructorConfig, Any]:
        """Parse self-supervised learning configuration"""
        ssl_config = deepcopy(config["self_supervised_learning_config"])
        ssl_model_config = ModelConfig(**ssl_config.pop("model"))
        ssl_optimizer_config = OptimizerConfig(**ssl_config.pop("optimizer"))
        ssl_lr_scheduler_conf = LRSchedulerConfig(**ssl_config.pop("lr_scheduler"))
        data_constructor_config = SelfSupervisedDataConstructorConfig(
            **ssl_config.pop("data_constructor_config")
        )

        reconstruction_loss_config: Dict[str, Any] = ssl_config.pop(
            "reconstruction_loss_config"
        )
        reconstruction_loss_type = reconstruction_loss_config.pop("type")
        reconstruction_loss_class = REGISTERED_RECONSTRUCTION_LOSS[
            reconstruction_loss_type
        ]
        reconstruction_loss = reconstruction_loss_class(**reconstruction_loss_config)

        return (
            ssl_model_config,
            ssl_optimizer_config,
            ssl_lr_scheduler_conf,
            data_constructor_config,
            reconstruction_loss,
        )

    def process(self, config: Dict[str, Dict]) -> Tuple[Dict, Dict, str]:
        """Process the config and return trainer_kwargs, train_args, and checkpoint"""
        config = deepcopy(config)
        # Parse individual components using parent class
        trainer_kwargs, train_args, checkpoint = super().process(config)

        # Parse self-supervised learning config specific to this class
        (
            ssl_model_config,
            ssl_optimizer_config,
            ssl_lr_scheduler_conf,
            data_constructor_config,
            reconstruction_loss,
        ) = self.parse_self_supervised_learning_config(config)

        # Add self-supervised learning specific configs to trainer_kwargs
        trainer_kwargs.update(
            {
                "ssl_model_config": ssl_model_config,
                "ssl_optimizer_config": ssl_optimizer_config,
                "ssl_lr_scheduler_conf": ssl_lr_scheduler_conf,
                "data_constructor_config": data_constructor_config,
                "reconstruction_loss": reconstruction_loss,
            }
        )

        return trainer_kwargs, train_args, checkpoint
