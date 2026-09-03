"""MAPPO config processor — skips epsilon_scheduler (not used by on-policy agents)."""

from typing import Dict, Tuple
from copy import deepcopy

from marlite.util.scheduler import Scheduler
from marlite.util.scheduler_config import SchedulerConfig
from marlite.config_processor.qmix_config_processor import (
    QMIXConfigProcessor,
    SemiSupervisedQMIXConfigProcessor,
)


class MAPPOConfigProcessor(QMIXConfigProcessor):
    """Config processor for MAPPO (on-policy) trainers.

    On-policy agents always sample from the policy distribution when
    collecting episodes; epsilon-based exploration is not used.
    """

    def parse_trainer_config(
        self, config: Dict[str, Dict]
    ) -> Tuple[Scheduler, Scheduler, Dict, Dict, str, str]:
        trainer_config = deepcopy(config["trainer"])
        train_args = trainer_config.pop("train_args")
        checkpoint = trainer_config.pop("checkpoint", None)
        trainer_type = trainer_config.pop("type")

        trainer_config.pop("epsilon_scheduler", None)
        trainer_config.pop("entropy_coef_scheduler", None)

        sample_ratio_scheduler = SchedulerConfig(
            **trainer_config.pop("sample_ratio_scheduler")
        ).get_scheduler()

        return (
            None,
            sample_ratio_scheduler,
            trainer_config,
            train_args,
            checkpoint,
            trainer_type,
        )

    def process(self, config: Dict[str, Dict]) -> Tuple[Dict, Dict, str]:
        config = deepcopy(config)
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
            _epsilon_scheduler,
            sample_ratio_scheduler,
            trainer_config,
            train_args,
            checkpoint,
            trainer_type,
        ) = self.parse_trainer_config(config)

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
            "sample_ratio_scheduler": sample_ratio_scheduler,
        }
        trainer_kwargs.update(trainer_config)

        return trainer_kwargs, train_args, checkpoint


class SemiSupervisedMAPPOConfigProcessor(
    SemiSupervisedQMIXConfigProcessor, MAPPOConfigProcessor
):
    """Config processor for SSLGroupConsensusMAPPO (SSL + on-policy MAPPO)."""

    def process(self, config: Dict[str, Dict]) -> Tuple[Dict, Dict, str]:
        config = deepcopy(config)
        trainer_kwargs, train_args, checkpoint = MAPPOConfigProcessor.process(
            self, config
        )
        gate_scheduler_config = trainer_kwargs.pop(
            "rl_consensus_gate_scheduler", None
        )
        if gate_scheduler_config is not None:
            gate_scheduler_config = deepcopy(gate_scheduler_config)
            trainer_kwargs["rl_consensus_gate_scheduler"] = SchedulerConfig(
                **gate_scheduler_config
            ).get_scheduler()
        (
            ssl_model_config,
            ssl_optimizer_config,
            ssl_lr_scheduler_conf,
            data_constructor_config,
            reconstruction_loss,
        ) = self.parse_self_supervised_learning_config(config)
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
