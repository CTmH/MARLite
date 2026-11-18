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
        self.parse_loss_mixer_config(trainer_kwargs)
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


class DistilledGroupCommunicationMAPPOConfigProcessor(MAPPOConfigProcessor):
    """Resolve the raw teacher from the same policy schema as the student."""

    def process(self, config):
        config = deepcopy(config)
        trainer = config["trainer"]
        teacher = trainer.pop("teacher")
        auxiliary_head = config.pop("auxiliary_head", None)
        trainer_kwargs, train_args, checkpoint = super().process(config)

        source = teacher.get("source")
        allowed = {
            "checkpoint": {"source", "actor_checkpoint"},
            "train": {"source", "initial_actor_checkpoint", "trainer", "critic",
                      "optimizer", "lr_scheduler"},
        }
        if source not in allowed or teacher.keys() - allowed[source]:
            raise ValueError("trainer.teacher must configure either checkpoint or train source")
        if source == "checkpoint" and not teacher.get("actor_checkpoint"):
            raise ValueError("A teacher checkpoint source requires actor_checkpoint")

        raw_config = deepcopy(config)
        raw_config["agent_group"]["message_mode"] = "raw"
        raw_actor, _, _ = self.parse_agent_group_config(raw_config)
        teacher_training = None
        if source == "train":
            raw_trainer = raw_config["trainer"]
            for key in ("offline", "online_distillation", "checkpoint"):
                raw_trainer.pop(key, None)
            raw_trainer.update(teacher.get("trainer", {}))
            raw_trainer.update(type="MAPPO", workdir=f"{trainer['workdir']}/teacher")
            if "checkpoint" in raw_trainer:
                raise ValueError("Use teacher.initial_actor_checkpoint to initialize a teacher")
            if "critic" in teacher:
                raw_config["critic"] = teacher["critic"]
            for key in ("optimizer", "lr_scheduler"):
                if key in teacher:
                    raw_config["agent_group"][key] = teacher[key]
            kwargs, args, _ = super().process(raw_config)
            if type(args.get("iterations")) is not int or args["iterations"] < 1:
                raise ValueError("A trained teacher needs at least one MAPPO iteration")
            teacher_training = (kwargs, args)

        trainer_kwargs.update(
            teacher=teacher, teacher_agent_config=raw_actor,
            teacher_training=teacher_training, auxiliary_head=auxiliary_head,
        )
        return trainer_kwargs, train_args, checkpoint
