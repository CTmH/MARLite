from copy import deepcopy
from typing import Dict, Type
from marlite.trainer.trainer import Trainer  # abstract base; keep for type hints
from marlite.trainer.qmix_trainer import QMIXTrainer
from marlite.trainer.graph_qmix_trainer import GraphQMIXTrainer
from marlite.trainer.msg_aggr_qmix_trainer import MsgAggrQMIXTrainer, ProbMsgAggrQMIXTrainer
from marlite.trainer.self_supervised_qmix_trainer import SelfSupervisedQMIXTrainer
from marlite.trainer.vae_graph_qmix_trainer import VAEGraphQMIXTrainer
from marlite.trainer.group_consensus_trainer import GroupConsensusTrainer
from marlite.trainer.vae_group_consensus_trainer import VAEGroupConsensusQMIXTrainer
from marlite.trainer.mappo_trainer import MAPPOTrainer
from marlite.config_processor import REGISTERED_PROCESSORS

REGISTERED_TRAINERS: Dict[str, Type[Trainer]] = {
    'QMIX': QMIXTrainer,
    'GraphQMIX': GraphQMIXTrainer,
    'MsgAggr': MsgAggrQMIXTrainer,
    'ProbMsgAggr': ProbMsgAggrQMIXTrainer,
    'VAEGraphQMIX': VAEGraphQMIXTrainer,
    'GroupConsensusQMIX': GroupConsensusTrainer,
    'VAEGroupConsensusQMIX': VAEGroupConsensusQMIXTrainer,
    'MAPPO': MAPPOTrainer,
}

class TrainerConfig:
    def __init__(self, config_dict: Dict[str, Dict]):
        self.config = deepcopy(config_dict)

        # Extract trainer_type to select the appropriate processor
        trainer_type = config_dict['trainer_config']['type']

        # Get the appropriate config processor
        processor_class = REGISTERED_PROCESSORS[trainer_type]
        processor = processor_class()
        self.trainer_class = REGISTERED_TRAINERS[trainer_type]

        # Process the configuration using the selected processor
        self.trainer_kwargs, self.train_args, self.checkpoint = processor.process(self.config)

        self.trainer = None

    def create_trainer(self) -> Trainer:
        self.trainer = self.trainer_class(**self.trainer_kwargs)
        if self.checkpoint:
            self.trainer.load_checkpoint(
                checkpoint=self.checkpoint
            )
        return self.trainer

    def run(self):
        self.create_trainer()
        if self.trainer:
            return self.trainer.train(**self.train_args)
        else:
            raise ValueError("Trainer not created. Please call create_trainer() first.")