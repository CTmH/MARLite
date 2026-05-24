from typing import Dict, Type
from marlite.config_processor.config_processor import ConfigProcessor
from marlite.config_processor.qmix_config_processor import QMIXConfigProcessor, SemiSupervisedQMIXConfigProcessor
from marlite.config_processor.mappo_config_processor import MAPPOConfigProcessor, SemiSupervisedMAPPOConfigProcessor

REGISTERED_PROCESSORS: Dict[str, Type[ConfigProcessor]] = {
    'QMIX': QMIXConfigProcessor,
    'GraphQMIX': QMIXConfigProcessor,
    'MsgAggr': QMIXConfigProcessor,
    'ProbMsgAggr': QMIXConfigProcessor,
    'VAEGraphQMIX': SemiSupervisedQMIXConfigProcessor,
    'GroupConsensusQMIX': QMIXConfigProcessor,
    'VAEGroupConsensusQMIX': SemiSupervisedQMIXConfigProcessor,
    'MAPPO': MAPPOConfigProcessor,
    'VAEGroupConsensusMAPPO': SemiSupervisedMAPPOConfigProcessor,
    'G2ANetMAPPO': MAPPOConfigProcessor,
}

__all__ = ['ConfigProcessor', 'REGISTERED_PROCESSORS']