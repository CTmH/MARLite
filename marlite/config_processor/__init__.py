from typing import Dict, Type
from marlite.config_processor.config_processor import ConfigProcessor
from marlite.config_processor.qmix_config_processor import QMIXConfigProcessor, SemiSupervisedQMIXConfigProcessor
from marlite.config_processor.mappo_config_processor import MAPPOConfigProcessor, SemiSupervisedMAPPOConfigProcessor
from marlite.config_processor.qtran_config_processor import QTRANConfigProcessor
from marlite.config_processor.qplex_config_processor import QPLEXConfigProcessor

REGISTERED_PROCESSORS: Dict[str, Type[ConfigProcessor]] = {
    'QMIX': QMIXConfigProcessor,
    'QTRAN': QTRANConfigProcessor,
    'QPLEX': QPLEXConfigProcessor,
    'GraphQMIX': QMIXConfigProcessor,
    'MsgAggr': QMIXConfigProcessor,
    'ProbMsgAggr': QMIXConfigProcessor,
    'VAEGraphQMIX': SemiSupervisedQMIXConfigProcessor,
    'GroupConsensusQMIX': QMIXConfigProcessor,
    'SSLGroupConsensusQMIX': SemiSupervisedQMIXConfigProcessor,
    'MAPPO': MAPPOConfigProcessor,
    'SSLGroupConsensusMAPPO': SemiSupervisedMAPPOConfigProcessor,
    'GraphMAPPO': MAPPOConfigProcessor,
    'G2ANetMAPPO': MAPPOConfigProcessor,  # backward-compatible alias
}

__all__ = ['ConfigProcessor', 'REGISTERED_PROCESSORS']