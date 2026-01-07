from typing import Dict, Type
from marlite.config_processor.config_processor import ConfigProcessor
from marlite.config_processor.qmix_config_processor import QMIXConfigProcessor, SemiSupervisedQMIXConfigProcessor
# Register processors for each trainer type
REGISTERED_PROCESSORS: Dict[str, Type[ConfigProcessor]] = {
    'QMIX': QMIXConfigProcessor,
    'GraphQMIX': QMIXConfigProcessor,
    'MsgAggr': QMIXConfigProcessor,
    'ProbMsgAggr': QMIXConfigProcessor,
    'VAEGraphQMIX': SemiSupervisedQMIXConfigProcessor,
}

__all__ = ['ConfigProcessor', 'REGISTERED_PROCESSORS']