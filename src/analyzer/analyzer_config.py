from copy import deepcopy
from .analyzer import Analyzer
from ..algorithm.agents.agent_group_config import AgentGroupConfig
from ..environment.env_config import EnvConfig
from ..rollout.rolloutmanager_config import RolloutManagerConfig
from ..util.scheduler import Scheduler

class AnalyzerConfig:
    def __init__(self, config_dict: dict):
        """Initialize Analyzer configuration from dictionary"""
        self.config = deepcopy(config_dict)

        # Core configuration components
        self.workdir = self.config['trainer_config']['workdir']
        self.agent_group_config = AgentGroupConfig(**self.config['agent_group_config'])
        self.env_config = EnvConfig(**self.config['env_config'])
        self.rolloutmanager_config = RolloutManagerConfig(**self.config['rollout_config'])

        self.registered_analyzers = {
            'Default': Analyzer
        }

    def create_analyzer(self, analyzer_type: str = 'Default') -> Analyzer:
        if analyzer_type in self.registered_analyzers:
            trainer_class = self.registered_analyzers[analyzer_type]
            analyzer = trainer_class(
                workdir=self.workdir,
                env_config=self.env_config,
                agent_group_config=self.agent_group_config,
                rolloutmanager_config=self.rolloutmanager_config,
            )
        else:
            raise ValueError(f"Unsupported algorithm: {analyzer_type}")
        return analyzer