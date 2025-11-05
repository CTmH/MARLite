from copy import deepcopy
from marlite.analyzer.analyzer import Analyzer
from ..algorithm.agents.agent_group_config import AgentGroupConfig
from ..environment.env_config import EnvConfig
from ..rollout.rolloutmanager_config import RolloutManagerConfig

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

    def create_analyzer(self, analyzer_type: str = 'Default', checkpoint: str = "best") -> Analyzer:
        """
        Create an analyzer instance.

        Parameters:
            analyzer_type: Type of analyzer to create (default: 'Default')
            checkpoint: Name of the checkpoint to load (e.g., 'best', '1', '2') — default is 'best'

        Returns:
            An initialized Analyzer instance
        """
        if analyzer_type in self.registered_analyzers:
            analyzer_class = self.registered_analyzers[analyzer_type]
            analyzer = analyzer_class(
                workdir=self.workdir,
                env_config=self.env_config,
                agent_group_config=self.agent_group_config,
                rolloutmanager_config=self.rolloutmanager_config,
                checkpoint=checkpoint,  # Pass checkpoint name
            )
        else:
            raise ValueError(f"Unsupported analyzer type: {analyzer_type}")
        return analyzer