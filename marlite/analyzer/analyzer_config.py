# marlite/analyzer/analyzer_config.py
from typing import Dict, Any, Type
from copy import deepcopy

from marlite.analyzer.analyzer import Analyzer
from marlite.analyzer.magent_analyzer import MAgentAnalyzer
from marlite.analyzer.smac_analyzer import SMACAnalyzer
from marlite.analyzer.sumo_analyzer import SUMOAnalyzer

# Registry of available analyzer classes
REGISTERED_ANALYZERS: Dict[str, Type] = {
    "default": Analyzer,
    "magent": MAgentAnalyzer,
    "smac": SMACAnalyzer,
    "sumo": SUMOAnalyzer,
}

class AnalyzerConfig:
    """
    Configuration class for creating analyzers based on type.

    This class allows dynamic creation of analyzer instances based on configuration.
    It follows the same pattern as EnvConfig but for analyzers.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize the analyzer configuration.

        Parameters:
            analyzer_config: Configuration parameters for the analyzer
        """
        analyzer_config = deepcopy(kwargs)
        self.analyzer_type = analyzer_config.pop('type', 'default')
        self.analyzer_config = analyzer_config

        if self.analyzer_type not in REGISTERED_ANALYZERS:
            raise ValueError(f"Unknown analyzer type: {self.analyzer_type}")

    def create_analyzer(self) -> 'Analyzer':
        """
        Create an analyzer instance based on the configuration.

        Returns:
            An instance of the appropriate analyzer class
        """
        # Get the analyzer class from the registry
        analyzer_class = REGISTERED_ANALYZERS[self.analyzer_type]

        # Create and return the analyzer instance
        return analyzer_class(**self.analyzer_config)