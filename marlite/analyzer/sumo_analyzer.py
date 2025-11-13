# marlite/analyzer/sumo_analyzer.py
import numpy as np
from typing import Dict, Any, List
from marlite.analyzer.analyzer import Analyzer

class SUMOAnalyzer(Analyzer):
    """EpisodeAnalyzer subclass for SUMO traffic simulation environments.
    Provides methods to analyze system-level metrics from simulation episodes.
    """

    def __init__(self):
        """
        Initialize the analyzer with an agent group.
        """
        super().__init__()

    def analyze_system_metrics(self, episodes: List[Dict], metric_name: str) -> Dict[str, Any]:
        """Analyze a specific system metric from SUMO simulation episodes.

        Extracts the final value of the specified metric from each episode and computes
        descriptive statistics using the parent class's _calculate_statistics method.

        Args:
            episodes: List of episodes to analyze. Each episode should contain
                    a list of 'infos' dictionaries with system metrics.
            metric_name: Name of the system metric to analyze (e.g., 'system_mean_speed')

        Returns:
            Dictionary containing statistical results for the specified metric.
            Returns None if no valid data is found.

        Example:
            >>> analyzer = SUMOAnalyzer(agent_group)
            >>> speed_results = analyzer.analyze_system_metrics(episodes, 'system_mean_speed')
            >>> waiting_time_results = analyzer.analyze_system_metrics(episodes, 'system_total_waiting_time')
        """
        if not episodes:
            return None

        values = []
        for episode in episodes:
            info_steps = episode.get('infos', [])
            if info_steps:
                final_step = info_steps[-1]
                if final_step:
                    agent_info = next(iter(final_step.values()))
                    if metric_name in agent_info:
                        values.append(agent_info[metric_name])

        return self._calculate_statistics(values)

    def __call__(self, episodes: List[Dict]) -> Dict[str, Any]:
        """Perform comprehensive analysis on system-level metrics from SUMO simulations.

        Analyzes multiple key system metrics by calling analyze_system_metrics for each.

        Args:
            episodes: List of episodes to analyze

        Returns:
            Dictionary containing statistical results for all system metrics
        """
        if not episodes:
            return {}

        # Define the system metrics to analyze
        comprehensive_results = {
            'reward': self.analyze_reward_distribution(episodes),
            'system_mean_speed': self.analyze_system_metrics(episodes, "system_mean_speed"),
            'system_total_waiting_time': self.analyze_system_metrics(episodes, "system_total_waiting_time"),
            'system_total_departed': self.analyze_system_metrics(episodes, "system_total_departed"),
            'system_total_arrived': self.analyze_system_metrics(episodes, "system_total_arrived"),
        }

        return comprehensive_results