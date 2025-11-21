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

    def analyze_accumulated_system_metrics(self, episodes: List[Dict], metric_name: str) -> Dict[str, Any]:
        """Analyze a specific accumulated system metric from SUMO simulation episodes.

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
            infos = episode.get('infos', [])
            if infos:
                final_step = infos[-1]
                if final_step:
                    for agent_info in final_step.values():
                        if metric_name in agent_info:
                            values.append(agent_info[metric_name])
                            break

        return self._calculate_statistics(values)

    def analyze_system_metrics_over_timestep(self, episodes: List[Dict], metric_name: str) -> Dict[str, Any]:
        """
        Analyze a system metric by first averaging it across all time steps in each episode,
        then computing statistics (mean, std, min, max, median) across all episodes.

        This method assumes that the metric is stored per-agent in each time step under the same name,
        and that all agents have identical values for system-level metrics (e.g., 'system_mean_speed').

        Args:
            episodes: List of episodes to analyze. Each episode should contain a list of 'infos' dictionaries.
                    Each info dictionary contains per-agent data, but system metrics are duplicated across agents.
            metric_name: Name of the system metric to analyze (e.g., 'system_mean_speed', 'system_total_waiting_time').

        Returns:
            Dictionary containing statistical results for the averaged metric across episodes.
            Returns None if no valid data is found.

        Example:
            >>> analyzer = SUMOAnalyzer(agent_group)
            >>> speed_results = analyzer.analyze_system_metrics_over_timestep(episodes, 'system_mean_speed')
            >>> waiting_time_results = analyzer.analyze_system_metrics_over_timestep(episodes, 'system_total_waiting_time')
        """
        if not episodes:
            return None

        # Extract the average value of the metric for each episode
        episode_averages = []
        for episode in episodes:
            info_steps = episode.get('infos', [])
            if not info_steps:
                continue  # Skip episodes with no info steps

            values = []
            for step in info_steps:
                # Since all agents have the same system metric, pick the first agent available
                for agent_id, agent_info in step.items():
                    if metric_name in agent_info:
                        values.append(agent_info[metric_name])
                        break  # Only need one agent's value
            if values:
                avg_value = np.mean(values)
                episode_averages.append(avg_value)

        # Use parent's _calculate_statistics method to compute stats across episodes
        return self._calculate_statistics(episode_averages)

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
            'system_mean_speed': self.analyze_system_metrics_over_timestep(episodes, "system_mean_speed"),
            'system_total_stopped': self.analyze_system_metrics_over_timestep(episodes, "system_total_stopped"),
            #'system_total_waiting_time': self.analyze_accumulated_system_metrics(episodes, "system_total_waiting_time"),
            'system_total_arrived': self.analyze_accumulated_system_metrics(episodes, "system_total_arrived"),
            'system_total_departed': self.analyze_accumulated_system_metrics(episodes, "system_total_departed"),
            'system_total_teleported': self.analyze_accumulated_system_metrics(episodes, "system_total_teleported")
        }

        return comprehensive_results