# marlite/analyzer/smac_episode_analyzer.py
import numpy as np
from typing import List, Dict, Any
from marlite.analyzer.analyzer import Analyzer

class SMACAnalyzer(Analyzer):
    """EpisodeAnalyzer subclass for SMAC environments"""

    def __init__(self, agent_group):
        super().__init__(agent_group)

    def analyze_unit_types(self, episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze performance by unit types.

        Parameters:
            episodes: List of episodes to analyze

        Returns:
            Dictionary with unit type performance statistics
        """
        unit_stats = {}

        # Extract unit type information from agent IDs
        unit_types = {}
        for agent_id in self.agent_group.agent_model_dict.keys():
            # SMAC agent IDs typically contain unit type information
            # e.g., 'marine_0', 'medivac_1', etc.
            if '_' in agent_id:
                unit_type = agent_id.split('_')[0]
                if unit_type not in unit_types:
                    unit_types[unit_type] = []
                unit_types[unit_type].append(agent_id)

        unit_stats['composition'] = {unit_type: len(agents) for unit_type, agents in unit_types.items()}

        # Analyze performance by unit type
        unit_performance = {}
        for unit_type, agents in unit_types.items():
            # Collect rewards for this unit type
            type_rewards = []
            for episode in episodes:
                for step_rewards in episode['rewards']:
                    unit_rewards = [reward for agent, reward in step_rewards.items() if agent in agents]
                    type_rewards.extend(unit_rewards)

            if type_rewards:
                unit_performance[unit_type] = {
                    'mean_reward': float(np.mean(type_rewards)),
                    'total_reward': float(np.sum(type_rewards)),
                    'positive_reward_count': sum(1 for r in type_rewards if r > 0),
                    'negative_reward_count': sum(1 for r in type_rewards if r < 0),
                }

        unit_stats['performance'] = unit_performance
        return unit_stats

    def analyze_kiting_behavior(self, episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze kiting behavior (maintaining distance while attacking).

        Parameters:
            episodes: List of episodes to analyze

        Returns:
            Dictionary with kiting behavior statistics
        """
        kiting_stats = {}

        # This would require access to position information
        # which might be available in the SMAC state representation
        # Implementation would depend on how positions are encoded

        return kiting_stats

    def analyze_micro_management(self, episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze micro-management skills like focus fire, retreats, etc.

        Parameters:
            episodes: List of episodes to analyze

        Returns:
            Dictionary with micro-management statistics
        """
        micro_stats = {}

        # Would analyze action patterns for advanced tactics
        # This requires understanding of SMAC action space

        return micro_stats

    def __call__(self, episodes) -> Dict[str, Any]:
        """
        Perform a comprehensive analysis including SMAC-specific metrics.

        Parameters:
            episodes: List of episodes to analyze

        Returns:
            Dictionary containing all analysis results
        """
        # Get base analysis results
        base_results = super().__call__(episodes)

        # Add SMAC-specific analyses
        smac_specific_results = {
            'unit_types': self.analyze_unit_types(episodes),
            'kiting_behavior': self.analyze_kiting_behavior(episodes),
            'micro_management': self.analyze_micro_management(episodes),
        }

        # Merge results
        comprehensive_results = {**base_results, **smac_specific_results}
        return comprehensive_results