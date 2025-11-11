# marlite/analyzer/episode_analyzer.py
import numpy as np
from typing import Dict, Any

class Analyzer:
    """
    EpisodeAnalyzer class to perform comprehensive analysis on episodes

    This class contains all the data analysis functionality that was previously
    in the Analyzer class, but separated to follow single responsibility principle.
    """

    def _calculate_statistics(self, data, include_total=False):
        """
        Calculate common statistics for numerical data

        Parameters:
            data: List or array of numerical data
            include_total: Whether to include total count in results

        Returns:
            Dictionary with statistical measures or None if data is empty
        """
        if not data:
            return None

        data = np.array(data)
        result = {
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'median': float(np.median(data)),
        }

        if include_total:
            result['total'] = len(data)

        return result

    def analyze_decision_distribution(self, episodes):
        """
        Analyze decision distribution of each agent

        Parameters:
            episodes: List of episodes to analyze

        Returns:
            Dictionary with action selection frequency for each agent
        """
        decision_stats = {}
        for episode in episodes:
            for step_actions in episode['actions']:
                for agent_id, action in step_actions.items():
                    if agent_id not in decision_stats:
                        decision_stats[agent_id] = {}
                    decision_stats[agent_id][action] = decision_stats[agent_id].get(action, 0) + 1

        # Normalize to get probabilities
        for agent_id, actions in decision_stats.items():
            total = sum(actions.values())
            if total > 0:
                for action in actions:
                    actions[action] /= total

        return decision_stats

    def analyze_reward_distribution(self, episodes):
        """
        Analyze reward distribution

        Parameters:
            episodes: List of episodes to analyze

        Returns:
            Dictionary with statistics on rewards
        """
        rewards = [ep['episode_reward'] for ep in episodes]
        return self._calculate_statistics(rewards)

    def analyze_edge_counts(self, episodes):
        """
        Analyze edge count statistics for each step

        Parameters:
            episodes: List of episodes to analyze

        Returns:
            Dictionary with edge count statistics
        """
        edge_counts = []
        for episode in episodes:
            for step_edges in episode['edge_indices']:
                if step_edges is not None and step_edges.shape[1] > 0:
                    edge_counts.append(step_edges.shape[1])
                else:
                    edge_counts.append(0)

        return self._calculate_statistics(edge_counts)

    def analyze_rewards_per_step(self, episodes, reward_condition):
        """
        Analyze reward occurrences per step across all episodes based on condition

        Parameters:
            episodes: List of episodes to analyze
            reward_condition: Function that takes a reward and returns True/False

        Returns:
            Dictionary with statistics of reward counts per step
        """
        step_counts = []
        for episode in episodes:
            for step_rewards in episode['rewards']:
                count = sum(1 for reward in step_rewards.values() if reward_condition(reward))
                step_counts.append(count)

        return self._calculate_statistics(step_counts, include_total=True)

    def analyze_positive_rewards_per_step(self, episodes):
        """
        Analyze positive reward occurrences per step across all episodes

        Parameters:
            episodes: List of episodes to analyze

        Returns:
            Dictionary with statistics of positive reward counts per step
        """
        return self.analyze_rewards_per_step(episodes, lambda x: x > 0)

    def analyze_negative_rewards_per_episode(self, episodes):
        """
        Analyze total negative reward occurrences per episode across all episodes

        Parameters:
            episodes: List of episodes to analyze

        Returns:
            Dictionary with statistics of total negative reward counts per episode
        """
        episode_counts = []
        for episode in episodes:
            episode_count = 0
            for step_rewards in episode['rewards']:
                count = sum(1 for reward in step_rewards.values() if reward < 0)
                episode_count += count
            episode_counts.append(episode_count)

        return self._calculate_statistics(episode_counts, include_total=True)

    def analyze_reward_counts(self, episodes, reward_condition):
        """
        Analyze the count of reward occurrences for each agent based on condition

        Parameters:
            episodes: List of episodes to analyze
            reward_condition: Function that takes a reward and returns True/False

        Returns:
            Dictionary with agent-wise counts of rewards meeting the condition
        """
        counts = {}
        for episode in episodes:
            for step_rewards in episode['rewards']:
                for agent_id, reward in step_rewards.items():
                    if reward_condition(reward):
                        if agent_id not in counts:
                            counts[agent_id] = 0
                        counts[agent_id] += 1

        return counts

    def analyze_positive_rewards(self, episodes):
        """
        Analyze the count of positive reward occurrences for each agent

        Parameters:
            episodes: List of episodes to analyze

        Returns:
            Dictionary with agent-wise counts of positive rewards
        """
        return self.analyze_reward_counts(episodes, lambda x: x > 0)

    def analyze_negative_rewards(self, episodes):
        """
        Analyze the count of negative reward occurrences for each agent

        Parameters:
            episodes: List of episodes to analyze

        Returns:
            Dictionary with agent-wise counts of negative rewards
        """
        return self.analyze_reward_counts(episodes, lambda x: x < 0)

    def analyze_surviving_agents(self, episodes):
        """
        Analyze the number of surviving agents at the end of each episode.

        Parameters:
            episodes: List of episodes to analyze

        Returns:
            Dictionary with statistics on the number of surviving agents per episode
        """
        survival_counts = []
        for episode in episodes:
            alive_mask_sequence = episode.get('alive_mask')
            if alive_mask_sequence is not None and len(alive_mask_sequence) > 0:
                final_alive_mask = alive_mask_sequence[-1]  # Last step's alive mask
                count = sum(1 for alive in final_alive_mask.values() if alive)
                survival_counts.append(count)

        return self._calculate_statistics(survival_counts, include_total=True)

    def analyze_win_rate(self, episodes):
        """
        Analyze the win rate across all episodes.

        Parameters:
            episodes: List of episodes to analyze

        Returns:
            Dictionary with statistics on win rate
        """
        win_results = []
        for episode in episodes:
            win_tag = episode.get('win_tag')
            if win_tag is not None:
                # Assume win_tag is a boolean or int (1 for win, 0 for loss)
                win_results.append(1 if win_tag else 0)

        if not win_results:
            return None

        data = np.array(win_results)
        win_rate = float(np.mean(data))
        return {
            'win_rate': win_rate,
            'wins': int(np.sum(data)),
            'losses': len(data) - int(np.sum(data)),
            'mean': win_rate # For compatibility
        }

    def __call__(self, episodes) -> Dict[str, Any]:
        """
        Perform comprehensive analysis on episodes

        Parameters:
            episodes: List of episodes to analyze

        Returns:
            Dictionary containing all analysis results
        """
        return {
            'decision_distribution': self.analyze_decision_distribution(episodes),
            'reward': self.analyze_reward_distribution(episodes),
            'edge_counts': self.analyze_edge_counts(episodes),
            'positive_rewards_per_step': self.analyze_positive_rewards_per_step(episodes),
            'negative_rewards_per_episode': self.analyze_negative_rewards_per_episode(episodes),
            'positive_rewards': self.analyze_positive_rewards(episodes),
            'negative_rewards': self.analyze_negative_rewards(episodes),
            'surviving_agents': self.analyze_surviving_agents(episodes),
            'win_rate': self.analyze_win_rate(episodes),
        }