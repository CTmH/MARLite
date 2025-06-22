import os
import numpy as np
import logging
from ..environment.env_config import EnvConfig
from ..rollout.rolloutmanager_config import RolloutManagerConfig
from ..algorithm.agents.agent_group_config import AgentGroupConfig

class Analyzer:
    def __init__(self,
                 workdir: str,
                 env_config: EnvConfig,
                 agent_group_config: AgentGroupConfig,
                 rolloutmanager_config: RolloutManagerConfig,
                 device: str = 'cpu'):
        """
        Analyzer class to load the best model and analyze various features of the model

        Parameters:
            workdir: Directory path where results are saved
            env_config: Environment configuration
            agent_group_config: Agent group configuration
            rolloutmanager_config: Rollout manager configuration
            device: Device to run on (cpu/cuda)
        """
        self.workdir = workdir
        self.env_config = env_config
        self.agent_group_config = agent_group_config
        self.rolloutmanager_config = rolloutmanager_config
        self.device = device

        # Directory paths
        self.checkpointdir = os.path.join(workdir, 'checkpoints')
        self.logdir = os.path.join(workdir, 'logs')

        # Create agent group
        self.agent_group = agent_group_config.get_agent_group()

        # Load best model
        self.load_best_model()

        logging.basicConfig(level=logging.INFO,
                           format='%(asctime)s - %(levelname)s - %(message)s',
                           handlers=[logging.StreamHandler()])

    def load_best_model(self):
        """Load the best model parameters from the work directory"""
        agent_path = os.path.join(self.checkpointdir, 'best', 'agent')
        self.agent_group.to(self.device)
        self.agent_group.load_params(agent_path)
        logging.info(f"Successfully loaded best model parameters: {agent_path}")

    def generate_episodes(self, num_episodes: int = 10, epsilon: float = 0.01):
        """
        Generate multiple episodes using the best model

        Parameters:
            num_episodes: Number of episodes to generate
            epsilon: Exploration rate

        Returns:
            List of generated episodes
        """
        self.agent_group.eval().to(self.device)
        manager = self.rolloutmanager_config.create_eval_manager(
            self.agent_group,
            self.env_config,
            epsilon
        )

        logging.info(f"Generating {num_episodes} episodes using the best model...")
        episodes = manager.generate_episodes(num_episodes)
        manager.cleanup()

        return episodes

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

        for agent_id, actions in decision_stats.items():
            total = sum(actions.values())
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
        return {
            'mean': float(np.mean(rewards)),
            'std': float(np.std(rewards)),
            'min': float(np.min(rewards)),
            'max': float(np.max(rewards)),
            'median': float(np.median(rewards)),
            'sum': float(np.sum(rewards))
        }

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

        if not edge_counts:
            return None

        edge_counts = np.array(edge_counts)
        return {
            'mean': float(np.mean(edge_counts)),
            'std': float(np.std(edge_counts)),
            'min': float(np.min(edge_counts)),
            'max': float(np.max(edge_counts)),
            'median': float(np.median(edge_counts)),
            'sum': float(np.sum(edge_counts))
        }

    def analyze_positive_rewards_per_step(self, episodes):
        """
        Analyze positive reward occurrences per step across all episodes

        Parameters:
            episodes: List of episodes to analyze

        Returns:
            Dictionary with statistics of positive reward counts per step
        """
        step_counts = []
        for episode in episodes:
            for step_rewards in episode['rewards']:
                count = sum(1 for reward in step_rewards.values() if reward > 0)
                step_counts.append(count)

        if not step_counts:
            return None

        data = np.array(step_counts)
        return {
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'median': float(np.median(data)),
            'sum': float(np.sum(data)),
            'total_steps': len(step_counts)
        }

    def analyze_negative_rewards_per_step(self, episodes):
        """
        Analyze negative reward occurrences per step across all episodes

        Parameters:
            episodes: List of episodes to analyze

        Returns:
            Dictionary with statistics of negative reward counts per step
        """
        step_counts = []
        for episode in episodes:
            for step_rewards in episode['rewards']:
                count = sum(1 for reward in step_rewards.values() if reward < 0)
                step_counts.append(count)

        if not step_counts:
            return None

        data = np.array(step_counts)
        return {
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'median': float(np.median(data)),
            'sum': float(np.sum(data)),
            'total_steps': len(step_counts)
        }

    def analyze_positive_rewards(self, episodes):
        """
        Analyze the count of positive reward occurrences for each agent

        Parameters:
            episodes: List of episodes to analyze

        Returns:
            Dictionary with agent-wise counts of positive rewards
        """
        positive_counts = {}
        for episode in episodes:
            for step_rewards in episode['rewards']:
                for agent_id, reward in step_rewards.items():
                    if reward > 0:
                        if agent_id not in positive_counts:
                            positive_counts[agent_id] = 0
                        positive_counts[agent_id] += 1
        return positive_counts

    def analyze_negative_rewards(self, episodes):
        """
        Analyze the count of negative reward occurrences for each agent

        Parameters:
            episodes: List of episodes to analyze

        Returns:
            Dictionary with agent-wise counts of negative rewards
        """
        negative_counts = {}
        for episode in episodes:
            for step_rewards in episode['rewards']:
                for agent_id, reward in step_rewards.items():
                    if reward < 0:
                        if agent_id not in negative_counts:
                            negative_counts[agent_id] = 0
                        negative_counts[agent_id] += 1
        return negative_counts

    def comprehensive_analysis(self, num_episodes: int = 100, epsilon: float = 0.01):
        """
        Perform a comprehensive analysis and return all results

        Parameters:
            num_episodes: Number of episodes to use for analysis
            epsilon: Exploration rate

        Returns:
            Dictionary containing all analysis results
        """
        episodes = self.generate_episodes(num_episodes, epsilon)

        return {
            'decision_distribution': self.analyze_decision_distribution(episodes),
            'state_values': self.analyze_state_values(episodes),
            'reward_distribution': self.analyze_reward_distribution(episodes),
            'edge_counts': self.analyze_edge_counts(episodes),
            'positive_rewards_per_step': self.analyze_positive_rewards_per_step(episodes),
            'negative_rewards_per_step': self.analyze_negative_rewards_per_step(episodes),
            'positive_rewards': self.analyze_positive_rewards(episodes),
            'negative_rewards': self.analyze_negative_rewards(episodes),
        }