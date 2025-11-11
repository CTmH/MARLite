# marlite/analyzer/analyzer.py
import os
from absl import logging
from marlite.environment import EnvConfig
from marlite.rollout import RolloutManagerConfig
from marlite.algorithm.agents import AgentGroupConfig
from marlite.analyzer.analyzer import Analyzer

class ExperimentAnalyzer:
    def __init__(
        self,
        workdir: str,
        env_config: EnvConfig,
        agent_group_config: AgentGroupConfig,
        rolloutmanager_config: RolloutManagerConfig,
        checkpoint: str = "best",
    ):
        """
        ExperimentAnalyzer class to load the best model and analyze various features of the model

        Parameters:
            workdir: Directory path where results are saved
            env_config: Environment configuration
            agent_group_config: Agent group configuration
            rolloutmanager_config: Rollout manager configuration
            checkpoint: Name of the checkpoint to load (e.g., 'best', '1', '2') — default is 'best'
        """
        self.workdir = workdir
        self.env_config = env_config
        self.agent_group_config = agent_group_config
        self.rolloutmanager_config = rolloutmanager_config
        self.checkpoint = checkpoint

        # Directory paths
        self.checkpointdir = os.path.join(workdir, 'checkpoints')
        self.logdir = os.path.join(workdir, 'logs')

        # Create agent group
        self.agent_group = agent_group_config.get_agent_group()

        # Create episode analyzer
        self.episode_analyzer = Analyzer(self.agent_group)

        # Load model from specified checkpoint
        self.load_checkpoint_model()

    def load_checkpoint_model(self):
        """Load model parameters from the specified checkpoint"""
        agent_path = os.path.join(self.checkpointdir, self.checkpoint, 'agent')
        self.agent_group.load_params(agent_path)
        logging.info(f"Successfully loaded model from checkpoint: {agent_path}")

    def generate_episodes(self, epsilon: float = 0.01):
        """
        Generate multiple episodes using the best model

        Parameters:
            epsilon: Exploration rate

        Returns:
            List of generated episodes
        """
        manager = self.rolloutmanager_config.create_eval_manager(
            self.agent_group,
            self.env_config,
            epsilon,
        )

        logging.info(f"Generating {manager.n_episodes} episodes using the best model...")
        episodes = manager.generate_episodes()
        manager.cleanup()

        return episodes

    def comprehensive_analysis(self, epsilon: float = 0.01):
        """
        Perform a comprehensive analysis and return all results

        Parameters:
            epsilon: Exploration rate

        Returns:
            Dictionary containing all analysis results
        """
        # Generate episodes
        episodes = self.generate_episodes(epsilon)

        # Analyze episodes using EpisodeAnalyzer
        return self.episode_analyzer(episodes)