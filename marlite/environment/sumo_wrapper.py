from typing import Dict, Any, Tuple, List
import numpy as np
from pettingzoo.utils import BaseParallelWrapper


class SUMOWrapper(BaseParallelWrapper):
    """
    A wrapper for SUMO environment that combines observations from all agents into a matrix.
    """

    def __init__(self, env):
        super().__init__(env)
        self.env.reset()
        self.agents = self.env.agents
        self.possible_agents = self.env.possible_agents
        self._state_matrix = np.array([]).reshape(0, 0)  # Cache for the state matrix

        # Calculate maximum feature dimension across all agents
        self.num_features = 0
        for agent in self.possible_agents:
            space = self.env.observation_space(agent)
            self.num_features = max(self.num_features, space.shape[0])


    def _observations_to_matrix(self, observations: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Convert dictionary of observations to a matrix.

        Args:
            observations: Dictionary mapping agent names to their observation vectors

        Returns:
            Matrix of shape [n_agents, num_features] with observations
        """
        if len(self.possible_agents) == 0:
            return np.array([]).reshape(0, 0)

        # Create a matrix to hold all observations
        state_matrix = np.zeros((len(self.possible_agents), self.num_features))

        # Fill the matrix with observations
        for i, agent in enumerate(self.possible_agents):
            if agent in observations and observations[agent] is not None:
                obs = observations[agent]
                # Handle case where observation is shorter than num_features
                if len(obs) <= self.num_features:
                    state_matrix[i, :len(obs)] = obs
                else:
                    # Truncate if observation is longer (should not happen normally)
                    state_matrix[i] = obs[:self.num_features]

        return state_matrix

    def reset(self, seed: int = None, options: Dict[str, Any] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset the environment and cache the state matrix.

        Args:
            seed: Random seed for reproducibility
            options: Additional reset options

        Returns:
            Tuple of (observations, infos)
        """
        observations, infos = self.env.reset(seed=seed, options=options)

        # Update agent list
        self.agents = self.env.agents

        # Convert observations to matrix and cache it
        self._state_matrix = self._observations_to_matrix(observations)

        return observations, infos

    def step(self, actions: Dict[str, Any]) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Any]
    ]:
        """
        Take a step in the environment and cache the state matrix.

        Returns:
            Tuple of (observations, rewards, terminations, truncations, infos)
        """
        observations, rewards, terminations, truncations, infos = self.env.step(actions)

        # Update agent list
        self.agents = self.env.agents

        # Convert observations to matrix and cache it
        self._state_matrix = self._observations_to_matrix(observations)

        return observations, rewards, terminations, truncations, infos

    def state(self) -> np.ndarray:
        """
        Get the global state of the environment as a matrix combining all agents' observations.

        Returns:
            np.ndarray: A matrix of shape [n_agents, num_features] containing all agents' observations
        """
        return self._state_matrix.copy()
