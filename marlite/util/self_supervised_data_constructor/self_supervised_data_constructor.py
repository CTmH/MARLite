from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Tuple


class SelfSupervisedDataConstructor(ABC):
    """
    Self-supervised data constructor base class.
    """

    def __init__(self, n_workers: int = 0):
        """
        Initialize the data constructor.

        Args:
            n_workers: Number of worker processes for parallel processing
        """
        self.n_workers = n_workers

    @abstractmethod
    def process(self, observations: np.ndarray, states: Optional[np.ndarray],
                edge_indices: np.ndarray, alive_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process the input data to construct self-supervised learning data.

        Args:
            observations: Array of shape (batch_size, n_agents, max_observed_entities, feature_dim)
            states: Optional array, not used in this implementation
            edge_indices: Array of shape (batch_size, 2, edge_num)
            alive_mask: Array of shape (batch_size, n_agents)

        Returns:
            A tuple containing:
            - Processed array of shape (batch_size, n_agents, max_entities_perception, feature_dim)
            - Mask array of shape (batch_size, n_agents, max_entities_perception) indicating padding
        """
        pass