import numpy as np
from typing import Dict
from pettingzoo.utils import BaseParallelWrapper
from marlite.util.env_util import ensure_all_agents_present

class SMACWrapper(BaseParallelWrapper):
    """
    A wrapper for SMAC PettingZoo environments that modifies the state() method
    to return flattened per-agent states stacked in possible_agents order.
    Missing agents are represented by zero rows.
    """
    def __init__(self, env):
        super().__init__(env)
        self.default_state_dict = {}
        for agent in env.possible_agents:
            space = env.state_space(agent)
            self.default_state_dict[agent] = np.zeros(space.shape, dtype=space.dtype)

    def state(self) -> np.ndarray:
        """
        Get per-agent states as a stacked numpy array.

        The original state() returns a dict with string keys and ndarray values.
        Rows follow possible_agents order; each state is flattened separately.

        Returns:
            np.ndarray: Array of shape (n_agents, flattened_state_size).
        """
        state_dict: Dict[str, np.ndarray] = self.env.state()
        state_dict = ensure_all_agents_present(state_dict, self.default_state_dict)

        sorted_arrays = [state_dict[key] for key in self.default_state_dict.keys()]

        flattened_arrays = [arr.flatten() for arr in sorted_arrays]

        return np.array(flattened_arrays)
