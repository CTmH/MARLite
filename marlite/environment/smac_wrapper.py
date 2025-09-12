import numpy as np
from typing import Dict
from pettingzoo.utils import BaseParallelWrapper
from marlite.util.env_util import ensure_all_agents_present

class SMACWrapper(BaseParallelWrapper):
    """
    A wrapper for SMAC PettingZoo environments that modifies the state() method
    to return a concatenated numpy array of all state components, sorted by key
    in alphabetical order.
    """
    def __init__(self, env):
        super().__init__(env)
        _ = env.reset()
        state = env.state()
        env.close()

        self.default_state_dict = {}
        for agent in env.possible_agents:
            if agent in state:
                self.default_state_dict[agent] = np.zeros_like(state[agent])
            else:
                # If agent not present in initial observations, use first available observation as template
                first_state = next(iter(state.values()))
                self.default_state_dict[agent] = np.zeros_like(first_state)

    def state(self) -> np.ndarray:
        """
        Get the global state as a concatenated numpy array.

        The original state() returns a dict with string keys and ndarray values.
        This method sorts the items by key alphabetically and concatenates
        the arrays along axis 0 (flattening if necessary).

        Returns:
            np.ndarray: Concatenated state vector.
        """
        state_dict: Dict[str, np.ndarray] = self.env.state()
        state_dict = ensure_all_agents_present(state_dict, self.default_state_dict)

        sorted_arrays = [state_dict[key] for key in self.default_state_dict.keys()]

        flattened_arrays = [arr.flatten() for arr in sorted_arrays]

        return np.array(flattened_arrays)