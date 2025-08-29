import numpy as np
from typing import Dict, Any
from .parallel_env_wrapper import ParallelEnvWrapper


class SMACWrapper(ParallelEnvWrapper):
    """
    A wrapper for SMAC PettingZoo environments that modifies the state() method
    to return a concatenated numpy array of all state components, sorted by key
    in alphabetical order.
    """
    def __init__(self, env):
        super().__init__(env)
    
    def state(self) -> np.ndarray:
        """
        Get the global state as a concatenated numpy array.
        
        The original state() returns a dict with string keys and ndarray values.
        This method sorts the items by key alphabetically and concatenates
        the arrays along axis 0 (flattening if necessary).
        
        Returns:
            np.ndarray: Concatenated state vector.
        """
        state_dict: Dict[str, np.ndarray] = self._env.state()
        
        # Sort by key alphabetically and concatenate values
        sorted_keys = sorted(state_dict.keys())
        sorted_arrays = [state_dict[key] for key in sorted_keys]
        
        # Flatten each array before concatenation to ensure 1D output
        flattened_arrays = [arr.flatten() for arr in sorted_arrays]
        
        # Concatenate all arrays into a single 1D array
        return np.concatenate(flattened_arrays, axis=0)