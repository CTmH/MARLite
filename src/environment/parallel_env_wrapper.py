import numpy as np
from typing import Dict, Any
from pettingzoo.utils.env import ParallelEnv


class ParallelEnvWrapper():
    """
    Base wrapper class for ParallelEnv environments.
    Provides delegation of all methods and attributes to the wrapped environment.
    """
    
    def __init__(self, env: ParallelEnv):
        """
        Initialize the wrapper.
        
        Args:
            env: The original environment to wrap.
        """
        self._env = env
        
        # Copy essential attributes from the original environment
        self.agents = env.agents
        self.possible_agents = env.possible_agents
        self.observation_spaces = getattr(env, 'observation_spaces', None)
        self.action_spaces = getattr(env, 'action_spaces', None)
        self.state_spaces = getattr(env, 'state_spaces', None)
    
    def __getattr__(self, name: str) -> Any:
        """
        Delegate attribute access to the original environment.
        This allows access to any methods or attributes not explicitly overridden.
        """
        if name.startswith('_'):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        return getattr(self._env, name)
    
    def reset(self, seed: int = None, options: Dict = None) -> tuple:
        """
        Reset the environment.
        
        Returns:
            tuple: Observations and infos.
        """
        return self._env.reset(seed=seed, options=options)
    
    def step(self, actions: Dict) -> tuple:
        """
        Take a step in the environment.
        
        Args:
            actions: Dictionary of actions keyed by agent name.
            
        Returns:
            tuple: Observations, rewards, terminations, truncations, and infos.
        """
        return self._env.step(actions)
    
    def state(self) -> Any:
        """
        Get the global state.
        
        Returns:
            Any: The state of the environment.
        """
        return self._env.state()
    
    def render(self, *args, **kwargs) -> Any:
        """Render the environment."""
        return self._env.render(*args, **kwargs)
    
    def close(self) -> None:
        """Close the environment."""
        self._env.close()
    
    def observation_space(self, agent: Any) -> Any:
        """Get the observation space for a specific agent."""
        return self._env.observation_space(agent)
    
    def action_space(self, agent: Any) -> Any:
        """Get the action space for a specific agent."""
        return self._env.action_space(agent)