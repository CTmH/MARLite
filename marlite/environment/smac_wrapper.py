import numpy as np
from typing import Callable, Dict
import logging
from pettingzoo.utils import BaseParallelWrapper
from pysc2.lib import protocol
from marlite.util.env_util import ensure_all_agents_present


SC2_RECOVERABLE_ERRORS = (
    protocol.ProtocolError,
    protocol.ConnectionError,
    ConnectionError,
    TimeoutError,
)


class SMACWrapper(BaseParallelWrapper):
    """
    A wrapper for SMAC PettingZoo environments that modifies the state() method
    to return flattened per-agent states stacked in possible_agents order.
    Missing agents are represented by zero rows.
    """
    # Keep backend-specific exception knowledge out of generic rollout workers.
    recoverable_errors = SC2_RECOVERABLE_ERRORS

    def __init__(
        self, env, env_factory: Callable | None = None, reset_retries: int = 0
    ):
        super().__init__(env)
        if reset_retries < 0:
            raise ValueError("reset_retries must be nonnegative")
        self.env_factory = env_factory
        # Rollout owns recovery/backoff by default. Opt-in local retries are
        # only for standalone wrapper users; stacking both policies retries twice.
        self.reset_retries = reset_retries
        self.default_state_dict = {}
        for agent in env.possible_agents:
            space = env.state_space(agent)
            self.default_state_dict[agent] = np.zeros(space.shape, dtype=space.dtype)

    def reset(self, seed=None, options=None):
        for attempt in range(self.reset_retries + 1):
            try:
                if attempt:
                    fresh = self.env_factory()
                    if tuple(fresh.possible_agents) != tuple(self.default_state_dict) or any(
                        fresh.state_space(agent).shape != state.shape
                        for agent, state in self.default_state_dict.items()
                    ):
                        fresh.close()
                        raise ValueError("SC2 agent names or state spaces changed on restart")
                    self.env = fresh
                return self.env.reset(seed=seed, options=options)
            except SC2_RECOVERABLE_ERRORS:
                try:
                    self.env.close()
                except Exception:
                    logging.warning("Failed to close disconnected SC2 environment", exc_info=True)
                if self.env_factory is None or attempt == self.reset_retries:
                    raise

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
