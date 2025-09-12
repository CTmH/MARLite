from copy import deepcopy
from marlite.rollout.rolloutmanager import RolloutManager
from marlite.rollout.persistent_env_rolloutmanager import PersistentEnvRolloutManager
from marlite.rollout.multiprocess_rolloutmanager import MultiProcessRolloutManager
from marlite.rollout.persistent_env_rollout import persistent_env_rollout
from marlite.rollout.multiprocess_rollout import multiprocess_rollout
from marlite.util.victory_checker import *
from marlite.environment import EnvConfig
from marlite.algorithm.agents import AgentGroup


# Module-level constants for registries
_MANAGER_REGISTRY = {
    "persistent-env": PersistentEnvRolloutManager,
    "multi-process": MultiProcessRolloutManager,
}

_WORKER_REGISTRY = {
    "persistent-env": persistent_env_rollout,
    "multi-process": multiprocess_rollout,
}

_VICTORY_CHECKER_REGISTRY = {
    "smac": check_smac_victory,
    "default": always_lose,
}

class RolloutManagerConfig:
    def __init__(self, **kwargs):
        self.config = deepcopy(kwargs)
        self.manager_type = self.config.pop('manager_type')
        self.worker_type = self.config.pop('worker_type')
        self.n_episodes = self.config.pop('n_episodes')
        self.n_eval_episodes = self.config.pop('n_eval_episodes', 100)

        # Validate manager type
        if self.manager_type not in _MANAGER_REGISTRY:
            raise ValueError(f"Unknown manager type: {self.manager_type}. "
                           f"Available options: {list(_MANAGER_REGISTRY.keys())}")

        # Validate worker type
        if self.worker_type not in _WORKER_REGISTRY:
            raise ValueError(f"Unknown worker type: {self.worker_type}. "
                           f"Available options: {list(_WORKER_REGISTRY.keys())}")

        self.manager_class = _MANAGER_REGISTRY[self.manager_type]
        self.worker_func = _WORKER_REGISTRY[self.worker_type]

        # Get victory checker name from config, default to "default"
        self.victory_checker_name = self.config.pop('victory_checker', 'default')

        # Validate victory checker name
        if self.victory_checker_name not in _VICTORY_CHECKER_REGISTRY:
            raise ValueError(f"Unknown victory checker: {self.victory_checker_name}. "
                           f"Available options: {list(_VICTORY_CHECKER_REGISTRY.keys())}")

    def create_manager(self, agent_group: AgentGroup, env_config: EnvConfig, epsilon: float) -> RolloutManager:
        manager = self.manager_class(
            worker_func=self.worker_func,
            env_config=env_config,
            agent_group=agent_group,
            n_episodes=self.n_episodes,
            epsilon=epsilon,
            check_victory=_VICTORY_CHECKER_REGISTRY[self.victory_checker_name],
            **self.config)
        return manager

    def create_eval_manager(self, agent_group: AgentGroup, env_config: EnvConfig, epsilon: float) -> RolloutManager:
        manager = self.manager_class(
            worker_func=self.worker_func,
            env_config=env_config,
            agent_group=agent_group,
            n_episodes=self.n_eval_episodes,
            epsilon=epsilon,
            check_victory=_VICTORY_CHECKER_REGISTRY[self.victory_checker_name],
            **self.config)
        return manager