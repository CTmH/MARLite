from copy import deepcopy
from .rolloutmanager import RolloutManager
from .multiprocess_rolloutmanager import MultiProcessRolloutManager
from .rollout_func import multiprocess_rollout
from ..environment.env_config import EnvConfig
from ..algorithm.agents.agent_group import AgentGroup


class RolloutManagerConfig:
    def __init__(self, **kwargs):
        self.config = deepcopy(kwargs)
        self.manager_type = self.config.pop('manager_type')
        self.worker_type = self.config.pop('worker_type')
        self.n_episodes = self.config.pop('n_episodes')
        self.n_eval_episodes = self.config.pop('n_eval_episodes', 100)
        self.registered_managers = {
            "multi-process": MultiProcessRolloutManager,
        }
        self.registered_workers = {
            "multi-process": multiprocess_rollout,
        }
        self.manager_class = self.registered_managers[self.manager_type]
        self.worker_func = self.registered_workers[self.worker_type]

    def create_manager(self, agent_group: AgentGroup, env_config: EnvConfig, epsilon: float) -> RolloutManager:
        manager = self.manager_class(
            worker_func = self.worker_func,
            env_config = env_config,
            agent_group = agent_group,
            n_episodes = self.n_episodes,
            epsilon = epsilon,
            **self.config)
        return manager

    def create_eval_manager(self, agent_group: AgentGroup, env_config: EnvConfig, epsilon: float) -> RolloutManager:
        manager = self.manager_class(
            worker_func = self.worker_func,
            env_config = env_config,
            agent_group = agent_group,
            n_episodes = self.n_eval_episodes,
            epsilon = epsilon,
            **self.config)
        return manager