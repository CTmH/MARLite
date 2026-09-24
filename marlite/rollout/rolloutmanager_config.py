from copy import deepcopy
from typing import Optional
from marlite.rollout.rolloutmanager import RolloutManager
from marlite.rollout.persistent_env_rolloutmanager import PersistentEnvRolloutManager
from marlite.rollout.multiprocess_rolloutmanager import MultiProcessRolloutManager
from marlite.rollout.persistent_env_rollout import persistent_env_rollout
from marlite.rollout.multiprocess_rollout import multiprocess_rollout
from marlite.util.victory_checker import *
from marlite.environment import EnvConfig
from marlite.algorithm.agents import AgentGroupConfig
from marlite.util.randomness import derive_seed, ROLLOUT_STREAM, EVALUATION_STREAM


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
    "battle_wrapper": check_battle_wrapper_victory,
    "default": always_lose,
}


class RolloutManagerConfig:
    def __init__(self, **kwargs):
        self.config = deepcopy(kwargs)
        self.manager_type = self.config.pop("manager_type")
        self.worker_type = self.config.pop("worker_type")
        self.n_episodes = self.config.pop("n_episodes")
        self.n_eval_episodes = self.config.pop("n_eval_episodes", 10)
        self.set_randomness()

        # Store the raw profile name (None or str) — the rollout worker
        # resolves it via resolve_required_attrs and resolve_phases.
        self.required_attrs = self.config.pop("required_attrs", None)

        if self.manager_type not in _MANAGER_REGISTRY:
            raise ValueError(
                f"Unknown manager type: {self.manager_type}. "
                f"Available options: {list(_MANAGER_REGISTRY.keys())}"
            )

        if self.worker_type not in _WORKER_REGISTRY:
            raise ValueError(
                f"Unknown worker type: {self.worker_type}. "
                f"Available options: {list(_WORKER_REGISTRY.keys())}"
            )

        self.manager_class = _MANAGER_REGISTRY[self.manager_type]
        self.worker_func = _WORKER_REGISTRY[self.worker_type]

        self.victory_checker_name = self.config.pop("victory_checker", "default")

        if self.victory_checker_name not in _VICTORY_CHECKER_REGISTRY:
            raise ValueError(
                f"Unknown victory checker: {self.victory_checker_name}. "
                f"Available options: {list(_VICTORY_CHECKER_REGISTRY.keys())}"
            )

    def set_randomness(self, seed=None, deterministic=False):
        """Trainer-owned streams, reset once at the start of an experiment."""
        self.seed = seed
        self.deterministic = deterministic
        self._collection_round = 0
        self._evaluation_round = 0

    def _randomness_kwargs(self, evaluation=False):
        round_index = self._evaluation_round if evaluation else self._collection_round
        stream = EVALUATION_STREAM if evaluation else ROLLOUT_STREAM
        seed = derive_seed(self.seed, stream, round_index)
        if evaluation:
            self._evaluation_round += 1
        else:
            self._collection_round += 1
        return {"seed": seed, "deterministic": self.deterministic}

    def create_manager(
        self,
        agent_group_config: AgentGroupConfig,
        serialized_agent_group_params: bytes,
        env_config: EnvConfig,
        epsilon: float,
    ) -> RolloutManager:
        manager = self.manager_class(
            worker_func=self.worker_func,
            env_config=env_config,
            agent_group_config=agent_group_config,
            serialized_agent_group_params=serialized_agent_group_params,
            n_episodes=self.n_episodes,
            epsilon=epsilon,
            check_victory=_VICTORY_CHECKER_REGISTRY[self.victory_checker_name],
            required_attrs=self.required_attrs,
            **self._randomness_kwargs(),
            **self.config,
        )
        return manager

    def create_eval_manager(
        self,
        agent_group_config: AgentGroupConfig,
        serialized_agent_group_params: bytes,
        env_config: EnvConfig,
        epsilon: float,
    ) -> RolloutManager:
        manager = self.manager_class(
            worker_func=self.worker_func,
            env_config=env_config,
            agent_group_config=agent_group_config,
            serialized_agent_group_params=serialized_agent_group_params,
            n_episodes=self.n_eval_episodes,
            epsilon=epsilon,
            check_victory=_VICTORY_CHECKER_REGISTRY[self.victory_checker_name],
            required_attrs=self.required_attrs,
            **self._randomness_kwargs(evaluation=True),
            **self.config,
        )
        return manager
