import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from typing import List, Any, Callable, Optional, Union
from concurrent.futures import ProcessPoolExecutor
from marlite.algorithm.agents import AgentGroupConfig
from marlite.environment.env_config import EnvConfig
from marlite.rollout.rolloutmanager import RolloutManager
from tqdm import tqdm


class MultiProcessRolloutManager(RolloutManager):
    def __init__(
        self,
        worker_func: Callable,
        env_config: EnvConfig,
        agent_group_config: AgentGroupConfig,
        serialized_agent_group_params: bytes,
        n_workers: int,
        n_episodes: int,
        traj_len: int,
        episode_limit: int,
        epsilon: float,
        device: Union[str, List[str]],
        check_victory: Callable,
        required_attrs: Optional[Union[str, List[str]]] = None,
    ):

        self.worker_func = worker_func
        self.env_config = env_config
        self.agent_group_config = agent_group_config
        self.serialized_agent_group_params = serialized_agent_group_params
        self.n_workers = n_workers
        self.n_episodes = n_episodes
        self.traj_len = traj_len
        self.episode_limit = episode_limit
        self.epsilon = epsilon
        self.device = device
        self.check_victory = check_victory
        self.required_attrs = required_attrs

    def generate_episodes(self) -> List[Any]:
        mp.set_start_method("spawn", force=True)

        shm = SharedMemory(
            create=True, size=len(self.serialized_agent_group_params)
        )
        shm.buf[: len(self.serialized_agent_group_params)] = (
            self.serialized_agent_group_params
        )
        shm_name = shm.name
        shm_size = len(self.serialized_agent_group_params)

        n_workers = min(self.n_workers, self.n_episodes)

        if isinstance(self.device, list):
            devices = [
                self.device[i % len(self.device)] for i in range(self.n_episodes)
            ]
        else:
            devices = [self.device] * self.n_episodes

        shm_info = (shm_name, shm_size)

        try:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                episodes = list(
                    tqdm(
                        executor.map(
                            self.worker_func,
                            [self.env_config] * self.n_episodes,
                            [self.agent_group_config] * self.n_episodes,
                            [shm_info] * self.n_episodes,
                            [self.traj_len] * self.n_episodes,
                            [self.episode_limit] * self.n_episodes,
                            [self.epsilon] * self.n_episodes,
                            devices,
                            [self.check_victory] * self.n_episodes,
                            [self.required_attrs] * self.n_episodes,
                        ),
                        total=self.n_episodes,
                        desc="Generating Episodes",
                    )
                )
        finally:
            shm.close()
            shm.unlink()

        episodes = [e for e in episodes if e]
        return episodes
