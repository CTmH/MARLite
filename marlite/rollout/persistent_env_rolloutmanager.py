import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from typing import List, Any, Callable, Optional, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
from marlite.algorithm.agents import AgentGroupConfig
from marlite.environment import EnvConfig
from marlite.rollout.rolloutmanager import RolloutManager
from tqdm import tqdm


class PersistentEnvRolloutManager(RolloutManager):
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

        episodes_per_worker = [
            self.n_episodes // self.n_workers
            + (1 if i < self.n_episodes % self.n_workers else 0)
            for i in range(self.n_workers)
        ]

        workers_with_episodes = [
            (i, n_episodes)
            for i, n_episodes in enumerate(episodes_per_worker)
            if n_episodes > 0
        ]
        n_active_workers = len(workers_with_episodes)

        n_workers = min(self.n_workers, n_active_workers)

        if isinstance(self.device, list):
            devices = [self.device[i % len(self.device)] for i in range(n_workers)]
        else:
            devices = [self.device] * n_workers

        shm_info = (shm_name, shm_size)

        episodes = []
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = [
                    executor.submit(
                        self.worker_func,
                        self.env_config,
                        self.agent_group_config,
                        shm_info,
                        n_episodes,
                        self.traj_len,
                        self.episode_limit,
                        self.epsilon,
                        devices[i],
                        self.check_victory,
                        self.required_attrs,
                    )
                    for i, (worker_idx, n_episodes) in enumerate(
                        workers_with_episodes[:n_workers]
                    )
                ]

                pbar = tqdm(total=self.n_episodes, desc="Generating Episodes")
                for future in as_completed(futures):
                    try:
                        worker_episodes = future.result()
                        episodes.extend(worker_episodes)
                        pbar.update(len(worker_episodes))
                    except Exception as e:
                        print(f"Worker failed with error: {e}")
                        continue
                pbar.close()
        finally:
            shm.close()
            shm.unlink()

        return episodes
