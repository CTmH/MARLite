from typing import List, Any, Callable, Union
from marlite.environment import EnvConfig
from marlite.algorithm.agents import AgentGroupConfig
import multiprocessing as mp
from tqdm import tqdm


class RolloutManager:
    def __init__(
        self,
        worker_func: Callable,
        env_config: EnvConfig,
        agent_group_config: AgentGroupConfig,
        serialized_agent_group_params: bytes,
        n_episodes: int,
        traj_len: int,
        episode_limit: int,
        epsilon: float,
        device: Union[str, List[str]],
    ):
        self.worker_func = worker_func
        self.env_config = env_config
        self.agent_group_config = agent_group_config
        self.serialized_agent_group_params = serialized_agent_group_params
        self.n_episodes = n_episodes
        self.traj_len = traj_len
        self.episode_limit = episode_limit
        self.epsilon = epsilon
        self.device = device

    def generate_episodes(self) -> List[Any]:
        shm = mp.shared_memory.SharedMemory(
            create=True, size=len(self.serialized_agent_group_params)
        )
        shm.buf[: len(self.serialized_agent_group_params)] = (
            self.serialized_agent_group_params
        )
        shm_name = shm.name

        episodes = []
        if isinstance(self.device, list):
            devices = [
                self.device[i % len(self.device)] for i in range(self.n_episodes)
            ]
        else:
            devices = [self.device] * self.n_episodes

        shm_info = (shm_name, len(self.serialized_agent_group_params))

        try:
            for i in tqdm(range(self.n_episodes), desc="Generating Episodes"):
                episode = self.worker_func(
                    self.env_config,
                    self.agent_group_config,
                    shm_info,
                    self.traj_len,
                    self.episode_limit,
                    self.epsilon,
                    devices[i],
                )
                episodes.append(episode)
        finally:
            shm.close()
            shm.unlink()

        return episodes
