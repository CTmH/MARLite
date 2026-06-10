from typing import List, Any, Callable, Optional, Union
from multiprocessing.shared_memory import SharedMemory
from marlite.environment import EnvConfig
from marlite.algorithm.agents import AgentGroupConfig
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
        required_attrs: Optional[Union[str, List[str]]] = None,
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
        self.required_attrs = required_attrs

    def generate_episodes(self) -> List[Any]:
        shm = SharedMemory(
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
                    required_attrs=self.required_attrs,
                )
                episodes.append(episode)
        finally:
            shm.close()
            shm.unlink()

        return episodes
