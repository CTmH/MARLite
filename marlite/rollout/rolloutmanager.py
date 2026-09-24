from typing import List, Any, Callable, Optional, Union
from multiprocessing.shared_memory import SharedMemory
from contextlib import nullcontext
from marlite.environment import EnvConfig
from marlite.algorithm.agents import AgentGroupConfig
from tqdm import tqdm
from marlite.util.randomness import derive_seed, preserve_rng_state


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
        seed: int | None = None,
        deterministic: bool = False,
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
        self.seed = seed
        self.deterministic = deterministic
        self._generation = 0

    def _episode_seeds(self):
        """Task IDs, not worker assignment or completion order, determine seeds."""
        seeds = [derive_seed(self.seed, self._generation, i) for i in range(self.n_episodes)]
        self._generation += 1
        return seeds

    def generate_episodes(self) -> List[Any]:
        shm = SharedMemory(
            create=True, size=len(self.serialized_agent_group_params)
        )
        shm.buf[: len(self.serialized_agent_group_params)] = (
            self.serialized_agent_group_params
        )
        shm_name = shm.name

        episodes = []
        episode_seeds = self._episode_seeds()
        if isinstance(self.device, list):
            devices = [
                self.device[i % len(self.device)] for i in range(self.n_episodes)
            ]
        else:
            devices = [self.device] * self.n_episodes

        shm_info = (shm_name, len(self.serialized_agent_group_params))

        try:
            for i in tqdm(range(self.n_episodes), desc="Generating Episodes"):
                # Unlike subprocess managers, this fallback executes in the
                # trainer process. Episode seeding must not reset training RNGs.
                rng_context = preserve_rng_state() if self.seed is not None else nullcontext()
                with rng_context:
                    episode = self.worker_func(
                        self.env_config,
                        self.agent_group_config,
                        shm_info,
                        self.traj_len,
                        self.episode_limit,
                        self.epsilon,
                        devices[i],
                        required_attrs=self.required_attrs,
                        seed=episode_seeds[i],
                        deterministic=self.deterministic,
                    )
                episodes.append(episode)
        finally:
            shm.close()
            shm.unlink()

        return episodes
