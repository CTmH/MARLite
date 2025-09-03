from typing import List, Any, Callable
from marlite.algorithm.agents import AgentGroup
from marlite.environment import EnvConfig
from tqdm import tqdm

class RolloutManager:
    def __init__(
        self,
        worker_func: Callable,
        env_config: EnvConfig,
        agent_group: AgentGroup,
        n_episodes: int,
        traj_len: int,
        episode_limit: int,
        epsilon: float,
        device: str
    ):
        self.worker_func = worker_func
        self.env_config = env_config
        self.agent_group = agent_group
        self.n_episodes = n_episodes
        self.traj_len = traj_len
        self.episode_limit = episode_limit
        self.epsilon = epsilon
        self.device = device

    def generate_episodes(self) -> List[Any]:
        episodes = []
        for _ in tqdm(range(self.n_episodes), desc="Generating Episodes"):
            episode = self.worker_func(
                self.env_config,
                self.agent_group,
                self.traj_len,
                self.episode_limit,
                self.epsilon,
                self.device
            )
            episodes.append(episode)
        return episodes

    def cleanup(self): # For compatibility
        return self