import torch.multiprocessing as mp
from typing import List, Any, Type
from concurrent.futures import ProcessPoolExecutor
from ..algorithm.agents.agent_group import AgentGroup
from ..environment.env_config import EnvConfig
from .multiprocess_rolloutworker import MultiProcessRolloutWorker, rollout
from tqdm import tqdm

class MultiProcessRolloutManager:
    def __init__(self,
                 worker_class: Type[MultiProcessRolloutWorker],
                 env_config: EnvConfig,
                 agent_group: AgentGroup,
                 n_workers: int,
                 n_episodes: int,
                 traj_len: int,
                 episode_limit: int,
                 epsilon: float,
                 device: str):
        
        self.env_config = env_config
        self.agent_group = agent_group
        self.n_workers = n_workers
        self.n_episodes = n_episodes
        self.traj_len = traj_len
        self.episode_limit = episode_limit
        self.epsilon = epsilon
        self.device = device

    def generate_episodes(self) -> List[Any]:
        mp.set_start_method('spawn', force=True)
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            episodes = list(tqdm(executor.map(
                rollout,
                [self.env_config] * self.n_episodes,
                [self.agent_group.share_memory()] * self.n_episodes,
                [self.traj_len] * self.n_episodes,
                [self.episode_limit] * self.n_episodes,
                [self.epsilon] * self.n_episodes,
                [self.device] * self.n_episodes
            ), total=self.n_episodes, desc="Generating Episodes"))
        return episodes
    
    def cleanup(self): # For compatibility
        return self