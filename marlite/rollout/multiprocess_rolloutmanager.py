import torch.multiprocessing as mp
import torch
from typing import List, Any, Callable
from concurrent.futures import ProcessPoolExecutor
from marlite.algorithm.agents.agent_group import AgentGroup
from marlite.environment.env_config import EnvConfig
from marlite.rollout.rolloutmanager import RolloutManager
from tqdm import tqdm

class MultiProcessRolloutManager(RolloutManager):
    def __init__(self,
                 worker_func: Callable,
                 env_config: EnvConfig,
                 agent_group: AgentGroup,
                 n_workers: int,
                 n_episodes: int,
                 traj_len: int,
                 episode_limit: int,
                 epsilon: float,
                 device: str,
                 check_victory: Callable):

        self.worker_func = worker_func
        self.env_config = env_config
        self.agent_group = agent_group
        self.n_workers = n_workers
        self.n_episodes = n_episodes
        self.traj_len = traj_len
        self.episode_limit = episode_limit
        self.epsilon = epsilon
        self.device = device
        self.check_victory = check_victory

    def generate_episodes(self) -> List[Any]:
        mp.set_start_method('spawn', force=True)
        n_workers = min(self.n_workers, self.n_episodes)

        # Handle CUDA device allocation when device is just "cuda" without device number
        if self.device == "cuda":
            if torch.cuda.is_available():
                num_cuda_devices = torch.cuda.device_count()
                # Distribute workers evenly across available CUDA devices
                devices = [f"cuda:{i % num_cuda_devices}" for i in range(self.n_episodes)]
            else:
                raise RuntimeError("CUDA is not available on this system")
        else:
            # Use the specified device for all workers
            devices = [self.device for _ in range(self.n_episodes)]

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            episodes = list(tqdm(executor.map(
                self.worker_func,
                [self.env_config] * self.n_episodes,
                [self.agent_group.share_memory()] * self.n_episodes,
                [self.traj_len] * self.n_episodes,
                [self.episode_limit] * self.n_episodes,
                [self.epsilon] * self.n_episodes,
                devices,
                [self.check_victory] * self.n_episodes
            ), total=self.n_episodes, desc="Generating Episodes"))
        episodes = [e for e in episodes if e]
        return episodes

    def cleanup(self): # For compatibility
        return self