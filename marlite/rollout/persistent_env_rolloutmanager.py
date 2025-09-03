import torch.multiprocessing as mp
from typing import List, Any, Callable
from concurrent.futures import ProcessPoolExecutor
from marlite.algorithm.agents import AgentGroup
from marlite.environment import EnvConfig
from marlite.rollout.rolloutmanager import RolloutManager
from tqdm import tqdm

class PersistentEnvRolloutManager(RolloutManager):
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

        # Calculate episodes per worker
        episodes_per_worker = [self.n_episodes // self.n_workers + (1 if i < self.n_episodes % self.n_workers else 0)
                              for i in range(self.n_workers)]

        # Filter out workers that would get 0 episodes
        workers_with_episodes = [(i, n_episodes) for i, n_episodes in enumerate(episodes_per_worker) if n_episodes > 0]
        n_active_workers = len(workers_with_episodes)

        # Only use as many workers as needed
        n_workers = min(self.n_workers, n_active_workers)

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Prepare arguments for each worker
            futures = [
                executor.submit(
                    self.worker_func,
                    self.env_config,
                    self.agent_group.share_memory(),
                    n_episodes,
                    self.traj_len,
                    self.episode_limit,
                    self.epsilon,
                    self.device,
                    self.check_victory
                )
                for _, n_episodes in workers_with_episodes[:n_workers]
            ]

            # Collect results with progress bar
            episodes = []
            for future in tqdm(futures, total=len(futures), desc="Generating Episodes"):
                worker_episodes = future.result()
                episodes.extend(worker_episodes)

        return episodes

    def cleanup(self): # For compatibility
        return self