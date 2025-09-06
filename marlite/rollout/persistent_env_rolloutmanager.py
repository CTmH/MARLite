import torch
import torch.multiprocessing as mp
from typing import List, Any, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
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

        # Handle CUDA device allocation when device is just "cuda" without device number
        if self.device == "cuda":
            if torch.cuda.is_available():
                num_cuda_devices = torch.cuda.device_count()
                # Distribute workers evenly across available CUDA devices
                devices = [f"cuda:{i % num_cuda_devices}" for i in range(n_workers)]
            else:
                raise RuntimeError("CUDA is not available on this system")
        else:
            # Use the specified device for all workers
            devices = [self.device for _ in range(n_workers)]

        episodes = []
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
                    devices[i],  # Use the assigned device for this worker
                    self.check_victory
                )
                for i, (worker_idx, n_episodes) in enumerate(workers_with_episodes[:n_workers])
            ]

            # Collect results with progress bar
            completed_count = 0
            pbar = tqdm(total=n_workers, desc="Generating Episodes")
            for future in as_completed(futures):
                try:
                    worker_episodes = future.result()
                    episodes.extend(worker_episodes)
                    completed_count += 1
                except Exception as e:
                    # Log the error but continue with other workers
                    print(f"Worker failed with error: {e}")
                    # Continue with remaining futures
                    completed_count += 1
                    continue
                # Update progress bar
                pbar.update(completed_count)
            pbar.close()

        return episodes

    def cleanup(self): # For compatibility
        return self