import multiprocessing as mp
import queue
from typing import List, Any, Type
from ..algorithm.agents.agent_group import AgentGroup
from ..environment.env_config import EnvConfig
from .multiprocess_rolloutworker import MultiProcessRolloutWorker
from .multithread_rolloutworker import MultiThreadRolloutWorker
from ..algorithm.agents.agent_group_config import AgentGroupConfig

class RolloutManager:
    def __init__(self,
                 worker_class: Type[MultiProcessRolloutWorker | MultiThreadRolloutWorker],
                 env_config: EnvConfig,
                 agent_group_config: AgentGroupConfig,
                 agent_model_params,
                 agent_fe_params,
                 n_workers: int,
                 n_episodes: int,
                 traj_len: int,
                 episode_limit: int,
                 epsilon: float,
                 device: str):

        self.worker_class = worker_class
        self.env_config = env_config
        self.agent_group_config = agent_group_config
        self.agent_model_params = agent_model_params
        self.agent_fe_params = agent_fe_params
        self.n_workers = n_workers
        self.n_episodes = n_episodes
        self.traj_len = traj_len
        self.episode_limit = episode_limit
        self.epsilon = epsilon
        self.device = device

        self.workers: List[MultiProcessRolloutWorker | MultiThreadRolloutWorker] = []
        self.episode_queue: mp.Queue | queue.Queue = None

    def start(self):
        raise NotImplementedError

    def join(self):
        raise NotImplementedError

    def generate_episodes(self) -> List[Any]:
        raise NotImplementedError

    def cleanup(self):
        raise NotImplementedError