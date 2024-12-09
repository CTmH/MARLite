from typing import Dict
from copy import deepcopy
from multiprocessing import Process, Queue, Pool

from ..algorithm.agents import AgentGroup
from ..environment.env_config import EnvConfig
from ..algorithm.model import ModelConfig
from ..rolloutWorker.episode_collector import RolloutWorker
from ..util.replay_buffer import ReplayBuffer

class Learner():
    def __init__(self, 
                 agents: Dict[str, str], 
                 env_config: EnvConfig, 
                 model_configs: ModelConfig, 
                 traj_len: int, 
                 n_workers: int, 
                 buffer_capacity: int = 50000,
                 episode_limit: int = 500,
                 device: str = 'cpu'):
        
        self.env_config = env_config
        self.model_configs = model_configs
        self.traj_len = traj_len
        self.n_workers = n_workers
        self.episode_limit = episode_limit
        self.device = device
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity, traj_len=self.traj_len)
        self.agents = agents
        self.target_agent_group = AgentGroup(agents=self.agents, env_config=self.env_config, model_configs=self.model_configs)
        # Set the same parameters for evaluation agents as target agents.
        self.target_models_params = self.target_agent_group.get_model_params()
        self.eval_agent_group = deepcopy(self.target_agent_group)  # Deep copy the target agent group to create evaluation agents.
        self.epsilon = 0.9
        self.n_episodes = 30

    def learn(self):
        raise NotImplementedError
    
    def save_model(self):
        raise NotImplementedError
    
    def load_model(self):
        raise NotImplementedError
    
    def collect_experience(self):

        rollout_workers = [RolloutWorker(env_config=self.env_config,
                                 agent_group=self.eval_agent_group,
                                 rnn_traj_len=self.traj_len) for _ in range(self.n_episodes)]

        with Pool(self.n_workers) as pool:
            episodes = pool.map(lambda worker: worker.generate_episode(self.episode_limit, self.epsilon), rollout_workers)

        for episode in episodes:
            self.replay_buffer.add_episode(episode)
        
        return self
