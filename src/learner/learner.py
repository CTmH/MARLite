import numpy as np
from typing import Dict
from copy import deepcopy
from multiprocessing import Process, Queue, Pool

from ..algorithm.agents import AgentGroup
from ..environment.env_config import EnvConfig
from ..algorithm.model import ModelConfig
from ..rolloutworker.rolloutworker import RolloutWorker
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
        # Critic
        self.target_critic = None
        self.eval_critic = None
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
    
    def __extract_batch(self, batch):
        # Extract necessary components from the trajectory
        observations = [traj['observations'] for traj in batch]
        states = [traj['states'] for traj in batch]
        actions = [traj['actions'] for traj in batch]
        rewards = [traj['rewards'] for traj in batch]
        next_state = [traj['next_states'][-1] for traj in batch] # Only need the next state from the last step of each trajectory
        next_observations = [traj['next_observations'][-1] for traj in batch] # Only need the next observation from the last step of each trajectory
        terminations = [traj['terminations'] for traj in batch]

        # Format Data

        # Observations
        # Nested list convert to numpy array (Batch Size, Time Step, Agent Number, Feature Dimensions) (B, T, N, F) -> (B, N, T, F)
        observations = [[[value for _, value in dict.items()] for dict in traj] for traj in observations]
        observations = np.array(observations)
        obs = obs.transpose(0,2,1,3)
        
        # Actions, Rewards, Terminations
        # Nested list convert to numpy array (Batch Size, Time Step, Agent Number) (B, T, N) -> (B, N, T)
        actions = [[[value for _, value in dict.items()] for dict in traj] for traj in actions]
        rewards = [[[value for _, value in dict.items()] for dict in traj] for traj in rewards]
        terminations = [[[value for _, value in dict.items()] for dict in traj] for traj in terminations]
        actions, rewards, terminations = np.array(actions), np.array(rewards), np.array(terminations)
        actions, rewards, terminations = actions.transpose(0,2,1), rewards.transpose(0,2,1), terminations.transpose(0,2,1)

        # States (Batch Size, Time Step, Feature Dimensions) (B, T, F)
        states = np.array(states)
        # Next State (Batch Size, Feature Dimensions) (B, F)
        next_state = np.array(next_state)
        # Next Observations (Batch Size, Agent Number, Feature Dimensions) (B, N, F)
        next_observations = [[value for _, value in dict.items()] for dict in next_observations]
        next_observations = np.array(next_observations)
        
        return observations, states, actions, rewards, next_state, next_observations, terminations
