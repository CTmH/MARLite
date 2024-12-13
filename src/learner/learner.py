import numpy as np
import torch
from typing import Dict
from copy import deepcopy
from multiprocessing import Process, Queue, Pool

from ..algorithm.agents import AgentGroup
from ..environment.env_config import EnvConfig
from ..algorithm.model import ModelConfig
from ..rolloutworker.rolloutworker import RolloutWorker
from ..util.replay_buffer import ReplayBuffer

def get_episode(worker, episode_limit, epsilon):
    return worker.generate_episode(episode_limit, epsilon)

class Learner():
    def __init__(self, 
                 agents: Dict[str, str], 
                 env_config: EnvConfig, 
                 model_configs: ModelConfig, 
                 critic_config: Dict[str, any],
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
        self.target_agent_group = AgentGroup(agents=self.agents, model_configs=self.model_configs, device=self.device)
        # Set the same parameters for evaluation agents as target agents.
        self.target_models_params = self.target_agent_group.get_model_params()
        self.eval_agent_group = deepcopy(self.target_agent_group)  # Deep copy the target agent group to create evaluation agents.
        # Critic
        self.target_critic = torch.nn.Module()
        self.eval_critic = torch.nn.Module()
        self.optimizer = None
        self.epsilon = 0.9
        self.gamma = 0.9

    def learn(self):
        raise NotImplementedError
    
    def save_model(self):
        raise NotImplementedError
    
    def load_model(self):
        raise NotImplementedError
    
    def collect_experience(self, n_episodes=10):
        """
        Collect experiences using multiple rollout workers.
        """
        job = RolloutWorker(env_config=self.env_config,
                                 agent_group=self.eval_agent_group,
                                 rnn_traj_len=self.traj_len)
        episodes = [job.generate_episode(self.episode_limit, self.epsilon) for _ in range(n_episodes)]

        for episode in episodes:
            self.replay_buffer.add_episode(episode)
        
        return self
    # TODO: parallelize the experience collection process.
    '''
    def collect_experience(self, n_episodes=10):
        """
        Collect experiences using multiple rollout workers.
        """
        rworker = RolloutWorker(env_config=self.env_config,
                                 agent_group=deepcopy(self.eval_agent_group),
                                 rnn_traj_len=self.traj_len)
        job = [[deepcopy(rworker), self.episode_limit, self.epsilon] for _ in range(n_episodes)]

        with Pool(self.n_workers) as pool:
            episodes = pool.starmap(get_episode, job)

        for episode in episodes:
            self.replay_buffer.add_episode(episode)
        
        return self
    '''
    def extract_batch(self, batch):
        # Extract necessary components from the trajectory
        observations = batch['observations']
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_state = batch['next_states']
        next_observations = batch['next_observations']
        terminations = batch['terminations']
        # Format Data

        # Observations
        # Nested list convert to numpy array (Time Step, Agent Number, Batch Size, Feature Dimensions) (T, N, B, F) -> (B, N, T, F)
        observations = [[value for _, value in agent.items()] for agent in observations]
        next_observations = [[value for _, value in agent.items()] for agent in next_observations]
        observations, next_observations = np.array(observations), np.array(next_observations)
        observations, next_observations = observations.transpose(2,1,0,3), next_observations.transpose(2,1,0,3)
        
        # Actions, Rewards, Terminations
        # Nested list convert to numpy array (Time Step, Agent Number, Batch Size) (T, N, B) -> (B, N, T)
        actions = [[value for _, value in agent.items()] for agent in actions]
        rewards = [[value for _, value in agent.items()] for agent in rewards]
        terminations = [[value for _, value in agent.items()] for agent in terminations]
        actions, rewards, terminations = np.array(actions), np.array(rewards), np.array(terminations)
        actions, rewards, terminations = actions.transpose(2,1,0), rewards.transpose(2,1,0), terminations.transpose(2,1,0)
        terminations = terminations.astype(int)  # Convert to int type for termination flags

        # States (Time Step, Batch Size, Feature Dimensions) (T, B, F) -> (B, T, F)
        states, next_state = np.array(states),np.array(next_state)
        states, next_state = states.transpose(1,0,2), next_state.transpose(1,0,2)

        return observations, states, actions, rewards, next_state, next_observations, terminations
    
    def update_eval_models(self):
        # Update the evaluation models with the latest weights from the training models
        agents_params = self.target_agent_group.get_model_params()
        self.eval_agent_group.set_model_params(agents_params)
        self.eval_critic.load_state_dict(self.target_critic.state_dict())