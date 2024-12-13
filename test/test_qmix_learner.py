import unittest
from unittest.mock import MagicMock

import torch

from src.learner.qmix_learner import QMIXLearner
from src.algorithm.model import RNNModel
from src.algorithm.agents import QMIXAgentGroup
from src.algorithm.model import ModelConfig
from src.rolloutworker.rolloutworker import RolloutWorker
from src.environment.mpe_env_config import MPEEnvConfig

class TestQMixLearner(unittest.TestCase):
    def setUp(self):
        self.capacity = 10
        self.traj_len = 5

        # Environment setup and model configuration
        self.env_config = MPEEnvConfig(env_config_dic={})
        self.env = self.env_config.create_env()
        self.env.reset()
        key = self.env.agents[0]
        self.obs_shape = self.env.observation_space(key).shape
        self.obs_shape = self.obs_shape[0]
        self.state_shape = self.env.state().shape[0] # Assuming state is a single vector for simplicity
        self.n_agents = len(self.env.agents) # Number of agents in the environment
        self.action_space_shape = self.env.action_space(key).n
        self.model_names = ["RNN0", "RNN0", "RNN1"]
        self.agents = {self.env.agents[i]: self.model_names[i] for i in range(len(self.env.agents))}
        self.env.close()

        # Model configuration
        self.model_layers = {
            "input_shape": self.obs_shape,
            "rnn_hidden_dim": 128,
            "output_shape": self.action_space_shape
        }

        self.model_configs = {
            "RNN0": ModelConfig(model_type="RNN",layers=self.model_layers),
            "RNN1": ModelConfig(model_type="RNN",layers=self.model_layers)
        }

        self.traj_len = 10
        self.num_workers = 4
        self.buffer_capacity = 50000
        self.episode_limit = 100
        self.device = 'cpu'

        self.critic_config = {
            'state_shape': self.state_shape,
            'n_agents': self.n_agents,
            'qmix_hidden_dim': 128,
            'hyper_hidden_dim': 64
        }

        # Initialize the QMixLearner with mocks
        self.learner = QMIXLearner(
            self.agents,
            self.env_config,
            self.model_configs,
            self.critic_config,
            self.traj_len,
            self.num_workers,
            self.buffer_capacity,
            self.episode_limit,
            self.device
        )

    def test_collect_experience(self):
        n_episodes = 4
        self.learner.collect_experience(n_episodes=n_episodes)
        self.assertNotEqual(len(self.learner.replay_buffer.buffer), 0)

    def test_learn(self):
        n_episodes = 20
        self.learner.collect_experience(n_episodes=n_episodes)
        self.learner.learn(sample_size=320, batch_size=32, epochs=10)

        
if __name__ == '__main__':
    unittest.main()