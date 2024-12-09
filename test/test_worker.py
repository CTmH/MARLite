import sys
import unittest
import torch
import numpy as np
from pettingzoo.mpe import simple_spread_v3

from src.algorithm.agents import QMIXAgentGroup
from src.rolloutWorker.episode_collector import RolloutWorker
from src.algorithm.model import ModelConfig
from src.environment.mpe_env_config import MPEEnvConfig

class TestRolloutWorker(unittest.TestCase):

    def setUp(self):
        # Environment setup and model configuration
        self.env_config = MPEEnvConfig(env_config_dic={})
        self.env = self.env_config.create_env()
        obs, _ = self.env.reset()
        key = self.env.agents[0]
        self.obs_shape = self.env.observation_space(key).shape
        self.obs_shape = self.obs_shape[0]
        self.action_space_shape = self.env.action_space(key).n
        self.model_names = ["RNN0", "RNN0", "RNN1"]
        self.agents = {self.env.agents[i]: self.model_names[i] for i in range(len(self.env.agents))}
        self.avail_actions = self.env.action_space
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
        
        # Initialize QMIXAgents
        self.agent_group = QMIXAgentGroup(agents=self.agents, model_configs=self.model_configs, device='cpu')
        self.worker = RolloutWorker(env_config=self.env_config, agent_group=self.agent_group)

    def test_generate_episode(self):
        # Test act method with epsilon = 0 (greedy policy)
        episode = self.worker.generate_episode(episode_limit=10)
        self.assertFalse(not isinstance(episode, dict))
        self.assertEqual(len(episode["rewards"]), 10)
        
if __name__ == '__main__':
    unittest.main()
