import unittest
import numpy as np
from pettingzoo.mpe import simple_spread_v3
from torch.utils.data import DataLoader
import torch

from src.util.replay_buffer import ReplayBuffer
from src.util.trajectory_dataset import TrajectoryDataset, TrajectoryDataLoader
from src.algorithm.agents import QMIXAgentGroup
from src.algorithm.model import ModelConfig
from src.rolloutworker.rolloutworker import RolloutWorker
from src.environment.mpe_env_config import MPEEnvConfig

class TestTrajectoryDataset(unittest.TestCase):

    def setUp(self):
        self.capacity = 10
        self.traj_len = 5
        self.buffer = ReplayBuffer(capacity=self.capacity, traj_len=self.traj_len)

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
        self.env.close()

        # Model configuration
        self.model_layers = {
            "model_type": "RNN",
            "input_shape": self.obs_shape,
            "rnn_hidden_dim": 128,
            "output_shape": self.action_space_shape
        }

        self.model_configs = {
            "RNN0": ModelConfig(**self.model_layers),
            "RNN1": ModelConfig(**self.model_layers)
        }
        self.feature_extractor_configs = {
            "RNN0": ModelConfig(model_type="Identity"),
            "RNN1": ModelConfig(model_type="Identity"),
        }
        
        # Initialize QMIXAgents
        self.agent_group = QMIXAgentGroup(agents=self.agents,
                                          model_configs=self.model_configs,
                                          feature_extractors_configs=self.feature_extractor_configs,
                                          optim=torch.optim.Adam,
                                          lr=1e-4,
                                          device='cpu')
        self.worker = RolloutWorker(env_config=self.env_config, agent_group=self.agent_group, rnn_traj_len=self.traj_len)
        self.buffer = ReplayBuffer(capacity=self.capacity, traj_len=self.traj_len)
        episode = self.worker.generate_episode(episode_limit=10, epsilon=0.5)
        self.buffer.add_episode(episode)
        self.dataset = self.buffer.sample(10)

    def test_getitem_normal_case(self):
        for sample in self.dataset:
            self.assertEqual(len(sample['observations']), self.traj_len)
            self.assertEqual(len(sample['actions']), self.traj_len)
            self.assertEqual(len(sample['rewards']), self.traj_len)
            self.assertEqual(len(sample['states']), self.traj_len)
            self.assertTrue(isinstance(sample['observations'][0], dict))
            self.assertTrue(isinstance(sample['actions'][0], dict))
            self.assertTrue(isinstance(sample['rewards'][0], dict))
            self.assertTrue(isinstance(sample['states'], list))

class TestTrajectoryDataloader(unittest.TestCase):

    def setUp(self):
        self.capacity = 10
        self.traj_len = 5
        self.buffer = ReplayBuffer(capacity=self.capacity, traj_len=self.traj_len)

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
        self.env.close()

        # Model configuration
        self.model_layers = {
            "model_type": "RNN",
            "input_shape": self.obs_shape,
            "rnn_hidden_dim": 128,
            "output_shape": self.action_space_shape
        }

        self.model_configs = {
            "RNN0": ModelConfig(**self.model_layers),
            "RNN1": ModelConfig(**self.model_layers)
        }
        self.feature_extractor_configs = {
            "RNN0": ModelConfig(model_type="Identity"),
            "RNN1": ModelConfig(model_type="Identity"),
        }

        # Initialize QMIXAgents
        self.agent_group = QMIXAgentGroup(agents=self.agents,
                                          model_configs=self.model_configs,
                                          feature_extractors_configs=self.feature_extractor_configs,
                                          optim=torch.optim.Adam,
                                          lr=1e-4,
                                          device='cpu')
        self.worker = RolloutWorker(env_config=self.env_config, agent_group=self.agent_group, rnn_traj_len=self.traj_len)
        self.buffer = ReplayBuffer(capacity=self.capacity, traj_len=self.traj_len)
        episode = self.worker.generate_episode(episode_limit=10, epsilon=0.5)
        self.buffer.add_episode(episode)
        self.dataset = self.buffer.sample(10)
        self.dataloader = TrajectoryDataLoader(dataset=self.dataset, batch_size=3, shuffle=True)

    def test_get_batch(self):
        for batch in self.dataloader:
            print(batch)

if __name__ == '__main__':
    unittest.main()