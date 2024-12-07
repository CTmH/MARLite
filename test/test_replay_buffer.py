import unittest
import numpy as np
from pettingzoo.mpe import simple_spread_v3

from src.util.replay_buffer import ReplayBuffer
from src.util.trajectory_dataset import TrajectoryDataset
from src.module.agents import QMIXAgentGroup
from src.module.model import ModelConfig
from src.rolloutworkers.worker import RolloutWorker
from src.environment.mpe_env_config import MPEEnvConfig

class TestReplayBuffer(unittest.TestCase):

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
        self.worker = RolloutWorker(env_config=self.env_config, agent_group=self.agent_group, rnn_traj_len=self.traj_len)


    def test_add_episode_too_short(self):
        self.buffer = ReplayBuffer(capacity=self.capacity, traj_len=self.traj_len)
        episode = self.worker.generate_episode(episode_limit=1, epsilon=0.5)
        self.buffer.add_episode(episode)
        self.assertEqual(len(self.buffer.episode_buffer), 0)
        self.assertEqual(len(self.buffer.buffer), 0)

    def test_add_episode_normal(self):
        self.buffer = ReplayBuffer(capacity=self.capacity, traj_len=self.traj_len)
        episode = self.worker.generate_episode(episode_limit=self.traj_len, epsilon=0.5)
        self.buffer.add_episode(episode)
        self.assertEqual(len(self.buffer.episode_buffer), 1)
        self.assertEqual(len(self.buffer.buffer), 4)

    def test_sample_with_data(self):
        self.buffer = ReplayBuffer(capacity=self.capacity, traj_len=self.traj_len)
        episode = self.worker.generate_episode(episode_limit=self.traj_len, epsilon=0.5)
        self.buffer.add_episode(episode)
        samples = self.buffer.sample(2)
        self.assertIsInstance(samples, TrajectoryDataset)
        self.assertEqual(len(samples), 2)

    def test_sample_more_than_available(self):
        self.buffer = ReplayBuffer(capacity=self.capacity, traj_len=self.traj_len)
        episode = self.worker.generate_episode(episode_limit=self.traj_len, epsilon=0.5)
        self.buffer.add_episode(episode)
        samples = self.buffer.sample(10)
        self.assertIsInstance(samples, TrajectoryDataset)
        self.assertEqual(len(samples), len(self.buffer.buffer))

if __name__ == '__main__':
    unittest.main()