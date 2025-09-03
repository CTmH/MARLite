import unittest
import numpy as np
import multiprocessing as mp

from marlite.replaybuffer.normal_replaybuffer import NormalReplayBuffer
from marlite.util.trajectory_dataset import TrajectoryDataLoader
from marlite.algorithm.agents.qmix_agent_group import QMIXAgentGroup
from marlite.algorithm.model import ModelConfig
from marlite.rollout.multiprocess_rollout import multiprocess_rollout
from marlite.environment import EnvConfig
from marlite.util.optimizer_config import OptimizerConfig

class TestTrajectoryDataset(unittest.TestCase):

    def setUp(self):
        self.capacity = 10
        self.traj_len = 5
        self.buffer = NormalReplayBuffer(capacity=self.capacity, traj_len=self.traj_len)

        # Environment setup and model configuration
        self.env_config = {"module_name": "mpe2", "env_name": "simple_spread_v3"}
        self.env_config = EnvConfig(**self.env_config)
        self.env = self.env_config.create_env()
        obs, _ = self.env.reset()
        key = self.env.agents[0]
        self.obs_shape = self.env.observation_space(key).shape
        self.obs_shape = self.obs_shape[0]
        self.action_space_shape = self.env.action_space(key).n
        self.model_names = ["RNN0", "RNN0", "RNN1"]
        self.agents = {self.env.agents[i]: self.model_names[i] for i in range(len(self.env.agents))}
        observations = {agent: [] for agent in self.env.agents}
        seq_length = 5
        for i in range(seq_length):
            actions = {agent: self.env.action_space(agent).sample() for agent in self.env.agents}
            obs, rewards, terminations, truncations, infos = self.env.step(actions)
            for agent in self.env.agents:
                observations[agent].append(obs[agent])
        self.observations = {key: np.array(value) for key, value in observations.items()}

        self.avail_actions = self.env.action_space
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

        self.optimizer_config = OptimizerConfig(type="Adam", lr=0.001)

        # Initialize QMIXAgents
        self.agent_group = QMIXAgentGroup(agent_model_dict=self.agents,
                                          model_configs=self.model_configs,
                                          feature_extractors_configs=self.feature_extractor_configs,
                                          optimizer_config=self.optimizer_config,
                                          device='cpu')
        self.episode_limit=10
        self.epsilon=0.5
        self.n_episodes = 5
        self.buffer = NormalReplayBuffer(capacity=self.capacity, traj_len=self.traj_len)
        episode = episode = multiprocess_rollout(env_config=self.env_config,
                                    agent_group=self.agent_group,
                                    rnn_traj_len=self.traj_len,
                                    episode_limit=self.episode_limit,
                                    epsilon=0.9,
                                    device='cpu')
        self.buffer.add_episode(episode)
        self.dataset = self.buffer.sample(10)

    def test_getitem_normal_case(self):
        for sample in self.dataset:
            self.assertEqual(len(sample['observations']), self.traj_len)
            self.assertEqual(len(sample['actions']), self.traj_len)
            self.assertEqual(len(sample['rewards']), self.traj_len)
            self.assertEqual(len(sample['states']), self.traj_len)
            self.assertEqual(len(sample['edge_indices']), self.traj_len)
            self.assertTrue(isinstance(sample['observations'][0], dict))
            self.assertTrue(isinstance(sample['actions'][0], dict))
            self.assertTrue(isinstance(sample['rewards'][0], dict))
            self.assertTrue(isinstance(sample['states'], list))

class TestTrajectoryDataloader(unittest.TestCase):

    def setUp(self):
        self.capacity = 10
        self.traj_len = 5
        self.buffer = NormalReplayBuffer(capacity=self.capacity, traj_len=self.traj_len)

        # Environment setup and model configuration
        self.env_config = {"module_name": "mpe2", "env_name": "simple_spread_v3"}
        self.env_config = EnvConfig(**self.env_config)
        self.env = self.env_config.create_env()
        obs, _ = self.env.reset()
        key = self.env.agents[0]
        self.obs_shape = self.env.observation_space(key).shape
        self.obs_shape = self.obs_shape[0]
        self.action_space_shape = self.env.action_space(key).n
        self.model_names = ["RNN0", "RNN0", "RNN1"]
        self.agents = {self.env.agents[i]: self.model_names[i] for i in range(len(self.env.agents))}
        observations = {agent: [] for agent in self.env.agents}
        seq_length = 5
        for i in range(seq_length):
            actions = {agent: self.env.action_space(agent).sample() for agent in self.env.agents}
            obs, rewards, terminations, truncations, infos = self.env.step(actions)
            for agent in self.env.agents:
                observations[agent].append(obs[agent])
        self.observations = {key: np.array(value) for key, value in observations.items()}

        self.avail_actions = self.env.action_space
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

        self.optimizer_config = OptimizerConfig(type="Adam", lr=0.001)

        # Initialize QMIXAgents
        self.agent_group = QMIXAgentGroup(agent_model_dict=self.agents,
                                          model_configs=self.model_configs,
                                          feature_extractors_configs=self.feature_extractor_configs,
                                          optimizer_config=self.optimizer_config,
                                          device='cpu')

        self.episode_limit=10
        self.epsilon=0.5
        self.n_episodes = 5
        self.buffer = NormalReplayBuffer(capacity=self.capacity, traj_len=self.traj_len)
        episode = episode = multiprocess_rollout(env_config=self.env_config,
                                    agent_group=self.agent_group,
                                    rnn_traj_len=self.traj_len,
                                    episode_limit=self.episode_limit,
                                    epsilon=0.9,
                                    device='cpu')
        self.buffer.add_episode(episode)
        self.dataset = self.buffer.sample(10)
        self.dataloader = TrajectoryDataLoader(dataset=self.dataset, batch_size=3, shuffle=True)

    def test_get_batch(self):
        for batch in self.dataloader:
            for key1, key2 in zip(batch.keys(), self.dataloader.attr):
                self.assertEqual(key1, key2)

if __name__ == '__main__':
    unittest.main()