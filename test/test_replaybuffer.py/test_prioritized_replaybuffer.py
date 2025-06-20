import unittest
import torch
import numpy as np
from queue import Queue

from src.replaybuffer.prioritized_replaybuffer import PrioritizedReplayBuffer
from src.util.trajectory_dataset import TrajectoryDataset
from src.algorithm.agents import QMIXAgentGroup
from src.algorithm.agents.agent_group_config import AgentGroupConfig
from src.algorithm.model import ModelConfig
from src.rollout.multithread_rolloutworker import MultiThreadRolloutWorker
from src.environment.env_config import EnvConfig
from src.util.optimizer_config import OptimizerConfig

class TestPrioritizedReplayBuffer(unittest.TestCase):

    def setUp(self):
        self.capacity = 10
        self.traj_len = 5
        self.buffer = PrioritizedReplayBuffer(capacity=self.capacity, traj_len=self.traj_len, priority_attr="all_agents_sum_rewards", alpha=1)

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

        self.traj_len = 5
        self.n_episodes = 2
        self.episode_limit = 10
        self.episode_queue = Queue()
        self.worker = MultiThreadRolloutWorker(env_config=self.env_config,
                                    agent_group=self.agent_group,
                                    episode_queue=self.episode_queue,
                                    n_episodes=self.n_episodes,
                                    rnn_traj_len=self.traj_len,
                                    episode_limit=self.episode_limit,
                                    epsilon=0.9,
                                    device='cpu')


    def test_add_episode_too_short(self):
        self.worker = MultiThreadRolloutWorker(env_config=self.env_config,
                                    agent_group=self.agent_group,
                                    episode_queue=self.episode_queue,
                                    n_episodes=self.n_episodes,
                                    rnn_traj_len=self.traj_len,
                                    episode_limit=1,
                                    epsilon=0.9,
                                    device='cpu')
        self.buffer = PrioritizedReplayBuffer(capacity=self.capacity, traj_len=self.traj_len, priority_attr="all_agents_sum_rewards", alpha=1)
        episode = self.worker.rollout()
        self.buffer.add_episode(episode)
        self.assertEqual(self.buffer.tail, -1)
        self.assertEqual(len(self.buffer.buffer), 0)

    def test_remove_episode(self):
        self.buffer = PrioritizedReplayBuffer(capacity=self.capacity, traj_len=self.traj_len, priority_attr="all_agents_sum_rewards", alpha=1)
        episode = self.worker.rollout()
        self.buffer.add_episode(episode)
        self.buffer.remove_episode(self.buffer.tail)
        self.assertEqual(self.buffer.tail, 0)
        self.assertEqual(len(self.buffer.buffer), 0)

    def test_add_episode_normal(self):
        self.buffer = PrioritizedReplayBuffer(capacity=self.capacity, traj_len=self.traj_len, priority_attr="all_agents_sum_rewards", alpha=1)
        episode = self.worker.rollout()
        self.buffer.add_episode(episode)
        self.assertTrue(self.buffer.episode_buffer[0] != None)
        self.assertEqual(self.buffer.tail, 0)
        self.assertEqual(len(self.buffer.buffer), self.traj_len * self.n_episodes)

    def test_add_episode_full_buffer(self):
        capacity = 3
        self.buffer = PrioritizedReplayBuffer(capacity=capacity, traj_len=self.traj_len, priority_attr="all_agents_sum_rewards", alpha=1)
        for i in range(capacity+1):
            episode = self.worker.rollout()
            self.buffer.add_episode(episode)

        self.assertEqual(len(self.buffer.episode_buffer), capacity)
        self.assertEqual(len(self.buffer.buffer), capacity * self.traj_len * self.n_episodes)

    def test_sample_with_data(self):
        self.buffer = PrioritizedReplayBuffer(capacity=self.capacity, traj_len=self.traj_len, priority_attr="all_agents_sum_rewards", alpha=1)
        episode = self.worker.rollout()
        self.buffer.add_episode(episode)
        samples = self.buffer.sample(2)
        self.assertIsInstance(samples, TrajectoryDataset)
        self.assertEqual(len(samples), 2)

    def test_sample_more_than_available(self):
        self.buffer = PrioritizedReplayBuffer(capacity=self.capacity, traj_len=self.traj_len, priority_attr="all_agents_sum_rewards", alpha=1)
        episode = self.worker.rollout()
        self.buffer.add_episode(episode)
        samples = self.buffer.sample(10)
        self.assertIsInstance(samples, TrajectoryDataset)
        self.assertEqual(len(samples), len(self.buffer.buffer))

if __name__ == '__main__':
    unittest.main()