import unittest
import torch
import numpy as np
import multiprocessing as mp

from src.algorithm.agents import QMIXAgentGroup
from src.rollout.multiprocess_rolloutworker import MultiProcessRolloutWorker
from src.algorithm.model import ModelConfig
from src.environment.env_config import EnvConfig
from src.util.optimizer_config import OptimizerConfig

class TestRolloutWorker(unittest.TestCase):

    def setUp(self):
        # Environment setup and model configuration
        self.env_config = {"module_name": "pettingzoo.mpe", "env_name": "simple_spread_v3"}
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
        self.agent_group = QMIXAgentGroup(agents=self.agents,
                                          model_configs=self.model_configs,
                                          feature_extractors_configs=self.feature_extractor_configs,
                                          optimizer_config=self.optimizer_config,
                                          device='cpu')

        self.traj_len = 5
        self.n_episodes = 2
        self.episode_limit = 10
        self.episode_queue = mp.Queue()
        self.worker = MultiProcessRolloutWorker(env_config=self.env_config,
                                    agent_group=self.agent_group,
                                    episode_queue=self.episode_queue,
                                    n_episodes=self.n_episodes,
                                    rnn_traj_len=self.traj_len,
                                    episode_limit=self.episode_limit,
                                    epsilon=0.9,
                                    device='cpu')

    def test_rollout(self):
        # Test act method with epsilon = 0 (greedy policy)
        episode = self.worker.rollout()
        self.assertFalse(not isinstance(episode, dict))
        self.assertEqual(len(episode["rewards"]), self.episode_limit)

    def test_run(self):
        self.worker.run()
        episodes = []
        while not self.episode_queue.empty():
            episode = self.episode_queue.get()
            self.assertFalse(not isinstance(episode, dict))
            self.assertEqual(len(episode["rewards"]), self.episode_limit)
            episodes.append(episode)

        self.assertEqual(len(episodes), self.n_episodes)

if __name__ == '__main__':
    unittest.main()
