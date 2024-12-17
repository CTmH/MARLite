import unittest
import os
from unittest.mock import MagicMock, patch, call, mock_open
from copy import deepcopy

import torch

from src.learner.qmix_learner import QMIXLearner
from src.algorithm.model import RNNModel
from src.algorithm.agents import QMIXAgentGroup
from src.algorithm.model import ModelConfig
from src.rolloutworker.rolloutworker import RolloutWorker
from src.environment.mpe_env_config import MPEEnvConfig

@patch('os.makedirs')
class TestQMixLearner(unittest.TestCase):
    def setUp(self):
        self.capacity = 10
        self.traj_len = 5
        self.n_episodes= 8
        self.workdir = "test/test_workdir"

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
        self.buffer_capacity = 500
        self.episode_limit = 200
        self.device = 'cpu'

        self.critic_config = {
            'state_shape': self.state_shape,
            'input_dim': self.n_agents * self.action_space_shape,
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
            self.n_episodes,
            self.workdir,
            self.device
        )

    def test_collect_experience(self):
        n_episodes = 4
        self.learner.collect_experience(n_episodes=n_episodes)
        self.assertNotEqual(len(self.learner.replay_buffer.buffer), 0)

    @patch('src.learner.learner.save_model')
    @patch('src.learner.save_results_to_csv')
    def test_learn(self):
        n_episodes = 20
        origin_agent_params = deepcopy(self.learner.target_agent_group.get_model_params())
        origin_critic_params = deepcopy(self.learner.target_critic.state_dict())
        self.learner.collect_experience(n_episodes=n_episodes)
        self.learner.learn(sample_size=320, batch_size=32, times=10)
        self.learner.update_params()
        agent_params = self.learner.target_agent_group.get_model_params()
        critic_params = self.learner.target_critic.state_dict()

        for w1, w2 in zip(critic_params.values(), origin_critic_params.values()):
            self.assertFalse(torch.equal(w1, w2))

        for model1, model2 in zip(agent_params.values(), origin_agent_params.values()):
            for w1, w2 in zip(model1.values(), model2.values()):
                self.assertFalse(torch.equal(w1, w2))

    def test_train(self):
        reward, _ = self.learner.evaluate()
        best_reward, _ = self.learner.train(epochs=5, target_reward=5)
        self.assertGreaterEqual(best_reward, reward)

    @patch('src.learner.learner.torch.save')
    def test_save_model(self, mock_torch_save):
        filename = "test_model"
        self.learner.save_model(filename)
        call_list = []
        # Check actor models
        for model_name, params in self.learner.target_agent_group.get_model_params().items():
            actor_path = os.path.join(self.learner.modeldir, f'actor_{model_name}_{filename}.pth')
            call_list.append(call(params, actor_path))
        # Check critic models
        critic_path = os.path.join(self.workdir, 'models', 'critic_test_model.pth')
        call_list.append(call(self.learner.target_critic.state_dict(), critic_path))
        #mock_torch_save.assert_has_calls(call_list)
        self.assertEqual(mock_torch_save.call_count, len(call_list))


        
if __name__ == '__main__':
    unittest.main()