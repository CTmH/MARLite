import unittest
import os
from unittest.mock import MagicMock, patch, call, mock_open
from copy import deepcopy

import torch

from src.trainer.qmix_trainer import QMIXTrainer
from src.algorithm.model import ModelConfig
from src.environment.mpe_env_config import MPEEnvConfig
from src.util.scheduler import Scheduler

class TestQMixTrainer(unittest.TestCase):
    def setUp(self):
        self.capacity = 10
        self.traj_len = 5
        self.n_episodes= 8
        self.workdir = "test_workdir"
        self.gamma = 0.95
        self.epochs = 5
        self.epsilon_scheduler = Scheduler(start_value=1.0, end_value=0.05, decay_steps=self.epochs, type="linear")
        self.sample_ratio_scheduler = Scheduler(start_value=1.0, end_value=0.3, decay_steps=self.epochs, type="linear")
        self.critic_lr = 0.01
        self.critic_optimizer = torch.optim.Adam

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
        self.state_shape = self.env.state().shape
        self.state_shape = self.state_shape[0]
        self.n_agents = self.env.num_agents
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
        self.critic_feature_extractors = ModelConfig(model_type="Identity")

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

        # Initialize the QMixTrainer
        self.trainer = QMIXTrainer(
            self.agents,
            self.env_config,
            self.model_configs,
            self.feature_extractor_configs,
            self.critic_config,
            self.critic_feature_extractors,
            self.epsilon_scheduler,
            self.sample_ratio_scheduler,
            self.traj_len,
            self.num_workers,
            self.epochs,
            self.buffer_capacity,
            self.episode_limit,
            self.n_episodes,
            self.gamma,
            self.critic_lr,
            self.critic_optimizer,
            self.workdir,
            self.device
        )


    def test_collect_experience(self):
        n_episodes = 4
        self.trainer.collect_experience(n_episodes, self.episode_limit, 0.9)
        self.assertNotEqual(len(self.trainer.replay_buffer.buffer), 0)

    def test_learn(self):
        n_episodes = 20
        origin_agent_params, origin_agent_fe_params = deepcopy(self.trainer.target_agent_group.get_model_params())
        origin_critic_params = deepcopy(self.trainer.target_critic.state_dict())
        self.trainer.collect_experience(n_episodes, self.episode_limit, 0.9)
        self.trainer.learn(sample_size=320, batch_size=32, times=10)
        self.trainer.update_params()
        agent_params, agent_fe_params = self.trainer.target_agent_group.get_model_params()
        critic_params = self.trainer.target_critic.state_dict()

        for w1, w2 in zip(critic_params.values(), origin_critic_params.values()):
            self.assertFalse(torch.equal(w1, w2))

        for model1, model2 in zip(agent_params.values(), origin_agent_params.values()):
            for w1, w2 in zip(model1.values(), model2.values()):
                self.assertFalse(torch.equal(w1, w2))
        for model1, model2 in zip(agent_fe_params.values(), origin_agent_fe_params.values()):
            for w1, w2 in zip(model1.values(), model2.values()):
                self.assertFalse(torch.equal(w1, w2))

    def test_train(self):
        reward, _ = self.trainer.evaluate()
        best_reward, _ = self.trainer.train(target_reward=5)
        self.assertGreaterEqual(best_reward, reward)

    @patch('src.trainer.trainer.torch.save')
    def test_save_model(self, mock_torch_save):
        checkpoint = "test_model"
        self.trainer.save_model(checkpoint)
        call_list = []
        # Check actor models
        agent_params, agent_fe_params = self.trainer.target_agent_group.get_model_params()
        for model_name, params in agent_params.items():
            actor_path = os.path.join(self.trainer.agentsdir, model_name, 'model',f'{checkpoint}.pth')
            call_list.append(call(params, actor_path))
        for model_name, params in agent_fe_params.items():
            actor_path = os.path.join(self.trainer.agentsdir, model_name, 'feature_extractor',f'{checkpoint}.pth')
            call_list.append(call(params, actor_path))
        # Check critic models
        critic_path = os.path.join(self.trainer.criticdir, 'model', f'{checkpoint}.pth')
        call_list.append(call(self.trainer.target_critic.state_dict(), critic_path))
        critic_path = os.path.join(self.trainer.criticdir, 'feature_extractor', f'{checkpoint}.pth')
        call_list.append(call(self.trainer.target_critic.state_dict(), critic_path))
        #mock_torch_save.assert_has_calls(call_list)
        self.assertEqual(mock_torch_save.call_count, len(call_list))

if __name__ == '__main__':
    unittest.main()