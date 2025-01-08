import unittest
import os
import yaml
from unittest.mock import MagicMock, patch, call, mock_open
from copy import deepcopy

import torch

from src.trainer.qmix_trainer import QMIXTrainer
from src.trainer.trainer_config import TrainerConfig
from src.algorithm.model import ModelConfig
from src.environment.env_config import EnvConfig
from src.util.scheduler import Scheduler

class TestQMixTrainer(unittest.TestCase):
    def setUp(self):
        self.config_path = 'test/config/qmix_default.yaml'
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.config['trainer_config']['train_args']['epochs'] = 2
        self.config['rollout_config']['n_episodes'] = 20
        self.config['rollout_config']['episode_limit'] = 30
        self.config['replaybuffer_config']['capacity'] = 50
        self.trainer_config = TrainerConfig(self.config)
        self.trainer = self.trainer_config.create_trainer()


    def test_collect_experience(self):
        n_episodes = 4
        self.trainer.collect_experience(0.9)
        self.assertNotEqual(len(self.trainer.replaybuffer.buffer), 0)

    def test_learn(self):
        n_episodes = 20
        origin_agent_params, origin_agent_fe_params = deepcopy(self.trainer.target_agent_group.get_model_params())
        origin_critic_params = deepcopy(self.trainer.target_critic.state_dict())
        self.trainer.collect_experience(0.9)
        self.trainer.learn(sample_size=320, batch_size=32, times=10)
        self.trainer.update_eval_model_params()
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
        best_reward, _ = self.trainer.train(epochs=2, target_reward=5)
        self.assertGreaterEqual(best_reward, reward)

    @patch('src.trainer.trainer.torch.save')
    def test_save_model(self, mock_torch_save):
        checkpoint = "save_test_model"
        self.trainer.save_current_model(checkpoint)
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
        #mock_torch_save.assert_has_calls(call_list)
        self.assertEqual(mock_torch_save.call_count, len(call_list))

    def test_load_model(self):
        checkpoint = "load_test_model"
        model_params, fe_params = self.trainer.target_agent_group.get_model_params()
        critic_params = deepcopy(self.trainer.target_critic.state_dict())
        self.trainer.load_model(checkpoint)
        
        # Check Cirtic
        for key in critic_params:
            loaded_critic_params = deepcopy(self.trainer.target_critic.state_dict())
            self.assertFalse(torch.equal(critic_params[key], loaded_critic_params[key]))
        # Check Agent
        loaded_model_params, loaded_fe_params = self.trainer.target_agent_group.get_model_params()
        for model_name in self.trainer.target_agent_group.models.keys():
            for key in model_params[model_name]:
                self.assertFalse(torch.equal(model_params[model_name][key], loaded_model_params[model_name][key]))
            for key in fe_params[model_name]:
                self.assertFalse(torch.equal(fe_params[model_name][key], loaded_fe_params[model_name][key]))

if __name__ == '__main__':
    unittest.main()