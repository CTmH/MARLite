import unittest
import os
import yaml
from copy import deepcopy
import tempfile
import torch
import pygame

from marlite.trainer import TrainerConfig

class TestQMixTrainer(unittest.TestCase):
    def setUp(self):
        self.config_path = 'test/config/qmix_kaz.yaml'
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.config['trainer_config']['train_args']['epochs'] = 2
        self.config['rollout_config']['n_episodes'] = 2
        self.config['rollout_config']['n_eval_episodes'] = 2
        self.config['rollout_config']['episode_limit'] = 20
        self.config['replaybuffer_config']['capacity'] = 5
        self.trainer_config = TrainerConfig(self.config)

    def test_collect_experience(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, 'logs')
            self.trainer.checkpointdir = os.path.join(self.trainer.workdir, 'checkpoints')
            n_episodes = 4
            self.trainer.collect_experience(0.9)
            self.assertNotEqual(len(self.trainer.replaybuffer.buffer), 0)

    def test_learn(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, 'logs')
            self.trainer.checkpointdir = os.path.join(self.trainer.workdir, 'checkpoints')
            origin_critic_params = deepcopy(self.trainer.target_critic.state_dict())
            self.trainer.collect_experience(0.9)
            self.trainer.learn(sample_size=32, batch_size=8, times=1)
            self.trainer.update_target_model_params()
            critic_params = self.trainer.target_critic.state_dict()

            for w1, w2 in zip(critic_params.values(), origin_critic_params.values()):
                if w1.requires_grad:
                    self.assertFalse(torch.equal(w1, w2))

    def test_save_load_checkpoint(self):
        checkpoint = 'test_checkpoint'
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, 'logs')
            self.trainer.checkpointdir = os.path.join(self.trainer.workdir, 'checkpoints')
            self.trainer.save_current_model(checkpoint)
            self.trainer.load_checkpoint(checkpoint)

    def test_train(self):
        pygame.font.init()
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, 'logs')
            self.trainer.checkpointdir = os.path.join(self.trainer.workdir, 'checkpoints')
            reward, _, _ = self.trainer.evaluate()
            best_reward, _ = self.trainer.train(epochs=2, target_reward=5)

    def test_target_update(self):

        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, 'logs')
            self.trainer.checkpointdir = os.path.join(self.trainer.workdir, 'checkpoints')

            # 使用随机参数初始化评估模型
            for param in self.trainer.eval_critic.parameters():
                torch.nn.init.ones_(param)
            for param in self.trainer.target_critic.parameters():
                torch.nn.init.zeros_(param)

            for (_, fe), (_, model) in zip(
                self.trainer.eval_agent_group.feature_extractors.items(),
                self.trainer.eval_agent_group.models.items()):
                for param in fe.parameters():
                    torch.nn.init.ones_(param)
                for param in model.parameters():
                    torch.nn.init.ones_(param)

            for (_, fe), (_, model) in zip(
                self.trainer.target_agent_group.feature_extractors.items(),
                self.trainer.target_agent_group.models.items()):
                for param in fe.parameters():
                    torch.nn.init.zeros_(param)
                for param in model.parameters():
                    torch.nn.init.zeros_(param)

            # 获取更新前的参数
            original_target_critic_params = deepcopy(self.trainer.target_critic.state_dict())

            original_target_agent_group_params = self.trainer.target_agent_group.get_agent_group_params()

            # 执行参数更新
            self.trainer.update_target_model_params()

            self.assertEqual(
                self.trainer.target_critic._modules.keys(),
                self.trainer.eval_critic._modules.keys()
            )
            # 验证critic参数更新及深拷贝正确性
            new_target_critic_params = deepcopy(self.trainer.target_critic.state_dict())
            for name in new_target_critic_params:
                self.assertTrue(torch.equal(
                    new_target_critic_params[name],
                    self.trainer.eval_critic.state_dict()[name]
                ))
                self.assertFalse(torch.equal(
                    new_target_critic_params[name],
                    original_target_critic_params[name]
                ))

            # 验证 agent group 参数更新及深拷贝正确性
            eval_agent_group_models = {'feature_extractor': self.trainer.eval_agent_group.feature_extractors,
                                       'model': self.trainer.eval_agent_group.models}
            new_target_agent_group_params = self.trainer.target_agent_group.get_agent_group_params()
            for key in eval_agent_group_models.keys():
                for model_name in self.trainer.eval_agent_group.models.keys():
                    for name in original_target_agent_group_params[key][model_name]:
                        self.assertTrue(torch.equal(
                            new_target_agent_group_params[key][model_name][name],
                            eval_agent_group_models[key][model_name].state_dict()[name]
                        ))
                        self.assertFalse(torch.equal(
                            new_target_agent_group_params[key][model_name][name],
                            original_target_agent_group_params[key][model_name][name],
                        ))

            # 验证深拷贝有效性：修改eval参数不应影响target
            for param in self.trainer.eval_critic.parameters():
                torch.nn.init.normal_(param)
            for name in new_target_critic_params:
                self.assertFalse(torch.equal(
                    self.trainer.eval_critic.state_dict()[name],
                    self.trainer.target_critic.state_dict()[name]
                ))
                self.assertFalse(torch.equal(
                    new_target_critic_params[name],
                    self.trainer.eval_critic.state_dict()[name]
                ))

            for (_, fe), (_, model) in zip(
                self.trainer.eval_agent_group.feature_extractors.items(),
                self.trainer.eval_agent_group.models.items()):
                for param in fe.parameters():
                    torch.nn.init.normal_(param)
                for param in model.parameters():
                    torch.nn.init.normal_(param)

            new_target_agent_group_params = self.trainer.target_agent_group.get_agent_group_params()
            for key in eval_agent_group_models.keys():
                for model_name in self.trainer.eval_agent_group.models.keys():
                    for name in original_target_agent_group_params[key][model_name]:
                        self.assertFalse(torch.equal(
                            new_target_agent_group_params[key][model_name][name],
                            eval_agent_group_models[key][model_name].state_dict()[name]
                        ))
                        self.assertFalse(torch.equal(
                            new_target_agent_group_params[key][model_name][name],
                            original_target_agent_group_params[key][model_name][name],
                        ))

    def test_data_parallel(self):
        self.config_path = 'test/config/qmix_kaz.yaml'
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.config['trainer_config']['train_args']['epochs'] = 2
        self.config['rollout_config']['n_episodes'] = 2
        self.config['rollout_config']['n_eval_episodes'] = 2
        self.config['rollout_config']['episode_limit'] = 2
        self.config['replaybuffer_config']['capacity'] = 2
        self.config['trainer_config']['use_data_parallel'] = True
        self.config['trainer_config']['train_device'] = 'cuda'
        self.trainer_config = TrainerConfig(self.config)
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, 'logs')
            self.trainer.checkpointdir = os.path.join(self.trainer.workdir, 'checkpoints')
            origin_critic_params = deepcopy(self.trainer.target_critic.state_dict())
            self.trainer.collect_experience(0.9)
            self.trainer.learn(sample_size=32, batch_size=8, times=1)
            self.trainer.update_target_model_params()
            critic_params = self.trainer.target_critic.state_dict()

            for w1, w2 in zip(critic_params.values(), origin_critic_params.values()):
                if w1.requires_grad:
                    self.assertFalse(torch.equal(w1, w2))

    def test_torch_compile(self):
        self.config_path = 'test/config/qmix_kaz.yaml'
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.config['trainer_config']['train_args']['epochs'] = 2
        self.config['rollout_config']['n_episodes'] = 2
        self.config['rollout_config']['n_eval_episodes'] = 2
        self.config['rollout_config']['episode_limit'] = 20
        self.config['replaybuffer_config']['capacity'] = 2
        self.config['trainer_config']['compile_models'] = True
        self.trainer_config = TrainerConfig(self.config)
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, 'logs')
            self.trainer.checkpointdir = os.path.join(self.trainer.workdir, 'checkpoints')
            origin_critic_params = deepcopy(self.trainer.target_critic.state_dict())
            self.trainer.collect_experience(0.9)
            self.trainer.learn(sample_size=32, batch_size=8, times=1)
            self.trainer.update_target_model_params()
            critic_params = self.trainer.target_critic.state_dict()

            for w1, w2 in zip(critic_params.values(), origin_critic_params.values()):
                if w1.requires_grad:
                    self.assertFalse(torch.equal(w1, w2))

if __name__ == '__main__':
    unittest.main()