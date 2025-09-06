import unittest
import os
import yaml
from copy import deepcopy
import tempfile
import torch

from marlite.trainer import TrainerConfig

class TestGNNObsCommQMIXTrainer(unittest.TestCase):
    def setUp(self):
        self.config_path = 'test/config/gnn_obs_comm_default.yaml'
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.config['trainer_config']['train_args']['epochs'] = 2
        self.config['rollout_config']['n_episodes'] = 2
        self.config['rollout_config']['episode_limit'] = 2
        self.config['replaybuffer_config']['capacity'] = 2
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
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, 'logs')
            self.trainer.checkpointdir = os.path.join(self.trainer.workdir, 'checkpoints')
            reward, _ = self.trainer.evaluate()
            best_reward, _ = self.trainer.train(epochs=2, target_reward=5)

    def test_data_parallel(self):
        self.config_path = 'test/config/gnn_obs_comm_default.yaml'
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.config['trainer_config']['train_args']['epochs'] = 2
        self.config['rollout_config']['n_episodes'] = 2
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
                self.assertFalse(torch.equal(w1, w2))

    def test_torch_compile(self):
        self.config_path = 'test/config/gnn_obs_comm_default.yaml'
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.config['trainer_config']['train_args']['epochs'] = 2
        self.config['rollout_config']['n_episodes'] = 2
        self.config['rollout_config']['episode_limit'] = 2
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
                self.assertFalse(torch.equal(w1, w2))

if __name__ == '__main__':
    unittest.main()