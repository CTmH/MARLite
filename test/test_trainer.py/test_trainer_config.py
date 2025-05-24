import unittest
import yaml
import tempfile
import os
from src.trainer.trainer_config import TrainerConfig
from src.trainer.trainer import Trainer

class TestTrainerConfig(unittest.TestCase):
    def setUp(self):
        self.config_path = 'test/config/qmix_default.yaml'
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.config['trainer_config']['train_args']['epochs'] = 2
        self.config['rollout_config']['n_episodes'] = 2
        self.config['rollout_config']['episode_limit'] = 30
        self.config['replaybuffer_config']['capacity'] = 50
        self.trainer_config = TrainerConfig(self.config)

    def test_create_learner(self):
        ret = self.trainer_config.create_trainer()
        self.assertTrue(isinstance(ret, Trainer))

    def test_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, 'logs')
            self.trainer.checkpointdir = os.path.join(self.trainer.workdir, 'checkpoints')
            reward, _ = self.trainer_config.trainer.evaluate()
            self.trainer_config.trainer.epochs = 2
            self.trainer_config.trainer.n_episodes = 20
            self.trainer_config.trainer.replaybuffer.capacity = 200
            best_reward, _ = self.trainer_config.run()
            self.assertNotEqual(best_reward, reward)

class TestTrainerConfigWithKAZConfig(unittest.TestCase):
    def setUp(self):
        self.config_path = 'test/config/qmix_kaz.yaml'
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            self.config['trainer_config']['train_args']['epochs'] = 2
            self.config['rollout_config']['n_episodes'] = 20
            self.config['rollout_config']['episode_limit'] = 30
            self.config['replaybuffer_config']['capacity'] = 50
            self.trainer_config = TrainerConfig(self.config)

    def test_create_learner(self):
        ret = self.trainer_config.create_trainer()
        self.assertTrue(isinstance(ret, Trainer))

    def test_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, 'logs')
            self.trainer.checkpointdir = os.path.join(self.trainer.workdir, 'checkpoints')
            reward, _ = self.trainer_config.trainer.evaluate()
            self.trainer_config.trainer.epochs = 2
            self.trainer_config.trainer.n_episodes = 20
            self.trainer_config.trainer.replaybuffer.capacity = 20
            best_reward, _ = self.trainer_config.run()
            self.assertNotEqual(best_reward, reward)

class TestTrainerConfigWithMagentPredator(unittest.TestCase):
    def setUp(self):
        self.config_path = 'test/config/qmix_adversarial_pursuit_predator.yaml'
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.config['trainer_config']['train_args']['epochs'] = 2
        self.config['rollout_config']['n_episodes'] = 20
        self.config['rollout_config']['episode_limit'] = 30
        self.config['replaybuffer_config']['capacity'] = 50
        self.trainer_config = TrainerConfig(self.config)

    def test_create_learner(self):
        ret = self.trainer_config.create_trainer()
        self.assertTrue(isinstance(ret, Trainer))

    def test_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, 'logs')
            self.trainer.checkpointdir = os.path.join(self.trainer.workdir, 'checkpoints')
            reward, _ = self.trainer_config.trainer.evaluate()
            self.trainer_config.trainer.epochs = 5
            self.trainer_config.trainer.n_episodes = 20
            self.trainer_config.trainer.replaybuffer.capacity = 200
            best_reward, _ = self.trainer_config.run()
            self.assertNotEqual(best_reward, reward)

class TestTrainerConfigWithMagentPrey(unittest.TestCase):
    def setUp(self):
        self.config_path = 'test/config/qmix_adversarial_pursuit_prey.yaml'
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.config['trainer_config']['train_args']['epochs'] = 2
        self.config['rollout_config']['n_episodes'] = 20
        self.config['rollout_config']['episode_limit'] = 30
        self.config['replaybuffer_config']['capacity'] = 50
        self.trainer_config = TrainerConfig(self.config)

    def test_create_learner(self):
        ret = self.trainer_config.create_trainer()
        self.assertTrue(isinstance(ret, Trainer))

    def test_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, 'logs')
            self.trainer.checkpointdir = os.path.join(self.trainer.workdir, 'checkpoints')
            reward, _ = self.trainer_config.trainer.evaluate()
            self.trainer_config.trainer.epochs = 5
            self.trainer_config.trainer.n_episodes = 2
            self.trainer_config.trainer.replaybuffer.capacity = 200
            best_reward, _ = self.trainer_config.run()
            #self.assertNotEqual(best_reward, reward)

class TestTrainerConfigWithMagentBattlefield(unittest.TestCase):
    def setUp(self):
        self.config_path = 'test/config/qmix_adversarial_pursuit_prey.yaml'
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.config['trainer_config']['train_args']['epochs'] = 2
        self.config['rollout_config']['n_episodes'] = 20
        self.config['rollout_config']['episode_limit'] = 30
        self.config['replaybuffer_config']['capacity'] = 50
        self.trainer_config = TrainerConfig(self.config)

    def test_create_learner(self):
        ret = self.trainer_config.create_trainer()
        self.assertTrue(isinstance(ret, Trainer))

    def test_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, 'logs')
            self.trainer.checkpointdir = os.path.join(self.trainer.workdir, 'checkpoints')
            reward, _ = self.trainer_config.trainer.evaluate()
            self.trainer_config.trainer.epochs = 2
            self.trainer_config.trainer.n_episodes = 20
            self.trainer_config.trainer.replaybuffer.capacity = 200
            best_reward, _ = self.trainer_config.run()
            #self.assertNotEqual(best_reward, reward)