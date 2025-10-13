import unittest
import yaml
import tempfile
import torch
from marlite.trainer import TrainerConfig, Trainer

class TestTrainerConfig(unittest.TestCase):
    def setUp(self):
        self.config_path = 'test/config/qmix_kaz.yaml'
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.config['trainer_config']['train_args']['epochs'] = 3
        self.config['rollout_config']['n_episodes'] = 2
        self.config['rollout_config']['n_eval_episodes'] = 2
        self.config['rollout_config']['episode_limit'] = 3
        self.config['replaybuffer_config']['capacity'] = 5
        self.trainer_config = TrainerConfig(self.config)

    def test_create_learner(self):
        ret = self.trainer_config.create_trainer()
        self.assertTrue(isinstance(ret, Trainer))

    def test_run(self):
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        with tempfile.TemporaryDirectory() as temp_dir:
            self.config['trainer_config']['train_args']['epochs'] = 2
            self.config['rollout_config']['n_episodes'] = 2
            self.config['rollout_config']['n_eval_episodes'] = 2
            self.config['rollout_config']['episode_limit'] = 3
            self.config['replaybuffer_config']['capacity'] = 5
            self.config['trainer_config']['workdir'] = temp_dir
            self.trainer_config = TrainerConfig(self.config)
            best_reward, _ = self.trainer_config.run()

class TestTrainerConfigWithKAZConfig(unittest.TestCase):
    def setUp(self):
        self.config_path = 'test/config/qmix_kaz.yaml'
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            self.config['trainer_config']['train_args']['epochs'] = 2
            self.config['rollout_config']['n_episodes'] = 2
            self.config['rollout_config']['n_eval_episodes'] = 2
            self.config['rollout_config']['episode_limit'] = 3
            self.config['replaybuffer_config']['capacity'] = 5
            self.trainer_config = TrainerConfig(self.config)

    def test_create_learner(self):
        ret = self.trainer_config.create_trainer()
        self.assertTrue(isinstance(ret, Trainer))

    def test_run(self):
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        with tempfile.TemporaryDirectory() as temp_dir:
            self.config['trainer_config']['train_args']['epochs'] = 2
            self.config['rollout_config']['n_episodes'] = 2
            self.config['rollout_config']['n_eval_episodes'] = 2
            self.config['rollout_config']['episode_limit'] = 3
            self.config['replaybuffer_config']['capacity'] = 5
            self.config['trainer_config']['workdir'] = temp_dir
            self.trainer_config = TrainerConfig(self.config)
            best_reward, _ = self.trainer_config.run()

class TestTrainerConfigWithMAgentPredator(unittest.TestCase):
    def setUp(self):
        self.config_path = 'test/config/qmix_adversarial_pursuit_predator.yaml'
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.config['trainer_config']['train_args']['epochs'] = 2
        self.config['rollout_config']['n_episodes'] = 2
        self.config['rollout_config']['n_eval_episodes'] = 2
        self.config['rollout_config']['episode_limit'] = 3
        self.config['replaybuffer_config']['capacity'] = 5
        if torch.cuda.is_available():
            self.config['trainer_config']['train_device'] = 'cuda'
            self.config['rollout_config']['device'] = 'cpu'
        self.trainer_config = TrainerConfig(self.config)

    def test_create_learner(self):
        ret = self.trainer_config.create_trainer()
        self.assertTrue(isinstance(ret, Trainer))

    def test_run(self):
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        with tempfile.TemporaryDirectory() as temp_dir:
            self.config['trainer_config']['train_args']['epochs'] = 2
            self.config['rollout_config']['n_episodes'] = 2
            self.config['rollout_config']['n_eval_episodes'] = 2
            self.config['rollout_config']['episode_limit'] = 3
            self.config['replaybuffer_config']['capacity'] = 5
            self.config['trainer_config']['workdir'] = temp_dir
            self.trainer_config = TrainerConfig(self.config)
            best_reward, _ = self.trainer_config.run()

class TestTrainerConfigWithMAgentPrey(unittest.TestCase):
    def setUp(self):
        self.config_path = 'test/config/qmix_adversarial_pursuit_prey.yaml'
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.config['trainer_config']['train_args']['epochs'] = 2
        self.config['rollout_config']['n_episodes'] = 2
        self.config['rollout_config']['n_eval_episodes'] = 2
        self.config['rollout_config']['episode_limit'] = 3
        self.config['replaybuffer_config']['capacity'] = 5
        if torch.cuda.is_available():
            self.config['trainer_config']['train_device'] = 'cuda'
            self.config['rollout_config']['device'] = 'cpu'
        self.trainer_config = TrainerConfig(self.config)

    def test_create_learner(self):
        ret = self.trainer_config.create_trainer()
        self.assertTrue(isinstance(ret, Trainer))

    def test_run(self):
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        with tempfile.TemporaryDirectory() as temp_dir:
            self.config['trainer_config']['train_args']['epochs'] = 2
            self.config['rollout_config']['n_episodes'] = 2
            self.config['rollout_config']['n_eval_episodes'] = 2
            self.config['rollout_config']['episode_limit'] = 3
            self.config['replaybuffer_config']['capacity'] = 5
            self.config['trainer_config']['workdir'] = temp_dir
            self.trainer_config = TrainerConfig(self.config)
            best_reward, _ = self.trainer_config.run()

class TestTrainerConfigWithMAgentBattlefield(unittest.TestCase):
    def setUp(self):
        self.config_path = 'test/config/qmix_adversarial_pursuit_prey.yaml'
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.config['trainer_config']['train_args']['epochs'] = 2
        self.config['rollout_config']['n_episodes'] = 2
        self.config['rollout_config']['n_eval_episodes'] = 2
        self.config['rollout_config']['episode_limit'] = 3
        self.config['replaybuffer_config']['capacity'] = 5
        if torch.cuda.is_available():
            self.config['trainer_config']['train_device'] = 'cuda'
            self.config['rollout_config']['device'] = 'cpu'
        self.trainer_config = TrainerConfig(self.config)

    def test_create_learner(self):
        ret = self.trainer_config.create_trainer()
        self.assertTrue(isinstance(ret, Trainer))

    def test_run(self):
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        with tempfile.TemporaryDirectory() as temp_dir:
            self.config['trainer_config']['train_args']['epochs'] = 2
            self.config['rollout_config']['n_episodes'] = 2
            self.config['rollout_config']['n_eval_episodes'] = 2
            self.config['rollout_config']['episode_limit'] = 3
            self.config['replaybuffer_config']['capacity'] = 5
            self.config['trainer_config']['workdir'] = temp_dir
            self.trainer_config = TrainerConfig(self.config)
            best_reward, _ = self.trainer_config.run()