import unittest
import os
from unittest.mock import MagicMock, patch, call, mock_open
from copy import deepcopy

import torch
from src.trainer.trainer_config import TrainerConfig

class TestTrainerConfig(unittest.TestCase):
    def setUp(self):
        self.config_path = 'config/qmix_default.yaml'
        self.trainer_config = TrainerConfig(self.config_path)

    def test_create_learner(self):
        ret = self.trainer_config.create_trainer()
        self.assertTrue(isinstance(ret, TrainerConfig))

    def test_run(self):
        self.trainer_config.create_trainer()
        reward, _ = self.trainer_config.trainer.evaluate()
        self.trainer_config.trainer.epochs = 5
        self.trainer_config.trainer.n_episodes = 20
        self.trainer_config.trainer.replay_buffer.capacity = 200
        best_reward, _ = self.trainer_config.run()
        self.assertGreaterEqual(best_reward, reward)

class TestTrainerConfigWithKAZConfig(unittest.TestCase):
    def setUp(self):
        self.config_path = 'config/qmix_kaz.yaml'
        self.trainer_config = TrainerConfig(self.config_path)

    def test_create_learner(self):
        ret = self.trainer_config.create_trainer()
        self.assertTrue(isinstance(ret, TrainerConfig))

    def test_run(self):
        self.trainer_config.create_trainer()
        reward, _ = self.trainer_config.trainer.evaluate()
        self.trainer_config.trainer.epochs = 5
        self.trainer_config.trainer.n_episodes = 20
        self.trainer_config.trainer.replay_buffer.capacity = 200
        best_reward, _ = self.trainer_config.run()
        self.assertGreaterEqual(best_reward, reward)