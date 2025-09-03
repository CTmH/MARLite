import unittest
import yaml
from marlite.environment import EnvConfig
from pettingzoo import ParallelEnv

class TestEnvConfig(unittest.TestCase):
    def setUp(self):
        self.config_path = 'test/config/gnn_default.yaml'
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        env_config = self.config['env_config']
        self.env_config = EnvConfig(**env_config)

    def test_create_env(self):
        ret = self.env_config.create_env()
        self.assertTrue(isinstance(ret, ParallelEnv))
