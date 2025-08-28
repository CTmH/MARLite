import unittest
from src.environment.env_config import EnvConfig
from pettingzoo import ParallelEnv

class TestEnvConfig(unittest.TestCase):
    def setUp(self):
        env_config = {"module_name": "mpe2", "env_name": "simple_speaker_listener_v4"}
        self.env_config = EnvConfig(**env_config)

    def test_create_env(self):
        ret = self.env_config.create_env()
        self.assertTrue(isinstance(ret, ParallelEnv))
