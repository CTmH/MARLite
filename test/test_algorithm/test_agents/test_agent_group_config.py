import sys    
print("In module products sys.path[0], __package__ ==", sys.path[0], __package__)
import unittest
import torch
import numpy as np
import yaml
from unittest.mock import MagicMock
from pettingzoo.mpe import simple_spread_v3

from src.algorithm.agents.agent_group import AgentGroup
from src.algorithm.agents.agent_group_config import AgentGroupConfig
from src.algorithm.model import ModelConfig
from src.util.optimizer_config import OptimizerConfig


class TestAgentGroupConfig(unittest.TestCase):

    def setUp(self):
        # Environment setup and model configuration
        self.env = simple_spread_v3.parallel_env(render_mode="human")

        # Agent group configuration
        config_path = 'config/qmix_default.yaml'
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        self.agent_group_config = AgentGroupConfig(**config['agent_group_config'])
        
    def test_get_agent_group(self):
        self.agent_group = self.agent_group_config.get_agent_group()
        self.assertIsInstance(self.agent_group, AgentGroup)

        
if __name__ == '__main__':
    unittest.main()
