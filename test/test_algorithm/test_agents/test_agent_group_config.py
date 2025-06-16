import sys
print("In module products sys.path[0], __package__ ==", sys.path[0], __package__)
import unittest
import torch
import numpy as np
import yaml
from unittest.mock import MagicMock
from mpe2 import simple_spread_v3

from src.algorithm.agents.agent_group import AgentGroup
from src.algorithm.agents.agent_group_config import AgentGroupConfig
from src.algorithm.model import ModelConfig
from src.util.optimizer_config import OptimizerConfig


class TestAgentGroupConfig(unittest.TestCase):

    def setUp(self):
        # Environment setup and model configuration
        self.env = simple_spread_v3.parallel_env(render_mode="human")

    def test_get_agent_group(self):
        # Agent group configuration
        config_path = 'test/config/qmix_default.yaml'
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        self.agent_group_config = AgentGroupConfig(**config['agent_group_config'])
        self.agent_group = self.agent_group_config.get_agent_group()
        self.assertIsInstance(self.agent_group, AgentGroup)

    def test_get_gnn_agent_group(self):
        # GNN agent group configuration
        config_path = 'test/config/gnn_default.yaml'
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        self.agent_group_config = AgentGroupConfig(**config['agent_group_config'])
        self.agent_group = self.agent_group_config.get_agent_group()
        self.assertIsInstance(self.agent_group, AgentGroup)


if __name__ == '__main__':
    unittest.main()
