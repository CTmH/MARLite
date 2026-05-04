import os
import unittest
import numpy as np
import yaml
import torch
from marlite.algorithm.agents import AgentGroupConfig
from marlite.algorithm.agents.group_consensus_agent_group import GroupConsensusAgentGroup


class TestGroupConsensusAgentGroup(unittest.TestCase):
    def setUp(self):
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "config", "group_consensus_default.yaml"
        )
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        self.agent_group_config = AgentGroupConfig(**config["agent_group_config"])
        self.agent_group = self.agent_group_config.get_agent_group()

    def test_agent_group_type(self):
        self.assertIsInstance(self.agent_group, GroupConsensusAgentGroup)


class TestGroupConsensusAgentGroupConfig(unittest.TestCase):
    def test_group_consensus_registration(self):
        from marlite.algorithm.agents.agent_group_config import registered_agent_groups
        self.assertIn("GroupConsensus", registered_agent_groups)


if __name__ == "__main__":
    unittest.main()
