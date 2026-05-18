import unittest
import yaml
from mpe2 import simple_spread_v3

from marlite.algorithm.agents import AgentGroupConfig, AgentGroup
from marlite.algorithm.agents.group_consensus_agent_group import GroupConsensusAgentGroup

class TestAgentGroupConfig(unittest.TestCase):

    def setUp(self):
        # Environment setup and model configuration
        self.env = simple_spread_v3.parallel_env(render_mode="human")

    def test_get_agent_group(self):
        # Agent group configuration
        config_path = 'test/config/qmix_default.yaml'
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        self.agent_group_config = AgentGroupConfig(**config['agent_group'])
        self.agent_group = self.agent_group_config.get_agent_group()
        self.assertIsInstance(self.agent_group, AgentGroup)

    def test_get_gnn_agent_group(self):
        # GNN agent group configuration
        config_path = 'test/config/gnn_default.yaml'
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        self.agent_group_config = AgentGroupConfig(**config['agent_group'])
        self.agent_group = self.agent_group_config.get_agent_group()
        self.assertIsInstance(self.agent_group, AgentGroup)

    def test_get_group_consensus_agent_group(self):
        # GroupConsensus agent group configuration
        config_path = 'test/config/group_consensus_default.yaml'

        with open(config_path) as f:
            conf = yaml.safe_load(f)
        self.agent_group_config = AgentGroupConfig(**conf['agent_group'])
        self.agent_group = self.agent_group_config.get_agent_group()
        self.assertIsInstance(self.agent_group, GroupConsensusAgentGroup)


if __name__ == '__main__':
    unittest.main()
