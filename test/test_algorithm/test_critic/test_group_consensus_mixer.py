import os
import unittest
import yaml
from marlite.algorithm.critic import CriticConfig
from marlite.algorithm.critic.group_consensus_mixer import GroupConsensusMixer


class TestGroupConsensusMixer(unittest.TestCase):
    def setUp(self):
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "config", "group_consensus_default.yaml"
        )
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        self.critic_config = CriticConfig(**config["critic_config"])
        self.critic = self.critic_config.get_critic()

    def test_critic_type(self):
        self.assertIsInstance(self.critic, GroupConsensusMixer)


class TestGroupConsensusMixerConfig(unittest.TestCase):
    def test_group_consensus_mixer_registration(self):
        from marlite.algorithm.critic.critic_config import registered_critic_creators
        self.assertIn("GroupConsensusMixer", registered_critic_creators)


if __name__ == "__main__":
    unittest.main()
