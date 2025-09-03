import unittest
import yaml
from marlite.replaybuffer import ReplayBufferConfig
from marlite.replaybuffer.normal_replaybuffer import NormalReplayBuffer
from marlite.replaybuffer.prioritized_replaybuffer import PrioritizedReplayBuffer

class TestReplayBufferConfig(unittest.TestCase):
    def test_create_normal_replaybuffer(self):
        config = """
        type: "Normal"
        capacity: 50000
        traj_len: 5
        """
        config = yaml.safe_load(config)
        replaybuffer_config = ReplayBufferConfig(**config)
        replaybuffer = replaybuffer_config.create_replaybuffer()
        self.assertIsInstance(replaybuffer, NormalReplayBuffer)

    def test_create_prioritized_replaybuffer(self):
        config = """
        type: "Prioritized"
        capacity: 50000
        traj_len: 5
        priority_attr: all_agents_sum_rewards
        alpha: 0.6
        """
        config = yaml.safe_load(config)
        replaybuffer_config = ReplayBufferConfig(**config)
        replaybuffer = replaybuffer_config.create_replaybuffer()
        self.assertIsInstance(replaybuffer, PrioritizedReplayBuffer)

    def test_invalid_replaybuffer_type(self):
        config = """
        type: "Unknown"
        capacity: 50000
        traj_len: 5
        priority_attr: all_agents_sum_rewards
        alpha: 0.6
        """
        config = yaml.safe_load(config)
        # 测试传入无效的回放缓冲区类型
        with self.assertRaises(ValueError):
            replaybuffer_config = ReplayBufferConfig(**config)
            replaybuffer = replaybuffer_config.create_replaybuffer()

if __name__ == "__main__":
    unittest.main()