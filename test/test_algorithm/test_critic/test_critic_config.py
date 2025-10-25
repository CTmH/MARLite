import unittest
import yaml
from marlite.algorithm.critic import CriticConfig
from marlite.algorithm.critic.mixer import QMixer, SeqQMixer

class TestCriticConfig(unittest.TestCase):

    def test_get_critic(self):
        config_path = 'test/config/qmix_default.yaml'
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        critic_config_dict = config['critic_config']
        critic_config_dict.pop('optimizer')
        if 'lr_scheduler' in critic_config_dict:
            critic_config_dict.pop('lr_scheduler')
        self.critic_config = CriticConfig(**critic_config_dict)
        self.critic = self.critic_config.get_critic()
        self.assertIsInstance(self.critic, QMixer)

    def test_get_seq_critic(self):
        config_path = 'test/config/seq_msg_aggr_smac.yaml'
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        critic_config_dict = config['critic_config']
        critic_config_dict.pop('optimizer')
        if 'lr_scheduler' in critic_config_dict:
            critic_config_dict.pop('lr_scheduler')
        self.critic_config = CriticConfig(**critic_config_dict)
        self.critic = self.critic_config.get_critic()
        self.assertIsInstance(self.critic, SeqQMixer)


if __name__ == '__main__':
    unittest.main()