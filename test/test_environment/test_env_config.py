import unittest
import yaml
from marlite.environment import EnvConfig
from pettingzoo import ParallelEnv

class TestEnvConfig(unittest.TestCase):
    def setUp(self):
        config = yaml.safe_load("""
environment:
  module_name: "magent2.environments"
  env_name: "adversarial_pursuit_v4"
  env_params:
    tag_penalty: 0.0
    extra_features: true
  wrapper:
    type: adversarial_pursuit_predator
    opponent_agent_group:
      type: "MAgentPrey"
      agent_list:
        prey_0: random1
        prey_1: random1
        prey_2: random1
        prey_3: random1
        prey_4: random1
        prey_5: random1
        prey_6: random1
        prey_7: random1
        prey_8: random1
        prey_9: random1
        prey_10: random1
        prey_11: random1
        prey_12: random1
        prey_13: random1
        prey_14: random1
        prey_15: random1
        prey_16: random1
        prey_17: random1
        prey_18: random1
        prey_19: random1
        prey_20: random1
        prey_21: random1
        prey_22: random1
        prey_23: random1
        prey_24: random1
        prey_25: random1
        prey_26: random1
        prey_27: random1
        prey_28: random1
        prey_29: random1
        prey_30: random1
        prey_31: random1
        prey_32: random1
        prey_33: random1
        prey_34: random1
        prey_35: random1
        prey_36: random1
        prey_37: random1
        prey_38: random1
        prey_39: random1
        prey_40: random1
        prey_41: random1
        prey_42: random1
        prey_43: random1
        prey_44: random1
        prey_45: random1
        prey_46: random1
        prey_47: random1
        prey_48: random1
        prey_49: random1
    opp_obs_queue_len: 1
    channel_first: true
""")
        env_config = config['environment']
        self.env_config = EnvConfig(**env_config)

    def test_create_env(self):
        ret = self.env_config.create_env()
        self.assertTrue(isinstance(ret, ParallelEnv))
