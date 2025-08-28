import unittest
import yaml
import numpy as np
from src.algorithm.agents.magent_agent_group import MagentPreyAgentGroup
from src.environment.env_config import EnvConfig

class TestMagentPreyAgentGroup(unittest.TestCase):

    def setUp(self):
        yaml_conf = """
            env_config:
                module_name: "custom"
                env_name: "adversarial_pursuit_prey"
                tag_penalty: 0.0
                extra_features: false
                opponent_agent_group_config:
                    type: "Random"
                    agent_list:
                        predator_0: model1
                        predator_1: model1
                        predator_2: model1
                        predator_3: model1
                        predator_4: model1
                        predator_5: model1
                        predator_6: model1
                        predator_7: model1
                        predator_8: model1
                        predator_9: model1
                        predator_10: model1
                        predator_11: model1
                        predator_12: model1
                        predator_13: model1
                        predator_14: model1
                        predator_15: model1
                        predator_16: model1
                        predator_17: model1
                        predator_18: model1
                        predator_19: model1
                        predator_20: model1
                        predator_21: model1
                        predator_22: model1
                        predator_23: model1
                        predator_24: model1
                opp_obs_queue_len: 5
        """
        conf_dict = yaml.safe_load(yaml_conf)
        conf_dict = conf_dict['env_config']
        self.env_conf = EnvConfig(**conf_dict)
        env = self.env_conf.create_env()
        self.agents = {f'prey_{i}': 'policy' for i in range(50)}
        self.avail_actions = {}
        self.magent_prey_agent_group = MagentPreyAgentGroup(self.agents)

    def test_act(self):
        env = self.env_conf.create_env()
        obs, _ = env.reset()
        obs = {key: np.expand_dims(value, axis=0) for key, value in obs.items()} # Add Time Dim
        ret = self.magent_prey_agent_group.act(obs, self.avail_actions, 0)
        actions = ret['actions']
        env.step(actions)

    def test_get_q_values_returns_self(self):
        ret = self.magent_prey_agent_group.forward({})
        result = ret['q_val']
        self.assertIs(result, None)

if __name__ == '__main__':
    unittest.main()