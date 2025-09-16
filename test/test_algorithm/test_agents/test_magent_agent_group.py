import unittest
import yaml
import numpy as np
from marlite.algorithm.agents.magent_agent_group import MagentPreyAgentGroup, MagentBattleAgentGroup
from marlite.environment import EnvConfig

class TestMagentPreyAgentGroup(unittest.TestCase):

    def setUp(self):
        yaml_conf = """
            env_config:
                module_name: "magent2.environments"
                env_name: "adversarial_pursuit_v4"
                env_config:
                    tag_penalty: -0.01
                    extra_features: true
                wrapper_config:
                    type: adversarial_pursuit_prey
                    opp_obs_queue_len: 5
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

        """
        conf_dict = yaml.safe_load(yaml_conf)
        conf_dict = conf_dict['env_config']
        self.env_conf = EnvConfig(**conf_dict)
        env = self.env_conf.create_env()
        self.agents = {f'prey_{i}': 'policy' for i in range(50)}
        self.avail_actions = {}
        self.magent_prey_agent_group = MagentPreyAgentGroup(self.agents)

    def test_act(self):
        traj_padding_mask = np.array([])
        env = self.env_conf.create_env()
        obs, _ = env.reset()
        obs = {key: np.expand_dims(value, axis=0) for key, value in obs.items()} # Add Time Dim
        ret = self.magent_prey_agent_group.act(obs, env.state(), self.avail_actions, traj_padding_mask, env.agents, 0)
        actions = ret['actions']
        env.step(actions)


class TestBattleAgentGroup(unittest.TestCase):

    def setUp(self):
        yaml_conf = """
            env_config:
                module_name: "magent2.environments"
                env_name: "battle_v4"
                env_config:
                    attack_penalty: -0.01
                    extra_features: true
                wrapper_config:
                    type: battle
                    opp_obs_queue_len: 5
                    opponent_agent_group_config:
                        type: "Random"
                        agent_list:
                            red_0: policy
                            red_1: policy
                            red_2: policy
                            red_3: policy
                            red_4: policy
                            red_5: policy
                            red_6: policy
                            red_7: policy
                            red_8: policy
                            red_9: policy
                            red_10: policy
                            red_11: policy
                            red_12: policy
                            red_13: policy
                            red_14: policy
                            red_15: policy
                            red_16: policy
                            red_17: policy
                            red_18: policy
                            red_19: policy
                            red_20: policy
                            red_21: policy
                            red_22: policy
                            red_23: policy
                            red_24: policy
                            red_25: policy
                            red_26: policy
                            red_27: policy
                            red_28: policy
                            red_29: policy
                            red_30: policy
                            red_31: policy
                            red_32: policy
                            red_33: policy
                            red_34: policy
                            red_35: policy
                            red_36: policy
                            red_37: policy
                            red_38: policy
                            red_39: policy
                            red_40: policy
                            red_41: policy
                            red_42: policy
                            red_43: policy
                            red_44: policy
                            red_45: policy
                            red_46: policy
                            red_47: policy
                            red_48: policy
                            red_49: policy
                            red_50: policy
                            red_51: policy
                            red_52: policy
                            red_53: policy
                            red_54: policy
                            red_55: policy
                            red_56: policy
                            red_57: policy
                            red_58: policy
                            red_59: policy
                            red_60: policy
                            red_61: policy
                            red_62: policy
                            red_63: policy
                            red_64: policy
                            red_65: policy
                            red_66: policy
                            red_67: policy
                            red_68: policy
                            red_69: policy
                            red_70: policy
                            red_71: policy
                            red_72: policy
                            red_73: policy
                            red_74: policy
                            red_75: policy
                            red_76: policy
                            red_77: policy
                            red_78: policy
                            red_79: policy
        """
        conf_dict = yaml.safe_load(yaml_conf)
        conf_dict = conf_dict['env_config']
        self.env_conf = EnvConfig(**conf_dict)
        env = self.env_conf.create_env()
        self.agents = {f'red_{i}': 'policy' for i in range(50)}
        self.avail_actions = {}
        self.magent_battle_agent_group = MagentBattleAgentGroup(self.agents)

    def test_act(self):
        traj_padding_mask = np.array([])
        env = self.env_conf.create_env()
        obs, _ = env.reset()
        obs = {key: np.expand_dims(value, axis=0) for key, value in obs.items()} # Add Time Dim
        ret = self.magent_battle_agent_group.act(obs, env.state(), self.avail_actions, traj_padding_mask, env.agents, 0)
        actions = ret['actions']
        env.step(actions)

if __name__ == '__main__':
    unittest.main()