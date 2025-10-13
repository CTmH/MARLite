import unittest
import yaml
import numpy as np
from marlite.algorithm.agents.magent_agent_group import MAgentPreyAgentGroup, MAgentBattleAgentGroup
from marlite.environment import EnvConfig

class TestMAgentPreyAgentGroup(unittest.TestCase):

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
        self.magent_prey_agent_group = MAgentPreyAgentGroup(self.agents)

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
                    map_size: 32
                    step_reward: -0.001
                    dead_penalty: -0.1
                    attack_penalty: -0.01
                    attack_opponent_reward: 0.5
                    extra_features: true
                wrapper_config:
                    type: battle
                    opp_obs_queue_len: 1
                    opponent_agent_group_config:
                        type: "MAgentBattle"
                        agent_list:
                            blue_0: policy
                            blue_1: policy
                            blue_2: policy
                            blue_3: policy
                            blue_4: policy
                            blue_5: policy
                            blue_6: policy
                            blue_7: policy
                            blue_8: policy
                            blue_9: policy
                            blue_10: policy
                            blue_11: policy
                            blue_12: policy
                            blue_13: policy
                            blue_14: policy
                            blue_15: policy
                            blue_16: policy
                            blue_17: policy
                            blue_18: policy
                            blue_19: policy
                            blue_20: policy
                            blue_21: policy
                            blue_22: policy
                            blue_23: policy
                            blue_24: policy
                            blue_25: policy
                            blue_26: policy
                            blue_27: policy
                            blue_28: policy
                            blue_29: policy
                            blue_30: policy
                            blue_31: policy
                            blue_32: policy
                            blue_33: policy
                            blue_34: policy
                            blue_35: policy
        """
        conf_dict = yaml.safe_load(yaml_conf)
        conf_dict = conf_dict['env_config']
        self.env_conf = EnvConfig(**conf_dict)
        env = self.env_conf.create_env()
        self.agents = {f'red_{i}': 'policy' for i in range(50)}
        self.avail_actions = {}
        self.magent_battle_agent_group = MAgentBattleAgentGroup(self.agents)

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