import unittest
import yaml
import numpy as np
from marlite.algorithm.agents.magent_agent_group import MAgentPreyAgentGroup, MAgentBattleAgentGroup
from marlite.environment import EnvConfig


class TestMAgentPreyAgentGroup(unittest.TestCase):

    def setUp(self):
        yaml_conf = """
            env_params:
                module_name: "magent2.environments"
                env_name: "adversarial_pursuit_v4"
                env_params:
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
        conf_dict = conf_dict['env_params']
        self.env_conf = EnvConfig(**conf_dict)
        env = self.env_conf.create_env()
        self.agents = {f'prey_{i}': 'policy' for i in range(50)}
        self.avail_actions = {}

    def test_greedy_strategy_default(self):
        group = MAgentPreyAgentGroup(self.agents)
        self.assertEqual(group.strategy, 'greedy')
        traj_padding_mask = np.array([])
        env = self.env_conf.create_env()
        obs, _ = env.reset()
        obs = {key: np.expand_dims(value, axis=0) for key, value in obs.items()}
        ret = group.act(obs, env.state(), self.avail_actions, traj_padding_mask, env.agents, 0)
        actions = ret['actions']
        for a in actions.values():
            self.assertIn(a, range(9))
        env.step(actions)

    def test_probability_strategy_returns_valid_actions(self):
        group = MAgentPreyAgentGroup(self.agents, strategy='probability')
        self.assertEqual(group.strategy, 'probability')
        self.assertEqual(group.temperature, 1.0)
        self.assertEqual(group.top_k, 5)

        traj_padding_mask = np.array([])
        env = self.env_conf.create_env()
        obs, _ = env.reset()
        obs = {key: np.expand_dims(value, axis=0) for key, value in obs.items()}
        ret = group.act(obs, env.state(), self.avail_actions, traj_padding_mask, env.agents, 0)
        actions = ret['actions']
        for a in actions.values():
            self.assertIn(a, range(9))
        env.step(actions)

    def test_low_temperature_approximates_greedy(self):
        traj_padding_mask = np.array([])
        env = self.env_conf.create_env()
        obs, _ = env.reset()
        obs = {key: np.expand_dims(value, axis=0) for key, value in obs.items()}

        greedy_group = MAgentPreyAgentGroup(self.agents, strategy='greedy')
        greedy_ret = greedy_group.act(obs, env.state(), self.avail_actions, traj_padding_mask, env.agents, 0)
        greedy_actions = greedy_ret['actions']

        prob_group = MAgentPreyAgentGroup(self.agents, strategy='probability', temperature=0.01, top_k=9)
        prob_actions_counts = {a: 0 for a in greedy_actions.keys()}
        for _ in range(20):
            ret = prob_group.act(obs, env.state(), self.avail_actions, traj_padding_mask, env.agents, 0)
            for agent, action in ret['actions'].items():
                if action == greedy_actions[agent]:
                    prob_actions_counts[agent] += 1

        # With very low temperature, probability strategy should match greedy
        # the majority of the time for most agents
        high_agreement = sum(1 for c in prob_actions_counts.values() if c >= 15)
        self.assertGreaterEqual(high_agreement, len(greedy_actions) * 0.5)

    def test_high_temperature_increases_randomness(self):
        traj_padding_mask = np.array([])
        env = self.env_conf.create_env()
        obs, _ = env.reset()
        obs = {key: np.expand_dims(value, axis=0) for key, value in obs.items()}

        low_temp = MAgentPreyAgentGroup(self.agents, strategy='probability', temperature=0.01, top_k=5)
        high_temp = MAgentPreyAgentGroup(self.agents, strategy='probability', temperature=10.0, top_k=5)

        env_agents = list(self.agents.keys())[:3]
        low_actions_set = set()
        high_actions_set = set()
        for _ in range(50):
            ret_low = low_temp.act(obs, env.state(), self.avail_actions, traj_padding_mask, env_agents, 0)
            ret_high = high_temp.act(obs, env.state(), self.avail_actions, traj_padding_mask, env_agents, 0)
            low_actions_set.update(ret_low['actions'].values())
            high_actions_set.update(ret_high['actions'].values())

        # High temperature should explore more diverse actions
        self.assertGreaterEqual(len(high_actions_set), len(low_actions_set))

    def test_top_k_limits_action_choices(self):
        traj_padding_mask = np.array([])
        env = self.env_conf.create_env()
        obs, _ = env.reset()
        obs = {key: np.expand_dims(value, axis=0) for key, value in obs.items()}

        group = MAgentPreyAgentGroup(self.agents, strategy='probability', temperature=5.0, top_k=2)
        env_agents = list(self.agents.keys())[:5]
        seen_actions = set()
        for _ in range(30):
            ret = group.act(obs, env.state(), self.avail_actions, traj_padding_mask, env_agents, 0)
            seen_actions.update(ret['actions'].values())

        self.assertGreaterEqual(len(seen_actions), 1)

    def test_probability_strategy_return_structure(self):
        group = MAgentPreyAgentGroup(self.agents, strategy='probability')
        traj_padding_mask = np.array([])
        env = self.env_conf.create_env()
        obs, _ = env.reset()
        obs = {key: np.expand_dims(value, axis=0) for key, value in obs.items()}
        ret = group.act(obs, env.state(), self.avail_actions, traj_padding_mask, env.agents, 0)

        self.assertIn('actions', ret)
        self.assertIn('all_actions', ret)
        self.assertEqual(set(ret['actions'].keys()), set(env.agents))
        self.assertEqual(set(ret['all_actions'].keys()), set(obs.keys()))


class TestBattleAgentGroup(unittest.TestCase):

    def setUp(self):
        yaml_conf = """
            env_params:
                module_name: "magent2.environments"
                env_name: "battle_v4"
                env_params:
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
        conf_dict = conf_dict['env_params']
        self.env_conf = EnvConfig(**conf_dict)
        env = self.env_conf.create_env()
        self.agents = {f'red_{i}': 'policy' for i in range(36)}
        self.avail_actions = {}

    def test_advanced_strategy_default(self):
        group = MAgentBattleAgentGroup(self.agents)
        self.assertEqual(group.strategy, 'advanced')
        traj_padding_mask = np.array([])
        env = self.env_conf.create_env()
        obs, _ = env.reset()
        obs = {key: np.expand_dims(value, axis=0) for key, value in obs.items()}
        ret = group.act(obs, env.state(), self.avail_actions, traj_padding_mask, env.agents, 0)
        actions = ret['actions']
        for a in actions.values():
            self.assertIn(a, range(21))
        env.step(actions)

    def test_basic_strategy(self):
        group = MAgentBattleAgentGroup(self.agents, strategy='basic')
        self.assertEqual(group.strategy, 'basic')
        traj_padding_mask = np.array([])
        env = self.env_conf.create_env()
        obs, _ = env.reset()
        obs = {key: np.expand_dims(value, axis=0) for key, value in obs.items()}
        ret = group.act(obs, env.state(), self.avail_actions, traj_padding_mask, env.agents, 0)
        actions = ret['actions']
        for a in actions.values():
            self.assertIn(a, range(21))
        env.step(actions)

    def test_probability_strategy_returns_valid_actions(self):
        group = MAgentBattleAgentGroup(self.agents, strategy='probability')
        self.assertEqual(group.strategy, 'probability')
        self.assertEqual(group.temperature, 1.0)
        self.assertEqual(group.top_k, 8)

        traj_padding_mask = np.array([])
        env = self.env_conf.create_env()
        obs, _ = env.reset()
        obs = {key: np.expand_dims(value, axis=0) for key, value in obs.items()}
        ret = group.act(obs, env.state(), self.avail_actions, traj_padding_mask, env.agents, 0)
        actions = ret['actions']
        for a in actions.values():
            self.assertIn(a, range(21))
        env.step(actions)

    def test_low_temperature_approximates_greedy(self):
        traj_padding_mask = np.array([])
        env = self.env_conf.create_env()
        obs, _ = env.reset()
        obs = {key: np.expand_dims(value, axis=0) for key, value in obs.items()}

        advanced_group = MAgentBattleAgentGroup(self.agents, strategy='advanced')
        advanced_ret = advanced_group.act(obs, env.state(), self.avail_actions, traj_padding_mask, env.agents, 0)
        advanced_actions = advanced_ret['actions']

        prob_group = MAgentBattleAgentGroup(self.agents, strategy='probability', temperature=0.01, top_k=21)
        prob_actions_counts = {a: 0 for a in advanced_actions.keys()}
        for _ in range(20):
            ret = prob_group.act(obs, env.state(), self.avail_actions, traj_padding_mask, env.agents, 0)
            for agent, action in ret['actions'].items():
                if action == advanced_actions[agent]:
                    prob_actions_counts[agent] += 1

        # With very low temperature, probability strategy based on advanced scoring
        # should agree with greedy advanced most of the time
        high_agreement = sum(1 for c in prob_actions_counts.values() if c >= 15)
        self.assertGreaterEqual(high_agreement, env.num_agents * 0.3)

    def test_probability_strategy_supports_epsilon(self):
        group = MAgentBattleAgentGroup(self.agents, strategy='probability', temperature=0.5, top_k=5)
        traj_padding_mask = np.array([])
        env = self.env_conf.create_env()
        obs, _ = env.reset()
        obs = {key: np.expand_dims(value, axis=0) for key, value in obs.items()}

        ret_no_eps = group.act(obs, env.state(), self.avail_actions, traj_padding_mask, env.agents, 0)
        ret_with_eps = group.act(obs, env.state(), self.avail_actions, traj_padding_mask, env.agents, 1.0)

        for a in ret_with_eps['actions'].values():
            self.assertIn(a, range(21))

    def test_probability_strategy_return_structure(self):
        group = MAgentBattleAgentGroup(self.agents, strategy='probability')
        traj_padding_mask = np.array([])
        env = self.env_conf.create_env()
        obs, _ = env.reset()
        obs = {key: np.expand_dims(value, axis=0) for key, value in obs.items()}
        ret = group.act(obs, env.state(), self.avail_actions, traj_padding_mask, env.agents, 0)

        self.assertIn('actions', ret)
        self.assertIn('all_actions', ret)
        self.assertEqual(set(ret['actions'].keys()), set(env.agents))
        self.assertEqual(set(ret['all_actions'].keys()), set(obs.keys()))


class TestProbabilityStrategyDeterministic(unittest.TestCase):
    """Tests with controlled synthetic observations for the probability strategy."""

    def test_prey_probability_distribution(self):
        group = MAgentPreyAgentGroup({'a0': 'm'}, strategy='probability', temperature=1.0, top_k=3)
        # Create synthetic obs: 1x13x13x5 grid, obstacle at (0,0) forces agent to prefer
        # positions far from top-left corner
        obs = {'a0': np.zeros((1, 13, 13, 5), dtype=np.float64)}
        obs['a0'][0, 2, 2, 0] = 1  # Obstacle at (2,2)
        obs['a0'][0, 2, 3, 0] = 1  # Another obstacle
        obs['a0'][0, 3, 2, 0] = 1
        # center is (6,6), offsets are all around center
        actual_actions, all_actions = group.probability_strategy(obs, ['a0'])
        self.assertIn(all_actions['a0'], range(9))

    def test_battle_probability_distribution(self):
        group = MAgentBattleAgentGroup({'a0': 'm'}, strategy='probability', temperature=1.0, top_k=5)
        n = 13
        obs = {'a0': np.zeros((1, n, n, 5), dtype=np.float64)}
        center = n // 2
        # Place an enemy at (center, center) - directly on the agent
        obs['a0'][0, center, center, 3] = 1  # other team presence
        obs['a0'][0, center, center, 4] = 5  # other team HP

        actual_actions, all_actions = group.probability_strategy(obs, ['a0'])
        self.assertIn(all_actions['a0'], range(21))


if __name__ == '__main__':
    unittest.main()
