import unittest
import numpy as np
from src.environment.smac_wrapper import SMACWrapper
from smac_pettingzoo import smacv2_pettingzoo_v1


class TestSMACv2Wrapper(unittest.TestCase):

    def setUp(self):
        env = smacv2_pettingzoo_v1.parallel_env("10gen_protoss_20_vs_23")
        self.wrapper = SMACWrapper(env)
        self.n_agent = 20

    def test_init(self):
        self.assertEqual(len(self.wrapper.agents), self.n_agent)
        self.assertEqual(self.wrapper.num_agents, self.n_agent)
        self.assertEqual(self.wrapper.max_num_agents, self.n_agent)

    def test_reset(self):
        observations, info = self.wrapper.reset()
        self.assertEqual(len(observations), self.n_agent)
        self.assertEqual(len(info), self.n_agent)

    def test_step(self):
        state, infos = self.wrapper.reset()
        avail_actions = {agent: infos[agent]['action_mask'] for agent in self.wrapper.agents}
        #actions = {agent: action_space.sample() for agent, action_space in self.wrapper.action_spaces.items()}
        actions = {}
        for agent, mask in avail_actions.items():
            # Sample from available actions only
            available_actions = np.where(mask == 1)[0]
            if len(available_actions) > 0:
                actions[agent] = np.random.choice(available_actions)
            else:
                # If no actions are available, use a default value
                actions[agent] = 0
        observations, rewards, terminations, truncations, infos = self.wrapper.step(actions)
        self.assertEqual(len(observations), self.n_agent)
        self.assertEqual(len(rewards), self.n_agent)
        self.assertEqual(len(terminations), self.n_agent)
        self.assertEqual(len(truncations), self.n_agent)
        self.assertEqual(len(infos), self.n_agent)

    def test_close(self):
        self.wrapper.close()

    def test_state(self):
        self.wrapper.reset()
        state = self.wrapper.state()
        self.assertIsInstance(state, np.ndarray)

    def test_observation_space(self):
        agent = self.wrapper.agents[0]
        obs_space = self.wrapper.observation_space(agent)
        self.assertEqual(obs_space, self.wrapper._env.observation_space(agent))

    def test_action_space(self):
        agent = self.wrapper.agents[0]
        act_space = self.wrapper.action_space(agent)
        self.assertEqual(act_space, self.wrapper._env.action_space(agent))

if __name__ == '__main__':
    unittest.main()