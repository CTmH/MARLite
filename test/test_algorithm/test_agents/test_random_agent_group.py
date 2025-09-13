import unittest
import numpy as np
from marlite.algorithm.agents.random_agent_group import RandomAgentGroup
from gymnasium.spaces import Discrete

class TestRandomAgentGroup(unittest.TestCase):

    def setUp(self):
        self.agents = {
            "agent1": "modelA",
            "agent2": "modelB"
        }
        self.random_agent_group = RandomAgentGroup(self.agents)

    def test_act_random_actions(self):

        traj_padding_mask = np.array([])
        state = np.array([])

        avail_actions = {
            "agent1": Discrete(5),
            "agent2": Discrete(5)
        }
        obs = {agent: np.zeros(10) for agent in avail_actions.keys()}

        ret = self.random_agent_group.act(obs, state, avail_actions, traj_padding_mask, self.agents, 0.1)
        actions = ret["actions"]

        # Check if the correct number of actions are returned
        self.assertEqual(len(actions), len(avail_actions))
        for action in actions.values():
            # Check if the action is within the available actions
            self.assertIn(action, range(5))

if __name__ == '__main__':
    unittest.main()