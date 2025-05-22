import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from src.algorithm.agents.random_agent_group import RandomAgentGroup
from gymnasium.spaces import Discrete

class TestRandomAgentGroup(unittest.TestCase):

    def setUp(self):
        self.agents = {
            "agent1": "modelA",
            "agent2": "modelB"
        }
        self.random_agent_group = RandomAgentGroup(self.agents)

    def test_act_random_actions(self):

        avail_actions = {
            "agent1": Discrete(5),
            "agent2": Discrete(5)
        }
        obs = {agent: np.zeros(10) for agent in avail_actions.keys()}

        actions = self.random_agent_group.act(obs, avail_actions, 0.1)
        
        # Check if the correct number of actions are returned
        self.assertEqual(len(actions), len(avail_actions))
        for action in actions.values():
            # Check if the action is within the available actions
            self.assertIn(action, range(5))

    def test_get_q_values_returns_self(self):
        result = self.random_agent_group.forward(None)
        self.assertIs(result, self.random_agent_group)

    def test_set_model_params_returns_self(self):
        model_params = {"modelA": {}, "modelB": {}}
        feature_extractor_params = {}
        result = self.random_agent_group.set_agent_group_params(model_params, feature_extractor_params)
        self.assertIs(result, self.random_agent_group)

    def test_get_model_params_returns_self(self):
        result = self.random_agent_group.get_agent_group_params()
        self.assertIs(result, self.random_agent_group)

    def test_zero_grad_returns_self(self):
        result = self.random_agent_group.zero_grad()
        self.assertIs(result, self.random_agent_group)

    def test_step_returns_self(self):
        result = self.random_agent_group.step()
        self.assertIs(result, self.random_agent_group)

if __name__ == '__main__':
    unittest.main()