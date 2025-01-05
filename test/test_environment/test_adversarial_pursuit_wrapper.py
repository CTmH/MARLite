import unittest
import numpy as np
from unittest.mock import MagicMock, patch
from magent2.environments import adversarial_pursuit_v4
from src.environment.adversarial_pursuit_wrapper import AdversarialPursuitPredator, AdversarialPursuitPrey
from src.algorithm.agents.random_agent_group import RandomAgentGroup

class TestAdversarialPursuitPredator(unittest.TestCase):
    
    def setUp(self):
        opponent_agents = {f'prey_{i}':'random' for i in range(50)}
        self.opponent_agents_with_model = {agent: 'model1' for agent in opponent_agents}
        self.opponent_agent_group_config = {
            "type": "Random",
            "agent_list": opponent_agents,
        }
        kwargs = {
            'opponent_agent_group_config': self.opponent_agent_group_config,
            'opp_obs_queue_len': 5
        }
        self.wrapper = AdversarialPursuitPredator(**kwargs)
        self.n_agent = 25
        self.n_oppo = 50
        
    def test_init(self):
        self.assertEqual(len(self.wrapper.agents), self.n_agent)
        self.assertEqual(self.wrapper.num_agents, self.n_agent)
        self.assertEqual(self.wrapper.max_num_agents, self.n_agent)
        self.assertEqual(len(self.wrapper.opponent_agents), self.n_oppo)

    def test_reset(self):
        observations, info = self.wrapper.reset()
        self.assertEqual(len(observations), self.n_agent)
        self.assertEqual(len(info), self.n_agent)

    def test_step(self):
        key = 'predator_0'
        actions = {agent: action_space.sample() for agent, action_space in self.wrapper.action_spaces.items()}
        observations, rewards, terminations, truncations, infos = self.wrapper.step(actions)
        self.assertEqual(len(observations), self.n_agent)
        self.assertEqual(len(rewards), self.n_agent)
        self.assertEqual(len(terminations), self.n_agent)
        self.assertEqual(len(truncations), self.n_agent)
        self.assertEqual(len(infos), self.n_agent)

    def test_render(self):
        self.wrapper.render()
        
    def test_close(self):
        self.wrapper.close()

    def test_state(self):
        state = self.wrapper.state()
        self.assertTrue(np.array_equal(state, self.wrapper.env.state()))

    def test_observation_space(self):
        agent = 'predator_0'
        obs_space = self.wrapper.observation_space(agent)
        self.assertEqual(obs_space, self.wrapper.env.observation_space(agent))

    def test_action_space(self):
        agent = 'predator_0'
        act_space = self.wrapper.action_space(agent)
        self.assertEqual(act_space, self.wrapper.env.action_space(agent))


class TestAdversarialPursuitPrey(unittest.TestCase):
    
    def setUp(self):
        opponent_agents = {f'predator_{i}':'random' for i in range(25)}
        self.opponent_agents_with_model = {agent: 'model1' for agent in opponent_agents}
        self.opponent_agent_group_config = {
            "type": "Random",
            "agent_list": opponent_agents,
        }
        kwargs = {
            'opponent_agent_group_config': self.opponent_agent_group_config,
            'opp_obs_queue_len': 5
        }
        self.wrapper = AdversarialPursuitPrey(**kwargs)
        self.n_agent = 50
        self.n_oppo = 25
        
    def test_init(self):
        self.assertEqual(len(self.wrapper.agents), self.n_agent)
        self.assertEqual(self.wrapper.num_agents, self.n_agent)
        self.assertEqual(self.wrapper.max_num_agents, self.n_agent)
        self.assertEqual(len(self.wrapper.opponent_agents), self.n_oppo)

    def test_reset(self):
        observations, info = self.wrapper.reset()
        self.assertEqual(len(observations), self.n_agent)
        self.assertEqual(len(info), self.n_agent)

    def test_step(self):
        key = 'prey_0'
        actions = {agent: action_space.sample() for agent, action_space in self.wrapper.action_spaces.items()}
        observations, rewards, terminations, truncations, infos = self.wrapper.step(actions)
        self.assertEqual(len(observations), self.n_agent)
        self.assertEqual(len(rewards), self.n_agent)
        self.assertEqual(len(terminations), self.n_agent)
        self.assertEqual(len(truncations), self.n_agent)
        self.assertEqual(len(infos), self.n_agent)

    def test_render(self):
        self.wrapper.render()
        
    def test_close(self):
        self.wrapper.close()

    def test_state(self):
        state = self.wrapper.state()
        self.assertTrue(np.array_equal(state, self.wrapper.env.state()))

    def test_observation_space(self):
        agent = 'prey_0'
        obs_space = self.wrapper.observation_space(agent)
        self.assertEqual(obs_space, self.wrapper.env.observation_space(agent))

    def test_action_space(self):
        agent = 'prey_0'
        act_space = self.wrapper.action_space(agent)
        self.assertEqual(act_space, self.wrapper.env.action_space(agent))

if __name__ == '__main__':
    unittest.main()