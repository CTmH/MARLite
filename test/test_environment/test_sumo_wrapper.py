import os
import unittest
import numpy as np
import sumo_rl
from marlite.environment.sumo_wrapper import SUMOWrapper

class TestSUMOWrapper(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Initialize the environment
        self.env = sumo_rl.parallel_env(
            net_file='.venv/lib/python3.13/site-packages/sumo_rl/nets/RESCO/grid4x4/grid4x4.net.xml',
            route_file='.venv/lib/python3.13/site-packages/sumo_rl/nets/RESCO/grid4x4/grid4x4_1.rou.xml',
            use_gui=False,
            num_seconds=3600
        )

        # Create the wrapper
        self.wrapper = SUMOWrapper(self.env)
        self.n_agent = len(self.wrapper.possible_agents)

    def test_init(self):
        """Test initialization of SUMOWrapper."""
        self.assertEqual(len(self.wrapper.agents), len(self.wrapper.possible_agents))
        self.assertGreater(self.n_agent, 0)
        self.assertTrue(hasattr(self.wrapper, 'num_features'))
        self.assertIsInstance(self.wrapper.num_features, int)
        self.assertGreater(self.wrapper.num_features, 0)

    def test_reset(self):
        """Test reset method."""
        observations, infos = self.wrapper.reset()

        # Check basic properties
        self.assertEqual(len(observations), len(self.wrapper.agents))
        self.assertEqual(len(infos), len(self.wrapper.agents))

        # Check that all observations are numpy arrays
        for agent, obs in observations.items():
            self.assertIsInstance(obs, np.ndarray)

    def test_step(self):
        """Test step method."""
        # First reset
        state, infos = self.wrapper.reset()

        # Create actions for all agents
        actions = {agent: 0 for agent in self.wrapper.agents}

        # Take a step
        observations, rewards, terminations, truncations, infos = self.wrapper.step(actions)

        # Check returned values
        self.assertEqual(len(observations), len(self.wrapper.agents))
        self.assertEqual(len(rewards), len(self.wrapper.agents))
        self.assertEqual(len(terminations), len(self.wrapper.agents))
        self.assertEqual(len(truncations), len(self.wrapper.agents))
        self.assertEqual(len(infos), len(self.wrapper.agents))

        # Check that all are dictionaries with correct keys
        for agent in self.wrapper.agents:
            self.assertIn(agent, observations)
            self.assertIn(agent, rewards)
            self.assertIn(agent, terminations)
            self.assertIn(agent, truncations)
            self.assertIn(agent, infos)

    def test_state_shape(self):
        """Test that state() returns matrix with correct shape."""
        # Reset to initialize
        self.wrapper.reset()

        # Get state matrix
        state_matrix = self.wrapper.state()

        # Check type and shape
        self.assertIsInstance(state_matrix, np.ndarray)
        self.assertEqual(state_matrix.shape[0], self.n_agent)
        self.assertEqual(state_matrix.shape[1], self.wrapper.num_features)
        obs, _ = self.env.reset()
        self.assertEqual(state_matrix.shape[1], next(iter(obs.values())).shape[0])

        # Take a few steps and check shape remains consistent
        for _ in range(3):
            actions = {agent: 0 for agent in self.wrapper.agents}
            self.wrapper.step(actions)
            state_matrix = self.wrapper.state()
            self.assertEqual(state_matrix.shape, (self.n_agent, self.wrapper.num_features))

    def test_state_consistency(self):
        """Test that state matrix is properly updated after each step."""
        self.wrapper.reset()

        # Get initial state
        initial_state = self.wrapper.state()

        # Take a step
        actions = {agent: self.wrapper.action_space(agent).sample() for agent in self.wrapper.agents}
        observations, rewards, terminations, truncations, infos = self.wrapper.step(actions)

        # Get new state
        new_state = self.wrapper.state()

        # State should have changed (unless the environment is deterministic and actions don't change state)
        # We can't guarantee they'll be different, but we can check the structure is maintained
        self.assertEqual(new_state.shape, initial_state.shape)
        self.assertFalse(np.array_equal(new_state, initial_state))

        for row, obs in zip(new_state, observations.values()):
            self.assertTrue(np.array_equal(row, obs))

    def test_observation_space(self):
        """Test observation_space method."""
        if len(self.wrapper.agents) > 0:
            agent = self.wrapper.agents[0]
            obs_space = self.wrapper.observation_space(agent)
            self.assertEqual(obs_space, self.wrapper.env.observation_space(agent))

    def test_action_space(self):
        """Test action_space method."""
        if len(self.wrapper.agents) > 0:
            agent = self.wrapper.agents[0]
            act_space = self.wrapper.action_space(agent)
            self.assertEqual(act_space, self.wrapper.env.action_space(agent))

if __name__ == '__main__':
    unittest.main()