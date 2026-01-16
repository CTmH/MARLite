import unittest
import numpy as np
from magent2.environments import adversarial_pursuit_v4
from marlite.environment.magent_wrapper import AdversarialPursuitPredator, AdversarialPursuitPrey

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
        self.wrapper = AdversarialPursuitPredator(env=adversarial_pursuit_v4.parallel_env(), **kwargs)
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
        actions = {agent: action_space.sample() for agent, action_space in self.wrapper.action_spaces.items()}
        self.wrapper.reset()
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

    def test_vector_state(self):
        """Test the vector state conversion functionality"""

        # Create a mock state with shape (L, L, 29)
        L = 10  # Map size
        state = np.zeros((L, L, 29), dtype=np.float16)

        # Define channel indices
        OBSTACLE_CHANNEL = 0
        TEAM_0_PRESENCE_CHANNEL = 1
        TEAM_0_HP_CHANNEL = 2
        TEAM_1_PRESENCE_CHANNEL = 3
        TEAM_1_HP_CHANNEL = 4
        BINARY_AGENT_ID_START = 5
        ONE_HOT_ACTION_START = 15
        LAST_REWARD_CHANNEL = 28

        # Add some obstacles
        state[0, 0, OBSTACLE_CHANNEL] = 1
        state[9, 9, OBSTACLE_CHANNEL] = 1

        # Add team 0 agents
        state[2, 2, TEAM_0_PRESENCE_CHANNEL] = 1
        state[2, 2, TEAM_0_HP_CHANNEL] = 100
        # Set binary agent ID for agent at (2,2) - agent ID 1
        state[2, 2, BINARY_AGENT_ID_START] = 0  # binary: 0000000000
        # Set one-hot action (action 0)
        state[2, 2, ONE_HOT_ACTION_START] = 1
        state[2, 2, LAST_REWARD_CHANNEL] = 1

        state[3, 3, TEAM_0_PRESENCE_CHANNEL] = 1
        state[3, 3, TEAM_0_HP_CHANNEL] = 80
        # Set binary agent ID for agent at (3,3) - agent ID 2
        state[3, 3, BINARY_AGENT_ID_START+1] = 1  # binary: 0000000010
        # Set one-hot action (action 1)
        state[3, 3, ONE_HOT_ACTION_START+1] = 1
        state[3, 3, LAST_REWARD_CHANNEL] = 0

        # Add team 1 agents
        state[7, 7, TEAM_1_PRESENCE_CHANNEL] = 1
        state[7, 7, TEAM_1_HP_CHANNEL] = 90
        # Set binary agent ID for agent at (7,7) - agent ID 3
        state[7, 7, BINARY_AGENT_ID_START] = 1
        state[7, 7, BINARY_AGENT_ID_START+1] = 1  # binary: 0000000011
        # Set one-hot action (action 2)
        state[7, 7, ONE_HOT_ACTION_START+2] = 1
        state[7, 7, LAST_REWARD_CHANNEL] = -1

        # Create a mock environment
        class MockEnv:
            def __init__(self):
                self.agents = [f'predator_{i}' for i in range(25)] + [f'prey{i}' for i in range(50)]
                self.possible_agents = self.agents[:]
                self.observation_spaces = {agent: (5, 5) for agent in self.possible_agents}
                self.action_spaces = {agent: (0, 5) for agent in self.possible_agents}
            def state(self):
                return state
            def observation_space(self, key):
                return self.observation_spaces[key]
            def action_space(self, key):
                return self.action_spaces[key]

        kwargs = {
            'opponent_agent_group_config': self.opponent_agent_group_config,
            'opp_obs_queue_len': 5,
            'vector_state': True
        }

        # Test the vector state conversion
        wrapper = AdversarialPursuitPredator(env=MockEnv(), **kwargs)

        # Get vector state
        vector_state = wrapper.state()

        # Verify feature dimensions
        expected_feature_dim = 1 + 1 + 1 + 2 + 13 + 12 * 3 # hp + team + last_reward + coords + action + nearby
        self.assertEqual(vector_state.shape[1], expected_feature_dim, f"Expected {expected_feature_dim} features, got {vector_state.shape[1]}")

        # Verify number of agents
        self.assertEqual(vector_state.shape[0], 75, f"Expected 75 agents, got {vector_state.shape[0]}")

        # Verify agent positions
        self.assertEqual(vector_state[0, 3], 2 / L)  # x position of first agent
        self.assertEqual(vector_state[0, 4], 2 / L)  # y position of first agent
        self.assertEqual(vector_state[2, 3], 3 / L)  # x position of second agent
        self.assertEqual(vector_state[2, 4], 3 / L)  # y position of second agent
        self.assertEqual(vector_state[3, 3], 7 / L)  # x position of third agent
        self.assertEqual(vector_state[3, 4], 7 / L)  # y position of third agent

        # Verify HP values
        self.assertEqual(vector_state[0, 0], 100)  # HP of first agent
        self.assertEqual(vector_state[2, 0], 80)   # HP of second agent
        self.assertEqual(vector_state[3, 0], 90)   # HP of third agent

        # Verify Team values
        self.assertEqual(vector_state[0, 1], 0)  # Team of first agent
        self.assertEqual(vector_state[2, 1], 0)   # Team of second agent
        self.assertEqual(vector_state[3, 1], 1)   # Team of third agent

        # Verify rewards
        self.assertEqual(vector_state[0, 2], 1)  # Reward for first agent
        self.assertEqual(vector_state[2, 2], 0)  # Reward for second agent
        self.assertEqual(vector_state[3, 2], -1) # Reward for third agent

        # Verify actions
        self.assertEqual(vector_state[0, 5], 1)  # Action 0 for first agent
        self.assertEqual(vector_state[2, 6], 1)  # Action 1 for second agent
        self.assertEqual(vector_state[3, 7], 1)  # Action 2 for third agent

    def test_vector_observation(self):
        """Test the vector observation conversion functionality"""

        # Create a mock observation with shape (H, W, C)
        H, W, C = 10, 10, 29  # Typical observation dimensions
        observation = np.zeros((H, W, C), dtype=np.float16)

        # Define channel indices for observation
        OBSERVATION_OBSTACLE_CHANNEL = 0
        OBSERVATION_TEAM_0_PRESENCE_CHANNEL = 1
        OBSERVATION_TEAM_1_PRESENCE_CHANNEL = 4

        # Add some obstacles
        observation[0, 0, OBSERVATION_OBSTACLE_CHANNEL] = 1
        observation[9, 9, OBSERVATION_OBSTACLE_CHANNEL] = 1
        observation[5, 5, OBSERVATION_OBSTACLE_CHANNEL] = 1

        # Add team 0 agents (predators)
        observation[2, 2, OBSERVATION_TEAM_0_PRESENCE_CHANNEL] = 1
        observation[2, 2, 2] = 100  # HP
        observation[2, 2, 3] = 1    # Team 0 presence

        observation[3, 3, OBSERVATION_TEAM_0_PRESENCE_CHANNEL] = 1
        observation[3, 3, 2] = 80   # HP
        observation[3, 3, 3] = 1    # Team 0 presence

        # Add team 1 agents (prey - opponents)
        observation[7, 7, OBSERVATION_TEAM_1_PRESENCE_CHANNEL] = 1
        observation[7, 7, 5] = 90   # HP
        observation[7, 7, 6] = 1    # Team 1 presence

        observation[8, 8, OBSERVATION_TEAM_1_PRESENCE_CHANNEL] = 1
        observation[8, 8, 5] = 70   # HP
        observation[8, 8, 6] = 1    # Team 1 presence

        # Create a mock environment that returns our observation
        class MockEnv:
            def __init__(self):
                self.agents = [f'predator_{i}' for i in range(25)] + [f'prey{i}' for i in range(50)]
                self.possible_agents = self.agents[:]
                self.observation_spaces = {agent: (5, 5) for agent in self.possible_agents}
                self.action_spaces = {agent: (0, 5) for agent in self.possible_agents}

            def observation_space(self, key):
                return self.observation_spaces[key]

            def action_space(self, key):
                return self.action_spaces[key]

            def reset(self, seed=None, options=None):
                # Return our mock observation for all agents
                obs_dict = {}
                for agent in self.agents:
                    obs_dict[agent] = observation.copy()
                return obs_dict, {}

            def step(self, actions):
                # Return our mock observation for all agents
                obs_dict = {}
                for agent in self.agents:
                    obs_dict[agent] = observation.copy()
                return obs_dict, {}, {}, {}, {}

            def state(self):
                return np.zeros((10, 10, 29), dtype=np.float16)

        # Create wrapper with vector_observation enabled
        kwargs = {
            'opponent_agent_group_config': self.opponent_agent_group_config,
            'opp_obs_queue_len': 5,
            'vector_observation': True,
            'max_vector_observation_records': 8
        }

        # Test the vector observation conversion
        wrapper = AdversarialPursuitPredator(env=MockEnv(), **kwargs)

        # Reset to get initial observations
        observations, _ = wrapper.reset()

        # Verify that observations are vectorized
        if observations:
            # Get first agent's observation
            first_agent_obs = list(observations.values())[0]

            # Check shape: should be (max_vector_observation_records, C)
            expected_shape = (8, C + 2) # Append relative position
            self.assertEqual(first_agent_obs.shape, expected_shape,
                           f"Expected shape {expected_shape}, got {first_agent_obs.shape}")

            # Verify the center is at (H//2, W//2) = (5, 5)
            center_y, center_x = H // 2, W // 2

            # Calculate Manhattan distances from center to each entity we placed
            # (0,0): distance = |0-5| + |0-5| = 10
            # (9,9): distance = |9-5| + |9-5| = 8
            # (5,5): distance = |5-5| + |5-5| = 0
            # (2,2): distance = |2-5| + |2-5| = 6
            # (3,3): distance = |3-5| + |3-5| = 4
            # (7,7): distance = |7-5| + |7-5| = 4
            # (8,8): distance = |8-5| + |8-5| = 6

            # The closest entities should be at (5,5), then (3,3) and (7,7) (both distance 4),
            # then (2,2), (8,8), (9,9), (0,0)

            # Since we have max_vector_observation_records=8, all should be included
            # Check that the first row contains the closest entity (5,5)
            self.assertTrue(np.array_equal(first_agent_obs[0][:-2], observation[5, 5]),
                          "First row should contain closest entity at (5,5)")

            # Check that the observation contains non-zero values (not all zeros)
            self.assertFalse(np.all(first_agent_obs == 0),
                           "Vectorized observation should not be all zeros")

            # Verify that the number of non-zero rows matches expected entities
            # We placed 7 entities total, so should have 7 non-zero rows
            non_zero_rows = np.sum(np.any(first_agent_obs != 0, axis=1))
            self.assertGreaterEqual(non_zero_rows, 7,
                                  f"Expected at least 7 non-zero rows, got {non_zero_rows}")
        else:
            self.fail("No observations returned from reset()")


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
        self.wrapper = AdversarialPursuitPrey(env=adversarial_pursuit_v4.parallel_env(), **kwargs)
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
        actions = {agent: action_space.sample() for agent, action_space in self.wrapper.action_spaces.items()}
        self.wrapper.reset()
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


class TestAdversarialPursuitPreyWitVectorState(unittest.TestCase):

    def setUp(self):
        opponent_agents = {f'predator_{i}':'random' for i in range(25)}
        self.opponent_agents_with_model = {agent: 'model1' for agent in opponent_agents}
        self.opponent_agent_group_config = {
            "type": "Random",
            "agent_list": opponent_agents,
        }
        kwargs = {
            'opponent_agent_group_config': self.opponent_agent_group_config,
            'opp_obs_queue_len': 5,
            'vector_state': True
        }
        self.wrapper = AdversarialPursuitPrey(env=adversarial_pursuit_v4.parallel_env(extra_features=True), **kwargs)
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
        actions = {agent: action_space.sample() for agent, action_space in self.wrapper.action_spaces.items()}
        self.wrapper.reset()
        observations, rewards, terminations, truncations, infos = self.wrapper.step(actions)
        self.assertEqual(len(observations), self.n_agent)
        self.assertEqual(len(rewards), self.n_agent)
        self.assertEqual(len(terminations), self.n_agent)
        self.assertEqual(len(truncations), self.n_agent)
        self.assertEqual(len(infos), self.n_agent)

    def test_close(self):
        self.wrapper.close()

    def test_state(self):
        state = self.wrapper.state()
        expected_feature_dim = 1 + 1 + 1 + 2 + 13 + 12 * 3 # hp + team + last_reward + coords + action + nearby
        self.assertTrue(state.shape == (75, expected_feature_dim))

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