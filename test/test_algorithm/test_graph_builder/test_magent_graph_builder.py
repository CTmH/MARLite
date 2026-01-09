import numpy as np
import unittest
from magent2.environments import adversarial_pursuit_v4
from marlite.algorithm.graph_builder import GraphBuilderConfig
from marlite.algorithm.graph_builder.magent_graph_builder import MAgentVecStateGraphBuilder

class TestMAgentGraphBuilder(unittest.TestCase):

    def test_process_batch_normal_case(self):
        config = {
            "type": "MAgent",
            "binary_agent_id_dim": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            "agent_presence_dim": [1],
            "comm_distance": 3,
            "distance_metric": "cityblock",
            "valid_node_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        }
        bs = 5
        n_predators = len(config['valid_node_list'])
        env = adversarial_pursuit_v4.parallel_env(map_size=45, minimap_mode=False, tag_penalty=-0.2,
        max_cycles=500, extra_features=True, render_mode='rgb_array')
        obs = env.reset()
        state = env.state()
        states = np.stack([state for _ in range(bs)])
        # Create instance of GraphBuilderConfig
        builder_config = GraphBuilderConfig(**config)

        # Call get_graph_builder method
        graph_builder = builder_config.get_graph_builder()
        adj_matrix, edge_index = graph_builder(states)
        self.assertEqual(adj_matrix.shape, np.zeros((bs, n_predators, n_predators)).shape)
        self.assertEqual(len(edge_index), bs)


class TestMAgentVecStateGraphBuilder(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a simple test configuration
        self.coord_dims = (0, 1)  # First two dimensions are coordinates
        self.hp_dim = 2  # Third dimension is health points
        self.team_dim = 3  # Fourth dimension is team
        self.selected_teams = [1, 2]  # We want to include teams 1 and 2
        self.comm_distance = 5.0
        self.distance_metric = 'euclidean'

        # Create a sample state with 4 agents
        # Format: [x, y, hp, team]
        self.sample_state = np.array([
            [0.0, 0.0, 10.0, 1],  # Agent 0: team 1, hp 10
            [3.0, 4.0, 5.0, 1],   # Agent 1: team 1, hp 5
            [10.0, 10.0, 0.0, 2], # Agent 2: team 2, hp 0 (should be filtered out)
            [1.0, 1.0, 8.0, 2]    # Agent 3: team 2, hp 8
        ])

        # Create the graph builder
        self.graph_builder = MAgentVecStateGraphBuilder(
            coord_dims=self.coord_dims,
            hp_dim=self.hp_dim,
            team_dim=self.team_dim,
            selected_teams=self.selected_teams,
            comm_distance=self.comm_distance,
            distance_metric=self.distance_metric
        )

    def test_initialization(self):
        """Test that the graph builder initializes correctly."""
        self.assertEqual(self.graph_builder.coord_dims, (0, 1))
        self.assertEqual(self.graph_builder.hp_dim, 2)
        self.assertEqual(self.graph_builder.team_dim, 3)
        self.assertEqual(self.graph_builder.selected_teams, [1, 2])
        self.assertEqual(self.graph_builder.comm_distance, 5.0)
        self.assertEqual(self.graph_builder.distance_metric, 'euclidean')
        self.assertEqual(self.graph_builder.n_workers, 8)  # Default value

    def test_process_single_batch_no_candidates(self):
        """Test processing when no agents belong to selected teams."""
        # Create state with no agents in selected teams
        state_no_candidates = np.array([
            [0.0, 0.0, 10.0, 3],  # Agent 0: team 3 (not selected)
            [3.0, 4.0, 5.0, 4],   # Agent 1: team 4 (not selected)
        ])

        adj_matrix, edge_indices = self.graph_builder._process_single_batch(state_no_candidates)

        # Should return empty matrices
        self.assertEqual(adj_matrix.shape, (0, 0))
        self.assertEqual(edge_indices.shape, (2, 0))

    def test_process_single_batch_no_valid_agents(self):
        """Test processing when agents exist but none are valid (hp <= 0)."""
        # Create state with agents in selected teams but all with hp <= 0
        state_no_valid = np.array([
            [0.0, 0.0, 0.0, 1],   # Agent 0: team 1, hp 0 (invalid)
            [3.0, 4.0, 0.0, 2],   # Agent 1: team 2, hp 0 (invalid)
        ])

        adj_matrix, edge_indices = self.graph_builder._process_single_batch(state_no_valid)

        # Should return zero matrix with correct size and empty edge indices
        expected_size = 2  # 2 candidate agents
        self.assertEqual(adj_matrix.shape, (expected_size, expected_size))
        self.assertEqual(edge_indices.shape, (2, 0))

    def test_process_single_batch_single_agent(self):
        """Test processing when only one valid agent exists."""
        # Create state with only one valid agent
        state_single = np.array([
            [0.0, 0.0, 10.0, 1],  # Agent 0: team 1, hp 10 (valid)
            [3.0, 4.0, 0.0, 2],   # Agent 1: team 2, hp 0 (invalid)
        ])

        adj_matrix, edge_indices = self.graph_builder._process_single_batch(state_single)

        # Should return matrix sized for max agent ID (1) with no edges
        expected_size = 2  # Max ID is 1 (agent 1)
        self.assertEqual(adj_matrix.shape, (expected_size, expected_size))
        # Check that all values are 0 (no connections)
        self.assertTrue(np.all(adj_matrix == 0))
        # Edge indices should be empty
        self.assertEqual(edge_indices.shape, (2, 0))

    def test_process_single_batch_multiple_agents_no_connections(self):
        """Test processing when multiple agents exist but no connections within distance."""
        # Create state with agents far apart
        state_far_apart = np.array([
            [0.0, 0.0, 10.0, 1],  # Agent 0: team 1, hp 10
            [10.0, 10.0, 5.0, 2], # Agent 1: team 2, hp 5
        ])

        # Set a small communication distance
        builder_small_dist = MAgentVecStateGraphBuilder(
            coord_dims=self.coord_dims,
            hp_dim=self.hp_dim,
            team_dim=self.team_dim,
            selected_teams=self.selected_teams,
            comm_distance=1.0,  # Very small distance
            distance_metric=self.distance_metric
        )

        adj_matrix, edge_indices = builder_small_dist._process_single_batch(state_far_apart)

        # Should return matrix sized for max agent ID (1) with no edges
        expected_size = 2  # Max ID is 1 (agent 1)
        self.assertEqual(adj_matrix.shape, (expected_size, expected_size))
        # Check that all values are 0 (no connections due to distance)
        self.assertTrue(np.all(adj_matrix == 0))
        # Edge indices should be empty
        self.assertEqual(edge_indices.shape, (2, 0))

    def test_process_single_batch_multiple_agents_with_connections(self):
        """Test processing when multiple agents exist and some are connected."""
        # Create state with agents close enough to connect
        state_close = np.array([
            [0.0, 0.0, 10.0, 1],  # Agent 0: team 1, hp 10
            [3.0, 4.0, 5.0, 1],   # Agent 1: team 1, hp 5
            [10.0, 10.0, 0.0, 2], # Agent 2: team 2, hp 0 (should be filtered out)
            [1.0, 1.0, 8.0, 2]    # Agent 3: team 2, hp 8
        ])

        adj_matrix, edge_indices = self.graph_builder._process_single_batch(state_close)

        # Should return matrix sized for max agent ID (3)
        expected_size = 4  # Max ID is 3 (agent 3)
        self.assertEqual(adj_matrix.shape, (expected_size, expected_size))

        # Check that we have connections between agents 0 and 1 (close together)
        # Agent 0 and 1 should be connected
        self.assertEqual(adj_matrix[0, 1], 1)
        self.assertEqual(adj_matrix[1, 0], 1)

        # Check that agents 2 and 3 are not connected (agent 2 has hp=0)
        # But agent 3 should be in the matrix (as it's a candidate)
        self.assertEqual(adj_matrix[3, 3], 0)  # No self-loop

        # Edge indices should contain bidirectional connections
        # Agent 0 connects to agent 1, so we expect edges [0,1] and [1,0]
        self.assertGreaterEqual(edge_indices.shape[1], 2)  # At least one bidirectional edge

    def test_forward_method_single_batch(self):
        """Test the forward method with a single batch."""
        # Test with a batch containing one state
        batch_states = np.array([self.sample_state])

        adj_matrices, edge_indices_list = self.graph_builder.forward(batch_states)

        # Should return one adjacency matrix and one edge index list
        self.assertEqual(len(adj_matrices), 1)
        self.assertEqual(len(edge_indices_list), 1)

        # Check that adjacency matrix is properly shaped
        adj_matrix = adj_matrices[0]
        self.assertEqual(adj_matrix.shape, (4, 4))  # Max ID is 3, so 4x4 matrix

        # Check that edge indices list contains one array
        edge_indices = edge_indices_list[0]
        # Should have 2 rows (source and destination) and some columns (edges)
        self.assertEqual(edge_indices.shape[0], 2)

    def test_forward_method_multiple_batches(self):
        """Test the forward method with multiple batches."""
        # Test with a batch containing two states
        batch_states = np.array([
            self.sample_state,
            self.sample_state
        ])

        adj_matrices, edge_indices_list = self.graph_builder.forward(batch_states)

        # Should return two adjacency matrices and two edge index lists
        self.assertEqual(len(adj_matrices), 2)
        self.assertEqual(len(edge_indices_list), 2)

        # Both should have same shape
        for adj_matrix in adj_matrices:
            self.assertEqual(adj_matrix.shape, (4, 4))

    def test_edge_indices_bidirectional(self):
        """Test that edge indices are bidirectional."""
        # Create a simple case with two agents that should connect
        simple_state = np.array([
            [0.0, 0.0, 10.0, 1],  # Agent 0: team 1, hp 10
            [1.0, 1.0, 5.0, 1],   # Agent 1: team 1, hp 5
        ])

        adj_matrix, edge_indices = self.graph_builder._process_single_batch(simple_state)

        # With distance 5.0, agents 0 and 1 should be connected (distance ~1.41)
        # Check adjacency matrix
        self.assertEqual(adj_matrix[0, 1], 1)
        self.assertEqual(adj_matrix[1, 0], 1)

        # Check edge indices - should contain both directions
        # There should be two edges: 0->1 and 1->0
        self.assertEqual(edge_indices.shape[1], 2)  # Two edges
        # First row should contain source agents (0, 1)
        # Second row should contain destination agents (1, 0)
        self.assertIn(0, edge_indices[0])
        self.assertIn(1, edge_indices[0])
        self.assertIn(1, edge_indices[1])
        self.assertIn(0, edge_indices[1])

    def test_team_filtering(self):
        """Test that team filtering works correctly."""
        # Create state with agents from different teams
        team_test_state = np.array([
            [0.0, 0.0, 10.0, 1],  # Agent 0: team 1 (selected)
            [3.0, 4.0, 5.0, 2],   # Agent 1: team 2 (selected)
            [6.0, 8.0, 8.0, 3],   # Agent 2: team 3 (not selected)
        ])

        # Create builder that only selects teams 1 and 2
        team_builder = MAgentVecStateGraphBuilder(
            coord_dims=self.coord_dims,
            hp_dim=self.hp_dim,
            team_dim=self.team_dim,
            selected_teams=[1, 2],
            comm_distance=self.comm_distance,
            distance_metric=self.distance_metric
        )

        adj_matrix, edge_indices = team_builder._process_single_batch(team_test_state)

        # Should create matrix sized for 2 candidate agents
        self.assertEqual(adj_matrix.shape, (2, 2))

        # Agent 2 should be filtered out due to team 3
        # Agents 0 and 1 should be processed
        # Check that we have proper connections between agents 0 and 1 if they're close enough
        # Distance between (0,0) and (3,4) is 5.0, which equals our comm_distance
        # So they should be connected
        self.assertEqual(adj_matrix[0, 1], 1)  # Connection from 0 to 1
        self.assertEqual(adj_matrix[1, 0], 1)  # Connection from 1 to 0

    def test_hp_filtering(self):
        """Test that HP filtering works correctly."""
        # Create state with agents having different HP values
        hp_test_state = np.array([
            [0.0, 0.0, 10.0, 1],  # Agent 0: team 1, hp 10 (valid)
            [3.0, 4.0, 0.0, 1],   # Agent 1: team 1, hp 0 (invalid)
            [6.0, 8.0, 5.0, 1],   # Agent 2: team 1, hp 5 (valid)
        ])

        adj_matrix, edge_indices = self.graph_builder._process_single_batch(hp_test_state)

        # Should create matrix sized for max ID (2)
        self.assertEqual(adj_matrix.shape, (3, 3))

        # Agent 1 should be filtered out due to hp=0
        # Agents 0 and 2 should remain and be processed
        # Check that we have proper connections between agents 0 and 2 if they're close enough
        # Distance between (0,0) and (6,8) is ~10.0, which exceeds our comm_distance
        # So they should NOT be connected
        self.assertEqual(adj_matrix[0, 2], 0)  # No connection
        self.assertEqual(adj_matrix[2, 0], 0)  # No connection


if __name__ == '__main__':
    unittest.main()