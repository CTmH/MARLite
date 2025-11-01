import numpy as np
import unittest
from marlite.algorithm.graph_builder.partial_graph_builder import PartialGraphMAgentBuilder, PartialGraphVectorStateBuilder
from marlite.algorithm.graph_builder.graph_util import extract_agent_positions_batch, build_partial_graph

class TestPartialGraphMAgentBuilder(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.n_agent = 25
        self.binary_agent_id_dim = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        self.agent_presence_dim = [1, 3]
        self.comm_distance = 10
        self.distance_metric = 'cityblock'
        self.valid_node_list = list(range(12))
        self.target_node_list = list(range(12, self.n_agent))  # New target node list

        # Create builder instance for testing
        self.builder = PartialGraphMAgentBuilder(
            binary_agent_id_dim=self.binary_agent_id_dim,
            agent_presence_dim=self.agent_presence_dim,
            comm_distance=self.comm_distance,
            valid_node_list=self.valid_node_list,
            target_node_list=self.target_node_list,
            distance_metric=self.distance_metric,
            n_workers=1,  # Use single worker for testing
            n_subgraphs=5,
            update_interval=5,
            channel_first=False
        )

        # Reset builder to initial state
        self.builder.reset()

    def _create_binary_id_little_endian(self, agent_id, num_bits=10):
        """Create binary ID using little-endian format (LSB first)"""
        binary_str = format(agent_id, f'0{num_bits}b')  # Get binary representation
        # Reverse the string to get little-endian (LSB first)
        return [int(x) for x in reversed(binary_str)]

    def test_process_batch_normal_case(self):
        """Test forward with normal case (agents present)."""
        # Create a sample batch state with agents present
        batch_size = 5
        height, width = 45, 45
        channels = 15  # Based on the dimensions used in the config

        # Initialize state with zeros
        states = np.zeros((batch_size, height, width, channels), dtype=np.int8)

        # Add some agents (set presence to 1 and valid binary IDs)
        for b in range(batch_size):
            # Place agents at random positions with IDs
            agent_positions = np.random.choice(height * width, size=self.n_agent, replace=False)
            for agent_id in range(self.n_agent):
                pos_idx = agent_positions[agent_id]
                i, j = divmod(pos_idx, width)  # Convert flat index to 2D coordinates

                # Set agent presence
                states[b, i, j, self.agent_presence_dim] = 1

                # Set binary agent ID using little-endian format
                binary_id = self._create_binary_id_little_endian(agent_id, len(self.binary_agent_id_dim))
                for k, bit in enumerate(binary_id):
                    states[b, i, j, self.binary_agent_id_dim[k]] = bit

        # Process the states
        adj_matrix, edge_index = self.builder(states)

        # Check output shapes
        expected_adj_shape = (batch_size, len(self.valid_node_list), len(self.valid_node_list))
        self.assertEqual(adj_matrix.shape, expected_adj_shape)
        self.assertEqual(len(edge_index), batch_size)

        # Check that we have some edges (not completely empty)
        total_edges = sum(ei.shape[1] for ei in edge_index)
        self.assertGreater(total_edges, 0, "Should have edges when agents are present")

    def test_process_batch_empty_case(self):
        """Test forward with empty case (no agents present)."""
        # Create a sample batch state with no agents (all zeros)
        batch_size = 3
        height, width = 45, 45
        channels = 15

        # All zeros means no agents present
        states = np.zeros((batch_size, height, width, channels), dtype=np.int8)

        # Process the states
        adj_matrix, edge_index = self.builder(states)

        # Check output shapes
        # For empty graphs, we should have:
        # - Adjacency matrix: (batch_size, 0, 0) since no valid nodes
        # - Edge index: list of arrays with shape (2, 0)
        max_node_id = max(self.valid_node_list)
        expected_adj_shape = (max_node_id + 1, max_node_id + 1)
        self.assertEqual(len(adj_matrix), batch_size)
        for adj in adj_matrix:
            self.assertEqual(adj.shape, expected_adj_shape)

        self.assertEqual(len(edge_index), batch_size)
        for ei in edge_index:
            self.assertEqual(ei.shape, (2, 0), "Empty graph should have empty edge index")

    def test_caching_mechanism(self):
        """Test the caching mechanism with update_interval."""

        self.builder.reset().eval()
        # Create a sample batch state with agents present
        batch_size = 5
        height, width = 45, 45
        channels = 15  # Based on the dimensions used in the config

        # Initialize state with zeros
        states = np.zeros((batch_size, height, width, channels), dtype=np.int8)

        # Add some agents (set presence to 1 and valid binary IDs)
        for b in range(batch_size):
            # Place agents at random positions with IDs
            agent_positions = np.random.choice(height * width, size=self.n_agent, replace=False)
            for agent_id in range(self.n_agent):
                pos_idx = agent_positions[agent_id]
                i, j = divmod(pos_idx, width)  # Convert flat index to 2D coordinates

                # Set agent presence
                states[b, i, j, self.agent_presence_dim] = 1

                # Set binary agent ID using little-endian format
                binary_id = self._create_binary_id_little_endian(agent_id, len(self.binary_agent_id_dim))
                for k, bit in enumerate(binary_id):
                    states[b, i, j, self.binary_agent_id_dim[k]] = bit

        # First call - should compute new graph
        adj_matrix_1, edge_index_1 = self.builder(states)

        # Subsequent calls within update interval - should return cached results
        for i in range(self.builder.update_interval - 1):
            adj_matrix_2, edge_index_2 = self.builder(np.zeros_like(states))

            # Should be equal to first result (cached)
            np.testing.assert_array_equal(adj_matrix_1, adj_matrix_2)
            for j in range(len(edge_index_1)):
                np.testing.assert_array_equal(edge_index_1[j], edge_index_2[j])

        # After update interval, should compute new graph
        adj_matrix_3, edge_index_3 = self.builder(np.zeros_like(states))

        # This should be different from cached result (empty vs previous)
        # Empty graph has shape (0,0) while previous had some nodes
        if adj_matrix_1.size > 0:  # If first graph wasn't empty
            self.assertFalse(np.array_equal(adj_matrix_3, adj_matrix_1))

    def test_n_workers_behavior(self):
        """Test that n_workers=1 uses sequential processing."""
        # Test with n_workers=1 (should use sequential processing)
        builder_seq = PartialGraphMAgentBuilder(
            binary_agent_id_dim=self.binary_agent_id_dim,
            agent_presence_dim=self.agent_presence_dim,
            comm_distance=self.comm_distance,
            distance_metric=self.distance_metric,
            n_workers=1,
            n_subgraphs=5,
            valid_node_list=self.valid_node_list,
            target_node_list=self.target_node_list,
            update_interval=5
        )

        # Create test states
        batch_size = 2
        height, width = 45, 45
        channels = 15
        states = np.zeros((batch_size, height, width, channels), dtype=np.float32)
        # Add agents
        for b in range(batch_size):
            states[b, b*10, b*10, self.agent_presence_dim] = 1
            # Set binary ID using little-endian
            binary_id = self._create_binary_id_little_endian(1, len(self.binary_agent_id_dim))
            states[b, b*10, b*10, self.binary_agent_id_dim[0]] = binary_id[0]

        # Process states
        adj_matrix, edge_index = builder_seq.forward(states)

        # Check results
        self.assertEqual(len(adj_matrix), batch_size)
        self.assertEqual(len(edge_index), batch_size)

    def test_extract_agent_positions_batch(self):
        """Test the extract_agent_positions_batch utility function."""
        # Create test state
        batch_size = 2
        height, width = 45, 45
        channels = 15
        states = np.zeros((batch_size, height, width, channels), dtype=np.float32)

        # Add agents with specific IDs
        # Format: [agent_id, coord_y, coord_x]
        test_positions = [
            [[1, 10, 10], [3, 20, 20]],  # Batch 0: agents with IDs 1 and 3
            [[5, 15, 15], [1, 25, 25]]   # Batch 1: agents with IDs 5 and 1
        ]

        for b, positions in enumerate(test_positions):
            for agent_id, i, j in positions:
                states[b, i, j, self.agent_presence_dim] = 1
                binary_id = self._create_binary_id_little_endian(agent_id, len(self.binary_agent_id_dim))
                for k, bit in enumerate(binary_id):
                    states[b, i, j, self.binary_agent_id_dim[k]] = bit

        # Extract positions
        coords_with_id = extract_agent_positions_batch(
            states, self.binary_agent_id_dim, self.agent_presence_dim
        )

        # Check results
        self.assertEqual(len(coords_with_id), batch_size)

        # Check first batch
        expected_coords_0 = np.array([[1, 10, 10], [3, 20, 20]])  # Output format still: [y, x, id]
        np.testing.assert_array_equal(coords_with_id[0], expected_coords_0)

        # Check second batch
        expected_coords_1 = np.array([[5, 15, 15], [1, 25, 25]])
        np.testing.assert_array_equal(coords_with_id[1], expected_coords_1)

    def test_build_partial_graph_direct_with_communities(self):
        """Test build_partial_graph with explicit spatial clustering into two communities."""

        # Define parameters
        comm_distance = 100  # Communication threshold
        distance_metric = 'cityblock'

        # Community 1: tightly clustered around (10, 10)
        community1_coords = np.array([
            [0, 1, 1],
            [1, 2, 1],
            [2, 2, 2]
        ])

        # Community 2: tightly clustered around (30, 30)
        community2_coords = np.array([
            [3, 30, 30],
            [4, 30, 31],
            [5, 31, 31]
        ])

        # Combine all coordinates
        coords_with_id = np.vstack([community1_coords, community2_coords])  # Shape: (6, 3)

        # All agents are valid and target nodes (for simplicity)
        valid_node_list = [0, 1, 3, 4]
        target_node_list = [2, 5]

        n_subgraphs = 2  # Expect two communities

        # Build partial graph with community detection
        adj_matrix, edge_index = build_partial_graph(
            coords_with_id=coords_with_id,
            comm_distance=comm_distance,
            distance_metric=distance_metric,
            n_subgraphs=n_subgraphs,
            valid_node_list=valid_node_list,
            target_node_list=target_node_list
        )

        expected_size = max(valid_node_list) + 1
        self.assertEqual(adj_matrix.shape, (expected_size, expected_size))
        self.assertEqual(edge_index.shape[0], 2)  # Should be (2, E)
        self.assertEqual(edge_index.shape[1], 2)

    def test_little_endian_binary_encoding(self):
        """Test that binary encoding follows little-endian format."""
        # Test agent ID 1: should be 1 in binary, which is 1 in little-endian
        binary_id_1 = self._create_binary_id_little_endian(1, 10)
        # In little-endian, the least significant bit (1) should be first
        self.assertEqual(binary_id_1[0], 1)  # LSB should be at index 0
        self.assertTrue(all(bit == 0 for bit in binary_id_1[1:]))  # Rest should be 0

        # Test agent ID 2: should be 10 in binary, which becomes 01 in little-endian
        binary_id_2 = self._create_binary_id_little_endian(2, 10)
        self.assertEqual(binary_id_2[1], 1)  # Second bit should be 1 (2^1)
        self.assertEqual(binary_id_2[0], 0)  # First bit should be 0
        self.assertTrue(all(bit == 0 for bit in binary_id_2[2:]))  # Rest should be 0

        # Test agent ID 3: should be 11 in binary, which stays 11 in little-endian
        binary_id_3 = self._create_binary_id_little_endian(3, 10)
        self.assertEqual(binary_id_3[0], 1)  # 2^0 = 1
        self.assertEqual(binary_id_3[1], 1)  # 2^1 = 2
        self.assertTrue(all(bit == 0 for bit in binary_id_3[2:]))  # Rest should be 0

        # Test agent ID 5: should be 101 in binary, which becomes 101 in little-endian
        binary_id_5 = self._create_binary_id_little_endian(5, 10)
        self.assertEqual(binary_id_5[0], 1)  # 2^0 = 1
        self.assertEqual(binary_id_5[1], 0)  # 2^1 = 2
        self.assertEqual(binary_id_5[2], 1)  # 2^2 = 4
        self.assertTrue(all(bit == 0 for bit in binary_id_5[3:]))  # Rest should be 0


class TestPartialGraphVectorStateBuilder(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.coord_dim = [5, 6]  # Coordinate dimensions in feature space
        self.hp_dim = 1  # Health point dimension
        self.comm_distance = 10
        self.distance_metric = 'cityblock'
        self.valid_node_list = list(range(12))
        self.target_node_list = list(range(12, 25))  # New target node list

        # Create builder instance for testing
        self.builder = PartialGraphVectorStateBuilder(
            coord_dim=self.coord_dim,
            hp_dim=self.hp_dim,
            comm_distance=self.comm_distance,
            valid_node_list=self.valid_node_list,
            target_node_list=self.target_node_list,
            distance_metric=self.distance_metric,
            n_workers=1,  # Use single worker for testing
            n_subgraphs=5,
            update_interval=5
        )

        # Reset builder to initial state
        self.builder.reset()

    def test_forward_normal_case(self):
        """Test forward with normal case (agents present)."""
        # Create a sample batch state with agents present
        batch_size = 5
        num_agents = 25
        feature_dim = 15  # Based on the dimensions used in the config

        # Initialize state with zeros
        states = np.zeros((batch_size, num_agents, feature_dim), dtype=np.float32)

        # Add some agents (set presence to 1 and valid coordinates)
        for b in range(batch_size):
            # Place agents at random positions with IDs
            agent_positions = np.random.choice(num_agents, size=num_agents, replace=False)
            for agent_id in range(num_agents):
                pos_idx = agent_positions[agent_id]

                # Set agent presence
                states[b, pos_idx, self.hp_dim] = 1

                # Set coordinates using little-endian format
                coords = np.random.randint(0, 45, size=2)
                states[b, pos_idx, self.coord_dim[0]] = coords[0]  # y coordinate
                states[b, pos_idx, self.coord_dim[1]] = coords[1]  # x coordinate

        # Process the states
        adj_matrix, edge_index = self.builder(states)

        # Check output shapes
        expected_adj_shape = (batch_size, len(self.valid_node_list), len(self.valid_node_list))
        self.assertEqual(adj_matrix.shape, expected_adj_shape)
        self.assertEqual(len(edge_index), batch_size)

        # Check that we have some edges (not completely empty)
        total_edges = sum(ei.shape[1] for ei in edge_index)
        self.assertGreater(total_edges, 0, "Should have edges when agents are present")

    def test_forward_empty_case(self):
        """Test forward with empty case (no agents present)."""
        # Create a sample batch state with no agents (all zeros)
        batch_size = 3
        num_agents = 25
        feature_dim = 15

        # All zeros means no agents present
        states = np.zeros((batch_size, num_agents, feature_dim), dtype=np.float32)

        # Process the states
        adj_matrix, edge_index = self.builder(states)

        # Check output shapes
        # For empty graphs, we should have:
        # - Adjacency matrix: (batch_size, 0, 0) since no valid nodes
        # - Edge index: list of arrays with shape (2, 0)
        max_node_id = max(self.valid_node_list)
        expected_adj_shape = (max_node_id + 1, max_node_id + 1)
        self.assertEqual(len(adj_matrix), batch_size)
        for adj in adj_matrix:
            self.assertEqual(adj.shape, expected_adj_shape)

        self.assertEqual(len(edge_index), batch_size)
        for ei in edge_index:
            self.assertEqual(ei.shape, (2, 0), "Empty graph should have empty edge index")

    def test_caching_mechanism(self):
        """Test the caching mechanism with update_interval."""

        self.builder.reset().eval()
        # Create a sample batch state with agents present
        batch_size = 5
        num_agents = 25
        feature_dim = 15

        # Initialize state with zeros
        states = np.zeros((batch_size, num_agents, feature_dim), dtype=np.float32)

        # Add some agents (set presence to 1 and valid coordinates)
        for b in range(batch_size):
            # Place agents at random positions with IDs
            agent_positions = np.random.choice(num_agents, size=num_agents, replace=False)
            for agent_id in range(num_agents):
                pos_idx = agent_positions[agent_id]

                # Set agent presence
                states[b, pos_idx, self.hp_dim] = 1

                # Set coordinates using little-endian format
                coords = np.random.randint(0, 45, size=2)
                states[b, pos_idx, self.coord_dim[0]] = coords[0]  # y coordinate
                states[b, pos_idx, self.coord_dim[1]] = coords[1]  # x coordinate

        # First call - should compute new graph
        adj_matrix_1, edge_index_1 = self.builder(states)

        # Subsequent calls within update interval - should return cached results
        for i in range(self.builder.update_interval - 1):
            adj_matrix_2, edge_index_2 = self.builder(np.zeros_like(states))

            # Should be equal to first result (cached)
            np.testing.assert_array_equal(adj_matrix_1, adj_matrix_2)
            for j in range(len(edge_index_1)):
                np.testing.assert_array_equal(edge_index_1[j], edge_index_2[j])

        # After update interval, should compute new graph
        adj_matrix_3, edge_index_3 = self.builder(np.zeros_like(states))

        # This should be different from cached result (empty vs previous)
        # Empty graph has shape (0,0) while previous had some nodes
        if adj_matrix_1.size > 0:  # If first graph wasn't empty
            self.assertFalse(np.array_equal(adj_matrix_3, adj_matrix_1))


if __name__ == '__main__':
    unittest.main()