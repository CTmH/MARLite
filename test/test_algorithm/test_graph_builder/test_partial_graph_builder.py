import numpy as np
import unittest
from magent2.environments import adversarial_pursuit_v4
from marlite.algorithm.graph_builder import GraphBuilderConfig

import numpy as np
import unittest
from marlite.algorithm.graph_builder.partial_graph_builder import PartialGraphMAgentBuilder

class TestPartialGraphMAgentBuilder(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.n_agent = 25
        self.binary_agent_id_dim = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        self.agent_presence_dim = [1, 3]
        self.comm_distance = 10
        self.distance_metric = 'cityblock'
        self.valid_node_list = list(range(1, self.n_agent+1))

        # Create builder instance for testing
        self.builder = PartialGraphMAgentBuilder(
            binary_agent_id_dim=self.binary_agent_id_dim,
            agent_presence_dim=self.agent_presence_dim,
            comm_distance=self.comm_distance,
            distance_metric=self.distance_metric,
            n_workers=1,  # Use single worker for testing
            n_subgraphs=5,
            valid_node_list=self.valid_node_list,
            update_interval=5,
            channel_first=False
        )

        # Reset builder to initial state
        self.builder.reset()

    def test_process_batch_normal_case(self):
        """Test _process_batch with normal case (agents present)."""
        # Create a sample batch state with agents present

        batch_size = 5
        height, width = 45, 45
        channels = 15  # Based on the dimensions used in the config

        # Initialize state with zeros
        states = np.zeros((batch_size, height, width, channels), dtype=np.float32)

        # Add some agents (set presence to 1 and valid binary IDs)
        for b in range(batch_size):
            # Place 6 agents at random positions with IDs 1-6
            agent_positions = np.random.choice(self.comm_distance * self.comm_distance, size=self.n_agent, replace=False)
            for agent_id in range(1, self.n_agent+1):
                pos_idx = agent_positions[agent_id - 1]
                i, j = divmod(pos_idx, width)  # Convert flat index to 2D coordinates

                # Set agent presence
                states[b, i, j, self.agent_presence_dim] = 1

                # Set binary agent ID
                binary_id = [int(x) for x in format(agent_id, '010b')]  # 10-bit binary
                for k, bit in enumerate(binary_id):
                    states[b, i, j, self.binary_agent_id_dim[k]] = bit

        # Process the states
        adj_matrix, edge_index = self.builder.forward(states)

        # Check output shapes
        expected_adj_shape = (batch_size, len(self.valid_node_list), len(self.valid_node_list))
        self.assertEqual(adj_matrix.shape, expected_adj_shape)
        self.assertEqual(len(edge_index), batch_size)

        # Check that we have some edges (not completely empty)
        total_edges = sum(ei.shape[1] for ei in edge_index)
        self.assertGreater(total_edges, 0, "Should have edges when agents are present")

    def test_process_batch_empty_case(self):
        """Test _process_batch with empty case (no agents present)."""
        # Create a sample batch state with no agents (all zeros)
        batch_size = 3
        height, width = 45, 45
        channels = 15

        # All zeros means no agents present
        states = np.zeros((batch_size, height, width, channels), dtype=np.float32)

        # Process the states
        adj_matrix, edge_index = self.builder.forward(states)

        # Check output shapes
        # For empty graphs, we should have:
        # - Adjacency matrix: (batch_size, 0, 0) since no valid nodes
        # - Edge index: list of arrays with shape (2, 0)
        self.assertEqual(len(adj_matrix), batch_size)
        for adj in adj_matrix:
            self.assertEqual(adj.shape, (0, 0), "Empty graph should have 0x0 adjacency matrix")

        self.assertEqual(len(edge_index), batch_size)
        for ei in edge_index:
            self.assertEqual(ei.shape, (2, 0), "Empty graph should have empty edge index")

    def test_caching_mechanism(self):
        """Test the caching mechanism with update_interval."""
        # Create a sample state
        batch_size = 2
        height, width = 45, 45
        channels = 15

        states = np.zeros((batch_size, height, width, channels), dtype=np.float32)
        # Add one agent to make it non-empty
        states[0, 10, 10, self.agent_presence_dim] = 1
        states[0, 10, 10, self.binary_agent_id_dim[0]] = 1  # Simple binary ID

        # First call - should compute new graph
        adj_matrix_1, edge_index_1 = self.builder.forward(states)

        # Subsequent calls within update interval - should return cached results
        for i in range(self.builder.update_interval - 1):
            adj_matrix_2, edge_index_2 = self.builder.forward(np.zeros_like(states))

            # Should be equal to first result (cached)
            np.testing.assert_array_equal(adj_matrix_1, adj_matrix_2)
            for j in range(len(edge_index_1)):
                np.testing.assert_array_equal(edge_index_1[j], edge_index_2[j])

        # After update interval, should compute new graph
        adj_matrix_3, edge_index_3 = self.builder.forward(np.zeros_like(states))

        # This should be different from cached result (empty vs previous)
        # Empty graph has shape (0,0) while previous had some nodes
        if adj_matrix_1.size > 0:  # If first graph wasn't empty
            self.assertNotEqual(adj_matrix_3.shape, adj_matrix_1.shape)

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
            states[b, b*10, b*10, self.binary_agent_id_dim[0]] = 1

        # Process states
        adj_matrix, edge_index = builder_seq.forward(states)

        # Check results
        self.assertEqual(len(adj_matrix), batch_size)
        self.assertEqual(len(edge_index), batch_size)

    def test_valid_node_list_filtering(self):
        """Test that valid_node_list properly filters nodes."""
        # Create builder with subset of valid nodes
        subset_valid_nodes = [0, 1, 2, 3, 4]
        builder_subset = PartialGraphMAgentBuilder(
            binary_agent_id_dim=self.binary_agent_id_dim,
            agent_presence_dim=self.agent_presence_dim,
            comm_distance=self.comm_distance,
            distance_metric=self.distance_metric,
            n_workers=1,
            n_subgraphs=5,
            valid_node_list=subset_valid_nodes,
            update_interval=5
        )

        # Create test states with agents having various IDs
        batch_size = 1
        height, width = 45, 45
        channels = 15
        states = np.zeros((batch_size, height, width, channels), dtype=np.float32)

        # Place agents with different IDs
        positions = [(10, 10), (20, 20), (30, 30)]
        agent_ids = [0, 5, 2]  # One inside valid list, one outside, one inside

        for pos, agent_id in zip(positions, agent_ids):
            i, j = pos
            states[0, i, j, self.agent_presence_dim] = 1
            # Set binary representation of agent_id
            binary_id = [int(x) for x in format(agent_id, '010b')]
            for k, bit in enumerate(binary_id):
                states[0, i, j, self.binary_agent_id_dim[k]] = bit

        # Process states
        adj_matrix, edge_index = builder_subset.forward(states)

        # With valid_node_list=[0,1,2,3,4], only agents with IDs 0 and 2 should be included
        # So adjacency matrix should be 2x2
        expected_size = len([aid for aid in agent_ids if aid in subset_valid_nodes])
        self.assertEqual(adj_matrix[0].shape, (expected_size, expected_size))

if __name__ == '__main__':
    unittest.main()

'''
class TestMAgentGraphBuilder(unittest.TestCase):

    def test_process_batch_normal_case(self):
        valid_node_list = [i for i in range(25)]
        update_interval = 5
        config = {
            "type": "PartialMAgent",
            "binary_agent_id_dim": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            "agent_presence_dim": [1, 3],
            "comm_distance": 10,
            "distance_metric": "cityblock",
            "n_workers": 8,
            "n_subgraphs": 5,
            "valid_node_list": valid_node_list,
            "update_interval": update_interval
        }
        bs = 5
        env = adversarial_pursuit_v4.parallel_env(map_size=45, minimap_mode=False, tag_penalty=-0.2,
        max_cycles=500, extra_features=True, render_mode='rgb_array')
        obs = env.reset()
        state = env.state()
        states = np.stack([state for _ in range(bs)])
        # Create instance of GraphBuilderConfig
        builder_config = GraphBuilderConfig(**config)

        # Call get_graph_builder method
        graph_builder = builder_config.get_graph_builder()
        graph_builder.eval().reset()
        adj_matrix, edge_index = graph_builder(states)
        self.assertEqual(adj_matrix.shape, np.zeros((bs, len(valid_node_list), len(valid_node_list))).shape)
        self.assertEqual(len(edge_index), bs)
        for i in range(update_interval-2):
            adj_matrix_next, edge_index_next = graph_builder(np.zeros_like(states))
            self.assertTrue(np.array_equal(adj_matrix, adj_matrix_next))
            self.assertTrue(np.array_equal(edge_index, edge_index_next))
'''