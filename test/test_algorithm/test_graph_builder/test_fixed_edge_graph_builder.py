import unittest
import numpy as np
from marlite.algorithm.graph_builder import GraphBuilderConfig
from marlite.algorithm.graph_builder.fixed_edge_graph_builder import FixedEdgeGraphBuilder


class TestFixedEdgeGraphBuilder(unittest.TestCase):

    def test_initialization_with_numpy_array(self):
        """Test initialization with numpy array edge indices."""
        edge_indices = np.array([[0, 1, 2], [1, 2, 0]])  # edges: 0->1, 1->2, 2->0

        builder = FixedEdgeGraphBuilder(
            edge_indices=edge_indices,
            num_nodes=3,
            add_self_loop=True,
            symmetric=True
        )

        self.assertEqual(builder.get_num_nodes(), 3)
        self.assertTrue(np.array_equal(builder.get_edge_indices(), edge_indices))
        self.assertEqual(builder.add_self_loop, True)
        self.assertEqual(builder.symmetric, True)

        # Check adjacency matrix
        expected_adj = np.array([
            [1, 1, 1],  # 0 connects to 0(self), 1, 2
            [1, 1, 1],  # 1 connects to 0, 1(self), 2
            [1, 1, 1]   # 2 connects to 0, 1, 2(self)
        ])
        self.assertTrue(np.array_equal(builder.get_adj_matrix(), expected_adj))

    def test_initialization_with_list(self):
        """Test initialization with list edge indices."""
        edge_list = [[0, 1, 2], [1, 2, 0]]

        builder = FixedEdgeGraphBuilder(
            edge_indices=edge_list,
            num_nodes=3,
            add_self_loop=False,
            symmetric=True
        )

        self.assertEqual(builder.get_num_nodes(), 3)
        self.assertTrue(np.array_equal(builder.get_edge_indices(), np.array(edge_list)))

        # Check adjacency matrix (no self-loops)
        expected_adj = np.array([
            [0, 1, 1],  # 0 connects to 1, 2
            [1, 0, 1],  # 1 connects to 0, 2
            [1, 1, 0]   # 2 connects to 0, 1
        ])
        self.assertTrue(np.array_equal(builder.get_adj_matrix(), expected_adj))

    def test_initialization_with_edge_list_format(self):
        """Test initialization with edge list format (num_edges, 2)."""
        edge_indices = np.array([[0, 1], [1, 2], [2, 0]])  # shape (3, 2)

        builder = FixedEdgeGraphBuilder(
            edge_indices=edge_indices,
            add_self_loop=True,
            symmetric=False  # Directed graph
        )

        # Should infer 3 nodes from max edge index
        self.assertEqual(builder.get_num_nodes(), 3)

        # Check adjacency matrix (directed, with self-loops)
        expected_adj = np.array([
            [1, 1, 0],  # 0 connects to 0(self), 1
            [0, 1, 1],  # 1 connects to 1(self), 2
            [1, 0, 1]   # 2 connects to 0, 2(self)
        ])
        self.assertTrue(np.array_equal(builder.get_adj_matrix(), expected_adj))

    def test_forward_method_single_batch(self):
        """Test forward method with single batch."""
        edge_indices = np.array([[0, 1], [2, 2]])  # edges: 0->2, 1->2

        builder = FixedEdgeGraphBuilder(
            edge_indices=edge_indices,
            num_nodes=4,
            add_self_loop=False,
            symmetric=True
        )

        # Create dummy states with batch size 1
        states = np.random.randn(1, 10, 5)

        batch_adj_matrix, batch_edge_indices = builder.forward(states)

        # Check shapes
        self.assertEqual(batch_adj_matrix.shape, (1, 4, 4))
        self.assertEqual(len(batch_edge_indices), 1)

        # Check content
        expected_adj = builder.get_adj_matrix()
        self.assertTrue(np.array_equal(batch_adj_matrix[0], expected_adj))
        self.assertTrue(np.array_equal(batch_edge_indices[0], edge_indices))

    def test_forward_method_multiple_batches(self):
        """Test forward method with multiple batches."""
        edge_indices = np.array([[0, 1, 2], [1, 2, 0]])

        builder = FixedEdgeGraphBuilder(
            edge_indices=edge_indices,
            num_nodes=3,
            add_self_loop=True,
            symmetric=True
        )

        # Create dummy states with batch size 3
        states = np.random.randn(3, 8, 6)

        batch_adj_matrix, batch_edge_indices = builder.forward(states)

        # Check shapes
        self.assertEqual(batch_adj_matrix.shape, (3, 3, 3))
        self.assertEqual(len(batch_edge_indices), 3)

        # Check that all batches have same adjacency matrix
        expected_adj = builder.get_adj_matrix()
        for i in range(3):
            self.assertTrue(np.array_equal(batch_adj_matrix[i], expected_adj))
            self.assertTrue(np.array_equal(batch_edge_indices[i], edge_indices))

    def test_empty_graph(self):
        """Test with empty graph (no edges)."""
        edge_indices = np.zeros((2, 0), dtype=np.int64)

        builder = FixedEdgeGraphBuilder(
            edge_indices=edge_indices,
            num_nodes=5,
            add_self_loop=True,
            symmetric=True
        )

        self.assertEqual(builder.get_num_nodes(), 5)
        self.assertEqual(builder.get_edge_indices().shape, (2, 0))

        # Check adjacency matrix (only self-loops)
        expected_adj = np.eye(5, dtype=np.int64)
        self.assertTrue(np.array_equal(builder.get_adj_matrix(), expected_adj))

        # Test forward
        states = np.random.randn(2, 7, 4)
        batch_adj_matrix, batch_edge_indices = builder.forward(states)

        self.assertEqual(batch_adj_matrix.shape, (2, 5, 5))
        self.assertEqual(len(batch_edge_indices), 2)
        self.assertEqual(batch_edge_indices[0].shape, (2, 0))

    def test_directed_graph(self):
        """Test directed graph (non-symmetric)."""
        edge_indices = np.array([[0, 0, 1], [1, 2, 2]])  # edges: 0->1, 0->2, 1->2

        builder = FixedEdgeGraphBuilder(
            edge_indices=edge_indices,
            num_nodes=3,
            add_self_loop=False,
            symmetric=False
        )

        # Check adjacency matrix (directed)
        expected_adj = np.array([
            [0, 1, 1],  # 0 connects to 1, 2
            [0, 0, 1],  # 1 connects to 2
            [0, 0, 0]   # 2 connects to none
        ])
        self.assertTrue(np.array_equal(builder.get_adj_matrix(), expected_adj))

    def test_node_inference(self):
        """Test node count inference from edge indices."""
        edge_indices = np.array([[0, 5, 10], [2, 7, 3]])

        builder = FixedEdgeGraphBuilder(
            edge_indices=edge_indices,
            add_self_loop=True,
            symmetric=True
        )

        # Should infer 11 nodes (max index is 10, so 0-10 inclusive)
        self.assertEqual(builder.get_num_nodes(), 11)

        # Check adjacency matrix shape
        self.assertEqual(builder.get_adj_matrix().shape, (11, 11))

    def test_reset_method(self):
        """Test reset method (should be no-op for fixed edge builder)."""
        edge_indices = np.array([[0, 1], [1, 0]])
        builder = FixedEdgeGraphBuilder(edge_indices=edge_indices)

        # Reset should return self
        reset_builder = builder.reset()
        self.assertIs(reset_builder, builder)

        # State should remain unchanged
        self.assertTrue(np.array_equal(builder.get_edge_indices(), edge_indices))

    def test_graph_builder_config_integration(self):
        """Test integration with GraphBuilderConfig."""
        config = {
            "type": "Fixed",
            "edge_indices": [[0, 1, 2], [1, 2, 0]],
            "num_nodes": 3,
            "add_self_loop": True,
            "symmetric": True
        }

        builder_config = GraphBuilderConfig(**config)
        graph_builder = builder_config.get_graph_builder()

        self.assertIsInstance(graph_builder, FixedEdgeGraphBuilder)

        # Test forward
        states = np.random.randn(2, 10, 5)
        batch_adj_matrix, batch_edge_indices = graph_builder.forward(states)

        self.assertEqual(batch_adj_matrix.shape, (2, 3, 3))
        self.assertEqual(len(batch_edge_indices), 2)

    def test_edge_cases(self):
        """Test various edge cases."""
        # Test 1: Single node graph
        edge_indices1 = np.zeros((2, 0), dtype=np.int64)
        builder1 = FixedEdgeGraphBuilder(
            edge_indices=edge_indices1,
            num_nodes=1,
            add_self_loop=True
        )
        self.assertEqual(builder1.get_num_nodes(), 1)
        self.assertTrue(np.array_equal(builder1.get_adj_matrix(), np.array([[1]])))

        # Test 2: Graph with only self-loops
        edge_indices2 = np.zeros((2, 0), dtype=np.int64)
        builder2 = FixedEdgeGraphBuilder(
            edge_indices=edge_indices2,
            num_nodes=3,
            add_self_loop=True
        )
        expected_adj2 = np.eye(3, dtype=np.int64)
        self.assertTrue(np.array_equal(builder2.get_adj_matrix(), expected_adj2))

        # Test 3: Graph without self-loops and no edges
        edge_indices3 = np.zeros((2, 0), dtype=np.int64)
        builder3 = FixedEdgeGraphBuilder(
            edge_indices=edge_indices3,
            num_nodes=2,
            add_self_loop=False
        )
        expected_adj3 = np.zeros((2, 2), dtype=np.int64)
        self.assertTrue(np.array_equal(builder3.get_adj_matrix(), expected_adj3))

    def test_edge_indices_smaller_than_num_nodes(self):
        """Test when edge_indices max index is smaller than num_nodes."""

        print("Testing edge_indices with max index smaller than num_nodes")
        print("=" * 60)

        # Case 1: edge_indices max index is 2, but num_nodes is specified as 5
        print("\nCase 1: edge_indices max index=2, num_nodes=5")
        edge_indices = np.array([[0, 1, 2], [1, 2, 0]])  # max index is 2
        num_nodes = 5  # specify larger number of nodes

        builder = FixedEdgeGraphBuilder(
            edge_indices=edge_indices,
            num_nodes=num_nodes,
            add_self_loop=True,
            symmetric=True
        )

        print(f"Number of nodes: {builder.get_num_nodes()}")
        print(f"Edge indices:\n{builder.get_edge_indices()}")
        print(f"Adjacency matrix shape: {builder.get_adj_matrix().shape}")
        print(f"Adjacency matrix:\n{builder.get_adj_matrix()}")

        # Verify adj_matrix shape is (5, 5)
        assert builder.get_adj_matrix().shape == (5, 5), f"Expected shape (5,5), got {builder.get_adj_matrix().shape}"

        # Verify connections between first 3 nodes are correct
        expected_submatrix = np.array([
            [1, 1, 1, 0, 0],  # Node 0: connects to 0,1,2
            [1, 1, 1, 0, 0],  # Node 1: connects to 0,1,2
            [1, 1, 1, 0, 0],  # Node 2: connects to 0,1,2
            [0, 0, 0, 1, 0],  # Node 3: only self-loop
            [0, 0, 0, 0, 1]   # Node 4: only self-loop
        ])

        print(f"\nExpected adjacency matrix:\n{expected_submatrix}")
        assert np.array_equal(builder.get_adj_matrix(), expected_submatrix), "Adjacency matrix incorrect!"

        # Test forward method
        print("\nTesting forward method...")
        states = np.random.randn(3, 10, 5)  # batch_size=3
        batch_adj_matrix, batch_edge_indices = builder.forward(states)

        print(f"Batch adjacency matrix shape: {batch_adj_matrix.shape}")
        print(f"Number of edge indices arrays: {len(batch_edge_indices)}")

        # Verify batch_adj_matrix shape is (3, 5, 5)
        assert batch_adj_matrix.shape == (3, 5, 5), f"Expected shape (3,5,5), got {batch_adj_matrix.shape}"

        # Verify all batches have the same adj_matrix
        for i in range(3):
            assert np.array_equal(batch_adj_matrix[i], builder.get_adj_matrix()), f"Batch {i} adjacency matrix mismatch"
            assert np.array_equal(batch_edge_indices[i], edge_indices), f"Batch {i} edge indices mismatch"

        # Case 2: No edges, but num_nodes is specified
        print("\n" + "=" * 60)
        print("Case 2: No edges, num_nodes=4")
        edge_indices2 = np.zeros((2, 0), dtype=np.int64)  # no edges
        num_nodes2 = 4

        builder2 = FixedEdgeGraphBuilder(
            edge_indices=edge_indices2,
            num_nodes=num_nodes2,
            add_self_loop=True,
            symmetric=True
        )

        print(f"Number of nodes: {builder2.get_num_nodes()}")
        print(f"Adjacency matrix shape: {builder2.get_adj_matrix().shape}")
        print(f"Adjacency matrix:\n{builder2.get_adj_matrix()}")

        # Verify adj_matrix is identity matrix (only self-loops)
        expected_adj2 = np.eye(4, dtype=np.int64)
        assert np.array_equal(builder2.get_adj_matrix(), expected_adj2), "Adjacency matrix should be identity matrix"

        # Case 3: edge_indices max index is 1, but num_nodes is specified as 10, without self-loops
        print("\n" + "=" * 60)
        print("Case 3: edge_indices max index=1, num_nodes=10, no self-loop")
        edge_indices3 = np.array([[0], [1]])  # single edge 0->1
        num_nodes3 = 10

        builder3 = FixedEdgeGraphBuilder(
            edge_indices=edge_indices3,
            num_nodes=num_nodes3,
            add_self_loop=False,
            symmetric=False  # directed graph
        )

        print(f"Number of nodes: {builder3.get_num_nodes()}")
        print(f"Adjacency matrix shape: {builder3.get_adj_matrix().shape}")

        # Verify adj_matrix[0,1] = 1, others are 0
        adj_matrix3 = builder3.get_adj_matrix()
        assert adj_matrix3[0, 1] == 1, "Edge 0->1 should exist"
        assert adj_matrix3[1, 0] == 0, "Edge 1->0 should not exist (directed)"

        # Verify no other connections exist
        for i in range(10):
            for j in range(10):
                if not (i == 0 and j == 1):
                    assert adj_matrix3[i, j] == 0, f"Unexpected edge at ({i},{j})"

        print("\n" + "=" * 60)
        print("All tests passed! ✅")
        print("Summary: When num_nodes is explicitly specified, batch_adj_matrix")
        print("         will have shape (batch_size, num_nodes, num_nodes) regardless")
        print("         of the max index in edge_indices.")


if __name__ == '__main__':
    unittest.main()