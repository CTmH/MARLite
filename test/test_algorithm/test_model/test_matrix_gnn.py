import unittest
import torch
from marlite.algorithm.model.matrix_gnn import MatrixGCNModel

class TestMatrixGCNModel(unittest.TestCase):
    def setUp(self):
        self.input_dim = 32
        self.hidden_dim = 64
        self.output_dim = 10
        self.num_layers = 2
        self.model = MatrixGCNModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            num_layers=self.num_layers
        )

        # Create dummy data
        self.batch_size = 4
        self.n_agents = 5

        self.x = torch.randn(self.batch_size, self.n_agents, self.input_dim)
        # Create a random weighted adjacency matrix and normalize it
        raw_adj = torch.rand(self.batch_size, self.n_agents, self.n_agents)
        self.weight_adj_matrix = raw_adj / raw_adj.sum(dim=-1, keepdim=True)

    def test_model_output_shape(self):
        """Test that the model produces output with correct shape."""
        output = self.model(self.x, self.weight_adj_matrix)
        expected_shape = (self.batch_size, self.n_agents, self.output_dim)
        self.assertEqual(output.shape, expected_shape,
                        f"Expected output shape {expected_shape}, got {output.shape}")

    def test_model_forward_pass(self):
        """Ensure no errors during forward pass."""
        try:
            _ = self.model(self.x, self.weight_adj_matrix)
        except Exception as e:
            self.fail(f"Forward pass failed with exception: {e}")

    def test_multiple_layers_effect(self):
        """Test that multiple layers produce different results than single layer."""
        # Create models with different number of layers
        model_1layer = MatrixGCNModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            num_layers=1
        )
        model_2layer = MatrixGCNModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            num_layers=2
        )

        # Set same weights for fair comparison
        model_2layer.output_layer.load_state_dict(model_1layer.output_layer.state_dict())

        output_1layer = model_1layer(self.x, self.weight_adj_matrix)
        output_2layer = model_2layer(self.x, self.weight_adj_matrix)

        # They should be different due to additional message passing
        diff = torch.norm(output_1layer - output_2layer).item()
        self.assertGreater(diff, 1e-6, "Single and multi-layer outputs are too similar")

    def test_identical_input_output_consistency(self):
        """Test that identical inputs produce identical outputs."""
        output1 = self.model(self.x, self.weight_adj_matrix)
        output2 = self.model(self.x, self.weight_adj_matrix)

        # Outputs should be identical
        self.assertTrue(torch.allclose(output1, output2, atol=1e-6))

    def test_different_adjacency_effect(self):
        """Test that different adjacency matrices produce different results."""
        # Create a different adjacency matrix
        raw_adj2 = torch.rand(self.batch_size, self.n_agents, self.n_agents)
        weight_adj_matrix2 = raw_adj2 / raw_adj2.sum(dim=-1, keepdim=True)

        # Ensure it's different from the original
        self.assertFalse(torch.allclose(self.weight_adj_matrix, weight_adj_matrix2))

        output1 = self.model(self.x, self.weight_adj_matrix)
        output2 = self.model(self.x, weight_adj_matrix2)

        # Results should be different
        diff = torch.norm(output1 - output2).item()
        self.assertGreater(diff, 1e-6, "Different adjacency matrices produce too similar results")

if __name__ == '__main__':
    unittest.main()
