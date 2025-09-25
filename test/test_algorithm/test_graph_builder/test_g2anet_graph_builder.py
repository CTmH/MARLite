import unittest
import torch
from marlite.algorithm.graph_builder.g2anet_graph_builder import G2ANetAttention, G2ANetGraphBuilder

class TestG2ANetGraphBuilder(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.n_agents = 5
        self.obs_dim = 10
        self.hidden_dim = 64

        # Create dummy encoded observations
        self.encoded_obs = torch.randn(self.batch_size, self.n_agents, self.obs_dim)

        # Initialize the graph builder
        self.graph_builder = G2ANetGraphBuilder(
            n_agents = self.n_agents,
            input_dim=self.obs_dim,
            hidden_dim=self.hidden_dim
        )

    def test_forward(self):
        # Test forward pass shape
        weight_adj_matrix, _ = self.graph_builder(self.encoded_obs)
        # Check shapes
        self.assertEqual(weight_adj_matrix.shape, (self.batch_size, self.n_agents, self.n_agents))

    def test_reset_method(self):
        # Test reset method returns the same object
        original_id = id(self.graph_builder)
        reset_result = self.graph_builder.reset()
        self.assertEqual(id(reset_result), original_id)

class TestG2ANetAttention(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.n_agents = 5
        self.obs_dim = 10
        self.hidden_dim = 64

        # Create dummy encoded observations
        self.encoded_obs = torch.randn(self.batch_size, self.n_agents, self.obs_dim)

        # Initialize the graph builder
        self.graph_builder = G2ANetAttention(
            n_agents = self.n_agents,
            input_dim=self.obs_dim,
            hidden_dim=self.hidden_dim
        )

    def test_forward_shape(self):
        # Test forward pass shape
        hard_attention_weights, soft_attention_weights = self.graph_builder(self.encoded_obs)

        # Check shapes
        self.assertEqual(hard_attention_weights.shape, (self.batch_size, self.n_agents, self.n_agents))
        self.assertEqual(soft_attention_weights.shape, (self.batch_size, self.n_agents, self.n_agents))

    def test_hard_attention_training_mode(self):
        """Test hard attention in training mode: should output probability distributions (sum=1)"""
        self.graph_builder.train()
        hard_attention_weights, _ = self.graph_builder(self.encoded_obs)

        # Each row should sum to ~1 (due to Gumbel-Softmax)
        hard_row_sums = torch.sum(hard_attention_weights, dim=-1)
        self.assertTrue(
            torch.allclose(hard_row_sums, torch.ones_like(hard_row_sums), atol=1e-5),
            f"Hard attention weights do not sum to 1 in training mode. Sums: {hard_row_sums}"
        )

    def test_hard_attention_eval_mode(self):
        """Test hard attention in eval mode: should output one-hot vectors (exactly one 1 per row)"""
        self.graph_builder.eval()
        hard_attention_weights, _ = self.graph_builder(self.encoded_obs)

        # In eval mode, each row must have exactly one 1 and rest 0s
        hard_max_values, hard_max_indices = torch.max(hard_attention_weights, dim=-1)

        # Max value should be 1 for all rows
        self.assertTrue(
            torch.allclose(hard_max_values, torch.ones(self.n_agents), atol=1e-5),
            f"Max values are not 1 in eval mode: {hard_max_values}"
        )

        # Sum of each row should also be 1
        hard_row_sums = torch.sum(hard_attention_weights, dim=-1)
        self.assertTrue(
            torch.allclose(hard_row_sums, torch.ones(self.n_agents), atol=1e-5),
            f"Row sums are not 1 in eval mode: {hard_row_sums}"
        )

    def test_soft_attention_shape(self):
        # Test soft attention weights shape
        _, soft_attention_weights = self.graph_builder(self.encoded_obs)

        # Soft attention weights should have shape (batch_size, n_agents, n_agents)
        self.assertEqual(soft_attention_weights.shape, (self.batch_size, self.n_agents, self.n_agents))

if __name__ == '__main__':
    unittest.main()