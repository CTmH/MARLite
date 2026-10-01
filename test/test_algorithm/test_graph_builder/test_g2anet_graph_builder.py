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
        """Forward graph is binary and identical during collection and training."""
        train_hard, train_soft = self.graph_builder.train()(self.encoded_obs)
        eval_hard, eval_soft = self.graph_builder.eval()(self.encoded_obs)
        torch.testing.assert_close(train_hard, eval_hard)
        torch.testing.assert_close(train_soft, eval_soft)
        self.assertTrue(((train_hard == 0) | (train_hard == 1)).all())

    def test_dead_and_isolated_agents(self):
        alive = torch.tensor([[True, False, True, False, False]]).expand(self.batch_size, -1)
        first = self.graph_builder(self.encoded_obs, alive)
        changed = self.encoded_obs.clone()
        changed[:, ~alive[0]] = 1000.
        second = self.graph_builder(changed, alive)
        torch.testing.assert_close(first[0] * first[1], second[0] * second[1])
        adjacency = first[0] * first[1]
        self.assertTrue((adjacency[:, ~alive[0]] == 0).all())
        self.assertTrue((adjacency[:, :, ~alive[0]] == 0).all())
        hard, soft = self.graph_builder(self.encoded_obs, torch.zeros_like(alive))
        self.assertTrue(torch.isfinite(hard * soft).all())
        self.assertTrue((hard * soft == 0).all())

    def test_closed_gates_can_learn_to_open(self):
        with torch.no_grad():
            self.graph_builder.hard_attention_fc.weight.zero_()
            self.graph_builder.hard_attention_fc.bias.fill_(-1.)
        hard, soft = self.graph_builder(self.encoded_obs)
        self.assertTrue((hard == 0).all())
        (hard * soft).sum().backward()
        self.assertGreater(self.graph_builder.hard_attention_fc.bias.grad.abs().sum(), 0.)

    def test_soft_attention_shape(self):
        # Test soft attention weights shape
        _, soft_attention_weights = self.graph_builder(self.encoded_obs)

        # Soft attention weights should have shape (batch_size, n_agents, n_agents)
        self.assertEqual(soft_attention_weights.shape, (self.batch_size, self.n_agents, self.n_agents))

if __name__ == '__main__':
    unittest.main()
