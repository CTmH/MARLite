import unittest
import torch
from marlite.algorithm.model.gnn import GCNModel, GATModel

class TestGCNModel(unittest.TestCase):
    def setUp(self):
        self.input_dim = 16
        self.hidden_dim = 64
        self.output_dim = 10
        self.model = GCNModel(self.input_dim, self.hidden_dim, self.output_dim)

        # Create dummy data
        self.num_nodes = 5
        self.num_edges = 10
        self.inputs = torch.randn(self.num_nodes, self.input_dim)
        self.edge_index = torch.randint(0, self.num_nodes, (2, self.num_edges))

    def test_model_output_shape(self):
        output = self.model(self.inputs, self.edge_index)
        self.assertEqual(output.shape, (self.num_nodes, self.output_dim))

    def test_model_forward_pass(self):
        # Ensure no errors during forward pass
        try:
            _ = self.model(self.inputs, self.edge_index)
        except Exception as e:
            self.fail(f"Forward pass failed with exception: {e}")

class TestGATModel(unittest.TestCase):
    def setUp(self):
        self.input_dim = 16
        self.hidden_dim = 64
        self.output_dim = 10
        self.model = GATModel(self.input_dim, self.hidden_dim, self.output_dim)

        # Create dummy data
        self.num_nodes = 5
        self.num_edges = 10
        self.inputs = torch.randn(self.num_nodes, self.input_dim)
        self.edge_index = torch.randint(0, self.num_nodes, (2, self.num_edges))

    def test_model_output_shape(self):
        output = self.model(self.inputs, self.edge_index)
        self.assertEqual(output.shape, (self.num_nodes, self.output_dim))

    def test_model_forward_pass(self):
        # Ensure no errors during forward pass
        try:
            _ = self.model(self.inputs, self.edge_index)
        except Exception as e:
            self.fail(f"Forward pass failed with exception: {e}")

if __name__ == '__main__':
    unittest.main()