import unittest
import torch
from torch import Tensor
from src.algorithm.model.gnn import GCNModel

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
    def test_batch_edge_index(self):
        batch_size = 2
        batch_edge_index = torch.randint(0, self.num_nodes, (batch_size, 2, self.num_edges))
        inputs = torch.randn(batch_size, self.num_nodes, self.input_dim)
        outputs = self.model(inputs, batch_edge_index)
        expected_shape = (batch_size, self.num_nodes, self.output_dim)
        self.assertEqual(outputs.shape, expected_shape)

if __name__ == '__main__':
    unittest.main()