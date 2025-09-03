import unittest
from marlite.algorithm.graph_builder import GraphBuilderConfig
from marlite.algorithm.graph_builder.magent_graph_builder import MagentGraphBuilder

class TestGraphBuilderConfig(unittest.TestCase):

    def test_get_graph_builder(self):
        # Initialize with valid parameters
        config = {
            "type": "Magent",
            "binary_agent_id_dim": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            "agent_presence_dim": [3],
            "comm_distance": 10,
            "distance_metric": "cityblock"
        }

        # Create instance of GraphBuilderConfig
        builder_config = GraphBuilderConfig(**config)

        # Call get_graph_builder method
        graph_builder = builder_config.get_graph_builder()

        # Verify that the correct class is used
        self.assertIsInstance(graph_builder, MagentGraphBuilder)

if __name__ == "__main__":
    unittest.main()