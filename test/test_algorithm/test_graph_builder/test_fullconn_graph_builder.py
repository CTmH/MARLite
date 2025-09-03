import numpy as np
import unittest
from magent2.environments import adversarial_pursuit_v4
from marlite.algorithm.graph_builder import GraphBuilderConfig

class TestFullConnGraphBuilder(unittest.TestCase):

    def test_process_batch_normal_case(self):
        config = {
            "type": "FullConn",
            "valid_node_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        }
        bs = 5
        num_nodes = len(config["valid_node_list"])
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
        self.assertTrue(np.array_equal(adj_matrix, np.ones((bs, num_nodes, num_nodes))))
        self.assertEqual(len(edge_index), bs)
        self.assertEqual(edge_index.shape[2], num_nodes * num_nodes)