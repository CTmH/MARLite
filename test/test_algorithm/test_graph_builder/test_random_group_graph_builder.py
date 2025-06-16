import numpy as np
import networkx as nx
import unittest
from magent2.environments import adversarial_pursuit_v4
from src.algorithm.graph_builder import GraphBuilderConfig

class TestRandomGroupGraphBuilder(unittest.TestCase):

    def test_process_batch_normal_case(self):
        group_num = 3
        config = {
            "type": "RandomGroup",
            "num_groups": group_num,
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
        adj_matrix, edge_indices = graph_builder(states)
        self.assertEqual(adj_matrix.shape, (bs, num_nodes, num_nodes))
        self.assertEqual(len(edge_indices), bs)
        G = nx.from_numpy_array(adj_matrix[0])
        num_components = nx.number_connected_components(G)
        self.assertGreaterEqual(num_components, group_num)
