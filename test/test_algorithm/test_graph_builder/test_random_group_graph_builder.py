import numpy as np
import torch
import networkx as nx
import unittest
from marlite.configuration.catalog import default_registry


class TestRandomGroupGraphBuilder(unittest.TestCase):
    def test_process_batch_normal_case(self):
        group_num = 3
        config = {
            "type": "RandomGroup",
            "num_groups": group_num,
            "valid_node_list": [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
            ],
        }
        bs = 5
        num_nodes = len(config["valid_node_list"])
        state = np.zeros((45, 45, 29), dtype=np.float32)
        state[2, 2, 1] = 1
        states = np.stack([state for _ in range(bs)])
        # Create instance of component registry
        builder_config = default_registry().build(
            {
                "type": "RandomGroupGraphBuilder",
                "params": {k: v for k, v in config.items() if k != "type"},
            }
        )

        # Call get_graph_builder method
        graph_builder = builder_config
        adj_matrix, edge_indices = graph_builder(torch.from_numpy(states).float())
        self.assertEqual(adj_matrix.shape, (bs, num_nodes, num_nodes))
        self.assertEqual(len(edge_indices), bs)
        G = nx.from_numpy_array(adj_matrix[0].cpu().numpy())
        num_components = nx.number_connected_components(G)
        self.assertGreaterEqual(num_components, group_num)
