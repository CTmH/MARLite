import numpy as np
import unittest
from magent2.environments import adversarial_pursuit_v4
from marlite.algorithm.graph_builder import GraphBuilderConfig

class TestMagentGraphBuilder(unittest.TestCase):

    def test_process_batch_normal_case(self):
        valid_node_list = [i for i in range(25)]
        update_interval = 5
        config = {
            "type": "PartialMagent",
            "binary_agent_id_dim": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            "agent_presence_dim": [1, 3],
            "comm_distance": 10,
            "distance_metric": "cityblock",
            "n_workers": 8,
            "n_subgraphs": 5,
            "valid_node_list": valid_node_list,
            "update_interval": update_interval
        }
        bs = 5
        env = adversarial_pursuit_v4.parallel_env(map_size=45, minimap_mode=False, tag_penalty=-0.2,
        max_cycles=500, extra_features=True, render_mode='rgb_array')
        obs = env.reset()
        state = env.state()
        states = np.stack([state for _ in range(bs)])
        # Create instance of GraphBuilderConfig
        builder_config = GraphBuilderConfig(**config)

        # Call get_graph_builder method
        graph_builder = builder_config.get_graph_builder()
        graph_builder.eval().reset()
        adj_matrix, edge_index = graph_builder(states)
        self.assertEqual(adj_matrix.shape, np.zeros((bs, len(valid_node_list), len(valid_node_list))).shape)
        self.assertEqual(len(edge_index), bs)
        for i in range(update_interval-2):
            adj_matrix_next, edge_index_next = graph_builder(np.zeros_like(states))
            self.assertTrue(np.array_equal(adj_matrix, adj_matrix_next))
            self.assertTrue(np.array_equal(edge_index, edge_index_next))