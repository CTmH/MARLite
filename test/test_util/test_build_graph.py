import numpy as np
import torch
from magent2.environments import adversarial_pursuit_v4
from src.environment.adversarial_pursuit_wrapper import AdversarialPursuitPredator
from src.algorithm.graph_builder.build_graph import *

def test_binary_to_decimal():
    binary_list = [1, 0, 1]
    assert binary_to_decimal(binary_list) == 5

def test_build_graph_from_map_2d():
    A = np.array([[0, -1, 2], [3, 4, -1], [-1, 5, 6]])
    threshold = 3
    adj_matrix, edge_index = build_graph_from_map_2d(A, threshold)
    expected_adj_matrix = np.array([[0, 0, 1, 1, 1, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 0],
                                    [1, 0, 0, 1, 1, 1, 1],
                                    [1, 0, 1, 0, 1, 1, 1],
                                    [1, 0, 1, 1, 0, 1, 1],
                                    [1, 0, 1, 1, 1, 0, 1],
                                    [0, 0, 1, 1, 1, 1, 0]])
    expected_edge_index = np.array([[0, 0, 0, 0, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5], [2, 3, 4, 5, 3, 4, 5, 6, 4, 5, 6, 5, 6, 6]])
    assert np.array_equal(adj_matrix, expected_adj_matrix)
    assert np.array_equal(edge_index, expected_edge_index)

def test_build_graph_from_state_with_binary_agent_id():
    env = adversarial_pursuit_v4.parallel_env(map_size=45, minimap_mode=False, tag_penalty=-0.2, max_cycles=500, extra_features=True)
    obs = env.reset()
    state = env.state()
    binary_agent_id_dim = [i for i in range(5,15)]
    agent_presence_dim = [1, 3]
    threshold = 5
    adj_matrix, edge_index = build_graph_from_state_with_binary_agent_id(state, binary_agent_id_dim, agent_presence_dim, threshold)
    assert np.array_equal(adj_matrix.shape, np.zeros((75, 75), dtype=np.int64).shape)
    assert np.array_equal(edge_index.shape[0], 2)

def test_filter_edge_index():
    edge_index = [[0, 0], [0, 1], [1, 2], [2, 3], [3, 0]]
    edge_index = np.array(edge_index).astype(np.int64).T
    node_ids = [0, 1, 3]
    filtered_edge_index = filter_edge_index(edge_index, node_ids)
    expected_filtered_edge_index = np.array([[0, 0], [0, 1], [3, 0]]).T.astype(np.int64)
    assert np.array_equal(filtered_edge_index, expected_filtered_edge_index)