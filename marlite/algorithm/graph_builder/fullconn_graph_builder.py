import numpy as np
from typing import Tuple, List, Union
from numpy import ndarray
from marlite.algorithm.graph_builder.graph_builder import GraphBuilder

class FullConnGraphBuilder(GraphBuilder):

    def __init__(
            self,
            valid_node_list: Union[list, None] = None,
            add_self_loop: bool = True):
        super().__init__()
        self.valid_node_list = valid_node_list
        self.add_self_loop = add_self_loop

    def forward(self, states: np.ndarray) -> Tuple[ndarray, List[ndarray]]:
        bs = states.shape[0]
        num_nodes = len(self.valid_node_list)
        edge_indices = [[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j]
        if self.add_self_loop:
            batch_adj_matrix = np.ones((bs, num_nodes, num_nodes),dtype=np.int64)
            edge_indices = edge_indices + [[i, i] for i in range(num_nodes)]
        else:
            batch_adj_matrix = np.ones((bs, num_nodes, num_nodes),dtype=np.int64) - np.eye(num_nodes, dtype=np.int64)

        edge_indices = np.array(edge_indices, dtype=np.int64).T
        batch_edge_indices = np.repeat(edge_indices[np.newaxis], bs, axis=0)
        return batch_adj_matrix, batch_edge_indices

    def reset(self):
        return self