from typing import Tuple, List, Union
from numpy import ndarray
import numpy as np
from marlite.algorithm.graph_builder.graph_builder import GraphBuilder

class RandomGroupGraphBuilder(GraphBuilder):

    def __init__(
            self,
            num_groups: int,
            valid_node_list: Union[list, None] = None,
            add_self_loop: bool = True):
        super().__init__()
        if valid_node_list is None:
            raise ValueError("valid_node_list must be provided for RandomGroupGraphBuilder")
        self.valid_node_list = valid_node_list
        self.add_self_loop = add_self_loop
        self.num_groups = num_groups

    def forward(self, states: np.ndarray) -> Tuple[ndarray, List[ndarray]]:
        bs = states.shape[0]
        num_nodes = len(self.valid_node_list)
        groups = np.random.randint(0, self.num_groups, size=len(self.valid_node_list))

        # Create adjacency mask based on groups
        group_mask = (groups[:, None] == groups[None, :]).astype(np.int64)

        if not self.add_self_loop:
            group_mask -= np.eye(num_nodes, dtype=np.int64)

        batch_adj_matrix = np.repeat(group_mask[np.newaxis], bs, axis=0)

        # Generate edge indices
        edge_indices = np.argwhere(group_mask).T

        batch_edge_indices = np.repeat(edge_indices[np.newaxis], bs, axis=0)

        return batch_adj_matrix, batch_edge_indices

    def reset(self):
        return self