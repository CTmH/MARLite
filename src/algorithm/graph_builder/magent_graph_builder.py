from concurrent.futures import ProcessPoolExecutor
import numpy as np
from typing import Union, Tuple, List
from numpy import ndarray
from scipy.spatial.distance import cdist
from copy import deepcopy
from .graph_builder import GraphBuilder
from .build_graph import binary_to_decimal

class MagentGraphBuilder(GraphBuilder):

    def __init__(
            self,
            binary_agent_id_dim: list,
            agent_presence_dim: list,
            comm_distance: int,
            distance_metric: str = 'cityblock',
            n_workers: int = 8,
            valid_node_list: Union[list, None] = None, # Suggestion: Add valid_node_list, otherwise isolated nodes will be ignored in the node mapping.
            update_interval: int = 1):
        super().__init__()
        self.binary_agent_id_dim = binary_agent_id_dim
        self.agent_presence_dim = agent_presence_dim
        self.comm_distance = comm_distance
        self.distance_metric = distance_metric
        self.n_workers = n_workers
        self.valid_node_list = valid_node_list

        self.update_interval = update_interval
        self.step_counter = 0
        self.cached_adj_matrix = None
        self.cached_edge_indices = None

    @staticmethod
    def _process_batch(
            batch_state,
            binary_agent_id_dim: list,
            agent_presence_dim: list,
            comm_distance: int,
            distance_metric: str,
            valid_node_list: Union[list, None] = None):
        """Process a single batch item"""
        binary_agent_id = batch_state[:, :, binary_agent_id_dim]
        agent_positions = np.apply_along_axis(binary_to_decimal, -1, binary_agent_id).astype(np.int64)
        agent_presence = batch_state[:, :, agent_presence_dim]
        agent_presence = agent_presence.astype(np.int64)
        agent_presence = agent_presence.sum(axis=-1)
        agent_positions = agent_positions * agent_presence + agent_presence - np.ones_like(agent_presence)
        threshold = comm_distance

        valid_elements = {val: (i, j) for i, row in enumerate(agent_positions)
                         for j, val in enumerate(row) if val >= 0}
        sorted_ids = sorted(valid_elements.keys())
        sorted_ids = np.array(sorted_ids, dtype=np.int64)
        coords = np.array([valid_elements[k] for k in sorted_ids], dtype=np.int64)

        # Map nodes in advance
        if valid_node_list is None:
            valid_node_list = sorted_ids.tolist()
        node_mapping = {old_id: new_id for new_id, old_id in enumerate(valid_node_list)}
        map_func = np.vectorize(lambda x: node_mapping.get(x))

        if len(coords) > 0:
            distances = cdist(coords, coords, metric=distance_metric)
            mask = (distances <= threshold) & (np.triu(np.ones_like(distances, dtype=bool), k=1))
            rows, cols = np.where(mask)

            # Use mapped IDs to build edges
            mapped_rows = map_func(sorted_ids[rows])
            mapped_cols = map_func(sorted_ids[cols])
            edge_index = np.vstack([mapped_rows, mapped_cols]).astype(np.int64)

            # Build adjacency matrix
            n = len(valid_node_list)
            adj_matrix = np.zeros((n, n), dtype=np.int64)
            adj_matrix[mapped_rows, mapped_cols] = 1
            adj_matrix[mapped_cols, mapped_rows] = 1
        else:
            raise ValueError("No valid nodes found in the state.")

        return adj_matrix, edge_index

    def forward(self, state: ndarray) -> Tuple[ndarray, List[ndarray]]:

        bs = state.shape[0]
        if not self.training:
            self.step_counter += 1
            if (self.step_counter % self.update_interval != 0
                and self.cached_adj_matrix is not None
                and self.cached_edge_indices is not None):
                return deepcopy(self.cached_adj_matrix), deepcopy(self.cached_edge_indices)

        n_workers = min(bs, self.n_workers)
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            results = list(executor.map(
                self._process_batch,
                [state[b] for b in range(bs)],
                [self.binary_agent_id_dim] * bs,
                [self.agent_presence_dim] * bs,
                [self.comm_distance] * bs,
                [self.distance_metric] * bs
            ))

        batch_adj_matrix, batch_edge_indices = zip(*results)
        batch_adj_matrix = np.array(batch_adj_matrix)
        batch_edge_indices = list(batch_edge_indices)

        if not self.training:
            self.cached_adj_matrix = batch_adj_matrix
            self.cached_edge_indices = batch_edge_indices

        return batch_adj_matrix, batch_edge_indices

    def reset(self):
        self.step_counter = 0
        self.cached_adj_matrix = None
        self.cached_edge_indices = None
        return self