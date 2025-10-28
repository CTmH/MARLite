import numpy as np
import networkx as nx
from typing import Union, Tuple, List
from numpy import ndarray
from copy  import deepcopy
from networkx.algorithms.community import greedy_modularity_communities
from scipy.spatial.distance import cdist
from concurrent.futures import ProcessPoolExecutor

from marlite.algorithm.graph_builder.graph_builder import GraphBuilder
from marlite.algorithm.graph_builder.graph_util import binary_to_decimal, build_partial_graph

class PartialGraphMAgentBuilder(GraphBuilder):

    def __init__(
            self,
            binary_agent_id_dim: list,
            agent_presence_dim: list,
            comm_distance: int,
            distance_metric: str = 'cityblock',
            n_workers: int = 8,
            n_subgraphs=2,
            valid_node_list: Union[list, None] = None,
            update_interval: int = 1,
            channel_first: bool = False):
        super().__init__()
        self.binary_agent_id_dim = binary_agent_id_dim
        self.agent_presence_dim = agent_presence_dim
        self.comm_distance = comm_distance
        self.distance_metric = distance_metric
        self.n_workers = n_workers
        self.n_subgraphs = n_subgraphs
        self.valid_node_list = valid_node_list

        self.update_interval = update_interval
        self.channel_first = channel_first
        self.step_counter = 0
        self.cached_adj_matrix = None
        self.cached_edge_indices = None

    @staticmethod
    def _process_batch(
            batch_state: np.ndarray,
            binary_agent_id_dim: list,
            agent_presence_dim: list,
            comm_distance: int,
            distance_metric: str,
            n_subgraphs: int,
            valid_node_list: Union[list, None] = None
    ):
        """Process a single batch item"""
        binary_agent_id = batch_state[:, :, binary_agent_id_dim]
        agent_positions = np.apply_along_axis(binary_to_decimal, -1, binary_agent_id).astype(np.int64)
        agent_presence = batch_state[:, :, agent_presence_dim]
        agent_presence = agent_presence.astype(np.int64)
        agent_presence = agent_presence.sum(axis=-1)
        agent_positions = agent_positions * agent_presence + agent_presence - np.ones_like(agent_presence)
        threshold = comm_distance

        # Collect valid element coordinates
        valid_elements = {val: (i, j) for i, row in enumerate(agent_positions)
                         for j, val in enumerate(row) if val >= 0}
        sorted_ids = np.array(sorted(valid_elements.keys()), dtype=np.int64)
        coords = np.array([valid_elements[k] for k in sorted_ids], dtype=np.int64)

        # Calculate distance matrix
        if len(coords) > 0:
            distances = cdist(coords, coords, metric=distance_metric)
            mask = (distances <= threshold) & np.triu(np.ones_like(distances, dtype=bool), k=1)
            rows, cols = np.where(mask)

            # Build adjacency matrix
            max_id = agent_positions.max()
            n = max_id + 1
            adj_matrix = np.zeros((n, n), dtype=np.int64)
            adj_matrix[sorted_ids[rows], sorted_ids[cols]] = 1
            adj_matrix[sorted_ids[cols], sorted_ids[rows]] = 1  # Symmetric connections

            # Generate edge index in COO format
            edge_index = np.vstack([sorted_ids[rows], sorted_ids[cols]]).astype(np.int64)
        else:
            adj_matrix = np.zeros((0, 0), dtype=np.int64)
            edge_index = np.zeros((2, 0), dtype=np.int64)

        if n_subgraphs > 1 and edge_index.shape[1] > 0:
            G = nx.Graph()
            edge_weights = 1.0 / distances[rows, cols]
            G.add_weighted_edges_from(zip(
                sorted_ids[rows],
                sorted_ids[cols],
                edge_weights
            ))
            # Run the greedy modularity community detection algorithm
            communities = list(greedy_modularity_communities(
                G=G,
                best_n=n_subgraphs,
                weight='weight'
            ))

            # Create a mapping from nodes to their communities
            node_community_map = {}
            for comm_id, comm in enumerate(communities):
                for node in comm:
                    node_community_map[node] = comm_id

            # Filter edges that connect nodes within the same community
            intra_edges = []
            for u, v in edge_index.T:
                if node_community_map.get(u) == node_community_map.get(v):
                    intra_edges.append([u, v])

            # Update the adjacency matrix and edge index with only intra-community connections
            edge_index = np.array(intra_edges).T if intra_edges else np.zeros((2,0), dtype=np.int64)
            adj_matrix = np.zeros_like(adj_matrix)
            adj_matrix[edge_index[0], edge_index[1]] = 1
            adj_matrix[edge_index[1], edge_index[0]] = 1  # Maintain symmetry

        if valid_node_list == None:
            return adj_matrix, edge_index

        # Filter nodes and edges based on valid_node_list
        valid_node_mask = np.isin(sorted_ids, valid_node_list)

        # Filter sorted_ids to only include valid nodes
        filtered_sorted_ids = sorted_ids[valid_node_mask]

        # Create a mapping from old node IDs to new compact IDs
        old_to_new_id_map = {old_id: new_id for new_id, old_id in enumerate(filtered_sorted_ids)}

        # Filter edge_index to only include edges between valid nodes
        valid_edge_mask = np.isin(edge_index, filtered_sorted_ids).all(axis=0)
        filtered_edge_index = edge_index[:, valid_edge_mask]

        # Remap edge_index to use the new compact IDs
        filtered_edge_index = np.vectorize(old_to_new_id_map.get)(filtered_edge_index)

        # Build the new adjacency matrix based on the filtered and remapped edges
        n_valid_nodes = len(filtered_sorted_ids)
        filtered_adj_matrix = np.zeros((n_valid_nodes, n_valid_nodes), dtype=np.int64)
        if filtered_edge_index.size > 0:
            filtered_adj_matrix[filtered_edge_index[0], filtered_edge_index[1]] = 1
            filtered_adj_matrix[filtered_edge_index[1], filtered_edge_index[0]] = 1  # Maintain symmetry

        return filtered_adj_matrix, filtered_edge_index

    def forward(self, state: ndarray) -> Tuple[ndarray, List[ndarray]]:
        if self.channel_first:
            state = np.transpose(state, (0, 2, 3, 1))
        bs = state.shape[0]
        if not self.training:
            self.step_counter += 1
            if (self.step_counter % self.update_interval != 0
                and self.cached_adj_matrix is not None
                and self.cached_edge_indices is not None):
                return deepcopy(self.cached_adj_matrix), deepcopy(self.cached_edge_indices)

        n_workers = min(bs, self.n_workers)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(
                self._process_batch,
                [state[b] for b in range(bs)],
                [self.binary_agent_id_dim] * bs,
                [self.agent_presence_dim] * bs,
                [self.comm_distance] * bs,
                [self.distance_metric] * bs,
                [self.n_subgraphs] * bs,
                [self.valid_node_list] * bs
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

# TODO
class TODOPartialGraphMAgentBuilder(GraphBuilder):

    def __init__(
            self,
            binary_agent_id_dim: list,
            agent_presence_dim: list,
            comm_distance: int,
            distance_metric: str = 'cityblock',
            n_workers: int = 8,
            n_subgraphs=2,
            valid_node_list: Union[list, None] = None,
            update_interval: int = 1,
            channel_first: bool = False):
        super().__init__()
        self.binary_agent_id_dim = binary_agent_id_dim
        self.agent_presence_dim = agent_presence_dim
        self.comm_distance = comm_distance
        self.distance_metric = distance_metric
        self.n_workers = n_workers
        self.n_subgraphs = n_subgraphs
        self.valid_node_list = valid_node_list

        self.update_interval = update_interval
        self.channel_first = channel_first
        self.step_counter = 0
        self.cached_adj_matrix = None
        self.cached_edge_indices = None

    def forward(self, state: ndarray) -> Tuple[ndarray, List[ndarray]]:
        if self.channel_first:
            state = np.transpose(state, (0, 2, 3, 1))
        bs = state.shape[0]
        if not self.training:
            self.step_counter += 1
            if (self.step_counter % self.update_interval != 0
                and self.cached_adj_matrix is not None
                and self.cached_edge_indices is not None):
                return deepcopy(self.cached_adj_matrix), deepcopy(self.cached_edge_indices)

        n_workers = min(bs, self.n_workers)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(
                self._process_single_batch,
                [state[b] for b in range(bs)]
            ))

        batch_adj_matrix, batch_edge_indices = zip(*results)
        batch_adj_matrix = np.array(batch_adj_matrix)
        batch_edge_indices = list(batch_edge_indices)

        if not self.training:
            self.cached_adj_matrix = batch_adj_matrix
            self.cached_edge_indices = batch_edge_indices

        return batch_adj_matrix, batch_edge_indices

    def _process_single_batch(self, batch_state: np.ndarray):
        """Process a single batch item for PartialGraphMAgentBuilder"""
        binary_agent_id = batch_state[:, self.binary_agent_id_dim]
        agent_positions = np.apply_along_axis(binary_to_decimal, -1, binary_agent_id).astype(np.int64)
        agent_presence = batch_state[:, self.agent_presence_dim]
        agent_presence = agent_presence.astype(np.int64)
        agent_presence = agent_presence.sum(axis=-1)
        agent_positions = agent_positions * agent_presence + agent_presence - np.ones_like(agent_presence)

        # Convert to coordinates format expected by _process_batch
        coords = agent_positions

        return build_partial_graph(
            coords=coords,
            comm_distance=self.comm_distance,
            distance_metric=self.distance_metric,
            n_subgraphs=self.n_subgraphs,
            valid_node_list=self.valid_node_list
        )

    def reset(self):
        self.step_counter = 0
        self.cached_adj_matrix = None
        self.cached_edge_indices = None
        return self

# TODO
class PartialGraphVectorBuilder(GraphBuilder):
    """
    Graph builder for vectorized states with shape (batch_size, num_agents, feature_dim).
    Extracts agent coordinates from specified positions in the feature dimension.
    """

    def __init__(
            self,
            coord_dim: Tuple[int, int],
            hp_dim: int,
            comm_distance: int,
            distance_metric: str = 'cityblock',
            n_workers: int = 8,
            n_subgraphs=2,
            valid_node_list: Union[list, None] = None,
            update_interval: int = 1):
        """
        Initialize the PartialGraphVectorBuilder.

        Args:
            coord_dim: 2-tuple specifying the positions in feature_dim that represent agent coordinates
            comm_distance: Communication distance threshold
            distance_metric: Distance metric for calculating distances between agents
            n_workers: Number of parallel workers for batch processing
            n_subgraphs: Number of subgraphs for community detection
            valid_node_list: List of valid node IDs to include in the graph
            update_interval: Interval for updating cached graphs during inference
        """
        super().__init__()
        self.coord_dim = coord_dim
        self.hp_dim = hp_dim
        self.comm_distance = comm_distance
        self.distance_metric = distance_metric
        self.n_workers = n_workers
        self.n_subgraphs = n_subgraphs
        self.valid_node_list = valid_node_list

        self.update_interval = update_interval
        self.step_counter = 0
        self.cached_adj_matrix = None
        self.cached_edge_indices = None

    def forward(self, state: ndarray) -> Tuple[ndarray, List[ndarray]]:
        """
        Build graphs from vectorized state.

        Args:
            state: Input state with shape (batch_size, num_agents, feature_dim)

        Returns:
            Tuple of (batch_adjacency_matrix, list_of_edge_indices)
        """
        bs = state.shape[0]
        if not self.training:
            self.step_counter += 1
            if (self.step_counter % self.update_interval != 0
                and self.cached_adj_matrix is not None
                and self.cached_edge_indices is not None):
                return deepcopy(self.cached_adj_matrix), deepcopy(self.cached_edge_indices)

        n_workers = min(bs, self.n_workers)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(
                self._process_single_batch,
                [state[b] for b in range(bs)]
            ))

        batch_adj_matrix, batch_edge_indices = zip(*results)
        batch_adj_matrix = np.array(batch_adj_matrix)
        batch_edge_indices = list(batch_edge_indices)

        if not self.training:
            self.cached_adj_matrix = batch_adj_matrix
            self.cached_edge_indices = batch_edge_indices

        return batch_adj_matrix, batch_edge_indices

    def _process_single_batch(self, state: np.ndarray):
        """Process a single batch item for PartialGraphVectorBuilder"""
        # Extract coordinates from the specified feature dimensions
        coords = state[:, list(self.coord_dim)]
        hp = state[:, list(self.hp_dim)]
        alive_agent_list = np.where(hp > 0)

        return build_partial_graph(
            coords=coords,
            comm_distance=self.comm_distance,
            distance_metric=self.distance_metric,
            n_subgraphs=self.n_subgraphs,
            valid_node_list=self.valid_node_list
        )

    def reset(self):
        """Reset the builder's internal state and cache"""
        self.step_counter = 0
        self.cached_adj_matrix = None
        self.cached_edge_indices = None
        return self