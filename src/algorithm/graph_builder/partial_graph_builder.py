import numpy as np
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from scipy.spatial.distance import cdist
from .magent_graph_builder import MagentGraphBuilder
from .build_graph import binary_to_decimal
from concurrent.futures import ProcessPoolExecutor, as_completed

class PartialGraphMagentBuilder(MagentGraphBuilder):

    def __init__(
            self,
            binary_agent_id_dim: list,
            agent_presence_dim: list,
            comm_distance: int,
            distance_metric: str = 'cityblock',
            n_workers: int = 8,
            n_subgraphs=2):
        super().__init__(
            binary_agent_id_dim,
            agent_presence_dim,
            comm_distance,
            distance_metric)
        self.binary_agent_id_dim = binary_agent_id_dim
        self.agent_presence_dim = agent_presence_dim
        self.comm_distance = comm_distance
        self.distance_metric = distance_metric
        self.n_workers = n_workers
        self.n_subgraphs = n_subgraphs

    @staticmethod
    def _process_batch(
            batch_state,
            binary_agent_id_dim,
            agent_presence_dim,
            comm_distance,
            distance_metric,
            n_subgraphs
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

        return adj_matrix, edge_index
        
    def forward(self, state):
        batch_adj_matrices = []
        batch_edge_indices = []
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [executor.submit(
                self._process_batch,
                state[b],
                self.binary_agent_id_dim,
                self.agent_presence_dim,
                self.comm_distance,
                self.distance_metric,
                self.n_subgraphs
            ) for b in range(state.shape[0])]
            
            for future in as_completed(futures):
                adj_matrix, edge_index = future.result()
                batch_adj_matrices.append(adj_matrix)
                batch_edge_indices.append(edge_index)
        
        return np.array(batch_adj_matrices), batch_edge_indices