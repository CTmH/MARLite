import numpy as np
from scipy.spatial.distance import cdist
from .graph_builder import GraphBuilder
from .build_graph import binary_to_decimal
from concurrent.futures import ThreadPoolExecutor, as_completed

class MagentGraphBuilder(GraphBuilder):

    def __init__(
            self,
            binary_agent_id_dim: list,
            agent_presence_dim: list,
            comm_distance: int,
            distance_metric: str = 'cityblock'):
        super().__init__()
        self.binary_agent_id_dim = binary_agent_id_dim
        self.agent_presence_dim = agent_presence_dim
        self.comm_distance = comm_distance
        self.distance_metric = distance_metric
    
    def _process_batch(self, batch_state):
        """Process a single batch item"""
        binary_agent_id = batch_state[:, :, self.binary_agent_id_dim]
        agent_positions = np.apply_along_axis(binary_to_decimal, -1, binary_agent_id).astype(np.int64)
        agent_presence = batch_state[:, :, self.agent_presence_dim]
        agent_presence = agent_presence.astype(np.int64)
        agent_presence = agent_presence.sum(axis=-1)
        agent_positions = agent_positions + agent_presence - 1
        threshold = self.comm_distance

        # Collect valid element coordinates
        valid_elements = {val: (i, j) for i, row in enumerate(agent_positions) 
                         for j, val in enumerate(row) if val >= 0}
        sorted_ids = sorted(valid_elements.keys())
        sorted_ids = np.array(sorted_ids, dtype=np.int64)
        coords = np.array([valid_elements[k] for k in sorted_ids], dtype=np.int64)
        
        # Calculate distance matrix
        if len(coords) > 0:
            distances = cdist(coords, coords, metric=self.distance_metric)
            mask = (distances <= threshold) & (np.triu(np.ones_like(distances, dtype=bool), k=1))
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
        
        return adj_matrix, edge_index
        
    def forward(self, state):
        batch_adj_matrices = []
        batch_edge_indices = []
        
        # Use thread pool to parallel process each batch
        with ThreadPoolExecutor() as executor:
            futures = []
            for b in range(state.shape[0]):
                # Submit each batch item for processing
                future = executor.submit(self._process_batch, state[b])
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                adj_matrix, edge_index = future.result()
                batch_adj_matrices.append(adj_matrix)
                batch_edge_indices.append(edge_index)
        
        return np.array(batch_adj_matrices), batch_edge_indices