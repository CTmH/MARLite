import numpy as np
import torch
from typing import Tuple, List, Union
from marlite.algorithm.graph_builder.graph_builder import GraphBuilder


class FixedEdgeGraphBuilder(GraphBuilder):
    """
    GraphBuilder that uses a fixed edge_indices to generate adjacency matrices.
    
    This builder initializes with a fixed edge_indices (adjacency list) and 
    generates batch_adj_matrix and batch_edge_indices based on this fixed structure.
    
    Args:
        edge_indices (Union[np.ndarray, List[List[int]]]): Fixed edge indices in COO format.
            Shape should be (2, num_edges) where edge_indices[0] are source nodes
            and edge_indices[1] are target nodes.
        num_nodes (int, optional): Number of nodes in the graph. If not provided,
            it will be inferred from edge_indices.
        add_self_loop (bool, optional): Whether to add self-loops to all nodes.
            Default is True.
        symmetric (bool, optional): Whether to make the adjacency matrix symmetric.
            If True, for each edge (i, j), edge (j, i) will also be added.
            Default is True.
    """

    def __init__(
            self,
            edge_indices: Union[np.ndarray, List[List[int]]],
            num_nodes: Union[int, None] = None,
            add_self_loop: bool = True,
            symmetric: bool = True):
        super().__init__()
        
        # Convert edge_indices to numpy array if it's a list
        if isinstance(edge_indices, list):
            edge_indices = np.array(edge_indices, dtype=np.int64)
        
        # Ensure edge_indices has correct shape
        if edge_indices.ndim == 2 and edge_indices.shape[0] != 2:
            # If shape is (num_edges, 2), transpose it to (2, num_edges)
            if edge_indices.shape[1] == 2:
                edge_indices = edge_indices.T
            else:
                raise ValueError(
                    f"edge_indices should have shape (2, num_edges) or (num_edges, 2), "
                    f"got {edge_indices.shape}"
                )
        
        self.edge_indices = edge_indices.astype(np.int64)
        
        # Determine number of nodes
        if num_nodes is None:
            self.num_nodes = int(np.max(self.edge_indices)) + 1 if self.edge_indices.size > 0 else 0
        else:
            self.num_nodes = num_nodes
        
        self.add_self_loop = add_self_loop
        self.symmetric = symmetric
        
        # Pre-compute the adjacency matrix for single graph
        self._precompute_adj_matrix()
    
    def _precompute_adj_matrix(self):
        """Pre-compute the adjacency matrix based on edge_indices."""
        if self.num_nodes == 0:
            self.adj_matrix = np.zeros((0, 0), dtype=np.int64)
            return
        
        # Initialize adjacency matrix
        self.adj_matrix = np.zeros((self.num_nodes, self.num_nodes), dtype=np.int64)
        
        # Add edges from edge_indices
        if self.edge_indices.size > 0:
            src_nodes = self.edge_indices[0]
            dst_nodes = self.edge_indices[1]
            self.adj_matrix[src_nodes, dst_nodes] = 1
            
            # Make symmetric if requested
            if self.symmetric:
                self.adj_matrix[dst_nodes, src_nodes] = 1
        
        # Add self-loops if requested
        if self.add_self_loop:
            np.fill_diagonal(self.adj_matrix, 1)
    
    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, List[np.ndarray]]:
        """
        Generate batch adjacency matrix and edge indices based on fixed edge_indices.
        
        Args:
            states (torch.Tensor): Input states with shape (batch_size, ...).
                The content of states is ignored, only batch_size is used.
        
        Returns:
            Tuple[torch.Tensor, List[np.ndarray]]: 
                - batch_adj_matrix: Shape (batch_size, num_nodes, num_nodes)
                - batch_edge_indices: List of edge indices arrays for each batch
        """
        bs = states.shape[0]
        
        # Create batch adjacency matrix by repeating the pre-computed adj_matrix
        batch_adj_matrix = np.repeat(self.adj_matrix[np.newaxis], bs, axis=0)
        
        # Create batch edge indices
        batch_edge_indices = []
        for _ in range(bs):
            # For each batch, we need to get the edge indices from the adjacency matrix
            if self.edge_indices.size > 0:
                # Use the original edge_indices
                batch_edge_indices.append(self.edge_indices.copy())
            else:
                # If no edges, return empty array
                batch_edge_indices.append(np.zeros((2, 0), dtype=np.int64))
        
        return torch.from_numpy(batch_adj_matrix), batch_edge_indices
    
    def reset(self):
        """Reset the builder (no-op for fixed edge builder)."""
        return self
    
    def get_edge_indices(self) -> np.ndarray:
        """Get the fixed edge indices."""
        return self.edge_indices.copy()
    
    def get_adj_matrix(self) -> np.ndarray:
        """Get the pre-computed adjacency matrix."""
        return self.adj_matrix.copy()
    
    def get_num_nodes(self) -> int:
        """Get the number of nodes."""
        return self.num_nodes