import torch
import numpy as np
from torch import Tensor
from typing import Tuple, List
from marlite.algorithm.graph_builder.graph_builder import GraphBuilder
from marlite.algorithm.model.g2anet_attention import G2ANetAttention

class G2ANetGraphBuilder(GraphBuilder):

    def __init__(
            self,
            n_agents: int,
            add_self_loop: bool = False,
            input_dim: int = None,
            hidden_dim: int = 64):
        super().__init__()
        self.attention_model = G2ANetAttention(
            n_agents,
            add_self_loop,
            input_dim,
            hidden_dim
        )

    def forward(self, encoded_obs: Tensor, alive_mask: Tensor=None) -> Tuple[Tensor, List[np.ndarray]]:
        hard_attention_weights, soft_attention_weights = self.attention_model(encoded_obs)
        weight_adj_matrix = hard_attention_weights * soft_attention_weights

        # Generate edge_indices using torch.nonzero for efficiency
        batch_size, n_nodes, _ = hard_attention_weights.shape
        edge_indices = []

        # Use torch.nonzero to find all (batch, src, dst) indices where hard_attention_weights > 0
        adj_matrix = hard_attention_weights.cpu()

        # Generate edge_indices from hard_attention_weights
        batch_size, n_nodes, _ = adj_matrix.shape
        edge_indices = []
        for b in range(batch_size):
            adj_b = adj_matrix[b]  # shape: (n_nodes, n_nodes)
            indices = torch.nonzero(adj_b, as_tuple=False)  # shape: (num_edges, 2)
            edge_indices.append(indices.permute(1, 0).numpy())

        return weight_adj_matrix, edge_indices
