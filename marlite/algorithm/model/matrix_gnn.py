import torch
import torch.nn as nn
from torch import Tensor

class MatrixGCNModel(nn.Module):
    """
    A graph neural network designed for multi-agent reinforcement learning.

    This module aggregates information from neighboring agents using a weighted adjacency matrix
    derived from soft attention mechanisms (e.g., from G2ANet). It performs message passing
    through multiple layers to update each agent's feature representation.

    Args:
        input_dim (int): Dimension of input node features.
        hidden_dim (int): Hidden layer dimension for intermediate representations.
        output_dim (int): Dimension of final output features per agent.
        num_layers (int): Number of stacked graph convolutional layers (default: 2).

    Input:
        x: (batch_size, n_agents, input_dim) - Node features for all agents
        weight_adj_matrix: (batch_size, n_agents, n_agents) - Weighted adjacency matrix
                          (typically from soft attention in G2ANet)

    Output:
        out: (batch_size, n_agents, output_dim) - Updated node features after message passing
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers

        # Stacked linear layers for message passing
        self.layers = nn.ModuleList([
            nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])

        # Final output projection
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Activation function (ReLU is commonly used in GNNs)
        self.activation = nn.GELU()

    def forward(self, x: Tensor, weight_adj_matrix: Tensor) -> Tensor:
        """
        Forward pass of the MultiAgentAttentionGNN.

        Args:
            x: (B, N, D_hidden) - Input node features for all agents in the batch
            weight_adj_matrix: (B, N, N) - Weighted adjacency matrix (soft attention weights)

        Returns:
            out: (B, N, D_out) - Updated node features after graph convolution
        """
        # Use the provided weight_adj_matrix directly (assumed normalized per row)

        # Perform message passing through multiple layers
        h = x
        for i in range(self.num_layers):
            # Graph convolution: h = A @ W @ h
            h = torch.matmul(weight_adj_matrix, h)  # (B, N, D_in) -> (B, N, D_in)
            h = self.layers[i](h)                # (B, N, D_in) -> (B, N, D_hid)
            h = self.activation(h)               # Apply non-linearity

        # Final output projection
        out = self.output_layer(h)  # (B, N, D_hid) -> (B, N, D_out)

        return out