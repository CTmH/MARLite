import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from typing import Tuple, List
from marlite.algorithm.graph_builder.graph_builder import GraphBuilder

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

class G2ANetAttention(nn.Module):

    def __init__(
            self,
            n_agents: int,
            add_self_loop: bool = False,
            input_dim: int = None,
            hidden_dim: int = 64):
        super().__init__()
        self.add_self_loop = add_self_loop
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_agents = n_agents

        # Hard attention (using bidirectional LSTM)
        self.hard_attention = nn.LSTM(
            input_size=2 * input_dim,
            hidden_size=hidden_dim,
            bidirectional=True,
            batch_first=True
        )
        self.hard_attention_fc = nn.Linear(2 * hidden_dim, 1)

        # Soft attention
        self.W_q = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(input_dim, hidden_dim, bias=False)

    def forward(self, encoded_obs: Tensor, alive_mask: Tensor=None) -> Tuple[Tensor, Tensor]:
        # observations shape: (batch_size, n_agents, obs_dim)
        batch_size, n_agents, obs_dim = encoded_obs.shape

        # Prepare pairs for hard attention: [h_i, h_j] for all i,j
        h_i = encoded_obs.unsqueeze(2).expand(-1, -1, n_agents, -1)  # (batch, n, n, h)
        h_j = encoded_obs.unsqueeze(1).expand(-1, n_agents, -1, -1)  # (batch, n, n, h)
        pairs = torch.cat([h_i, h_j], dim=-1)  # (batch, n, n, 2h)

        # Hard attention processing
        pair_seq = pairs.view(batch_size * n_agents, n_agents, -1)
        lstm_out, _ = self.hard_attention(pair_seq)  # (batch*n, n, 2h)
        hard_scores = self.hard_attention_fc(lstm_out).squeeze(-1)  # (batch*n, n)
        hard_scores = hard_scores.view(batch_size, n_agents, n_agents)  # (batch, n, n)

        # Apply Gumbel-Softmax for discrete edge selection
        if self.training:
            # During training: Gumbel-Softmax
            hard_scores_flat = hard_scores.view(-1, n_agents)  # (B*N, N)
            hard_attention_weights = F.gumbel_softmax(hard_scores_flat, tau=1.0, hard=False, dim=-1)
            hard_attention_weights = hard_attention_weights.view(batch_size, n_agents, n_agents)
        else:
            # During evaluation: argmax
            hard_attention_weights = torch.zeros_like(hard_scores)
            max_indices = torch.argmax(hard_scores, dim=-1)
            for i in range(batch_size):
                for j in range(n_agents):
                    hard_attention_weights[i, j, max_indices[i, j]] = 1.0
            # Vectorized version using scatter_
            hard_attention_weights = torch.zeros_like(hard_scores)
            max_indices = torch.argmax(hard_scores, dim=-1)
            hard_attention_weights.scatter_(2, max_indices.unsqueeze(2), 1.0)

        # Add self-loops if requested
        if self.add_self_loop:
            identity = torch.eye(n_agents, device=hard_attention_weights.device)
            hard_attention_weights = hard_attention_weights + identity.unsqueeze(0)

        # Soft attention - compute attention weights
        Q = self.W_q(encoded_obs)  # (batch, n, h)
        K = self.W_k(encoded_obs)  # (batch, n, h)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(1, 2))  # (batch, n, n)
        scores = scores * hard_attention_weights  # apply hard attention mask

        # Apply softmax
        soft_attention_weights = F.softmax(scores, dim=-1)  # (batch, n, n)

        return hard_attention_weights, soft_attention_weights