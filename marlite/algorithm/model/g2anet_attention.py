import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple

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

        if alive_mask is not None:
            encoded_obs = encoded_obs.masked_fill(
                ~alive_mask.to(device=encoded_obs.device, dtype=torch.bool).unsqueeze(-1), 0.
            )

        # Prepare pairs for hard attention: [h_i, h_j] for all i,j
        h_i = encoded_obs.unsqueeze(2).expand(-1, -1, n_agents, -1)  # (batch, n, n, h)
        h_j = encoded_obs.unsqueeze(1).expand(-1, n_agents, -1, -1)  # (batch, n, n, h)
        pairs = torch.cat([h_i, h_j], dim=-1)  # (batch, n, n, 2h)

        # Hard attention processing
        pair_seq = pairs.view(batch_size * n_agents, n_agents, -1)
        lstm_out, _ = self.hard_attention(pair_seq)  # (batch*n, n, 2h)
        hard_scores = self.hard_attention_fc(lstm_out).squeeze(-1)  # (batch*n, n)
        hard_scores = hard_scores.view(batch_size, n_agents, n_agents)  # (batch, n, n)

        # A deterministic Bernoulli gate is used in BOTH collection and learning.
        # Its straight-through derivative trains graph selection without injecting
        # unrecorded Gumbel noise into PPO's importance-sampling denominator.
        probability = hard_scores.sigmoid()
        binary_gate = (probability >= 0.5).to(probability.dtype)
        gates = binary_gate + (probability - probability.detach())

        live = (torch.ones((batch_size, n_agents), device=encoded_obs.device, dtype=torch.bool)
                if alive_mask is None else alive_mask.to(device=encoded_obs.device, dtype=torch.bool))
        valid = live.unsqueeze(2) & live.unsqueeze(1)
        diagonal = torch.eye(n_agents, device=encoded_obs.device, dtype=torch.bool).unsqueeze(0)
        gates = gates.masked_fill(diagonal, 1.0 if self.add_self_loop else 0.0)
        gates = gates * valid

        query = self.W_q(encoded_obs)
        key = self.W_k(encoded_obs)
        scores = torch.matmul(query, key.transpose(1, 2)) / self.hidden_dim ** 0.5
        valid = valid & (torch.ones_like(diagonal) if self.add_self_loop else ~diagonal)
        # Empty rows (dead or isolated agents) remain finite and send no messages.
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        base_weights = F.softmax(scores, dim=-1) * valid
        mass = (gates * base_weights).sum(-1, keepdim=True)
        denominator = torch.where(mass.detach() > 0, mass, torch.ones_like(mass))
        # GraphBuilder multiplies these factors. Keeping a finite soft factor on
        # unselected edges also lets the straight-through gate learn to open them.
        soft_weights = base_weights / denominator.clamp_min(1e-8)
        return gates, soft_weights
