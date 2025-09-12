import torch
import torch.nn as nn
from marlite.algorithm.model.masked_model import MaskedModel

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, batch_first=True):
        super(SelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=batch_first)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        attn_output, attn_output_weights = self.attention(x, x, x,
                                                          attn_mask=attn_mask,
                                                          key_padding_mask=key_padding_mask)
        return attn_output

class SelfAttentionLearnablePE(MaskedModel):
    def __init__(self, embed_dim, num_heads, max_seq_len=100, dropout=0.1, batch_first=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=batch_first
        )

        # Fixed sinusoidal position encoding (not learnable)
        self.position_embed = nn.Parameter(_generate_sinusoidal_encoding(max_seq_len, embed_dim))

    def forward(self, x: torch.Tensor, attn_mask=None, key_padding_mask=None):
        """
        x: [batch_size, num_agents, input_dim]
        returns: [batch_size, num_agents, embedding_dim] (unpooled attention output)
        """

        # Add learnable position encoding: [B,N,E] + [1,1,E] -> [B,N,E]
        x = x + self.position_embed

        # Self-attention computation with Q=K=V=x
        # Returns attended output while preserving sequence dimension
        attn_output, _ = self.attention(
            query=x,
            key=x,
            value=x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask
        )

        return attn_output  # Return full sequence [B,N,E]

class SelfAttentionFixedPE(nn.Module):
    def __init__(self, embed_dim, num_heads, max_seq_len=100, dropout=0.1, batch_first=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=batch_first
        )

        # Fixed sinusoidal position encoding (not learnable)
        # Generates positional encodings using sine and cosine functions of different frequencies
        self.register_buffer('position_embed', _generate_sinusoidal_encoding(max_seq_len, embed_dim).unsqueeze(0))

    def forward(self, x: torch.Tensor, attn_mask=None, key_padding_mask=None):
        """
        x: [batch_size, num_agents, input_dim]
        returns: [batch_size, num_agents, embedding_dim] (unpooled attention output)
        """
        seq_len = x.size(1)

        # Add fixed position encoding: [B,N,E] + [1,N,E] -> [B,N,E]
        # Use only the first 'seq_len' positions of precomputed encoding
        x = x + self.position_embed[:, :seq_len, :]

        # Self-attention computation with Q=K=V=x
        # Returns attended output while preserving sequence dimension
        attn_output, _ = self.attention(
            query=x,
            key=x,
            value=x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask
        )

        return attn_output  # Return full sequence [B,N,E]

def _generate_sinusoidal_encoding(max_len, d_model) -> torch.Tensor:
    """Generate fixed sinusoidal position encoding matrix"""
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    # Add batch dimension: [max_len, d_model] -> [1, max_len, d_model]
    return pe