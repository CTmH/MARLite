import torch
import torch.nn as nn
import torch.nn.functional as F
from marlite.algorithm.model.masked_model import MaskedModel
from marlite.algorithm.model.self_attention import SelfAttentionLearnablePE, SelfAttentionFixedPE
from marlite.algorithm.model import AttentionModel

class ResAttentionStateEncoder(MaskedModel):
    def __init__(self, input_dim, embed_dim, num_heads, max_seq_len, dropout=0.1):
        super(ResAttentionStateEncoder, self).__init__()
        self.linear = nn.Linear(input_dim, embed_dim)
        self.attention = SelfAttentionFixedPE(embed_dim, num_heads, max_seq_len=max_seq_len, dropout=dropout, batch_first=True)
        self.attention_weights = nn.Linear(embed_dim, 1)

        # Layer normalization for stability
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, alive_mask:torch.Tensor=None):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len(n_agents), input_dim]
            attn_mask: Attention mask for causal masking [batch_size, seq_len(n_agents)]
            key_padding_mask: Mask for padding tokens

        Returns:
            Global embedding of shape [batch_size, embed_dim]
        """
        # When all agents are dead, to avoid undefined results from Softmax.
        if alive_mask != None:
            all_false_rows = ~alive_mask.any(dim=1)
            not_alive_mask = ~torch.where(all_false_rows.unsqueeze(1), True, alive_mask)
        else:
            not_alive_mask = torch.zeros((x.size(0),x.size(1)), dtype=torch.bool)

        x = self.linear(x)  # [B, N, D]

        # Self-attention block (Pre-LN)
        x_norm = self.norm1(x)
        attn_out = self.attention(x_norm, key_padding_mask=not_alive_mask)
        x = x + self.dropout(attn_out)

        # FFN block (Pre-LN)
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.dropout(ffn_out)

        # Weighted pooling with learned attention
        weights = self.attention_weights(self.norm3(attn_out)).squeeze(-1)  # [B, N]
        weights = weights.masked_fill(not_alive_mask, -torch.inf)
        weights = torch.softmax(weights, dim=-1)  # [B, N]
        x = torch.sum(x * weights.unsqueeze(-1), dim=1)  # [B, D]

        return x


class ResAttentionObsEncoder(AttentionModel):

    def __init__(self, input_dim, output_dim, embed_dim, num_heads, max_seq_len, dropout=0.1):
        super(ResAttentionObsEncoder, self).__init__()
        self.linear1 = nn.Linear(input_dim, embed_dim)
        self.linear2 = nn.Linear(embed_dim, output_dim)
        self.attention = SelfAttentionFixedPE(embed_dim, num_heads, max_seq_len=max_seq_len, dropout=dropout, batch_first=True)
        self.attention_weights = nn.Linear(embed_dim, 1)

        # Layer normalization for stability
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, padding_mask:torch.Tensor=None):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len(n_agents), input_dim]
            attn_mask: Attention mask for causal masking [batch_size, seq_len(n_agents)]
            key_padding_mask: Mask for padding tokens

        Returns:
            Global embedding of shape [batch_size, embed_dim]
        """
        # When all agents are dead, to avoid undefined results from Softmax.
        """
        if padding_mask != None:
            all_false_rows = ~padding_mask.any(dim=1)
            not_alive_mask = ~torch.where(all_false_rows.unsqueeze(1), True, padding_mask)
        else:
            #not_alive_mask = torch.zeros((x.size(0),x.size(1)), dtype=torch.bool, device=x.device)
            not_alive_mask = create_key_padding_mask(x).to(x.device)
        """
        x = self.linear1(x)  # [B, N, D]

        # Self-attention block (Pre-LN)
        x_norm = self.norm1(x)
        attn_out = self.attention(x_norm, key_padding_mask=padding_mask)
        x = x + self.dropout(attn_out)

        # FFN block (Pre-LN)
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.dropout(ffn_out)

        # Weighted pooling with learned attention
        weights = self.attention_weights(self.norm3(attn_out)).squeeze(-1)  # [B, N]
        weights = weights.masked_fill(padding_mask, -torch.inf)
        weights = torch.softmax(weights, dim=-1)  # [B, N]
        x = torch.sum(x * weights.unsqueeze(-1), dim=1)  # [B, D]

        x = self.linear2(F.gelu(x))

        return x

def create_key_padding_mask(tensor: torch.Tensor):
    """
    Creates a key padding mask for attention mechanisms based on whether each time step
    in the tensor is a zero vector (i.e., all features are 0).

    Args:
        tensor (torch.Tensor): Input tensor of shape [batch_size, time_step, feature_dim]

    Returns:
        torch.BoolTensor: Key padding mask of shape [batch_size, time_step], where True
                         indicates that the corresponding time step should be masked.
                         Note: If an entire batch has all zero vectors across all time steps,
                         only the last time step of that batch is marked as True.
    """
    with torch.no_grad():
        # Check if each time step contains any non-zero values
        is_nonzero = tensor.abs().sum(dim=-1) > 0  # Shape: [batch_size, time_step]

        # Create initial mask: True means "this time step should be masked"
        key_padding_mask = ~is_nonzero  # Shape: [batch_size, time_step]

        # Special case: if a batch has all zero vectors, mask only the last time step
        all_zeros_in_batch = (~is_nonzero).all(dim=1)  # Shape: [batch_size]

        # Set the last time step to True for batches that are entirely zero
        key_padding_mask[all_zeros_in_batch, -1] = False

    return key_padding_mask