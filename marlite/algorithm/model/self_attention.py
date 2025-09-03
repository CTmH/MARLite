import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, batch_first=True):
        super(SelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=batch_first)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        attn_output, attn_output_weights = self.attention(x, x, x,
                                                          attn_mask=attn_mask,
                                                          key_padding_mask=key_padding_mask)
        return attn_output