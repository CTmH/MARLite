"""Scalar message quantization and permutation-invariant group attention."""

import torch
from torch import nn


class ScalarMessageCodec(nn.Module):
    """Hard scalar codes in the forward pass, straight-through gradients in training.

    Replay codes are unpacked uint8 values; pack/unpack define wire payloads.
    """

    def __init__(self, hidden_dim, message_dim=32, bits=4, *, encoder=None, decoder=None):
        super().__init__()
        if bits not in (1, 2, 4, 8) or message_dim <= 0 or message_dim * bits % 8:
            raise ValueError(
                "Messages must contain a whole number of bytes; bits must divide 8"
            )
        self.bits, self.message_dim = bits, message_dim
        self.encoder = encoder if encoder is not None else nn.Linear(hidden_dim, message_dim)
        self.decoder = decoder if decoder is not None else nn.Sequential(
            nn.Linear(message_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.register_buffer("_decode_dtype", torch.empty(0), persistent=False)

    def forward(self, features):
        levels = (1 << self.bits) - 1
        soft = (self.encoder(features).tanh() + 1) * levels / 2
        if soft.shape != (*features.shape[:-1], self.message_dim):
            raise ValueError("Codec encoder must preserve leading axes and output message_dim codes")
        hard = soft.round()
        quantized = soft + (hard - soft).detach()
        return self._decode_levels(quantized), hard.to(torch.uint8)

    def _decode_levels(self, codes):
        levels = (1 << self.bits) - 1
        return self.decoder(codes * (2 / levels) - 1)

    def decode(self, codes):
        """Decode received integer codes without rerunning the sender encoder."""
        self._validate_codes(codes)
        return self._decode_levels(codes.to(dtype=self._decode_dtype.dtype))

    def _validate_codes(self, codes):
        if codes.dtype != torch.uint8 or codes.shape[-1] != self.message_dim:
            raise ValueError("Expected uint8 codes of configured message width")
        if torch.any(codes.to(torch.int64) >= (1 << self.bits)):
            raise ValueError("Code exceeds quantizer range")

    def pack(self, codes):
        self._validate_codes(codes)
        width = 8 // self.bits
        pieces = codes.to(torch.int64).reshape(*codes.shape[:-1], -1, width)
        shift = torch.arange(width, device=codes.device) * self.bits
        return (pieces << shift).sum(-1).to(torch.uint8)

    def unpack(self, payload):
        if (
            payload.dtype != torch.uint8
            or payload.shape[-1] != self.message_dim * self.bits // 8
        ):
            raise ValueError("Invalid packed payload")
        shifts = torch.arange(8 // self.bits, device=payload.device) * self.bits
        return (
            ((payload.to(torch.int64).unsqueeze(-1) >> shifts) & ((1 << self.bits) - 1))
            .flatten(-2)
            .to(torch.uint8)
        )


class GroupSetFusion(nn.Module):
    """Fuse each live sender set into shared slots without sender-order features.

    Empty group slots stay zero and are excluded through the returned mask.
    """

    def __init__(self, hidden_dim=128, heads=4, slots=8, inducing_points=16):
        super().__init__()
        self.inducing = nn.Parameter(
            torch.randn(inducing_points, hidden_dim) / hidden_dim**0.5
        )
        self.induce_attention = nn.MultiheadAttention(
            hidden_dim, heads, batch_first=True, dropout=0
        )
        self.induce_norm = nn.LayerNorm(hidden_dim)
        self.queries = nn.Parameter(torch.randn(slots, hidden_dim) / hidden_dim**0.5)
        self.attention = nn.MultiheadAttention(
            hidden_dim, heads, batch_first=True, dropout=0
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def _fuse_members(self, members):
        batch = members.shape[0]
        inducing = self.inducing.unsqueeze(0).expand(batch, -1, -1)
        queries = self.queries.unsqueeze(0).expand(batch, -1, -1)
        induced, _ = self.induce_attention(
            inducing, members, members, need_weights=False
        )
        induced = self.induce_norm(induced + inducing)
        value, _ = self.attention(queries, induced, induced, need_weights=False)
        return self.norm(value + queries)

    def forward(self, tokens, groups, alive):
        if groups.shape != tokens.shape[:2] or alive.shape != groups.shape:
            raise ValueError("Group and alive shapes must match batch/agent axes")
        b, n, h = tokens.shape
        if torch.any(alive & ((groups < 0) | (groups >= n))):
            raise ValueError("Alive agents require group IDs in [0, number of agents)")
        memory = tokens.new_zeros(b, n, self.queries.shape[0], h)
        valid = torch.zeros(b, n, dtype=torch.bool, device=tokens.device)
        for row in range(b):
            for group in groups[row, alive[row]].unique().tolist():
                members = alive[row] & (groups[row] == group)
                memory[row, group] = self._fuse_members(tokens[row, members].unsqueeze(0))[0]
                valid[row, group] = True
        return memory, valid
