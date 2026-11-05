"""Finite-rate messages, set fusion, and a context-conditioned neural process."""

import torch
from torch import nn
from torch.nn import functional as F


class ScalarMessageCodec(nn.Module):
    """Hard scalar codes in the forward pass, straight-through gradients in training.

    Replay codes are unpacked uint8 values; pack/unpack define wire payloads.
    """

    def __init__(self, hidden_dim, message_dim=32, bits=4):
        super().__init__()
        if bits not in (1, 2, 4, 8) or message_dim <= 0 or message_dim * bits % 8:
            raise ValueError(
                "Messages must contain a whole number of bytes; bits must divide 8"
            )
        self.bits, self.message_dim = bits, message_dim
        self.encoder = nn.Linear(hidden_dim, message_dim)
        self.decoder = nn.Sequential(
            nn.Linear(message_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, features):
        levels = (1 << self.bits) - 1
        soft = (self.encoder(features).tanh() + 1) * levels / 2
        hard = soft.round()
        quantized = soft + (hard - soft).detach()
        return self.decoder(quantized * (2 / levels) - 1), hard.to(torch.uint8)

    def pack(self, codes):
        if codes.dtype != torch.uint8 or codes.shape[-1] != self.message_dim:
            raise ValueError("Expected uint8 codes of configured message width")
        if torch.any(codes.to(torch.int64) >= (1 << self.bits)):
            raise ValueError("Code exceeds quantizer range")
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

    def forward(self, tokens, groups, alive):
        if groups.shape != tokens.shape[:2] or alive.shape != groups.shape:
            raise ValueError("Group and alive shapes must match batch/agent axes")
        if torch.any(alive & ((groups < 0) | (groups >= tokens.shape[1]))):
            raise ValueError("Alive agents require group IDs in [0, number of agents)")
        b, n, h = tokens.shape
        memory = tokens.new_zeros(b, n, self.queries.shape[0], h)
        valid = torch.zeros(b, n, dtype=torch.bool, device=tokens.device)
        for row in range(b):
            for group in groups[row, alive[row]].unique().tolist():
                members = alive[row] & (groups[row] == group)
                query = self.queries.unsqueeze(0)
                selected = tokens[row, members].unsqueeze(0)
                induced, _ = self.induce_attention(
                    self.inducing.unsqueeze(0), selected, selected, need_weights=False
                )
                induced = self.induce_norm(induced + self.inducing.unsqueeze(0))
                value, _ = self.attention(query, induced, induced, need_weights=False)
                memory[row, group] = self.norm(value + query)[0]
                valid[row, group] = True
        return memory, valid


class RelationalNeuralProcess(nn.Module):
    """Message-conditioned prior with a label-conditioned training posterior.

    A single latent draw is shared across a group's queries. The policy reads
    only deterministic prior parameters, keeping posterior labels out of actions.
    """

    def __init__(self, hidden_dim=128, query_dim=4, latent_dim=32):
        super().__init__()
        self.prior = nn.Linear(hidden_dim, 2 * latent_dim)
        self.target_encoder = nn.Sequential(
            nn.Linear(query_dim + 1, hidden_dim), nn.GELU()
        )
        self.posterior = nn.Linear(2 * hidden_dim, 2 * latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim + query_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.policy_projection = nn.Linear(2 * latent_dim, hidden_dim)

    @staticmethod
    def distribution(params):
        mean, log_std = params.chunk(2, -1)
        return torch.distributions.Normal(mean, log_std.clamp(-8, 4).exp())

    def policy_context(self, context):
        # The action path never reads target labels or samples the posterior.
        return context + self.policy_projection(self.prior(context))

    def loss(self, context, queries, labels, mask):
        if queries.shape[:-1] != labels.shape or labels.shape != mask.shape:
            raise ValueError("Relation queries, labels and mask must align")
        if not torch.isfinite(queries).all() or not torch.isfinite(labels[mask]).all():
            raise ValueError("Relation data must be finite")
        if torch.any((labels[mask] < 0) | (labels[mask] > 1)):
            raise ValueError("Bernoulli labels must lie in [0, 1]")
        safe_labels = torch.where(mask, labels, torch.zeros_like(labels))
        target = self.target_encoder(
            torch.cat([queries, safe_labels.unsqueeze(-1)], -1)
        )
        count = mask.sum(-1)
        target = (target * mask.unsqueeze(-1)).sum(-2) / count.clamp_min(1).unsqueeze(
            -1
        )
        prior = self.distribution(self.prior(context))
        posterior = self.distribution(self.posterior(torch.cat([context, target], -1)))
        z = posterior.rsample()  # One sample is shared across this group's queries.
        features = (
            torch.cat([context, z], -1)
            .unsqueeze(-2)
            .expand(*queries.shape[:-1], context.shape[-1] + z.shape[-1])
        )
        logits = self.decoder(torch.cat([features, queries], -1)).squeeze(-1)
        nll = (
            F.binary_cross_entropy_with_logits(logits, safe_labels, reduction="none")
            * mask
        ).sum(-1)
        kl = torch.distributions.kl_divergence(posterior, prior).sum(-1)
        valid = count > 0
        return ((nll + kl) * valid).sum() / count.sum().clamp_min(1)

    def predict(self, context, queries, samples=32):
        """Prior predictive samples; each latent is shared by all group queries."""
        if samples < 1:
            raise ValueError("samples must be positive")
        z = self.distribution(self.prior(context)).rsample((samples,))
        context = context.unsqueeze(0).expand(samples, *context.shape)
        features = torch.cat([context, z], -1).unsqueeze(-2)
        features = features.expand(
            *features.shape[:-2], queries.shape[-2], features.shape[-1]
        )
        queries = queries.unsqueeze(0).expand(samples, *queries.shape)
        return self.decoder(torch.cat([features, queries], -1)).squeeze(-1).sigmoid()
