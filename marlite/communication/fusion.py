"""Tensor-only group and set fusion with explicit latent randomness."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class CommunicationOutput:
    features: Tensor
    diagnostics: dict[str, Tensor]


def gaussian(parameters):
    if parameters.shape[-1] % 2:
        raise ValueError("Gaussian parameters must contain mean and log variance")
    mean, log_var = parameters.chunk(2, -1)
    return mean, log_var.clamp(-10.0, 10.0)


def draw_latent(mean, log_var, context, sampled):
    if not sampled:
        return mean
    noise = context.get("latent_noise")
    if noise is None or noise.shape != mean.shape or not torch.isfinite(noise).all():
        raise ValueError(
            "Sampled policy latents require recorded finite latent_noise of matching shape"
        )
    return mean + (0.5 * log_var).exp() * noise


class MaskedMeanPool(nn.Module):
    def forward(self, tokens, alive):
        return tokens.masked_fill(~alive[..., None], 0).sum(-2) / alive.sum(
            -1, keepdim=True
        ).clamp_min(1)


class MaskedMaxPool(nn.Module):
    def forward(self, tokens, alive):
        maximum = tokens.masked_fill(~alive[..., None], -torch.inf).max(-2).values
        return torch.where(alive.any(-1, keepdim=True), maximum, torch.zeros_like(maximum))


class AttentionPool(nn.Module):
    """One learned query reads a live set; no sender-order embeddings."""

    def __init__(self, hidden_dim, heads=4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) / hidden_dim**0.5)
        self.attention = nn.MultiheadAttention(hidden_dim, heads, dropout=0.0, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens, alive):
        empty = ~alive.any(-1)
        safe = alive.clone()
        safe[:, 0] |= empty
        values = tokens.masked_fill(~alive[..., None], 0)
        query = self.query.expand(tokens.shape[0], -1, -1)
        read, _ = self.attention(query, values, values, key_padding_mask=~safe, need_weights=False)
        return self.norm(read[:, 0] + query[:, 0]).masked_fill(empty[:, None], 0)


class SetCommunication(nn.Module):
    """Pool all available senders, then project a deterministic or Gaussian summary."""

    def __init__(self, pool, projection, *, latent_dim, probabilistic=False, sampled_policy=False):
        super().__init__()
        if sampled_policy and not probabilistic:
            raise ValueError("Sampling requires a Gaussian summary")
        self.pool, self.projection = pool, projection
        self.latent_dim = latent_dim
        self.probabilistic, self.sampled_policy = probabilistic, sampled_policy

    def sample_context(self, alive, generator):
        if self.sampled_policy:
            return {
                "latent_noise": torch.randn(
                    *alive.shape[:-1], 1, self.latent_dim, generator=generator
                )
            }
        return {}

    def forward(self, messages, alive, context, *, queries=None):
        leading, n = alive.shape[:-1], alive.shape[-1]
        flat = messages.reshape(-1, n, messages.shape[-1])
        pooled = self.pool(flat, alive.reshape(-1, n))
        parameters = self.projection(pooled).reshape(*leading, 1, -1)
        if self.probabilistic:
            mean, log_var = gaussian(parameters)
        else:
            mean, log_var = parameters, torch.zeros_like(parameters)
        if mean.shape[-1] != self.latent_dim:
            raise ValueError("Set projection output does not match latent_dim")
        group_valid = alive.any(-1, keepdim=True)
        summary = draw_latent(mean, log_var, context, self.sampled_policy)
        summary = summary.masked_fill(~group_valid[..., None], 0)
        per_agent = summary.expand(*leading, n, -1).masked_fill(~alive[..., None], 0)
        return CommunicationOutput(
            per_agent,
            {
                "group_mu": mean.masked_fill(~group_valid[..., None], 0),
                "group_log_var": log_var.masked_fill(~group_valid[..., None], 0),
                "group_sample": summary,
                "group_valid": group_valid,
                "aggregated_msg": summary[..., 0, :],
            },
        )


class GaussianConsensus(nn.Module):
    """Diagonal group fusion with fixed capacity and scatter-based aggregation.

    CI implements the former information-proportional diagonal rule. This is
    not a calibration theorem for learned or cross-dimension correlated beliefs.
    No B*T*N*G*L membership tensor is materialized.
    """

    def __init__(self, latent_dim, n_groups, mode="bayesian", *, sampled_policy=False):
        super().__init__()
        if mode not in ("ae", "sample_mean", "bayesian", "ci"):
            raise ValueError("Unsupported consensus rule")
        if (
            type(n_groups) is not int
            or n_groups < 1
            or type(latent_dim) is not int
            or latent_dim < 1
        ):
            raise ValueError("Group and latent capacities must be positive")
        if mode == "ae" and sampled_policy:
            raise ValueError("AE consensus has no stochastic latent")
        self.latent_dim, self.n_groups, self.mode = latent_dim, n_groups, mode
        self.sampled_policy = sampled_policy

    def sample_context(self, alive, generator):
        if self.sampled_policy:
            return {
                "latent_noise": torch.randn(
                    *alive.shape[:-1], self.n_groups, self.latent_dim, generator=generator
                )
            }
        return {}

    def forward(self, messages, alive, context, *, queries=None):
        groups = context.get("groups")
        if groups is None or groups.shape != alive.shape or groups.dtype != torch.long:
            raise ValueError("Consensus requires recorded int64 groups matching live slots")
        if (alive & ((groups < 0) | (groups >= self.n_groups))).any():
            raise ValueError("A live agent has an out-of-capacity group")
        leading, n = alive.shape[:-1], alive.shape[-1]
        if self.mode == "ae":
            mean, log_var = messages, torch.zeros_like(messages)
        else:
            mean, log_var = gaussian(messages)
        if mean.shape != (*alive.shape, self.latent_dim):
            raise ValueError("Consensus message width does not match latent_dim")
        mu, lv = mean.reshape(-1, n, self.latent_dim), log_var.reshape(-1, n, self.latent_dim)
        live = alive.reshape(-1, n)
        gid = groups.reshape(-1, n).clamp_min(0)
        # Dead sentinel IDs must not be used as scatter indices, even if positive.
        gid = gid.masked_fill(~live, 0)
        index = gid[..., None].expand(-1, -1, self.latent_dim)
        capacity = (mu.shape[0], self.n_groups, self.latent_dim)
        count = mu.new_zeros(mu.shape[0], self.n_groups).scatter_add_(1, gid, live.to(mu))
        valid = count > 0

        def grouped_sum(value):
            value = value.masked_fill(~live[..., None], 0)
            return value.new_zeros(capacity).scatter_add_(1, index, value)

        if self.mode in ("ae", "sample_mean"):
            group_mean = grouped_sum(mu) / count.clamp_min(1)[..., None]
        if self.mode == "ae":
            group_lv = torch.zeros_like(group_mean)
        else:
            scores = lv if self.mode == "sample_mean" else -lv
            maximum = mu.new_full(capacity, -torch.inf).scatter_reduce_(
                1,
                index,
                scores.masked_fill(~live[..., None], -torch.inf),
                reduce="amax",
                include_self=True,
            )
            maximum = maximum.masked_fill(~valid[..., None], 0)
            scaled = (scores - maximum.gather(1, index)).exp().masked_fill(~live[..., None], 0)
            mass = grouped_sum(scaled)
            if self.mode == "sample_mean":
                group_lv = (
                    maximum + mass.clamp_min(1e-12).log() - 2 * count.clamp_min(1)[..., None].log()
                )
            elif self.mode == "bayesian":
                group_mean = grouped_sum(scaled * mu) / mass.clamp_min(1e-12)
                group_lv = -maximum - mass.clamp_min(1e-12).log()
            else:
                squared_mass = grouped_sum(scaled.square())
                group_mean = grouped_sum(scaled.square() * mu) / squared_mass.clamp_min(1e-12)
                group_lv = (
                    -maximum - squared_mass.clamp_min(1e-12).log() + mass.clamp_min(1e-12).log()
                )
        group_mean = group_mean.masked_fill(~valid[..., None], 0).reshape(
            *leading, self.n_groups, -1
        )
        group_lv = group_lv.masked_fill(~valid[..., None], 0).reshape_as(group_mean)
        group_valid = valid.reshape(*leading, self.n_groups)
        sample = draw_latent(group_mean, group_lv, context, self.sampled_policy)
        sample = sample.masked_fill(~group_valid[..., None], 0)
        selected = sample.flatten(0, len(leading) - 1).gather(1, index)
        selected = selected.reshape(*alive.shape, self.latent_dim).masked_fill(~alive[..., None], 0)
        return CommunicationOutput(
            selected,
            {
                "agent_mu": mean.masked_fill(~alive[..., None], 0),
                "agent_log_var": log_var.masked_fill(~alive[..., None], 0),
                "group_mu": group_mean,
                "group_log_var": group_lv,
                "group_sample": sample,
                "group_valid": group_valid,
                "groups": groups,
            },
        )
