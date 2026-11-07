"""Experimental CP04 adaptations sharing the complementary-pursuit wire budget."""

import torch
from torch import nn
from torch.nn import functional as F


CP04_BASELINES = ("masia", "ss_gigc", "coalign")
_GRID = 4
_FEATURES = 3 * _GRID**2
_LATENT = 16


def group_members(groups, alive):
    ids = torch.arange(alive.shape[1], device=alive.device)
    if torch.any(alive & ((groups < 0) | (groups >= len(ids)))):
        raise ValueError("Live group IDs must lie in [0,N)")
    return (groups[:, None] == ids[None, :, None]) & alive[:, None]


def gaussian_product(parameters, members):
    """Fuse diagonal Gaussian leaves; empty groups have zero output and are masked."""
    means, logvars = parameters.chunk(2, dim=-1)
    precision = torch.exp(-logvars)[:, None] * members[..., None]
    total = precision.sum(2)
    valid = members.any(2)
    safe_total = torch.where(valid[..., None], total, 1.0)
    mean = (precision * means[:, None]).sum(2) / safe_total
    return torch.cat((mean, -safe_total.log()), dim=-1), valid


def diagonal_kl(p, q=None):
    p_mean, p_logvar = p.chunk(2, -1)
    q_mean, q_logvar = (torch.zeros_like(p_mean), torch.zeros_like(p_logvar))
    if q is not None:
        q_mean, q_logvar = q.chunk(2, -1)
    return 0.5 * (
        q_logvar - p_logvar + (p_logvar - q_logvar).exp()
        + (p_mean - q_mean).square() * (-q_logvar).exp() - 1
    ).mean(-1)


def _masked_mean(values, valid):
    return (values * valid).sum() / valid.sum().clamp_min(1)


def _state_grid(state):
    if state.ndim != 4 or state.shape[-1] != 25:
        raise ValueError("CP04 paper adaptations require native 25-channel state")
    return F.adaptive_avg_pool2d(state[..., [0, 3, 1]].movedim(-1, 1), _GRID).flatten(1)


def _observed_union(frame, members, map_size):
    """Coarsen the union of positive local wall/ally/prey observations by max."""
    b, n, h, w, channels = frame.shape
    if channels != 29:
        raise ValueError("Observation union requires native 29-channel frames")
    center = (frame[:, :, h // 2, w // 2, -2:] * map_size).round().long()
    y, x = torch.meshgrid(
        torch.arange(h, device=frame.device), torch.arange(w, device=frame.device),
        indexing="ij",
    )
    offsets = torch.stack((x.flatten() - w // 2, y.flatten() - h // 2), -1)
    coords = center[:, :, None] + offsets
    inside = ((coords >= 0) & (coords < map_size)).all(-1)
    bins = ((coords * _GRID) // map_size).clamp(0, _GRID - 1)
    index = (bins[..., 0] * _GRID + bins[..., 1]).reshape(b, 1, n * h * w, 1)
    positives = frame[..., [0, 1, 4]].flatten(2, 3).clamp(0, 1)
    visible = positives[:, None] * (members[..., None] & inside[:, None])[..., None]
    union = frame.new_zeros(b, n, _GRID**2, 3)
    union.scatter_reduce_(
        2, index.expand(b, n, -1, 3),
        visible.reshape(b, n, n * h * w, 3), reduce="amax",
    )
    return union.flatten(2)


def _sample(parameters):
    mean, logvar = parameters.chunk(2, -1)
    return mean + torch.randn_like(mean) * (0.5 * logvar).exp()


class CP04Baseline(nn.Module):
    """A method-specific group context and auxiliary objective for the shared actor."""

    def __init__(self, method, hidden_dim, action_dim, map_size):
        super().__init__()
        self.method = method
        self.map_size = map_size
        self.action_dim = action_dim
        if method == "masia":
            self.gate = nn.Linear(hidden_dim, hidden_dim)
            self.reconstruction = nn.Linear(hidden_dim, _FEATURES)
            self.forecast = nn.Sequential(
                nn.Linear(hidden_dim + action_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, _FEATURES),
            )
        elif method == "ss_gigc":
            self.reconstruction = nn.Linear(_LATENT, _FEATURES)
        elif method == "coalign":
            self.collaborative = nn.Linear(hidden_dim, 2 * _LATENT)
            self.global_belief = nn.Sequential(
                nn.Linear(_FEATURES, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 2 * _LATENT)
            )
            self.global_reconstruction = nn.Linear(_LATENT, _FEATURES)
            self.readout = nn.Linear(_LATENT, hidden_dim)
        else:
            raise ValueError("Unknown experimental CP04 adaptation")

    @staticmethod
    def _bound(parameters):
        mean, logvar = parameters.chunk(2, -1)
        return torch.cat((mean, logvar.clamp(-6.0, 4.0)), -1)

    def context(self, memory):
        context = memory.mean(-2)
        if self.method == "coalign":
            belief = self._bound(self.collaborative(context))
            return self.readout(belief[..., :_LATENT]).unsqueeze(-2), belief
        return memory, context

    def read(self, query, memory):
        # MASIA-style role-dependent extraction from the permutation-invariant context.
        return self.gate(query).sigmoid() * memory.mean(-2)

    def loss(self, ret, batch, alive, groups):
        valid = ret["group_valid"]
        context = ret["baseline_context"]
        members = group_members(groups, alive)
        device = context.device
        if self.method == "ss_gigc":
            obs = batch["observations"][:, -1].to(device=device, dtype=torch.float32)
            target = _observed_union(obs, members, self.map_size)
            logits = self.reconstruction(_sample(context))
            reconstruction = F.binary_cross_entropy_with_logits(
                logits, target, reduction="none",
            ).mean(-1)
            return _masked_mean(reconstruction, valid) + 0.01 * (
                _masked_mean(diagonal_kl(ret["decoded_tokens"]), alive)
                + _masked_mean(diagonal_kl(context), valid)
            )

        current = _state_grid(batch["states"][:, -1].to(device=device, dtype=torch.float32))
        if self.method == "masia":
            actions = batch["actions"][:, -1].to(device=device, dtype=torch.long)
            one_hot = F.one_hot(actions.masked_fill(~alive, 0), self.action_dim).float()
            action_summary = (members.float() @ one_hot) / members.sum(-1, keepdim=True).clamp_min(1)
            following = _state_grid(
                batch["next_states"][:, -1].to(device=device, dtype=torch.float32)
            )
            error = (self.reconstruction(context).sigmoid() - current[:, None]).square().mean(-1)
            prediction = self.forecast(torch.cat((context, action_summary), -1)).sigmoid()
            return _masked_mean(error + (prediction - following[:, None]).square().mean(-1), valid)

        global_belief = self._bound(self.global_belief(current))
        reconstruction = (
            self.global_reconstruction(_sample(global_belief)).sigmoid() - current
        ).square().mean()
        alignment = diagonal_kl(context, global_belief.detach()[:, None])
        return reconstruction + 0.01 * diagonal_kl(global_belief).mean() + _masked_mean(alignment, valid)
