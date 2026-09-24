"""Categorical policies shared by MAPPO collection and learning."""

import torch
from torch.distributions import Categorical


def masked_categorical(logits, action_mask=None, active_mask=None, actions=None):
    """Mask unavailable actions; inactive rows use a harmless uniform policy.

    ``actions`` optionally validates replayed actions against the collection
    mask. Callers must exclude inactive rows from policy and entropy losses.
    """
    active = torch.ones(logits.shape[:-1], dtype=torch.bool, device=logits.device)
    if active_mask is not None:
        if active_mask.shape != active.shape:
            raise ValueError("active_mask shape must match logits without the action axis")
        active = active_mask.to(device=logits.device, dtype=torch.bool)
    mask = torch.ones_like(logits, dtype=torch.bool)
    if action_mask is not None:
        if action_mask.shape != logits.shape:
            raise ValueError("action_mask shape must match logits")
        mask = action_mask.to(device=logits.device, dtype=torch.bool)
    if (active & ~mask.any(dim=-1)).any():
        raise ValueError("An active agent has no available actions")
    if actions is not None:
        if actions.shape != active.shape:
            raise ValueError("actions shape must match the policy batch shape")
        actions = actions.to(device=logits.device, dtype=torch.long)
        if ((actions < 0) | (actions >= logits.shape[-1])).any():
            raise ValueError("Action index outside the policy action space")
        if (active & ~mask.gather(-1, actions.unsqueeze(-1)).squeeze(-1)).any():
            raise ValueError("A replayed action is unavailable under its collection mask")
    safe_logits = torch.where(active.unsqueeze(-1), logits, 0.0)
    safe_mask = mask | ~active.unsqueeze(-1)
    return Categorical(logits=safe_logits.masked_fill(~safe_mask, -torch.inf))
