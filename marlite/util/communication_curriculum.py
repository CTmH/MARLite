"""Action distillation and offline sender-replacement scheduling."""

import torch
from marlite.util.action_distribution import masked_categorical


def action_distillation(student, teacher, alive, available=None):
    target = masked_categorical(teacher, available, active_mask=alive)
    prediction = masked_categorical(student, available, active_mask=alive)
    loss = torch.distributions.kl_divergence(target, prediction)
    return (loss * alive).sum() / alive.sum().clamp_min(1)


def sender_replacement_mask(groups, alive, replacement, stage, update, generator=None):
    """Select live senders within each recorded group, independently of labels."""
    if groups.shape != alive.shape or groups.ndim != 2 or alive.dtype != torch.bool:
        raise ValueError("Group indices and boolean alive mask must have shape B,N")
    if torch.any(alive & ((groups < 0) | (groups >= groups.shape[1]))):
        raise ValueError("Live group indices must lie in [0,N)")
    if replacement == "all" or stage == 2:
        return alive.clone()
    selected = torch.zeros_like(alive)
    for b in range(groups.shape[0]):
        for g in groups[b, alive[b]].unique():
            indices = torch.where(alive[b] & (groups[b] == g))[0]
            if stage == 0:
                index = update % len(indices)
                indices = indices[index : index + 1]
            else:
                count = (len(indices) + 1) // 2
                indices = indices[
                    torch.randperm(len(indices), device=indices.device, generator=generator)[:count]
                ]
            selected[b, indices] = True
    return selected
