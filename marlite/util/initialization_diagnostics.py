"""Optional pre-training diagnostics on real, already-collated model inputs."""

from copy import deepcopy

import torch

from marlite.util.action_distribution import masked_categorical
from marlite.util.randomness import preserve_rng_state


@torch.no_grad()
def inspect_initial_outputs(model, *args, action_mask=None, active_mask=None, **kwargs):
    """Return scalar statistics without changing RNGs, modes or model caches.

    Pass the same positional/keyword arguments used by the model's forward.
    This explicit, one-off diagnostic copies the model and disables dropout;
    do not call it per minibatch. Run before compiling the model. Masks refer
    to the current step, not the full observation history.
    """
    with preserve_rng_state():
        result = deepcopy(model).eval()(*args, **kwargs)
    metrics = {}
    if "action_logits" in result:
        logits = result["action_logits"]
        active = torch.ones_like(logits[..., 0], dtype=torch.bool)
        if active_mask is not None:
            active = active_mask.to(device=logits.device, dtype=torch.bool)
        distribution = masked_categorical(logits, action_mask, active)
        choices = (action_mask.to(logits.device).bool().sum(-1) if action_mask is not None
                   else torch.full_like(logits[..., 0], logits.shape[-1]))
        # A forced action has entropy zero but no meaningful H / log(|A|).
        free_choices = active & (choices > 1)
        if free_choices.any():
            entropy = distribution.entropy()[free_choices]
            metrics["policy_normalized_entropy"] = (
                entropy / choices[free_choices].float().log()
            ).mean().item()
    for name in ("agent_mu", "agent_log_var", "group_mu", "group_log_var", "group_consensus"):
        if name not in result:
            continue
        values = result[name]
        if name.startswith("agent_") and active_mask is not None:
            values = values[active_mask.to(device=values.device, dtype=torch.bool)]
        if name.startswith("group_") and "group_indices" in result:
            indices = result["group_indices"].to(values.device)
            membership = indices.unsqueeze(-1) == torch.arange(values.shape[1], device=values.device)
            if active_mask is not None:
                membership &= active_mask.to(device=values.device, dtype=torch.bool).unsqueeze(-1)
            values = values[membership.any(dim=1)]
        if values.numel():
            metrics[f"{name}_mean"] = values.float().mean().item()
            metrics[f"{name}_std"] = values.float().std(unbiased=False).item()
    return metrics
