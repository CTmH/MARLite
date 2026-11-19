"""Optional initialization diagnostics on actual typed policy outputs."""

from copy import deepcopy

import torch

from marlite.algorithm.sequence_agent_group import PolicyOutput
from marlite.util.action_distribution import masked_categorical
from marlite.util.randomness import preserve_rng_state


@torch.no_grad()
def inspect_initial_outputs(model, *args, action_mask=None, active_mask=None, **kwargs):
    """Inspect a copy without mutating caller-owned state, modes or RNGs.

    Inputs follow the model's ordinary step/sequence API. Masks must cover the
    same leading dimensions as its outputs; statistics ignore dead/group padding.
    This copies the model once and is unsuitable for per-minibatch logging.
    """
    with preserve_rng_state():
        result = deepcopy(model).eval()(*deepcopy(args), **deepcopy(kwargs))
    if not isinstance(result, PolicyOutput):
        raise TypeError("Initialization diagnostics require a PolicyOutput")
    metrics = {}
    if result.output_kind == "logits":
        logits = result.values
        active = (
            torch.ones_like(logits[..., 0], dtype=torch.bool)
            if active_mask is None
            else active_mask.to(device=logits.device, dtype=torch.bool)
        )
        distribution = masked_categorical(logits, action_mask, active_mask=active)
        choices = (
            action_mask.to(logits.device).bool().sum(-1)
            if action_mask is not None
            else torch.full_like(logits[..., 0], logits.shape[-1])
        )
        free = active & (choices > 1)
        if free.any():
            metrics["policy_normalized_entropy"] = float(
                (distribution.entropy()[free] / choices[free].float().log()).mean()
            )
    diagnostics = result.diagnostics
    for name in ("agent_mu", "agent_log_var", "group_mu", "group_log_var", "group_sample"):
        if name not in diagnostics:
            continue
        values = diagnostics[name]
        mask = diagnostics.get("group_valid") if name.startswith("group_") else active_mask
        if mask is not None:
            values = values[mask.to(device=values.device, dtype=torch.bool)]
        if values.numel():
            metrics[f"{name}_mean"] = float(values.float().mean())
            metrics[f"{name}_std"] = float(values.float().std(unbiased=False))
    return metrics
