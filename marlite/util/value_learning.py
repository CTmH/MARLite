"""Available-action reductions and QTRAN-alt constraints shared by workers."""

import torch


def last_action_mask(batch, key, device):
    mask = batch.get(key)
    return mask[:, -1].to(device=device, dtype=torch.bool) if isinstance(mask, torch.Tensor) else None


def masked_action_values(q_values, available=None, alive=None):
    """Exclude illegal actions; dead agents have a finite zero-valued no-op."""
    alive = (torch.ones_like(q_values[..., 0], dtype=torch.bool) if alive is None
             else alive.to(device=q_values.device, dtype=torch.bool))
    available = (torch.ones_like(q_values, dtype=torch.bool) if available is None
                 else available.to(device=q_values.device, dtype=torch.bool))
    if available.shape != q_values.shape or alive.shape != q_values.shape[:-1]:
        raise ValueError("Action/alive mask shapes must match the Q-value table")
    if (alive & ~available.any(-1)).any():
        raise ValueError("An alive agent has no available actions")
    available = available & alive.unsqueeze(-1)
    available = available.clone()
    available[..., 0] |= ~alive
    safe_q = torch.where(alive.unsqueeze(-1), q_values, 0.)
    return safe_q.masked_fill(~available, -torch.inf)


def joint_q_from_counterfactual(q_per_action, actions, alive):
    """All active heads estimate the same joint Q: average, never sum them."""
    actions = actions.masked_fill(~alive, 0)
    selected = q_per_action.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    return (selected * alive).sum(-1) / alive.sum(-1).clamp_min(1)


def qtran_constraints(q_values, enc_out, actions, q_counterfactual, value,
                      critic, alive, available=None, mask_optimal_samples=True):
    """QTRAN-alt optimality and counterfactual non-optimality constraints.

    The optimal constraint always uses the complete feasible greedy joint action,
    independently of the sampled action. The optional mask only excludes sampled
    greedy joint actions from the non-optimal term. Detached joint-Q estimates
    leave constraint gradients in the individual utilities and state value net.
    """
    alive = alive.to(device=q_values.device, dtype=torch.bool)
    feasible = masked_action_values(q_values, available, alive)
    greedy_q, greedy_actions = feasible.max(-1)
    with torch.no_grad():
        greedy_cf = critic(enc_out.detach(), greedy_actions, alive_mask=alive)["q_per_action"]
        greedy_joint = joint_q_from_counterfactual(greedy_cf, greedy_actions, alive)
    state_value = value.reshape(-1)
    optimal_error = greedy_q.sum(-1) - greedy_joint + state_value
    active_sample = alive.any(-1)
    optimal_loss = (optimal_error.square() * active_sample).sum() / active_sample.sum().clamp_min(1)

    safe_actions = actions.masked_fill(~alive, 0)
    chosen_q = q_values.gather(-1, safe_actions.unsqueeze(-1)).squeeze(-1) * alive
    other_q = chosen_q.sum(-1, keepdim=True) - chosen_q
    differences = feasible + other_q.unsqueeze(-1) - q_counterfactual.detach() + state_value[:, None, None]
    differences = differences.masked_fill(~torch.isfinite(feasible), torch.inf)
    minimum = differences.min(-1).values
    active = alive
    if mask_optimal_samples:
        optimal_sample = ((safe_actions == greedy_actions) | ~alive).all(-1)
        active = active & ~optimal_sample.unsqueeze(-1)
    nonoptimal_loss = (minimum.square() * active).sum() / active.sum().clamp_min(1)
    return optimal_loss, nonoptimal_loss
