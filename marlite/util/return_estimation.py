"""Return estimators independent of the policy's sequence architecture."""

import torch


def generalized_advantage_estimation(
    rewards, values, next_values, bootstrap, continuation, gamma, gae_lambda
):
    """Compute detached team advantages along time (axis 0).

    Bootstrap is allowed at an artificial truncation, but the trace must stop
    there. Individual deaths do not end a team trace. Inputs contain real
    transitions only, not history-window padding. ``bootstrap`` permits V(s'),
    while ``continuation`` permits the next transition's advantage. Returned
    advantages and value targets have the same shape as rewards.
    """
    if not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1:
        raise ValueError("gamma and gae_lambda must be in [0, 1]")
    if any(x.shape != rewards.shape for x in (values, next_values, bootstrap, continuation)):
        raise ValueError("GAE inputs must have identical shapes")
    with torch.no_grad():
        can_bootstrap = bootstrap.bool() & (gamma != 0)
        delta = rewards + torch.where(can_bootstrap, gamma * next_values, 0) - values
        trace_discount = gamma * gae_lambda
        continue_trace = continuation.bool() & (trace_discount != 0)
        advantages = torch.zeros_like(delta)
        next_advantage = torch.zeros_like(delta[0]) if len(delta) else 0
        for t in reversed(range(len(delta))):
            next_advantage = delta[t] + trace_discount * torch.where(
                continue_trace[t], next_advantage, 0
            )
            advantages[t] = next_advantage
        return advantages, advantages + values
