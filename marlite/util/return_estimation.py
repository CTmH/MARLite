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


def td_target(batch, rewards, next_values, gamma, terminated):
    """Use dataset-provided n-step returns; accept legacy one-step batches."""
    if "td_rewards" in batch:
        rewards = batch["td_rewards"].to(next_values)
        discount = batch["td_discount"].to(next_values)
    else:
        discount = gamma * (1 - terminated.to(next_values))
    # Avoid 0 * NaN at terminal states (some mixers cannot evaluate all-dead inputs).
    return rewards + discount * torch.where(discount != 0, next_values, 0)


@torch.no_grad()
def bootstrap_values(critic, batch, device):
    """Evaluate full next windows only for nonterminal, nonempty teams."""
    next_alive_mask = batch["next_alive_mask"].to(device=device, dtype=torch.bool)
    team_terminated = batch["terminations"][:, -1].to(device=device, dtype=torch.bool).all(-1)
    can_bootstrap = ~team_terminated & next_alive_mask[:, -1].any(-1)
    next_values = torch.zeros(len(next_alive_mask), device=device)
    if can_bootstrap.any():
        states = batch["next_states"].to(device=device, dtype=torch.float32)
        padding = batch["next_timestep_padding_mask"].to(device=device, dtype=torch.bool)
        next_values[can_bootstrap] = critic(
            states[can_bootstrap], next_alive_mask[can_bootstrap], padding[can_bootstrap]
        )["v"].reshape(-1)
    return next_values


def ppo_targets(batch, values, critic, gamma, aggregate_rewards):
    """Consume fixed PPO labels, with a one-step fallback for direct worker use."""
    if "advantages" in batch:
        return batch["advantages"].to(values), batch["value_targets"].to(values)
    training = critic.training
    try:
        critic.eval()
        next_values = bootstrap_values(critic, batch, values.device)
    finally:
        critic.train(training)
    with torch.no_grad():
        rewards = aggregate_rewards(batch["rewards"][:, -1].to(values))
        returns = rewards + gamma * next_values
        return returns - values.detach(), returns
