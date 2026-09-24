"""Shared post-step boundary handling for both rollout implementations."""


def finish_transition(
    env, observed_agents, terminations, truncations, previous_state, win_tag, at_limit
):
    """Return the actual next state, physical survivors, and collection boundary.

    PettingZoo can empty env.agents at a timeout although units are alive.
    Termination wins if both flags are true (including dead SMAC units).
    A missing final observation/state at a truncation is an error: silently
    substituting the previous state would train on an invalid bootstrap target.
    Termination/truncation dictionaries are updated in place for storage.
    """
    if win_tag:
        terminations.update({agent: True for agent in terminations})
    team_terminated = all(terminations.values())
    all_agents_done = all(
        terminations[agent] or truncations[agent] for agent in terminations
    )
    episode_ended = team_terminated or not env.agents or at_limit or all_agents_done
    if episode_ended and not team_terminated:
        truncations.update({agent: True for agent in truncations})
    survivors = [
        agent for agent, terminated in terminations.items()
        if not terminated and (episode_ended or agent in env.agents)
    ]
    if not team_terminated and not set(survivors).issubset(observed_agents):
        raise ValueError("Missing final observations for live agents; cannot bootstrap")
    try:
        next_state = env.state()
    except Exception:
        if not team_terminated:
            raise
        # This value is never used for bootstrap after a genuine termination.
        next_state = previous_state
    return next_state, survivors, episode_ended
