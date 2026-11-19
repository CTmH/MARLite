"""Recurrent prefix reconstruction shared by policy, mixer and value updates."""

import torch

from marlite.algorithm.sequence_agent_group import DecisionInput
from marlite.communication.context import frame_context


def initial_states(model, episodes, sources, *, device, strategy, burn_in,
                   team=False, cache_prefix="actor_state."):
    """Build detached states independently for this model/version.

    Only PPO opts into behavior-state caches. Target networks must use their own
    zero-burnin or prefix-recompute path, never behavior or online hidden state.
    """
    states = []
    for source in sources:
        episode = episodes[source.episode_id]
        start = source.start
        begin = 0 if strategy == "prefix_recompute" else max(0, start - burn_in)
        state = (model.initial_state(1, device, torch.float32) if team else
                 model.initial_state(1, len(episode.agent_ids), device, torch.float32))
        if strategy == "cached_burnin":
            for key, initial in state.items():
                name = cache_prefix + key
                if name not in episode.frames:
                    raise ValueError(f"Missing behavior recurrent state: {name}")
                value = episode.frames[name][begin:begin + 1].to(device)
                if value.shape != initial.shape or value.dtype != initial.dtype:
                    raise ValueError(f"Invalid behavior state specification: {name}")
                state[key] = value
        if begin < start and state:
            with torch.no_grad():
                alive = episode.frames["alive"][begin:start][None].to(device)
                if team:
                    reset = torch.zeros(1, start - begin, device=device, dtype=torch.bool)
                    reset[:, 0] = begin == 0
                    state = model.encode_sequence(
                        episode.frames["states"][begin:start][None].to(device),
                        state, alive.any(-1), reset).state
                else:
                    inputs = DecisionInput(
                        episode.frames["observations"][begin:start][None].to(device), alive,
                        episode.frames["agent_reset"][begin:start][None].to(device),
                        frame_context(episode.frames, start=begin, stop=start, batched=False, device=device))
                    state = model.encode_sequence(inputs, state).state
        states.append(state)
    return {key: torch.cat([s[key].detach() for s in states], 0) for key in states[0]}


def validate_optimizer_ownership(pairs):
    """Reject omitted, duplicated or foreign optimizer parameters at construction."""
    owned = set()
    for model, optimizer in pairs:
        expected = {id(p) for p in model.parameters() if p.requires_grad}
        actual = ([] if optimizer is None else
                  [id(p) for group in optimizer.param_groups for p in group["params"]])
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError("Optimizer parameters must exactly match their trainable component")
        if owned.intersection(actual):
            raise ValueError("An optimizer parameter cannot be owned by multiple components")
        owned.update(actual)
