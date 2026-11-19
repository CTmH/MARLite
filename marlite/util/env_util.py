import numpy as np
from typing import Dict, Any


def ensure_all_agents_present(data_dict: dict, default_values: dict) -> Dict[str, Any]:
    """
    Ensure that the dictionary contains all possible agents.
    If any agent is missing, add it with the corresponding default value.
    Maintains the order of possible_agents.
    """

    result = {}
    for agent in default_values.keys():
        if agent in data_dict:
            result[agent] = data_dict[agent]
        elif agent in default_values:
            result[agent] = default_values[agent]
        else:
            # Fallback: use first available value as template
            if data_dict:
                first_value = next(iter(data_dict.values()))
                result[agent] = np.zeros_like(first_value)
            elif default_values:
                first_default = next(iter(default_values.values()))
                result[agent] = np.zeros_like(first_default)
    return result


def precompute_manhattan_offsets(dist: int):
    """Precompute offsets for positions with Manhattan distance."""
    offsets = []
    for dx in range(-dist, dist + 1):
        for dy in range(-dist, dist + 1):
            if abs(dx) + abs(dy) <= dist and (dx != 0 or dy != 0):
                offsets.append((dx, dy))
    return np.array(offsets, dtype=int)


def current_action_mask(observation, info, action_count):
    """Read the current mask from a view or its info (e.g. SMAC).

    A wrapper-provided observation mask takes precedence: padding adapters may
    expand its width while retaining the native info dictionary for diagnostics.
    """
    from collections.abc import Mapping

    mask = observation.get("action_mask") if isinstance(observation, Mapping) else None
    if mask is None and isinstance(info, Mapping):
        mask = info.get("action_mask")
    if mask is None:
        return np.ones(action_count, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (action_count,):
        raise ValueError("Action mask width must match the agent's discrete action space")
    return mask
