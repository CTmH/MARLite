"""Aggregate update means by their own valid label populations."""


def summarize_updates(updates, *, default_count, count_overrides=None):
    """Optional objectives may start halfway through an iteration.

    Missing measurements are absent, not zero-valued samples. A loss's explicit
    ``name_count`` takes precedence over the default RL population.
    """
    if not updates:
        return {}
    overrides = count_overrides or {}
    result = {}
    for key in sorted(set().union(*(update.keys() for update in updates))):
        measured = [update for update in updates if key in update]
        if key.endswith("_count") or key in ("agent_steps", "team_steps"):
            result[key] = sum(update[key] for update in measured)
        else:
            count = key + "_count"
            if not all(count in update for update in measured):
                count = overrides.get(key, default_count)
            total = sum(update[count] for update in measured)
            result[key] = sum(update[key] * update[count] for update in measured) / max(total, 1)
    return result
