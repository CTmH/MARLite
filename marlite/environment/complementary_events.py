"""Training-only attack queries for north-facing complementary-pursuit agents.

Queries specify the complete candidate attack set; omitted agents stay. Damage
is applied once per target, before recovery and movement, as in native dynamics.
"""

import numpy as np

ATTACK_OFFSETS = tuple(
    (x, y) for y in range(-2, 3) for x in range(-2, 3) if 0 < abs(x) + abs(y) <= 2
)
MOVE_OFFSETS = ((0, -1), (-1, 0), (0, 0), (1, 0), (0, 1))


def attack_outcomes(
    pursuers, evaders, actions, effective_damage=10.0, ineffective_damage=0.2
):
    """Return target events and missed attackers without mutating a simulator.

    pursuers maps actor names to (x,y); evaders maps native IDs to (x,y,hp).
    Events distinguish complementary hits from lethal hits, including solo kills.
    """
    if not all(
        np.isfinite(d) and d >= 0 for d in (effective_damage, ineffective_damage)
    ):
        raise ValueError("Damage must be finite and nonnegative")
    by_position = {tuple(v[:2]): key for key, v in evaders.items()}
    hits, missed = {}, []
    for agent, action in actions.items():
        if (
            agent not in pursuers
            or isinstance(action, (bool, np.bool_))
            or not isinstance(action, (int, np.integer))
            or not 0 <= action < 29
        ):
            raise ValueError("Invalid actor or attack action")
        if action < 5:
            continue
        dx, dy = ATTACK_OFFSETS[(action - 5) % 12]
        x, y = pursuers[agent]
        target = by_position.get((x + dx, y + dy))
        if target is None:
            missed.append(agent)
        else:
            hits.setdefault(target, []).append((agent, (action - 5) // 12))
    events = {}
    for target, attackers in hits.items():
        modes = [mode for _, mode in attackers]
        effective = 0 in modes and 1 in modes
        damage = effective_damage if effective else ineffective_damage
        events[target] = {
            "attackers": tuple(a for a, _ in attackers),
            "effective": effective,
            "captured": evaders[target][2] - damage <= 0,
            "capture_boundary": abs(evaders[target][2] - damage)
            <= 8 * np.finfo(np.float32).eps * max(1.0, abs(evaders[target][2]), damage),
            "critical": tuple(
                a for a, m in attackers if effective and modes.count(m) == 1
            ),
        }
    return events, tuple(missed)


def local_escape_action(observation, rng, minimap_mode):
    """Greedily maximize visible threat distance using only the evader's window."""
    enemy_channel = 4 if minimap_mode else 3
    threats = np.argwhere(observation[..., enemy_channel] > 0)
    if len(threats) == 0:
        return int(rng.integers(5))
    center = np.array(observation.shape[:2]) // 2
    choices, scores = [], []
    for action, (dx, dy) in enumerate(MOVE_OFFSETS):
        cell = center + [dy, dx]
        if np.any(cell < 0) or np.any(cell >= observation.shape[:2]):
            continue
        if observation[*cell, 0] > 0:
            continue
        choices.append(action)
        scores.append(np.abs(threats - cell).sum(-1).min())
    if not choices:
        return 2
    best = np.flatnonzero(np.asarray(scores) == max(scores))
    return int(choices[int(rng.choice(best))])
