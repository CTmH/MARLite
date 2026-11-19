import numpy as np
from marlite.environment.complementary_events import (
    attack_outcomes,
    ATTACK_OFFSETS,
    local_escape_action,
)


def test_different_targets_and_solo_capture():
    positions = {"a": (5, 5), "b": (10, 10)}
    targets = {0: (5, 7, 5), 1: (10, 12, 5)}
    actions = {
        "a": 5 + ATTACK_OFFSETS.index((0, 2)),
        "b": 17 + ATTACK_OFFSETS.index((0, 2)),
    }
    events, _ = attack_outcomes(positions, targets, actions, ineffective_damage=6)
    assert all(not e["effective"] and e["captured"] for e in events.values())


def test_escape_reads_only_local_view():
    obs = np.zeros((3, 3, 7))
    obs[0, 1, 4] = 1
    assert local_escape_action(obs, np.random.default_rng(1), True) in (1, 3, 4)


def test_query_constructor_and_singleton_groups():
    import torch
    from marlite.util.complementary_targets import ComplementaryAttackQueries
    from marlite.algorithm.graph_builder.graph_util import build_partial_groups

    state = torch.zeros(1, 15, 15, 25)
    state[0, 7, 7, 1] = 1
    state[0, 7, 7, 2] = 1
    state[0, 7, 5, 3] = state[0, 5, 7, 3] = 1
    constructor = ComplementaryAttackQueries(n_queries=3)
    target = torch.tensor([[[[7, 7]] * 3]])
    positions = torch.tensor([[[[[7, 5], [5, 7], [9, 7]]] * 3]])
    modes = torch.tensor([[[[0, 1, 0]] * 3]])
    active = torch.tensor([[[[True, True, False]] * 3]])
    result = constructor.evaluate(
        state,
        torch.tensor([[True]]),
        target,
        positions,
        modes,
        active,
        torch.tensor([[[0, 1, 2]]]),
    )
    assert result["relation_labels"].tolist() == [[[1.0, 1.0, 1.0]]]
    assert result["relation_queries"].shape == (1, 1, 3, 14)
    assert result["relation_mask"].all()
    groups = build_partial_groups(np.array([[5, 2, 3]]), 8, "cityblock", 2, [5], [])
    assert groups.tolist() == [0]
    groups = build_partial_groups(np.array([[5, 2, 3], [6, 20, 20]]), 8, "cityblock", 2, [5, 6], [])
    assert groups[0] != groups[1] and (groups >= 0).all()
