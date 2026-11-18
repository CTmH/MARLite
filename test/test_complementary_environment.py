import numpy as np
import pytest
from magent2.environments.complementary_pursuit_v1 import parallel_env as NativeEnv
from marlite.environment.complementary_pursuit import parallel_env
from marlite.environment.complementary_events import (
    attack_outcomes,
    ATTACK_OFFSETS,
    local_escape_action,
)


class Scene(NativeEnv):
    def __init__(self, modes, damage=10):
        self.modes = modes
        self.positions = [(7, 5), (5, 7), (9, 7)][: len(modes)]
        super().__init__(map_size=15, effective_damage=damage)

    def generate_map(self):
        evader, pursuer = self.env.get_handles()
        self.env.add_agents(evader, "custom", pos=[(7, 7, 3), (12, 12, 3)])
        self.env.add_agents(pursuer, "custom", pos=[(*p, 3) for p in self.positions])


@pytest.mark.parametrize("modes", ["A", "AA", "AB", "AAB"])
@pytest.mark.parametrize("damage", [1.0, 5.0, 10.0])
def test_attack_queries_match_native(modes, damage):
    env = Scene(modes, damage)
    env.reset(seed=42)
    positions = {f"pursuer_{i}": p for i, p in enumerate(env.positions)}
    actions = {
        a: 5 + 12 * (modes[i] == "B") + ATTACK_OFFSETS.index((7 - p[0], 7 - p[1]))
        for i, (a, p) in enumerate(positions.items())
    }
    events, missed = attack_outcomes(
        positions, {0: (7, 7, 5), 1: (12, 12, 5)}, actions, effective_damage=damage
    )
    _, rewards, terms, _, _ = env.step(actions)
    assert events[0]["captured"] == terms["evader_0"]
    assert events[0]["effective"] == ("A" in modes and "B" in modes)
    assert not missed
    pool = 0.2 if events[0]["effective"] else 0.02
    pool += 1.0 if events[0]["captured"] else 0.0
    for agent in actions:
        critical = events[0]["critical"]
        expected = pool / len(actions) + (
            0.2 / len(critical) if agent in critical else 0.0
        )
        assert rewards[agent] == pytest.approx(expected)
    env.close()


@pytest.mark.parametrize("policy", ["random", "local_escape"])
def test_population_permissions_reset_and_truncation(policy):
    env = parallel_env(
        map_size=15,
        n_pursuers=3,
        n_evaders=5,
        minimap_mode=True,
        extra_features=True,
        max_cycles=3,
        prey_policy=policy,
    )
    first, _ = env.reset(seed=7)
    second, _ = env.reset(seed=7)
    assert len(second) == 3
    for a in first:
        np.testing.assert_array_equal(first[a], second[a])
        assert not second[a][..., [3, 6]].any()
        assert env.action_space(a).n == 29
    for _ in range(3):
        _, _, _, _, infos = env.step({a: 2 for a in env.agents})
    assert not env.agents and env.captured == 0
    assert all(i["capture_fraction"] == 0 for i in infos.values())
    env.reset(seed=19)
    state = env.state()
    walls = np.asarray(env.env.env._get_walls_info())
    assert state[..., 0].sum() == len(walls)
    assert np.all(state[walls[:, 0], walls[:, 1], 0] == 1)
    env.close()


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
    groups = build_partial_groups(
        np.array([[5, 2, 3], [6, 20, 20]]), 8, "cityblock", 2, [5, 6], []
    )
    assert groups[0] != groups[1] and (groups >= 0).all()


def test_wrapper_reports_capture_on_the_final_transition():
    from marlite.environment.complementary_pursuit import ComplementaryPursuit

    env = ComplementaryPursuit(Scene("AB"))
    env.reset(seed=1)
    actions = {
        "pursuer_0": 5 + ATTACK_OFFSETS.index((0, 2)),
        "pursuer_1": 17 + ATTACK_OFFSETS.index((2, 0)),
    }
    _, _, _, _, infos = env.step(actions)
    assert env.captured == 1
    assert all(i["capture_fraction"] == 0.5 for i in infos.values())
    env.close()
