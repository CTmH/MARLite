"""Environment-edge migration: heterogeneity, opponents and native action schemas."""

import numpy as np
import pytest
from gymnasium import spaces

from marlite.environment.controllers import (
    RandomController,
    PreyEscapeController,
    BattleController,
    FrozenPolicyController,
)
from marlite.environment.padding import PadAgentView
from marlite.environment.sumo_wrapper import SUMOWrapper
from marlite.environment.magent_wrapper import BattleWrapper
from test.test_environment.test_battle_win_bonus import MockBattleEnv
from test.test_sequence_mappo import make_trainer


class HeterogeneousEnv:
    possible_agents = ("a", "b", "c")

    def __init__(self, selection="all"):
        self.selection, self.time = selection, 0
        self.observation_spaces = {
            a: spaces.Box(-10.0, 10.0, (n,), dtype=np.float32)
            for a, n in zip(self.possible_agents, (2, 4, 2))
        }
        self.action_spaces = {
            a: spaces.Discrete(n) for a, n in zip(self.possible_agents, (2, 3, 2))
        }
        self.agents = []
        self.resets = 0

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _obs(self, agents):
        return {
            a: np.arange(self.observation_spaces[a].shape[0], dtype=np.float32) + self.time
            for a in agents
        }

    def state(self):
        return np.array([self.time, 1, 2, 3, 4, 5], dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.time, self.resets = 0, self.resets + 1
        self.agents = (
            list(self.possible_agents)
            if self.selection == "all"
            else (["a", "c"] if self.selection == "small" else ["b"])
        )
        return self._obs(self.possible_agents), {}

    def step(self, actions):
        assert set(actions) == set(self.agents)
        assert all(self.action_spaces[a].contains(action) for a, action in actions.items())
        before = list(self.agents)
        self.time += 1
        terms = {
            a: self.time >= 4 or (self.selection == "all" and a == "c" and self.time >= 2)
            for a in before
        }
        self.agents = [a for a in before if not terms[a]]
        return (
            self._obs(before),
            {a: float(action == 0) for a, action in actions.items()},
            terms,
            {a: False for a in before},
            {},
        )

    def close(self):
        pass


def test_padding_keeps_native_indices_and_distinguishes_unobserved_zero():
    native = HeterogeneousEnv()
    env = PadAgentView(native)
    obs, _ = env.reset(seed=1)
    assert obs["a"]["observation"].tolist() == [0, 1, 0, 0, 1, 1, 0, 0]
    assert obs["b"]["observation"].shape == (8,)
    assert obs["a"]["action_mask"].tolist() == [1, 1, 0]
    assert env.action_space("a").n == 3
    with pytest.raises(ValueError, match="Padded"):
        env.step({"a": 2, "b": 0, "c": 0})
    env.step({"a": 1, "b": 2, "c": 0})


def test_sumo_constructor_does_not_start_or_reset_simulation():
    native = HeterogeneousEnv()
    env = SUMOWrapper(native)
    assert native.resets == 0
    obs, _ = env.reset(seed=2)
    assert env.state().shape == (3, 4)
    assert env.state()[0].tolist() == [0, 1, 0, 0]
    wrapped = PadAgentView(env)
    padded, _ = wrapped.reset(seed=2)
    assert padded["a"]["observation"].shape == padded["b"]["observation"].shape


@pytest.mark.parametrize(
    "controller",
    [RandomController(), PreyEscapeController("probability"), BattleController("probability")],
)
def test_controller_rng_is_local_and_episode_seeded(controller):
    width = 9 if isinstance(controller, PreyEscapeController) else 21
    obs = {"x": np.zeros((13, 13, 5), dtype=np.float32)}
    obs["x"][6, 6, 2] = 6
    available = {"x": spaces.Discrete(width)}
    before = np.random.get_state()
    controller.reset(seed=9, agents=("x",))
    first = [controller.actions(obs, available, ["x"])["x"] for _ in range(5)]
    controller.reset(seed=9, agents=("x",))
    assert first == [controller.actions(obs, available, ["x"])["x"] for _ in range(5)]
    after = np.random.get_state()
    assert np.array_equal(before[1], after[1]) and before[2:] == after[2:]


def test_battle_movement_and_attack_indices_do_not_overlap():
    controller = BattleController("basic")
    assert controller.scorer.move_start_idx == 0
    assert controller.scorer.do_nothing_action == 6
    assert tuple(controller.scorer.move_offsets[6]) == (0, 0)
    obs = np.zeros((13, 13, 5), dtype=np.float32)
    obs[5, 5, 3] = 1
    result = controller.actions({"x": obs}, {"x": spaces.Discrete(21)}, ["x"])
    assert result["x"] == 13  # First of eight attack offsets; not a movement.


def test_wrapper_resets_and_combines_controller_actions():
    native = MockBattleEnv()
    wrapped = BattleWrapper(native, opponent=RandomController())
    wrapped.reset(seed=123)
    result = wrapped.step({"red_0": 0, "red_1": 0})
    assert set(result[0]) == {"red_0", "red_1"}
    assert wrapped.opponent is not None


def test_frozen_policy_controller_owns_and_resets_recurrent_state():
    policy = make_trainer().learner.actor
    controller = FrozenPolicyController(policy, stochastic=True)
    obs = {"a": np.ones(3, dtype=np.float32), "b": np.ones(3, dtype=np.float32) * 2}
    availability = {a: spaces.Discrete(3) for a in obs}
    controller.reset(seed=91, agents=tuple(obs))
    first = [controller.actions(obs, availability, list(obs)) for _ in range(3)]
    assert any(value.abs().sum() for value in controller.state.values())
    controller.reset(seed=91, agents=tuple(obs))
    assert controller.state is None
    assert first == [controller.actions(obs, availability, list(obs)) for _ in range(3)]
    assert not any(p.requires_grad for p in policy.parameters())
