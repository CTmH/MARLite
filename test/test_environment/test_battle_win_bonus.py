import unittest

import numpy as np
from gymnasium.spaces import Box, Discrete

from marlite.environment.magent_wrapper import BattleWrapper


class MockBattleEnv:
    """Minimal stand-in for the magent2 battle parallel env.

    Behavior is driven by two optional flags:
    - victory_at: step index at which every blue agent is eliminated while
      red agents survive (red wins).
    - draw_at: step index at which both teams are wiped out in the same step
      (no red agent survives, so no one can receive a bonus).
    """

    def __init__(self, n_red=2, n_blue=2, max_cycles=100, no_max_cycles=False):
        self.possible_agents = (
            [f"red_{i}" for i in range(n_red)]
            + [f"blue_{i}" for i in range(n_blue)]
        )
        self.agents = self.possible_agents[:]
        self._obs_shape = (3, 3, 2)
        self.observation_spaces = {
            agent: Box(low=0.0, high=2.0, shape=self._obs_shape, dtype=np.float32)
            for agent in self.possible_agents
        }
        self.action_spaces = {agent: Discrete(21) for agent in self.possible_agents}
        self.frames = 0
        self.victory_at = None
        self.draw_at = None
        if not no_max_cycles:
            self.max_cycles = max_cycles

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def state(self):
        return np.zeros((self._obs_shape[0], self._obs_shape[1], 3), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.frames = 0
        self.agents = self.possible_agents[:]
        return (
            {a: np.zeros(self._obs_shape, dtype=np.float32) for a in self.agents},
            {a: {} for a in self.agents},
        )

    def step(self, actions):
        self.frames += 1
        terminal = (self.victory_at is not None and self.frames >= self.victory_at) or (
            self.draw_at is not None and self.frames >= self.draw_at
        )
        if terminal:
            observations = {
                a: np.zeros(self._obs_shape, dtype=np.float32) for a in self.agents
            }
            if self.draw_at is not None and self.frames >= self.draw_at:
                rewards = {}
            else:
                rewards = {
                    a: 1.0 for a in self.agents if a.startswith("red_")
                }
            terminations = {a: True for a in self.agents}
            truncations = {a: False for a in self.agents}
            infos = {a: {} for a in self.agents}
            self.agents = []
            return observations, rewards, terminations, truncations, infos

        observations = {
            a: np.zeros(self._obs_shape, dtype=np.float32) for a in self.agents
        }
        rewards = {
            a: (1.0 if a.startswith("red_") else 0.0) for a in self.agents
        }
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        infos = {a: {} for a in self.agents}
        return observations, rewards, terminations, truncations, infos


class TestBattleWinBonus(unittest.TestCase):

    def _make_wrapper(
        self,
        max_cycles=100,
        victory_at=None,
        draw_at=None,
        win_bonus_factor=0.0,
        no_max_cycles=False,
    ):
        env = MockBattleEnv(
            n_red=2, n_blue=2, max_cycles=max_cycles, no_max_cycles=no_max_cycles
        )
        env.victory_at = victory_at
        env.draw_at = draw_at
        opponent_agent_group_config = {
            "type": "Random",
            "agent_list": {"blue_0": "random1", "blue_1": "random1"},
        }
        return BattleWrapper(
            env=env,
            opponent_agent_group_config=opponent_agent_group_config,
            opp_obs_queue_len=2,
            win_bonus_factor=win_bonus_factor,
        )

    def _run_steps(self, wrapper, n):
        """Reset the wrapper and run n steps, returning the last rewards dict."""
        wrapper.reset()
        last_rewards = {}
        for _ in range(n):
            _, last_rewards, _, _, _ = wrapper.step({})
        return last_rewards

    def test_early_win_bonus(self):
        max_cycles, victory_at, factor = 100, 40, 10.0
        expected = 1.0 + factor * (max_cycles - victory_at) / max_cycles
        wrapper = self._make_wrapper(
            max_cycles=max_cycles, victory_at=victory_at, win_bonus_factor=factor
        )

        pre_rewards = self._run_steps(wrapper, victory_at - 1)
        self.assertEqual(set(pre_rewards), {"red_0", "red_1"})
        for reward in pre_rewards.values():
            self.assertAlmostEqual(reward, 1.0)

        wrapper.reset()
        for _ in range(victory_at - 1):
            wrapper.step({})
        _, win_rewards, _, _, _ = wrapper.step({})

        self.assertEqual(set(win_rewards), {"red_0", "red_1"})
        for reward in win_rewards.values():
            self.assertAlmostEqual(reward, expected)
        self.assertEqual(wrapper.agents, [])

    def test_no_victory_no_bonus(self):
        wrapper = self._make_wrapper(max_cycles=100, win_bonus_factor=10.0)
        rewards = self._run_steps(wrapper, 50)
        self.assertEqual(set(rewards), {"red_0", "red_1"})
        for reward in rewards.values():
            self.assertAlmostEqual(reward, 1.0)

    def test_victory_at_max_cycles_yields_zero_bonus(self):
        wrapper = self._make_wrapper(
            max_cycles=100, victory_at=100, win_bonus_factor=10.0
        )
        rewards = self._run_steps(wrapper, 100)
        for reward in rewards.values():
            self.assertAlmostEqual(reward, 1.0)

    def test_default_factor_disabled(self):
        wrapper = self._make_wrapper(max_cycles=100, victory_at=40)
        rewards = self._run_steps(wrapper, 40)
        for reward in rewards.values():
            self.assertAlmostEqual(reward, 1.0)

    def test_draw_awards_no_bonus(self):
        wrapper = self._make_wrapper(
            max_cycles=100, draw_at=40, win_bonus_factor=10.0
        )
        rewards = self._run_steps(wrapper, 40)
        self.assertEqual(rewards, {})

    def test_step_counter_resets(self):
        wrapper = self._make_wrapper(
            max_cycles=100, victory_at=40, win_bonus_factor=10.0
        )
        self._run_steps(wrapper, 40)
        self.assertEqual(wrapper._step_count, 40)
        wrapper.reset()
        self.assertEqual(wrapper._step_count, 0)
        self._run_steps(wrapper, 10)
        self.assertEqual(wrapper._step_count, 10)

    def test_missing_max_cycles_raises(self):
        with self.assertRaises(ValueError):
            self._make_wrapper(no_max_cycles=True, win_bonus_factor=10.0)
        wrapper = self._make_wrapper(no_max_cycles=True)
        self.assertEqual(wrapper.win_bonus_factor, 0.0)


if __name__ == "__main__":
    unittest.main()
