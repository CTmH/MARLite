"""Unit tests for marlite.rollout.phases."""

import pickle
import unittest
import numpy as np

from marlite.rollout.phases import (
    RolloutPhases,
    PHASE_REGISTRY,
    resolve_phases,
    FULL_PHASES,
    QMIX_PHASES,
    MAPPO_PHASES,
    GRAPH_QMIX_PHASES,
    GRAPH_MAPPO_PHASES,
    GROUP_CONSENSUS_PHASES,
    _build_custom_phases,
    _pre_step_filtered,
    _post_step_filtered,
    _terminal_filtered,
    _next_attr_filtered,
    _finalize_filtered,
)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _make_full_context():
    """Build a rich context dict containing all possible keys."""
    return {
        "alive_mask": {"a": True, "b": True},
        "observations": {"a": np.zeros(4), "b": np.zeros(4)},
        "states": np.zeros(8),
        "edge_indices": np.ones((2, 3)),
        "group_indices": {"a": 0, "b": 1},
        "actions": {"a": 1, "b": 2},
        "all_log_probs": {"a": -0.5, "b": -0.3},
        "log_probs": {"a": -0.5, "b": -0.3},
        "avail_actions": {"a": np.ones(5), "b": np.ones(5)},
        "infos": {"a": {}, "b": {}},
    }


def _make_episode_with_lists(*attr_names):
    """Return an episode dict with empty lists for the given attr names."""
    ep = {}
    for name in attr_names:
        ep[name] = []
    return ep


# Episode-level attrs that are scalars, not lists
_EPISODE_SCALAR_ATTRS = {"win_tag", "episode_length", "episode_reward"}

# All possible timestep attributes the full profile might touch
_ALL_TIMESTEP_ATTRS = [
    "alive_mask", "observations", "states", "edge_indices", "group_indices",
    "actions", "all_log_probs", "log_probs", "avail_actions", "infos",
    "next_alive_mask", "next_avail_actions", "next_edge_indices",
    "next_group_indices", "next_states", "next_observations",
    "rewards", "terminations", "truncations", "all_agents_sum_rewards",
]


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------

class TestResolvePhases(unittest.TestCase):
    """Tests for resolve_phases()."""

    def test_none_returns_full(self):
        """resolve_phases(None) returns FULL_PHASES."""
        self.assertIs(resolve_phases(None), FULL_PHASES)

    def test_all_registered_profiles(self):
        """Every key in PHASE_REGISTRY resolves to the correct instance."""
        for name, expected in PHASE_REGISTRY.items():
            with self.subTest(profile=name):
                self.assertIs(resolve_phases(name), expected)

    def test_unknown_profile_raises(self):
        """An unknown profile name raises ValueError."""
        with self.assertRaises(ValueError):
            resolve_phases("nonexistent")


class TestRolloutPhasesStructure(unittest.TestCase):
    """Verify RolloutPhases has the expected five fields."""

    def test_fields_present(self):
        self.assertTrue(hasattr(FULL_PHASES, "pre_step"))
        self.assertTrue(hasattr(FULL_PHASES, "post_step"))
        self.assertTrue(hasattr(FULL_PHASES, "terminal"))
        self.assertTrue(hasattr(FULL_PHASES, "next_attr"))
        self.assertTrue(hasattr(FULL_PHASES, "finalize"))

    def test_all_fields_callable(self):
        for name in ("pre_step", "post_step", "terminal", "next_attr",
                     "finalize"):
            fn = getattr(FULL_PHASES, name)
            self.assertTrue(callable(fn), f"{name} is not callable")


class TestPreStepPhase(unittest.TestCase):
    """Tests for the pre_step phase across profiles."""

    def setUp(self):
        self.ctx = _make_full_context()

    def test_full_profile_collects_all_optional(self):
        """FULL_PHASES.pre_step appends all pre-step attrs."""
        ep = _make_episode_with_lists(*_ALL_TIMESTEP_ATTRS)
        FULL_PHASES.pre_step(ep, self.ctx)

        self.assertEqual(len(ep["alive_mask"]), 1)
        self.assertEqual(len(ep["observations"]), 1)
        self.assertEqual(len(ep["states"]), 1)
        self.assertEqual(len(ep["actions"]), 1)
        self.assertEqual(len(ep["avail_actions"]), 1)
        self.assertEqual(len(ep["infos"]), 1)
        # Optional — full profile collects them
        self.assertEqual(len(ep["edge_indices"]), 1)
        self.assertEqual(len(ep["group_indices"]), 1)
        self.assertEqual(len(ep["all_log_probs"]), 1)
        self.assertEqual(len(ep["log_probs"]), 1)

    def test_qmix_profile_skips_optionals(self):
        """QMIX_PHASES.pre_step does NOT collect edge, group, or log probs."""
        ep = _make_episode_with_lists(*_ALL_TIMESTEP_ATTRS)
        QMIX_PHASES.pre_step(ep, self.ctx)

        # Essential are collected
        self.assertEqual(len(ep["alive_mask"]), 1)
        self.assertEqual(len(ep["observations"]), 1)
        # Optional are NOT collected
        self.assertEqual(len(ep["edge_indices"]), 0)
        self.assertEqual(len(ep["group_indices"]), 0)
        self.assertEqual(len(ep["all_log_probs"]), 0)
        self.assertEqual(len(ep["log_probs"]), 0)

    def test_mappo_profile_collects_log_probs_skips_graph(self):
        """MAPPO_PHASES.pre_step collects log_probs but not edge_indices."""
        ep = _make_episode_with_lists(*_ALL_TIMESTEP_ATTRS)
        MAPPO_PHASES.pre_step(ep, self.ctx)

        self.assertEqual(len(ep["all_log_probs"]), 1)
        self.assertEqual(len(ep["log_probs"]), 1)
        self.assertEqual(len(ep["edge_indices"]), 0)
        self.assertEqual(len(ep["group_indices"]), 0)

    def test_graph_qmix_profile_collects_edges_skips_log_probs(self):
        """GRAPH_QMIX_PHASES.pre_step collects edge_indices only."""
        ep = _make_episode_with_lists(*_ALL_TIMESTEP_ATTRS)
        GRAPH_QMIX_PHASES.pre_step(ep, self.ctx)

        self.assertEqual(len(ep["edge_indices"]), 1)
        self.assertEqual(len(ep["all_log_probs"]), 0)
        self.assertEqual(len(ep["group_indices"]), 0)

    def test_graph_mappo_profile_collects_both(self):
        """GRAPH_MAPPO_PHASES collects edge_indices AND log_probs."""
        ep = _make_episode_with_lists(*_ALL_TIMESTEP_ATTRS)
        GRAPH_MAPPO_PHASES.pre_step(ep, self.ctx)

        self.assertEqual(len(ep["edge_indices"]), 1)
        self.assertEqual(len(ep["all_log_probs"]), 1)
        self.assertEqual(len(ep["log_probs"]), 1)
        self.assertEqual(len(ep["group_indices"]), 0)


class TestPostStepPhase(unittest.TestCase):
    """post_step phase collects env.step() result attrs."""

    def test_post_step_collects_standard_attrs(self):
        ctx = {
            "rewards": {"a": 1.0, "b": 0.5},
            "terminations": {"a": False, "b": False},
            "truncations": {"a": False, "b": False},
            "observations": {"a": np.zeros(4), "b": np.zeros(4)},
            "all_agents_sum_rewards": 1.5,
        }
        ep = _make_episode_with_lists(
            "rewards", "terminations", "truncations",
            "next_observations", "all_agents_sum_rewards",
        )
        # All profiles use the same post_step
        FULL_PHASES.post_step(ep, ctx)

        self.assertEqual(len(ep["rewards"]), 1)
        self.assertEqual(len(ep["terminations"]), 1)
        self.assertEqual(len(ep["truncations"]), 1)
        self.assertEqual(len(ep["next_observations"]), 1)
        self.assertEqual(len(ep["all_agents_sum_rewards"]), 1)


class TestTerminalPhase(unittest.TestCase):
    """terminal phase collects next-* attrs with defaults/reuse."""

    def test_full_terminal_collects_all(self):
        """FULL terminal collects next_avail_actions, next_alive_mask, edge, group."""
        ep = _make_episode_with_lists(
            "next_avail_actions", "next_alive_mask",
            "next_edge_indices", "next_group_indices",
        )
        ctx = {
            "default_avail_actions": {"a": None},
            "default_alive_mask": {"a": False},
            "edge_indices": np.ones((2, 1)),
            "group_indices": {"a": 0},
        }
        FULL_PHASES.terminal(ep, ctx)
        self.assertEqual(len(ep["next_avail_actions"]), 1)
        self.assertEqual(len(ep["next_alive_mask"]), 1)
        self.assertEqual(len(ep["next_edge_indices"]), 1)
        self.assertEqual(len(ep["next_group_indices"]), 1)

    def test_essential_terminal_skips_optional(self):
        """QMIX terminal only collects next_avail_actions and next_alive_mask."""
        ep = _make_episode_with_lists(
            "next_avail_actions", "next_alive_mask",
            "next_edge_indices", "next_group_indices",
        )
        ctx = {
            "default_avail_actions": {},
            "default_alive_mask": {},
            "edge_indices": np.ones((2, 1)),
            "group_indices": {},
        }
        QMIX_PHASES.terminal(ep, ctx)
        self.assertEqual(len(ep["next_avail_actions"]), 1)
        self.assertEqual(len(ep["next_alive_mask"]), 1)
        self.assertEqual(len(ep["next_edge_indices"]), 0)
        self.assertEqual(len(ep["next_group_indices"]), 0)

    def test_graph_terminal_collects_edges(self):
        """GRAPH_QMIX terminal also collects next_edge_indices."""
        ep = _make_episode_with_lists(
            "next_avail_actions", "next_alive_mask",
            "next_edge_indices", "next_group_indices",
        )
        ctx = {
            "default_avail_actions": {},
            "default_alive_mask": {},
            "edge_indices": np.ones((2, 1)),
            "group_indices": {},
        }
        GRAPH_QMIX_PHASES.terminal(ep, ctx)
        self.assertEqual(len(ep["next_edge_indices"]), 1)
        self.assertEqual(len(ep["next_group_indices"]), 0)


class TestNextAttrPhase(unittest.TestCase):
    """next_attr phase collects attrs after agent.act()."""

    def test_essential_next_collects_basic(self):
        """Essential next collects next_alive_mask and next_avail_actions."""
        ep = _make_episode_with_lists(
            "next_alive_mask", "next_avail_actions",
            "next_edge_indices", "next_group_indices",
        )
        ctx = {
            "alive_mask": {"a": True},
            "avail_actions": {"a": np.ones(5)},
            "edge_indices": np.ones((2, 1)),
            "group_indices": {"a": 0},
        }
        QMIX_PHASES.next_attr(ep, ctx)
        self.assertEqual(len(ep["next_alive_mask"]), 1)
        self.assertEqual(len(ep["next_avail_actions"]), 1)
        self.assertEqual(len(ep["next_edge_indices"]), 0)
        self.assertEqual(len(ep["next_group_indices"]), 0)

    def test_graph_next_collects_edges(self):
        """GRAPH_QMIX next_attr also collects next_edge_indices."""
        ep = _make_episode_with_lists(
            "next_alive_mask", "next_avail_actions",
            "next_edge_indices", "next_group_indices",
        )
        ctx = {
            "alive_mask": {},
            "avail_actions": {},
            "edge_indices": np.ones((2, 1)),
            "group_indices": {},
        }
        GRAPH_QMIX_PHASES.next_attr(ep, ctx)
        self.assertEqual(len(ep["next_edge_indices"]), 1)
        self.assertEqual(len(ep["next_group_indices"]), 0)

    def test_group_next_collects_group(self):
        """GROUP_CONSENSUS next_attr also collects next_group_indices."""
        ep = _make_episode_with_lists(
            "next_alive_mask", "next_avail_actions",
            "next_edge_indices", "next_group_indices",
        )
        ctx = {
            "alive_mask": {},
            "avail_actions": {},
            "edge_indices": np.ones((2, 1)),
            "group_indices": {"a": 0},
        }
        GROUP_CONSENSUS_PHASES.next_attr(ep, ctx)
        self.assertEqual(len(ep["next_group_indices"]), 1)
        self.assertEqual(len(ep["next_edge_indices"]), 0)


class TestFinalizePhase(unittest.TestCase):
    """finalize phase sets episode-level scalars."""

    def test_finalize_sets_correctly(self):
        ep = {"win_tag": False, "episode_length": 0, "episode_reward": 0}
        ctx = {
            "win_tag": True,
            "episode_length": 42,
            "episode_reward": 105.5,
        }
        FULL_PHASES.finalize(ep, ctx)
        self.assertTrue(ep["win_tag"])
        self.assertEqual(ep["episode_length"], 42)
        self.assertEqual(ep["episode_reward"], 105.5)

    def test_finalize_overwrites_previous(self):
        """finalize overwrites, it does not append."""
        ep = {"win_tag": True, "episode_length": 10, "episode_reward": 1.0}
        ctx = {"win_tag": False, "episode_length": 7, "episode_reward": 3.0}
        FULL_PHASES.finalize(ep, ctx)
        self.assertFalse(ep["win_tag"])
        self.assertEqual(ep["episode_length"], 7)
        self.assertEqual(ep["episode_reward"], 3.0)


class TestPickleSafety(unittest.TestCase):
    """All RolloutPhases instances must be pickle-safe for multiprocessing."""

    def test_all_profiles_are_pickleable(self):
        for name, phases in PHASE_REGISTRY.items():
            with self.subTest(profile=name):
                data = pickle.dumps(phases)
                restored = pickle.loads(data)
                self.assertIsInstance(restored, RolloutPhases)

    def test_pickle_roundtrip_preserves_callables(self):
        """After pickle roundtrip, phase functions are still usable."""
        data = pickle.dumps(QMIX_PHASES)
        restored = pickle.loads(data)
        ep = _make_episode_with_lists(*_ALL_TIMESTEP_ATTRS)
        ctx = _make_full_context()
        restored.pre_step(ep, ctx)
        self.assertEqual(len(ep["alive_mask"]), 1,
                         "Restored pre_step should still work")


class TestCustomListPhases(unittest.TestCase):
    """Tests for resolve_phases with explicit attribute lists."""

    def test_resolve_phases_with_list(self):
        """resolve_phases(list) returns a custom RolloutPhases."""
        result = resolve_phases(["states", "actions", "rewards"])
        self.assertIsInstance(result, RolloutPhases)

    def test_resolve_phases_with_tuple(self):
        """resolve_phases(tuple) is also accepted."""
        result = resolve_phases(("alive_mask", "observations"))
        self.assertIsInstance(result, RolloutPhases)

    def test_custom_phases_collects_only_listed_attrs_pre_step(self):
        """Custom list pre_step only collects attributes in the list."""
        phases = resolve_phases(["alive_mask", "observations", "actions"])
        ep = _make_episode_with_lists(*_ALL_TIMESTEP_ATTRS)
        ctx = _make_full_context()
        phases.pre_step(ep, ctx)

        # Listed — collected
        self.assertEqual(len(ep["alive_mask"]), 1)
        self.assertEqual(len(ep["observations"]), 1)
        self.assertEqual(len(ep["actions"]), 1)
        # Not listed — skipped
        self.assertEqual(len(ep["states"]), 0)
        self.assertEqual(len(ep["avail_actions"]), 0)
        self.assertEqual(len(ep["infos"]), 0)
        self.assertEqual(len(ep["edge_indices"]), 0)
        self.assertEqual(len(ep["all_log_probs"]), 0)

    def test_custom_phases_collects_only_listed_attrs_post_step(self):
        """Custom list post_step only collects listed attrs."""
        phases = resolve_phases(["rewards", "terminations"])
        ep = _make_episode_with_lists(
            "rewards", "terminations", "truncations", "next_observations",
            "all_agents_sum_rewards",
        )
        ctx = {
            "rewards": {"a": 1.0},
            "terminations": {"a": False},
            "truncations": {"a": False},
            "observations": {"a": None},
            "all_agents_sum_rewards": 1.0,
        }
        phases.post_step(ep, ctx)
        self.assertEqual(len(ep["rewards"]), 1)
        self.assertEqual(len(ep["terminations"]), 1)
        self.assertEqual(len(ep["truncations"]), 0)
        self.assertEqual(len(ep["next_observations"]), 0)

    def test_custom_phases_collects_only_listed_attrs_terminal(self):
        """Custom list terminal only collects listed next-* attrs."""
        phases = resolve_phases(["next_avail_actions"])
        ep = _make_episode_with_lists(
            "next_avail_actions", "next_alive_mask",
            "next_edge_indices", "next_group_indices",
        )
        ctx = {
            "default_avail_actions": {"a": None},
            "default_alive_mask": {"a": False},
            "edge_indices": np.ones((2, 1)),
            "group_indices": {"a": 0},
        }
        phases.terminal(ep, ctx)
        self.assertEqual(len(ep["next_avail_actions"]), 1)
        self.assertEqual(len(ep["next_alive_mask"]), 0)
        self.assertEqual(len(ep["next_edge_indices"]), 0)

    def test_custom_phases_collects_only_listed_attrs_next(self):
        """Custom list next_attr only collects listed attrs."""
        phases = resolve_phases(["next_edge_indices"])
        ep = _make_episode_with_lists(
            "next_alive_mask", "next_avail_actions",
            "next_edge_indices", "next_group_indices",
        )
        ctx = {
            "alive_mask": {},
            "avail_actions": {},
            "edge_indices": np.ones((2, 1)),
            "group_indices": {},
        }
        phases.next_attr(ep, ctx)
        self.assertEqual(len(ep["next_edge_indices"]), 1)
        self.assertEqual(len(ep["next_alive_mask"]), 0)
        self.assertEqual(len(ep["next_avail_actions"]), 0)

    def test_custom_phases_collects_only_listed_attrs_finalize(self):
        """Custom list finalize only sets listed episode-level attrs."""
        phases = resolve_phases(["episode_reward", "win_tag"])
        ep = {"win_tag": False, "episode_length": 0, "episode_reward": 0}
        ctx = {"win_tag": True, "episode_length": 42, "episode_reward": 100.0}
        phases.finalize(ep, ctx)
        self.assertTrue(ep["win_tag"])
        self.assertEqual(ep["episode_reward"], 100.0)
        self.assertEqual(ep["episode_length"], 0, "Not listed — should not be set")

    def test_custom_list_includes_graph_attrs(self):
        """A custom list with edge_indices and group_indices collects them."""
        phases = resolve_phases([
            "alive_mask", "observations", "states", "actions",
            "avail_actions", "infos", "edge_indices", "group_indices",
            "rewards", "terminations", "truncations", "next_observations",
            "all_agents_sum_rewards", "next_states",
            "next_avail_actions", "next_alive_mask",
            "next_edge_indices", "next_group_indices",
            "episode_reward", "win_tag", "episode_length",
        ])
        ep = _make_episode_with_lists(*_ALL_TIMESTEP_ATTRS)
        ctx = _make_full_context()
        phases.pre_step(ep, ctx)
        self.assertEqual(len(ep["edge_indices"]), 1)
        self.assertEqual(len(ep["group_indices"]), 1)
        self.assertEqual(len(ep["all_log_probs"]), 0,
                         "log_probs not in list — should be skipped")

    def test_custom_phases_are_pickle_safe(self):
        """Custom phases built from a list are pickle-safe for multiprocessing."""
        phases = resolve_phases(["alive_mask", "observations", "states",
                                  "actions", "rewards", "terminations",
                                  "episode_reward", "win_tag", "episode_length"])
        data = pickle.dumps(phases)
        restored = pickle.loads(data)
        self.assertIsInstance(restored, RolloutPhases)
        # Verify the restored phases still work
        ep = _make_episode_with_lists(*_ALL_TIMESTEP_ATTRS)
        ctx = _make_full_context()
        restored.pre_step(ep, ctx)
        self.assertEqual(len(ep["alive_mask"]), 1)
        self.assertEqual(len(ep["all_log_probs"]), 0,
                         "Restored phases should still filter correctly")


if __name__ == "__main__":
    unittest.main()
