import unittest
import numpy as np
from marlite.algorithm.group_builder.label_propagation_group_builder import (
    MAgentLabelPropagationGroupBuilder,
)


class TestMAgentLabelPropagationGroupBuilder(unittest.TestCase):
    """Tests for MAgentLabelPropagationGroupBuilder including dead agents and group merging."""

    BINARY_DIM = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    PRESENCE_DIM = [1, 3]

    def _build_state(self, agents, H=20, W=20, C=37):
        """Build a single-sample state grid.

        Args:
            agents: List of (y, x, agent_id) tuples.
            H, W, C: grid dimensions.
        Returns:
            ndarray shape (H, W, C).
        """
        state = np.zeros((H, W, C), dtype=np.float16)
        for y, x, agent_id in agents:
            state[y, x, 1] = 1  # TEAM_0_PRESENCE
            # Consecutive binary encoding: agent_id = decimal value of bits
            bits = [(agent_id >> i) & 1 for i in range(len(self.BINARY_DIM))]
            for i, bit in enumerate(bits):
                if bit:
                    state[y, x, self.BINARY_DIM[i]] = 1
        return state

    def _make_builder(self, comm_distance=5, n_agents=4, n_groups=None):
        return MAgentLabelPropagationGroupBuilder(
            binary_agent_id_dim=self.BINARY_DIM,
            agent_presence_dim=self.PRESENCE_DIM,
            comm_distance=comm_distance,
            distance_metric="cityblock",
            valid_node_list=list(range(n_agents)),
            n_groups=n_groups,
        )

    # ── Basic group formation ────────────────────────────────────────────

    def test_all_agents_close_single_group(self):
        """All agents within communication range → single group."""
        state = self._build_state([(5, 5, 0), (6, 6, 1), (4, 7, 2), (7, 4, 3)])
        builder = self._make_builder(comm_distance=5, n_agents=4)
        result = builder(np.expand_dims(state, 0))
        self.assertEqual(result.shape, (1, 4))
        np.testing.assert_array_equal(result[0], [0, 0, 0, 0])

    def test_agents_separated_two_groups(self):
        """Two clusters far apart → two groups."""
        state = self._build_state([(2, 2, 0), (3, 3, 1), (15, 15, 2), (16, 16, 3)])
        builder = self._make_builder(comm_distance=3, n_agents=4)
        result = builder(np.expand_dims(state, 0))
        self.assertEqual(result.shape, (1, 4))
        self.assertEqual(result[0, 0], result[0, 1])
        self.assertEqual(result[0, 2], result[0, 3])
        self.assertNotEqual(result[0, 0], result[0, 2])

    # ── Dead agent handling ──────────────────────────────────────────────

    def test_dead_agent_gets_minus_one(self):
        """Dead agent (absent from state) gets group ID -1."""
        # Only agents 0, 1, 3 are present; agent 2 is dead
        state = self._build_state([(3, 3, 0), (4, 4, 1), (10, 10, 3)])
        builder = self._make_builder(comm_distance=5, n_agents=4)
        result = builder(np.expand_dims(state, 0))
        # agent 2 should be -1
        self.assertEqual(result[0, 2], -1)
        # alive agents should be in a group
        self.assertGreaterEqual(result[0, 0], 0)
        self.assertGreaterEqual(result[0, 1], 0)
        self.assertGreaterEqual(result[0, 3], 0)

    def test_all_dead_all_minus_one(self):
        """Empty state → all agents get -1."""
        state = self._build_state([])
        builder = self._make_builder(n_agents=4)
        result = builder(np.expand_dims(state, 0))
        np.testing.assert_array_equal(result[0], [-1, -1, -1, -1])

    def test_single_survivor(self):
        """Only one agent alive → single-agent group, others -1."""
        state = self._build_state([(5, 5, 0)])
        builder = self._make_builder(comm_distance=5, n_agents=4)
        result = builder(np.expand_dims(state, 0))
        self.assertEqual(result[0, 0], 0)
        self.assertEqual(result[0, 1], -1)
        self.assertEqual(result[0, 2], -1)
        self.assertEqual(result[0, 3], -1)

    # ── Group merging (n_groups constraint) ──────────────────────────────

    def test_no_merge_when_under_limit(self):
        """Two actual groups, n_groups=3 → no merge needed."""
        state = self._build_state([(2, 2, 0), (3, 3, 1), (15, 15, 2), (16, 16, 3)])
        builder = self._make_builder(comm_distance=3, n_agents=4, n_groups=3)
        result = builder(np.expand_dims(state, 0))
        self.assertIn(result[0, 0], [0, 1])
        self.assertIn(result[0, 2], [0, 1])
        self.assertNotEqual(result[0, 0], result[0, 2])

    def test_merge_when_over_limit(self):
        """Four groups (one per agent), n_groups=2 → merge to 2 groups."""
        # 4 agents far apart → 4 groups of size 1
        state = self._build_state([(1, 1, 0), (3, 1, 1), (1, 3, 2), (3, 3, 3)])
        builder = self._make_builder(comm_distance=1, n_agents=4, n_groups=2)
        result = builder(np.expand_dims(state, 0))
        unique_groups = set(r for r in result[0] if r >= 0)
        self.assertLessEqual(len(unique_groups), 2,
            f"Expected ≤2 groups after merge, got {unique_groups}")

    def test_merge_to_single_group(self):
        """Four isolated agents, n_groups=1 → all merged into group 0."""
        state = self._build_state([(2, 2, 0), (6, 2, 1), (10, 2, 2), (14, 2, 3)])
        builder = self._make_builder(comm_distance=1, n_agents=4, n_groups=1)
        result = builder(np.expand_dims(state, 0))
        alive = result[0] >= 0
        np.testing.assert_array_equal(
            result[0][alive], np.zeros(alive.sum(), dtype=result.dtype)
        )

    def test_merge_consecutive_labels(self):
        """After merge, group labels must be consecutive from 0."""
        # 6 agents far apart → 6 groups, merge to 3
        state = self._build_state([
            (1, 1, 0), (1, 5, 1), (1, 9, 2),
            (9, 1, 3), (9, 5, 4), (9, 9, 5),
        ])
        builder = self._make_builder(comm_distance=1, n_agents=6, n_groups=3)
        result = builder(np.expand_dims(state, 0))
        alive = result[0] >= 0
        unique = sorted(set(result[0][alive]))
        self.assertEqual(unique, list(range(len(unique))),
            f"Labels should be consecutive from 0, got {unique}")

    def test_merge_closest_first(self):
        """Merging should combine the SPATIALLY closest groups first."""
        # 3 agents in a line: a--b----c
        # b is closer to a than to c
        # n_groups=2 → should merge a+b into one, c stays as second group
        state = self._build_state([(5, 5, 0), (5, 7, 1), (5, 15, 2)])
        builder = self._make_builder(comm_distance=1, n_agents=3, n_groups=2)
        result = builder(np.expand_dims(state, 0))
        # Agents 0 and 1 are closest → same group
        self.assertEqual(result[0, 0], result[0, 1],
            "Agents 0 and 1 are closest, should be merged into same group")
        self.assertNotEqual(result[0, 0], result[0, 2],
            "Agent 2 is far, should be in different group")

    def test_merge_with_dead_agents(self):
        """Dead agents excluded from groups; merge still works on survivors."""
        state = self._build_state([(2, 2, 0), (6, 2, 1), (14, 2, 3)])
        # Agent 2 dead, 3 survivors far apart → 3 groups, merge to 1
        builder = self._make_builder(comm_distance=1, n_agents=4, n_groups=1)
        result = builder(np.expand_dims(state, 0))
        self.assertEqual(result[0, 2], -1, "Agent 2 should be dead (-1)")
        # Agents 0, 1, 3 all merged into group 0
        self.assertEqual(result[0, 0], 0)
        self.assertEqual(result[0, 1], 0)
        self.assertEqual(result[0, 3], 0)

    def test_n_groups_none_no_merge(self):
        """n_groups=None preserves original connected_components output."""
        state = self._build_state([(1, 1, 0), (1, 5, 1), (1, 9, 2)])
        builder = self._make_builder(comm_distance=1, n_agents=3, n_groups=None)
        result = builder(np.expand_dims(state, 0))
        alive = result[0] >= 0
        unique = sorted(set(result[0][alive]))
        self.assertEqual(len(unique), 3,
            "Without n_groups, each isolated agent should be its own group")

    # ── Batch processing ─────────────────────────────────────────────────

    def test_batch_independence(self):
        """Each batch element gets independent group assignments."""
        state_a = self._build_state([(2, 2, 0), (3, 3, 1)])  # one group
        state_b = self._build_state([(2, 2, 0), (15, 15, 1)])  # two groups
        states = np.stack([state_a, state_b], axis=0)
        builder = self._make_builder(comm_distance=3, n_agents=2)
        result = builder(states)
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result[0, 0], result[0, 1], "Batch 0: single group")
        self.assertNotEqual(result[1, 0], result[1, 1], "Batch 1: two groups")

    # ── Edge cases ───────────────────────────────────────────────────────

    def test_n_groups_equals_actual_groups(self):
        """n_groups == n_components → no merge, preserve labels."""
        state = self._build_state([(2, 2, 0), (15, 15, 1)])
        builder = self._make_builder(comm_distance=3, n_agents=2, n_groups=2)
        result = builder(np.expand_dims(state, 0))
        self.assertNotEqual(result[0, 0], result[0, 1])

    def test_merge_preserves_dead_labels(self):
        """Merging should not change -1 labels for dead agents."""
        state = self._build_state([(1, 1, 0), (1, 5, 1), (1, 9, 2), (9, 9, 3)])
        # agents 0,1,2,3 grouped; merge to 2
        builder = self._make_builder(comm_distance=1, n_agents=4, n_groups=2)
        result = builder(np.expand_dims(state, 0))
        # All are alive, no -1
        self.assertTrue(np.all(result[0] >= 0))

    def test_reset_method(self):
        """reset() returns self (required by GroupBuilder interface)."""
        builder = self._make_builder()
        self.assertIs(builder.reset(), builder)


class TestMergeExcessGroupsStatic(unittest.TestCase):
    """Unit tests for _merge_excess_groups static method."""

    def test_single_merge_2_of_3(self):
        coords = np.array([[0, 0], [1, 1], [10, 10]], dtype=np.float64)
        labels = np.array([0, 1, 2], dtype=np.int8)
        merged = MAgentLabelPropagationGroupBuilder._merge_excess_groups(
            coords, labels, n_groups=2
        )
        # Groups 0 and 1 are closest (dist=1.4 vs dist to group 2=14.1)
        self.assertEqual(merged[0], merged[1], "Groups 0 and 1 should merge")
        self.assertNotEqual(merged[0], merged[2], "Group 2 should remain separate")
        self.assertEqual(set(merged), {0, 1}, "Two group IDs after merge")

    def test_double_merge_4_to_1(self):
        coords = np.array([[0, 0], [2, 0], [4, 0], [6, 0]], dtype=np.float64)
        labels = np.array([0, 1, 2, 3], dtype=np.int8)
        merged = MAgentLabelPropagationGroupBuilder._merge_excess_groups(
            coords, labels, n_groups=1
        )
        np.testing.assert_array_equal(merged, np.zeros(4, dtype=np.int8))

    def test_merge_consecutive_relabel(self):
        coords = np.array([[0, 0], [1, 1], [10, 0], [10, 1], [20, 0]], dtype=np.float64)
        labels = np.array([0, 1, 2, 3, 4], dtype=np.int8)  # 5 groups
        merged = MAgentLabelPropagationGroupBuilder._merge_excess_groups(
            coords, labels, n_groups=2
        )
        # After merging to 2, labels should be 0 and 1
        self.assertEqual(set(merged), {0, 1})


if __name__ == "__main__":
    unittest.main()
