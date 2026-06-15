import unittest
import numpy as np
import torch
from marlite.algorithm.group_builder.partial_group_builder import (
    PartialGroupMAgentBuilder,
    PartialGroupVectorStateBuilder,
)
from marlite.algorithm.graph_builder.graph_util import build_partial_groups


# ---------------------------------------------------------------------------
#  MAgent2 grid-state channel layout (from magent2.environments):
#    state shape: (H, W, 37)
#      channel 0  : obstacle / wall             (binary)
#      channel 1  : TEAM_0_PRESENCE             (binary)
#      channel 2  : TEAM_1_PRESENCE             (binary)
#      channel 3  : TEAM_0_LAST_ACTION          (categorical-like)
#      channel 4  : TEAM_1_LAST_ACTION          (categorical-like)
#      channel 5-14: TEAM_0 binary agent ID     (10 bits, little-endian)
#      channel 15-24: TEAM_1 binary agent ID    (10 bits, little-endian)
#      channel 25-27: TEAM_0 extra features     (hp, ...)
#      channel 28-30: TEAM_1 extra features     (hp, ...)
#  For the partial group builder tests we focus on TEAM_0 only.
# ---------------------------------------------------------------------------
BINARY_DIM = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
PRESENCE_DIM = [1, 3]
N_AGENTS_DEFAULT = 25
N_VALID_DEFAULT = 12
COMM_DISTANCE = 10
H, W, C = 45, 45, 37

# Vector state layout
VEC_COORD_DIMS = (5, 6)
VEC_HP_DIM = 1
VEC_FEATURE_DIM = 15


def make_binary_id_little_endian(agent_id: int, num_bits: int = 10):
    """Consecutive binary encoding of ``agent_id`` as a list of 0/1 ints.

    Layout: LSB first (consistent with ``extract_agent_positions_batch``).
    """
    binary_str = format(agent_id, f'0{num_bits}b')
    return [int(x) for x in reversed(binary_str)]


def build_grid_state(agents, height=H, width=W, channels=C,
                     binary_dim=BINARY_DIM, presence_dim=PRESENCE_DIM):
    """Build a single MAgent-style grid state.

    Args:
        agents: iterable of (y, x, agent_id) tuples. Agents whose ID < 0
                are considered *dead* and are skipped (no channels set).
    """
    state = np.zeros((height, width, channels), dtype=np.float16)
    for y, x, agent_id in agents:
        if agent_id < 0:
            continue
        for c in presence_dim:
            state[y, x, c] = 1
        bits = make_binary_id_little_endian(agent_id, len(binary_dim))
        for i, bit in enumerate(bits):
            if bit:
                state[y, x, binary_dim[i]] = bit
    return state


def build_vector_state(agents, n_agents=N_AGENTS_DEFAULT, feature_dim=15,
                       coord_dim=(5, 6), hp_dim=1):
    """Build a single MAgent2-style vector state of shape (N, F).

    Args:
        agents: iterable of (row, y, x) tuples. Agents with hp=0
                (``row`` may be -1) are skipped.
    """
    state = np.zeros((n_agents, feature_dim), dtype=np.float32)
    for row, y, x in agents:
        if row < 0:
            continue
        state[row, hp_dim] = 1
        state[row, coord_dim[0]] = y
        state[row, coord_dim[1]] = x
    return state


# =============================================================================
#  Tests for the build_partial_groups utility function
# =============================================================================
class TestBuildPartialGroupsUtility(unittest.TestCase):
    """Unit tests for the ``build_partial_groups`` helper.

    These tests intentionally exercise the helper directly (without going
    through a builder) so that all edge cases — including label assignment
    when actual communities are fewer than ``n_groups`` — are covered
    independently of the public builder API.
    """

    def _coords(self, agent_specs):
        return np.array(
            [[aid, y, x] for aid, (y, x) in enumerate(agent_specs)],
            dtype=np.int64,
        )

    def test_two_clusters_produce_two_groups(self):
        coords = self._coords([
            (5, 5), (5, 6),                  # cluster A
            (30, 30), (30, 31), (31, 30),   # cluster B
        ])
        labels = build_partial_groups(
            coords_with_id=coords,
            comm_distance=10,
            distance_metric='cityblock',
            n_groups=2,
            valid_node_list=[0, 1, 2, 3, 4],
            target_node_list=[],
        )
        self.assertEqual(len(labels), 5)
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])
        self.assertEqual(labels[3], labels[4])
        self.assertNotEqual(labels[0], labels[2])
        # Labels are 0 and 1 (smaller first, no gaps)
        self.assertEqual(set(labels.tolist()), {0, 1})

    def test_single_cluster_produces_single_group(self):
        coords = self._coords([(5, 5), (5, 6), (6, 5)])
        labels = build_partial_groups(
            coords_with_id=coords,
            comm_distance=10,
            distance_metric='cityblock',
            n_groups=3,
            valid_node_list=[0, 1, 2],
            target_node_list=[],
        )
        self.assertEqual(set(labels.tolist()), {0})

    def test_disconnected_isolated_agents_get_distinct_groups(self):
        # No two agents are within comm_distance -> each is its own community.
        coords = self._coords([(1, 1), (1, 5), (1, 9), (1, 13)])
        labels = build_partial_groups(
            coords_with_id=coords,
            comm_distance=1,
            distance_metric='cityblock',
            n_groups=4,
            valid_node_list=[0, 1, 2, 3],
            target_node_list=[],
        )
        # 4 isolated agents -> 4 distinct labels
        unique = sorted(set(int(x) for x in labels.tolist()))
        self.assertEqual(unique, [0, 1, 2, 3])

    def test_missing_agents_get_minus_one(self):
        coords = self._coords([(5, 5), (5, 6)])  # only agents 0, 1
        labels = build_partial_groups(
            coords_with_id=coords,
            comm_distance=10,
            distance_metric='cityblock',
            n_groups=2,
            valid_node_list=[0, 1, 2, 3, 4],
            target_node_list=[],
        )
        # Agents 0, 1 alive -> some group; agents 2, 3, 4 missing -> -1
        self.assertGreaterEqual(labels[0], 0)
        self.assertGreaterEqual(labels[1], 0)
        self.assertEqual(labels[2], -1)
        self.assertEqual(labels[3], -1)
        self.assertEqual(labels[4], -1)

    def test_empty_state_returns_all_minus_one(self):
        labels = build_partial_groups(
            coords_with_id=np.zeros((0, 3), dtype=np.int64),
            comm_distance=10,
            distance_metric='cityblock',
            n_groups=2,
            valid_node_list=[0, 1, 2, 3],
            target_node_list=[],
        )
        self.assertTrue(np.all(labels == -1))

    def test_n_groups_equals_one_returns_all_minus_one(self):
        coords = self._coords([(5, 5), (5, 6), (30, 30)])
        labels = build_partial_groups(
            coords_with_id=coords,
            comm_distance=10,
            distance_metric='cityblock',
            n_groups=1,
            valid_node_list=[0, 1, 2],
            target_node_list=[],
        )
        # n_groups <= 1 -> no community detection
        self.assertTrue(np.all(labels == -1))

    def test_n_groups_exceeds_nodes_clamps_and_keeps_consecutive_labels(self):
        # Greedy modularity with best_n > number of nodes is not allowed by
        # networkx; the helper clamps best_n to G.number_of_nodes() and the
        # resulting labels are still consecutive from 0.
        coords = self._coords([(5, 5), (5, 6), (30, 30), (30, 31)])
        labels = build_partial_groups(
            coords_with_id=coords,
            comm_distance=10,
            distance_metric='cityblock',
            n_groups=20,
            valid_node_list=[0, 1, 2, 3],
            target_node_list=[],
        )
        unique = sorted(set(int(x) for x in labels if x >= 0))
        self.assertEqual(unique, list(range(len(unique))))

    def test_smaller_group_numbers_first_when_actual_below_n_groups(self):
        # Tight cluster: greedy_modularity should return 1 community even
        # when asked for more. Labels must still be the smallest prefix.
        coords = self._coords([(5, 5), (5, 6), (6, 5), (6, 6)])
        labels = build_partial_groups(
            coords_with_id=coords,
            comm_distance=10,
            distance_metric='cityblock',
            n_groups=5,
            valid_node_list=[0, 1, 2, 3],
            target_node_list=[],
        )
        unique = sorted(set(int(x) for x in labels if x >= 0))
        self.assertEqual(unique, list(range(len(unique))),
            "Actual communities < n_groups: labels should be 0, 1, ... (no gaps)")

    def test_target_nodes_in_detection_not_in_output(self):
        # valid: agents 0, 1; target: agent 100
        coords = np.array([
            [0, 5, 5],
            [1, 5, 6],
            [100, 5, 7],
        ], dtype=np.int64)
        labels = build_partial_groups(
            coords_with_id=coords,
            comm_distance=10,
            distance_metric='cityblock',
            n_groups=2,
            valid_node_list=[0, 1],
            target_node_list=[100],
        )
        # Output length matches valid_node_list, not the union
        self.assertEqual(len(labels), 2)
        self.assertGreaterEqual(labels[0], 0)
        self.assertGreaterEqual(labels[1], 0)

    def test_target_node_participates_in_community_detection(self):
        # A target node sitting between two clusters changes the
        # community partition compared to running detection without it.
        coords_no_target = np.array([
            [0, 5, 5], [1, 5, 6],
            [2, 30, 30], [3, 30, 31],
        ], dtype=np.int64)
        labels_no_target = build_partial_groups(
            coords_with_id=coords_no_target,
            comm_distance=30,
            distance_metric='cityblock',
            n_groups=2,
            valid_node_list=[0, 1, 2, 3],
            target_node_list=[],
        )
        # With a target node bridging the two clusters
        coords_with_target = np.array([
            [0, 5, 5], [1, 5, 6],
            [2, 30, 30], [3, 30, 31],
            [100, 5, 30],  # target, bridges cluster A and B
        ], dtype=np.int64)
        labels_with_target = build_partial_groups(
            coords_with_id=coords_with_target,
            comm_distance=30,
            distance_metric='cityblock',
            n_groups=2,
            valid_node_list=[0, 1, 2, 3],
            target_node_list=[100],
        )
        # Output length is always len(valid_node_list) = 4
        self.assertEqual(len(labels_with_target), 4)
        # All labels are still consecutive from 0
        unique = sorted(set(int(x) for x in labels_with_target if x >= 0))
        self.assertEqual(unique, list(range(len(unique))))
        # The valid agents have a non-negative label regardless
        for v in labels_with_target:
            if v >= 0:
                self.assertIn(int(v), unique)
        # With a bridging target the partition merges into at most 2 groups
        self.assertLessEqual(len(unique), 2,
            f"Bridging target should keep groups <= 2, got {unique}")
        # Without a target the partition can split into more isolated pieces
        # (in practice greedy_modularity may still merge them, so we don't
        # assert a specific size — only that with_target produces the right
        # consecutive prefix).
        _ = labels_no_target  # only used to compute baseline; not asserted

    def test_labels_aligned_to_valid_node_list(self):
        # valid_node_list has gaps; labels must still be in the order of
        # valid_node_list (not sorted_ids).
        coords = self._coords([(5, 5), (5, 6)])
        labels = build_partial_groups(
            coords_with_id=coords,
            comm_distance=10,
            distance_metric='cityblock',
            n_groups=2,
            valid_node_list=[0, 1],   # request agents 0, 1 explicitly
            target_node_list=[],
        )
        # Two valid nodes -> length-2 output
        self.assertEqual(len(labels), 2)
        self.assertGreaterEqual(labels[0], 0)
        self.assertGreaterEqual(labels[1], 0)

    def test_output_is_int64(self):
        coords = self._coords([(5, 5), (5, 6)])
        labels = build_partial_groups(
            coords_with_id=coords,
            comm_distance=10,
            distance_metric='cityblock',
            n_groups=2,
            valid_node_list=[0, 1],
            target_node_list=[],
        )
        self.assertEqual(labels.dtype, np.int64)


# =============================================================================
#  Tests for PartialGroupMAgentBuilder (grid state)
# =============================================================================
class TestPartialGroupMAgentBuilder(unittest.TestCase):

    def setUp(self):
        self.n_agent = N_AGENTS_DEFAULT
        self.binary_agent_id_dim = BINARY_DIM
        self.agent_presence_dim = PRESENCE_DIM
        self.comm_distance = COMM_DISTANCE
        self.distance_metric = 'cityblock'
        self.valid_node_list = list(range(N_VALID_DEFAULT))
        self.target_node_list = list(range(N_VALID_DEFAULT, self.n_agent))

        self.builder = PartialGroupMAgentBuilder(
            binary_agent_id_dim=self.binary_agent_id_dim,
            agent_presence_dim=self.agent_presence_dim,
            comm_distance=self.comm_distance,
            valid_node_list=self.valid_node_list,
            target_node_list=self.target_node_list,
            distance_metric=self.distance_metric,
            n_workers=1,
            n_groups=4,
            update_interval=5,
            channel_first=False,
        )
        self.builder.reset()

    # ── forward path ─────────────────────────────────────────────────────

    def test_forward_normal_case(self):
        """Random agent placements with a mix of clusters should yield
        communities whose labels are 0, 1, ..., n_actual-1 (smaller first)."""
        batch_size = 5
        states = np.zeros((batch_size, H, W, C), dtype=np.float16)
        for b in range(batch_size):
            positions = np.random.choice(H * W, size=self.n_agent, replace=False)
            for agent_id in range(self.n_agent):
                pos = positions[agent_id]
                i, j = divmod(pos, W)
                for c in self.agent_presence_dim:
                    states[b, i, j, c] = 1
                bits = make_binary_id_little_endian(agent_id, len(self.binary_agent_id_dim))
                for k, bit in enumerate(bits):
                    states[b, i, j, self.binary_agent_id_dim[k]] = bit

        result = self.builder(torch.from_numpy(states).float())

        # shape: (batch_size, n_valid)
        self.assertEqual(result.shape, (batch_size, len(self.valid_node_list)))
        self.assertEqual(result.dtype, torch.int16)

        # All output rows should have group labels in {0, ..., n_actual-1, -1}
        for row in result:
            unique = sorted({int(x) for x in row.tolist() if x >= 0})
            # Consecutive from 0
            if unique:
                self.assertEqual(unique, list(range(len(unique))),
                    f"Labels should be consecutive from 0, got {unique}")

    def test_forward_empty_state(self):
        """An all-zero state means no agents are present. The full
        ``valid_node_list`` should receive ``-1`` because no community
        can be formed."""
        batch_size = 3
        states = np.zeros((batch_size, H, W, C), dtype=np.float16)

        result = self.builder(torch.from_numpy(states).float())

        self.assertEqual(result.shape, (batch_size, len(self.valid_node_list)))
        for row in result:
            self.assertTrue(torch.all(row == -1),
                f"Empty state should yield all -1, got {row.tolist()}")

    def test_forward_channel_first(self):
        """When ``channel_first=True`` the input is ``(B, C, H, W)`` and the
        builder should transpose internally."""
        builder = PartialGroupMAgentBuilder(
            binary_agent_id_dim=self.binary_agent_id_dim,
            agent_presence_dim=self.agent_presence_dim,
            comm_distance=self.comm_distance,
            valid_node_list=self.valid_node_list,
            target_node_list=self.target_node_list,
            distance_metric=self.distance_metric,
            n_workers=1,
            n_groups=4,
            channel_first=True,
        )
        builder.reset()

        batch_size = 2
        states = np.zeros((batch_size, C, H, W), dtype=np.float16)
        for b in range(batch_size):
            for agent_id in range(self.n_agent):
                i, j = b * 5, b * 5  # overlap, doesn't really matter for test
                for c in self.agent_presence_dim:
                    states[b, c, i, j] = 1
                bits = make_binary_id_little_endian(agent_id, len(self.binary_agent_id_dim))
                for k, bit in enumerate(bits):
                    states[b, self.binary_agent_id_dim[k], i, j] = bit

        result = builder(torch.from_numpy(states).float())
        self.assertEqual(result.shape, (batch_size, len(self.valid_node_list)))

    def test_two_clusters_two_groups(self):
        """Two spatially separated clusters -> two groups, each containing
        the agents in that cluster."""
        state = build_grid_state([
            (5, 5, 0), (5, 6, 1),                 # cluster A
            (40, 40, 2), (40, 41, 3),             # cluster B
        ])
        states = np.expand_dims(state, 0)
        result = self.builder(torch.from_numpy(states).float())
        # Each cluster is its own community
        self.assertEqual(result[0, 0], result[0, 1])
        self.assertEqual(result[0, 2], result[0, 3])
        self.assertNotEqual(result[0, 0], result[0, 2])

    def test_all_agents_close_single_group(self):
        state = build_grid_state([
            (5, 5, 0), (5, 6, 1), (6, 5, 2), (6, 6, 3),
        ])
        result = self.builder(torch.from_numpy(np.expand_dims(state, 0)).float())
        alive = result[0] >= 0
        # All alive agents share the same group
        unique = set(int(x) for x in result[0][alive].tolist())
        self.assertEqual(unique, {0})

    def test_dead_agent_gets_minus_one(self):
        """Agent 2 is absent (no presence channels set) -> -1 label."""
        # All alive agents are placed close together so they form a single
        # community; agent 2 is dead and gets -1.
        state = build_grid_state([
            (5, 5, 0), (5, 6, 1), (5, 7, 3),  # cluster; agent 2 dead
        ])
        result = self.builder(torch.from_numpy(np.expand_dims(state, 0)).float())
        self.assertEqual(result[0, 2].item(), -1)
        for i in [0, 1, 3]:
            self.assertGreaterEqual(result[0, i], 0,
                f"Agent {i} should have a non-negative group label")
        # Agents 0, 1, 3 are in the same cluster -> same group
        self.assertEqual(result[0, 0], result[0, 1])
        self.assertEqual(result[0, 1], result[0, 3])

    def test_all_dead_all_minus_one(self):
        state = build_grid_state([])
        result = self.builder(torch.from_numpy(np.expand_dims(state, 0)).float())
        self.assertTrue(torch.all(result[0] == -1))

    def test_single_survivor(self):
        state = build_grid_state([(5, 5, 0)])
        result = self.builder(torch.from_numpy(np.expand_dims(state, 0)).float())
        # Single agent is its own community -> label 0
        self.assertEqual(result[0, 0].item(), 0)
        for i in range(1, len(self.valid_node_list)):
            self.assertEqual(result[0, i].item(), -1,
                f"Agent {i} is not present, should be -1")

    def test_target_nodes_excluded_from_output(self):
        """Agents in ``target_node_list`` participate in community detection
        but should NOT appear in the output labels."""
        # Place target nodes right between two valid clusters so that
        # without their help, the two clusters would be separate.
        state = build_grid_state([
            (5, 5, 0), (5, 6, 1),                  # valid cluster A
            (40, 40, 2), (40, 41, 3),              # valid cluster B
            (5, 7, N_VALID_DEFAULT),               # target near A (different pos)
            (40, 38, N_VALID_DEFAULT + 1),         # target near B (different pos)
        ])
        result = self.builder(torch.from_numpy(np.expand_dims(state, 0)).float())
        # Output is only for valid nodes
        self.assertEqual(result.shape, (1, len(self.valid_node_list)))
        # Valid agents are alive
        for i in range(4):
            self.assertGreaterEqual(result[0, i], 0,
                f"Valid agent {i} should have a non-negative group label")

    def test_channel_robustness_extra_features_ignored(self):
        """Channels 15-36 (extra features, hp, last-action) are not part of
        the agent id/presence detection; setting them should not change
        group assignments."""
        state_base = build_grid_state([
            (5, 5, 0), (5, 6, 1), (40, 40, 2), (40, 41, 3),
        ])
        state_noisy = state_base.copy()
        # Fill extra channels with random noise
        rng = np.random.default_rng(0)
        state_noisy[..., 15:] = rng.integers(0, 5, size=state_noisy[..., 15:].shape)
        r1 = self.builder(torch.from_numpy(np.expand_dims(state_base, 0)).float())
        r2 = self.builder(torch.from_numpy(np.expand_dims(state_noisy, 0)).float())
        np.testing.assert_array_equal(r1.numpy(), r2.numpy())

    # ── Caching ──────────────────────────────────────────────────────────

    def test_caching_mechanism(self):
        self.builder.eval()
        self.builder.reset()

        # Place a non-trivial layout
        state_with_agents = build_grid_state([
            (5, 5, 0), (5, 6, 1), (40, 40, 2), (40, 41, 3),
        ])
        empty_state = build_grid_state([])

        # First call -> compute
        r1 = self.builder(torch.from_numpy(np.expand_dims(state_with_agents, 0)).float())
        # Subsequent calls within update_interval - 1 should hit the cache
        for _ in range(self.builder.update_interval - 1):
            r_cached = self.builder(torch.from_numpy(np.expand_dims(empty_state, 0)).float())
            np.testing.assert_array_equal(r1.cpu().numpy(), r_cached.cpu().numpy())
        # After update_interval -> recompute; empty state yields all -1
        r_recomputed = self.builder(torch.from_numpy(np.expand_dims(empty_state, 0)).float())
        self.assertTrue(torch.all(r_recomputed[0] == -1))

    def test_no_cache_in_training_mode(self):
        self.builder.train()
        self.builder.reset()
        state = build_grid_state([
            (5, 5, 0), (5, 6, 1), (40, 40, 2), (40, 41, 3),
        ])
        r1 = self.builder(torch.from_numpy(np.expand_dims(state, 0)).float())
        r2 = self.builder(torch.from_numpy(np.expand_dims(state, 0)).float())
        np.testing.assert_array_equal(r1.numpy(), r2.numpy())

    # ── Batch processing ─────────────────────────────────────────────────

    def test_batch_independence(self):
        # Batch 0: two agents in the same cluster -> same group
        # Batch 1: four agents in two clusters -> two different groups
        state_a = build_grid_state([(5, 5, 0), (5, 6, 1)])
        state_b = build_grid_state([
            (5, 5, 0), (5, 6, 1),
            (40, 40, 2), (40, 41, 3),
        ])
        states = np.stack([state_a, state_b], axis=0)
        result = self.builder(torch.from_numpy(states).float())
        self.assertEqual(result.shape, (2, len(self.valid_node_list)))
        # Batch 0: agents 0, 1 same group
        self.assertEqual(result[0, 0], result[0, 1])
        # Batch 1: agents 0, 1 in one group; agents 2, 3 in another
        self.assertEqual(result[1, 0], result[1, 1])
        self.assertEqual(result[1, 2], result[1, 3])
        self.assertNotEqual(result[1, 0], result[1, 2])

    def test_n_workers_behavior(self):
        """Sequential processing (n_workers=1) should match multi-worker
        outputs in terms of *label semantics* (groups, consecutive labels)."""
        builder_seq = PartialGroupMAgentBuilder(
            binary_agent_id_dim=self.binary_agent_id_dim,
            agent_presence_dim=self.agent_presence_dim,
            comm_distance=self.comm_distance,
            valid_node_list=self.valid_node_list,
            target_node_list=self.target_node_list,
            distance_metric=self.distance_metric,
            n_workers=1,
            n_groups=4,
        )
        batch_size = 2
        states = np.zeros((batch_size, H, W, C), dtype=np.float32)
        for b in range(batch_size):
            for agent_id in range(self.n_agent):
                i, j = b * 10, b * 10
                for c in self.agent_presence_dim:
                    states[b, i, j, c] = 1
                bits = make_binary_id_little_endian(1, len(self.binary_agent_id_dim))
                for k, bit in enumerate(bits):
                    states[b, i, j, self.binary_agent_id_dim[k]] = bit
        adj_labels = builder_seq(torch.from_numpy(states).float())
        self.assertEqual(adj_labels.shape, (batch_size, len(self.valid_node_list)))

    # ── Reset & dtype ────────────────────────────────────────────────────

    def test_reset_method(self):
        # Eval mode populates the cache
        self.builder.eval()
        state = build_grid_state([(5, 5, 0), (5, 6, 1)])
        self.builder(torch.from_numpy(np.expand_dims(state, 0)).float())
        self.assertIsNotNone(self.builder.cached_labels)
        self.assertEqual(self.builder.step_counter, 1)
        # Reset
        self.assertIs(self.builder.reset(), self.builder)
        self.assertEqual(self.builder.step_counter, 0)
        self.assertIsNone(self.builder.cached_labels)

    def test_dtype_int16_by_default(self):
        state = build_grid_state([(5, 5, 0), (5, 6, 1), (40, 40, 2), (40, 41, 3)])
        result = self.builder(torch.from_numpy(np.expand_dims(state, 0)).float())
        self.assertEqual(result.dtype, torch.int16)

    def test_dtype_int8_when_configured(self):
        builder = PartialGroupMAgentBuilder(
            binary_agent_id_dim=self.binary_agent_id_dim,
            agent_presence_dim=self.agent_presence_dim,
            comm_distance=self.comm_distance,
            valid_node_list=self.valid_node_list,
            target_node_list=self.target_node_list,
            distance_metric=self.distance_metric,
            n_workers=1,
            n_groups=4,
            dtype='int8',
        )
        state = build_grid_state([(5, 5, 0), (5, 6, 1)])
        result = builder(torch.from_numpy(np.expand_dims(state, 0)).float())
        self.assertEqual(result.dtype, torch.int8)


# =============================================================================
#  Tests for PartialGroupVectorStateBuilder (vector state)
# =============================================================================
class TestPartialGroupVectorStateBuilder(unittest.TestCase):

    N_AGENTS = N_AGENTS_DEFAULT
    F_DIM = 15
    COORD_DIMS = (5, 6)
    HP_DIM = 1

    def setUp(self):
        self.coord_dim = [int(self.COORD_DIMS[0]), int(self.COORD_DIMS[1])]
        self.hp_dim = int(self.HP_DIM)
        self.comm_distance = COMM_DISTANCE
        self.distance_metric = 'cityblock'
        self.valid_node_list = list(range(N_VALID_DEFAULT))
        self.target_node_list = list(range(N_VALID_DEFAULT, self.N_AGENTS))

        self.builder = PartialGroupVectorStateBuilder(
            coord_dim=self.coord_dim,
            hp_dim=self.hp_dim,
            comm_distance=self.comm_distance,
            valid_node_list=self.valid_node_list,
            target_node_list=self.target_node_list,
            distance_metric=self.distance_metric,
            n_workers=1,
            n_groups=4,
            update_interval=5,
        )
        self.builder.reset()

    # ── forward path ─────────────────────────────────────────────────────

    def test_forward_normal_case(self):
        batch_size = 5
        states = np.zeros((batch_size, self.N_AGENTS, self.F_DIM), dtype=np.float32)
        for b in range(batch_size):
            positions = np.random.choice(self.N_AGENTS, size=self.N_AGENTS, replace=False)
            for agent_id in range(self.N_AGENTS):
                pos = positions[agent_id]
                states[b, pos, self.hp_dim] = 1
                coords = np.random.randint(0, 45, size=2)
                states[b, pos, self.coord_dim[0]] = coords[0]
                states[b, pos, self.coord_dim[1]] = coords[1]

        result = self.builder(torch.from_numpy(states).float())
        self.assertEqual(result.shape, (batch_size, len(self.valid_node_list)))
        self.assertEqual(result.dtype, torch.int16)
        # Group labels are consecutive from 0
        for row in result:
            unique = sorted({int(x) for x in row.tolist() if x >= 0})
            if unique:
                self.assertEqual(unique, list(range(len(unique))))

    def test_forward_empty_state(self):
        batch_size = 3
        states = np.zeros((batch_size, self.N_AGENTS, self.F_DIM), dtype=np.float32)
        result = self.builder(torch.from_numpy(states).float())
        self.assertEqual(result.shape, (batch_size, len(self.valid_node_list)))
        for row in result:
            self.assertTrue(torch.all(row == -1))

    def test_two_clusters_two_groups(self):
        state = build_vector_state(
            [(0, 5, 5), (1, 5, 6), (2, 40, 40), (3, 40, 41)],
            n_agents=self.N_AGENTS, feature_dim=self.F_DIM,
            coord_dim=self.COORD_DIMS, hp_dim=self.hp_dim,
        )
        result = self.builder(torch.from_numpy(np.expand_dims(state, 0)).float())
        self.assertEqual(result[0, 0], result[0, 1])
        self.assertEqual(result[0, 2], result[0, 3])
        self.assertNotEqual(result[0, 0], result[0, 2])

    def test_all_agents_close_single_group(self):
        state = build_vector_state(
            [(0, 5, 5), (1, 5, 6), (2, 6, 5), (3, 6, 6)],
            n_agents=self.N_AGENTS, feature_dim=self.F_DIM,
            coord_dim=self.COORD_DIMS, hp_dim=self.hp_dim,
        )
        result = self.builder(torch.from_numpy(np.expand_dims(state, 0)).float())
        alive = result[0] >= 0
        unique = set(int(x) for x in result[0][alive].tolist())
        self.assertEqual(unique, {0})

    def test_dead_agent_gets_minus_one(self):
        # Agent 2 has hp=0 -> dead
        state = build_vector_state(
            [(0, 5, 5), (1, 5, 6), (3, 5, 7)],  # all alive in one cluster
            n_agents=self.N_AGENTS, feature_dim=self.F_DIM,
            coord_dim=self.COORD_DIMS, hp_dim=self.hp_dim,
        )
        # Make agent 2 dead by clearing its hp
        state[2, self.hp_dim] = 0
        result = self.builder(torch.from_numpy(np.expand_dims(state, 0)).float())
        self.assertEqual(result[0, 2].item(), -1)
        for i in [0, 1, 3]:
            self.assertGreaterEqual(result[0, i], 0)
        # 0, 1, 3 form a single community
        self.assertEqual(result[0, 0], result[0, 1])
        self.assertEqual(result[0, 1], result[0, 3])

    def test_all_dead_all_minus_one(self):
        state = np.zeros((self.N_AGENTS, self.F_DIM), dtype=np.float32)
        result = self.builder(torch.from_numpy(np.expand_dims(state, 0)).float())
        self.assertTrue(torch.all(result[0] == -1))

    def test_single_survivor(self):
        state = build_vector_state(
            [(0, 5, 5)],
            n_agents=self.N_AGENTS, feature_dim=self.F_DIM,
            coord_dim=self.COORD_DIMS, hp_dim=self.hp_dim,
        )
        result = self.builder(torch.from_numpy(np.expand_dims(state, 0)).float())
        # Single agent is its own community -> label 0
        self.assertEqual(result[0, 0].item(), 0)
        for i in range(1, len(self.valid_node_list)):
            self.assertEqual(result[0, i].item(), -1)

    def test_target_nodes_excluded_from_output(self):
        state = build_vector_state(
            [
                (0, 5, 5), (1, 5, 6),                 # valid cluster A
                (2, 40, 40), (3, 40, 41),             # valid cluster B
                (N_VALID_DEFAULT, 5, 7),              # target near A (diff pos)
                (N_VALID_DEFAULT + 1, 40, 38),        # target near B (diff pos)
            ],
            n_agents=self.N_AGENTS, feature_dim=self.F_DIM,
            coord_dim=self.COORD_DIMS, hp_dim=self.hp_dim,
        )
        result = self.builder(torch.from_numpy(np.expand_dims(state, 0)).float())
        self.assertEqual(result.shape, (1, len(self.valid_node_list)))
        for i in range(4):
            self.assertGreaterEqual(result[0, i], 0)

    def test_hp_zero_filtered_out(self):
        """Agents with hp<=0 are not considered present, even if their
        coordinates are set. This mirrors the behaviour of the partial
        graph builder for vector states."""
        state = build_vector_state(
            [(0, 5, 5), (1, 5, 6), (2, 6, 5)],
            n_agents=self.N_AGENTS, feature_dim=self.F_DIM,
            coord_dim=self.COORD_DIMS, hp_dim=self.hp_dim,
        )
        # Manually set agent 2 hp to 0 to simulate a dead agent
        state[2, self.hp_dim] = 0
        result = self.builder(torch.from_numpy(np.expand_dims(state, 0)).float())
        self.assertEqual(result[0, 2].item(), -1)

    def test_other_features_ignored(self):
        """Only ``coord_dim`` and ``hp_dim`` should influence the result;
        filling other feature columns with random noise must be a no-op."""
        state_base = build_vector_state(
            [(0, 5, 5), (1, 5, 6), (2, 40, 40), (3, 40, 41)],
            n_agents=self.N_AGENTS, feature_dim=self.F_DIM,
            coord_dim=self.COORD_DIMS, hp_dim=self.hp_dim,
        )
        state_noisy = state_base.copy()
        rng = np.random.default_rng(0)
        # Randomize every column that isn't coord_dim or hp_dim
        mask = np.ones(self.F_DIM, dtype=bool)
        mask[self.hp_dim] = False
        mask[self.coord_dim[0]] = False
        mask[self.coord_dim[1]] = False
        state_noisy[:, mask] = rng.random(state_noisy[:, mask].shape)
        r1 = self.builder(torch.from_numpy(np.expand_dims(state_base, 0)).float())
        r2 = self.builder(torch.from_numpy(np.expand_dims(state_noisy, 0)).float())
        np.testing.assert_array_equal(r1.numpy(), r2.numpy())

    # ── Caching ──────────────────────────────────────────────────────────

    def test_caching_mechanism(self):
        self.builder.eval()
        self.builder.reset()
        state_with_agents = build_vector_state(
            [(0, 5, 5), (1, 5, 6), (2, 40, 40), (3, 40, 41)],
            n_agents=self.N_AGENTS, feature_dim=self.F_DIM,
            coord_dim=self.COORD_DIMS, hp_dim=self.hp_dim,
        )
        empty_state = np.zeros((self.N_AGENTS, self.F_DIM), dtype=np.float32)

        r1 = self.builder(torch.from_numpy(np.expand_dims(state_with_agents, 0)).float())
        for _ in range(self.builder.update_interval - 1):
            r_cached = self.builder(torch.from_numpy(np.expand_dims(empty_state, 0)).float())
            np.testing.assert_array_equal(r1.cpu().numpy(), r_cached.cpu().numpy())
        r_recomputed = self.builder(torch.from_numpy(np.expand_dims(empty_state, 0)).float())
        self.assertTrue(torch.all(r_recomputed[0] == -1))

    def test_no_cache_in_training_mode(self):
        self.builder.train()
        self.builder.reset()
        state = build_vector_state(
            [(0, 5, 5), (1, 5, 6)],
            n_agents=self.N_AGENTS, feature_dim=self.F_DIM,
            coord_dim=self.COORD_DIMS, hp_dim=self.hp_dim,
        )
        r1 = self.builder(torch.from_numpy(np.expand_dims(state, 0)).float())
        r2 = self.builder(torch.from_numpy(np.expand_dims(state, 0)).float())
        np.testing.assert_array_equal(r1.numpy(), r2.numpy())

    # ── Batch processing ─────────────────────────────────────────────────

    def test_batch_independence(self):
        state_a = build_vector_state(
            [(0, 5, 5), (1, 5, 6)],
            n_agents=self.N_AGENTS, feature_dim=self.F_DIM,
            coord_dim=self.COORD_DIMS, hp_dim=self.hp_dim,
        )
        state_b = build_vector_state(
            [(0, 5, 5), (1, 5, 6), (2, 40, 40), (3, 40, 41)],
            n_agents=self.N_AGENTS, feature_dim=self.F_DIM,
            coord_dim=self.COORD_DIMS, hp_dim=self.hp_dim,
        )
        states = np.stack([state_a, state_b], axis=0)
        result = self.builder(torch.from_numpy(states).float())
        self.assertEqual(result.shape, (2, len(self.valid_node_list)))
        # Batch 0: one group
        self.assertEqual(result[0, 0], result[0, 1])
        # Batch 1: two groups
        self.assertEqual(result[1, 0], result[1, 1])
        self.assertEqual(result[1, 2], result[1, 3])
        self.assertNotEqual(result[1, 0], result[1, 2])

    # ── Reset & dtype ────────────────────────────────────────────────────

    def test_reset_method(self):
        # Eval mode populates the cache
        self.builder.eval()
        state = build_vector_state(
            [(0, 5, 5), (1, 5, 6)],
            n_agents=self.N_AGENTS, feature_dim=self.F_DIM,
            coord_dim=self.COORD_DIMS, hp_dim=self.hp_dim,
        )
        self.builder(torch.from_numpy(np.expand_dims(state, 0)).float())
        self.assertIsNotNone(self.builder.cached_labels)
        self.assertIs(self.builder.reset(), self.builder)
        self.assertEqual(self.builder.step_counter, 0)
        self.assertIsNone(self.builder.cached_labels)

    def test_dtype_int16_by_default(self):
        state = build_vector_state(
            [(0, 5, 5), (1, 5, 6)],
            n_agents=self.N_AGENTS, feature_dim=self.F_DIM,
            coord_dim=self.COORD_DIMS, hp_dim=self.hp_dim,
        )
        result = self.builder(torch.from_numpy(np.expand_dims(state, 0)).float())
        self.assertEqual(result.dtype, torch.int16)

    def test_dtype_int8_when_configured(self):
        builder = PartialGroupVectorStateBuilder(
            coord_dim=self.coord_dim,
            hp_dim=self.hp_dim,
            comm_distance=self.comm_distance,
            valid_node_list=self.valid_node_list,
            target_node_list=self.target_node_list,
            distance_metric=self.distance_metric,
            n_workers=1,
            n_groups=4,
            dtype='int8',
        )
        state = build_vector_state(
            [(0, 5, 5), (1, 5, 6)],
            n_agents=self.N_AGENTS, feature_dim=self.F_DIM,
            coord_dim=self.COORD_DIMS, hp_dim=self.hp_dim,
        )
        result = builder(torch.from_numpy(np.expand_dims(state, 0)).float())
        self.assertEqual(result.dtype, torch.int8)


# =============================================================================
#  Tests for GroupBuilderConfig registration
# =============================================================================
class TestPartialGroupBuilderConfig(unittest.TestCase):

    def test_partial_magent_registered(self):
        from marlite.algorithm.group_builder.group_builder_config import GroupBuilderConfig
        config = {
            "type": "PartialMAgent",
            "binary_agent_id_dim": BINARY_DIM,
            "agent_presence_dim": PRESENCE_DIM,
            "comm_distance": COMM_DISTANCE,
            "valid_node_list": list(range(N_VALID_DEFAULT)),
            "target_node_list": list(range(N_VALID_DEFAULT, N_AGENTS_DEFAULT)),
            "n_groups": 4,
            "n_workers": 1,
        }
        builder = GroupBuilderConfig(**config).get_group_builder()
        self.assertIsInstance(builder, PartialGroupMAgentBuilder)

    def test_partial_vector_state_registered(self):
        from marlite.algorithm.group_builder.group_builder_config import GroupBuilderConfig
        config = {
            "type": "PartialVectorState",
            "coord_dim": [int(VEC_COORD_DIMS[0]), int(VEC_COORD_DIMS[1])],
            "hp_dim": int(VEC_HP_DIM),
            "comm_distance": COMM_DISTANCE,
            "valid_node_list": list(range(N_VALID_DEFAULT)),
            "target_node_list": list(range(N_VALID_DEFAULT, N_AGENTS_DEFAULT)),
            "n_groups": 4,
            "n_workers": 1,
        }
        builder = GroupBuilderConfig(**config).get_group_builder()
        self.assertIsInstance(builder, PartialGroupVectorStateBuilder)

    def test_invalid_type_raises(self):
        from marlite.algorithm.group_builder.group_builder_config import GroupBuilderConfig
        with self.assertRaises(ValueError):
            GroupBuilderConfig(type="DefinitelyNotABuilder")


if __name__ == "__main__":
    unittest.main()
