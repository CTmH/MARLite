import numpy as np
import unittest
from marlite.util.self_supervised_data_constructor.magent_vec_obs_data_constructor import MagentVecObsDataConstructor


class TestMagentVecObsDataConstructor(unittest.TestCase):
    """Test suite for MagentVecObsDataConstructor."""

    def test_process_empty_edges(self):
        """Test behavior when edge_indices is empty (no communication)."""
        # Setup: batch_size=1, 2 agents, max_observed_entities=2, feature_dim=3, max_entities_perception=4
        observations = np.array([[
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],  # agent 0: two entities
            [[7.0, 8.0, 9.0], [0.0, 0.0, 0.0]],  # agent 1: one real + one zero-padded
        ]], dtype=np.float32)  # shape: (1, 2, 2, 3)

        states = None
        # Empty edge_indices: (1, 2, 0)
        edge_indices = np.empty((1, 2, 0), dtype=int)
        alive_mask = np.array([[True, True]], dtype=bool)

        constructor = MagentVecObsDataConstructor(max_entities_perception=4, max_observed_entities=2, n_workers=0)

        result = constructor.process(
            observations=observations,
            states=states,
            edge_indices=edge_indices,
            alive_mask=alive_mask
        )

        # Expected: each agent sees only its own 2 entities → padded to 4
        expected_agent0 = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ], dtype=np.float32)

        expected_agent1 = np.array([
            [7.0, 8.0, 9.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ], dtype=np.float32)

        np.testing.assert_array_equal(result[0, 0], expected_agent0)
        np.testing.assert_array_equal(result[0, 1], expected_agent1)

    def test_process_with_incoming_edges(self):
        """Test with one incoming edge: agent1 -> agent0."""
        # batch_size=1, 2 agents, max_observed_entities=2, feature_dim=2
        observations = np.array([[
            [[1.0, 1.0], [2.0, 2.0]],  # agent0: [e0, e1]
            [[3.0, 3.0], [4.0, 4.0]],  # agent1: [e2, e3]
        ]], dtype=np.float32)  # (1, 2, 2, 2)

        # edge_indices: one edge from agent1 to agent0 → [0, 1] means source=1, target=0
        edge_indices = np.array([[[1], [0]]], dtype=int)  # shape: (1, 2, 1)

        alive_mask = np.array([[True, True]], dtype=bool)

        constructor = MagentVecObsDataConstructor(max_entities_perception=5, max_observed_entities=2, n_workers=0)

        result = constructor.process(
            observations=observations,
            states=None,
            edge_indices=edge_indices,
            alive_mask=alive_mask
        )

        # agent0 sees: self([1,1],[2,2]) + agent1([3,3],[4,4]) → 4 entities → pad to 5
        expected_agent0 = np.array([
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [0.0, 0.0],
        ], dtype=np.float32)

        # agent1 has no incoming edges → only self
        expected_agent1 = np.array([
            [3.0, 3.0],
            [4.0, 4.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ], dtype=np.float32)

        np.testing.assert_array_equal(result[0, 0], expected_agent0)
        np.testing.assert_array_equal(result[0, 1], expected_agent1)

    def test_process_with_self_loop_and_duplicates(self):
        """Test self-loop and duplicate observations across agents."""
        # batch_size=1, 2 agents, max_observed_entities=2, feature_dim=2
        observations = np.array([[
            [[1.0, 1.0], [2.0, 2.0]],  # agent0
            [[1.0, 1.0], [3.0, 3.0]],  # agent1 — shares [1,1] with agent0
        ]], dtype=np.float32)

        # Self-loop on agent0 (0->0) + edge from agent1 to agent0 (1->0)
        edge_indices = np.array([[[0, 1], [0, 0]]], dtype=int)  # two edges: (0→0) and (1→0)

        alive_mask = np.array([[True, True]], dtype=bool)

        constructor = MagentVecObsDataConstructor(max_entities_perception=3, max_observed_entities=2, n_workers=0)

        result = constructor.process(
            observations=observations,
            states=None,
            edge_indices=edge_indices,
            alive_mask=alive_mask
        )

        # agent0 sees: self([1,1],[2,2]) + self-loop([1,1],[2,2]) + agent1([1,1],[3,3])
        # All vectors: [1,1], [2,2], [1,1], [2,2], [1,1], [3,3] → unique: [1,1], [2,2], [3,3] → exactly 3
        expected_agent0 = np.array([
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
        ], dtype=np.float32)

        # agent1 has no incoming edges → only self → [1,1], [3,3] → pad to 3
        expected_agent1 = np.array([
            [1.0, 1.0],
            [3.0, 3.0],
            [0.0, 0.0],
        ], dtype=np.float32)

        np.testing.assert_array_equal(result[0, 0], expected_agent0)
        np.testing.assert_array_equal(result[0, 1], expected_agent1)

    def test_process_with_zero_padding_and_truncation(self):
        """Test removal of zero-padded rows and truncation to max_entities_perception."""
        # batch_size=1, 1 agent, max_observed_entities=4, but only first 2 are non-zero
        observations = np.array([[[
            [1.0, 0.0],  # real
            [0.0, 1.0],  # real
            [0.0, 0.0],  # padding
            [0.0, 0.0],  # padding
        ]]], dtype=np.float32)  # (1, 1, 4, 2)

        edge_indices = np.empty((1, 2, 0), dtype=int)  # no edges
        alive_mask = np.array([[True]])

        constructor = MagentVecObsDataConstructor(max_entities_perception=2, max_observed_entities=4, n_workers=0)

        result = constructor.process(
            observations=observations,
            states=None,
            edge_indices=edge_indices,
            alive_mask=alive_mask
        )

        # Only [1,0] and [0,1] survive → exactly 2 → no pad needed
        expected = np.array([[[
            [1.0, 0.0],
            [0.0, 1.0],
        ]]], dtype=np.float32)

        np.testing.assert_array_equal(result, expected)

    def test_process_parallel_vs_sequential_consistency(self):
        """Ensure n_workers=0 and n_workers=1 produce identical results."""
        # Larger test case: batch_size=1, 3 agents, max_observed_entities=2, feature_dim=2
        observations = np.array([[[
            [1.0, 1.0], [2.0, 2.0]],  # a0
            [[3.0, 3.0], [4.0, 4.0]],  # a1
            [[5.0, 5.0], [6.0, 6.0]],  # a2
        ]], dtype=np.float32)

        # Edges: a1->a0, a2->a0, a0->a1
        edge_indices = np.array([[
            [1, 2, 0],  # sources
            [0, 0, 1],  # targets → a0 gets from a1,a2; a1 gets from a0
        ]], dtype=int)

        alive_mask = np.array([[True, True, True]], dtype=bool)

        constructor_seq = MagentVecObsDataConstructor(max_entities_perception=6, max_observed_entities=2, n_workers=0)
        constructor_par = MagentVecObsDataConstructor(max_entities_perception=6, max_observed_entities=2, n_workers=1)

        result_seq = constructor_seq.process(
            observations=observations,
            states=None,
            edge_indices=edge_indices,
            alive_mask=alive_mask
        )

        result_par = constructor_par.process(
            observations=observations,
            states=None,
            edge_indices=edge_indices,
            alive_mask=alive_mask
        )

        np.testing.assert_array_equal(result_seq, result_par)

    def test_process_dead_agent_ignored(self):
        """Dead agents contribute no observations and are skipped."""
        observations = np.array([[[
            [1.0, 1.0], [2.0, 2.0]],  # a0 (alive)
            [[3.0, 3.0], [4.0, 4.0]],  # a1 (dead)
            [[5.0, 5.0], [6.0, 6.0]],  # a2 (alive)
        ]], dtype=np.float32)

        # Edges: a1->a0 (but a1 is dead, so ignored), a2->a0
        edge_indices = np.array([[
            [1, 2],  # sources
            [0, 0],  # targets
        ]], dtype=int)

        alive_mask = np.array([[True, False, True]], dtype=bool)  # a1 dead

        constructor = MagentVecObsDataConstructor(max_entities_perception=4, max_observed_entities=2, n_workers=0)

        result = constructor.process(
            observations=observations,
            states=None,
            edge_indices=edge_indices,
            alive_mask=alive_mask
        )

        # a0 sees: self([1,1],[2,2]) + a2([5,5],[6,6]) → 4 entities → exact fit
        expected_a0 = np.array([
            [1.0, 1.0],
            [2.0, 2.0],
            [5.0, 5.0],
            [6.0, 6.0],
        ], dtype=np.float32)

        # a1 is dead → all zeros
        expected_a1 = np.zeros((4, 2), dtype=np.float32)

        # a2 has no incoming edges → only self → pad to 4
        expected_a2 = np.array([
            [5.0, 5.0],
            [6.0, 6.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ], dtype=np.float32)

        np.testing.assert_array_equal(result[0, 0], expected_a0)
        np.testing.assert_array_equal(result[0, 1], expected_a1)
        np.testing.assert_array_equal(result[0, 2], expected_a2)

    def test_process_complex_scenario(self):
        """Test a complex scenario with multiple agents and edges."""
        # batch_size=1, 4 agents, max_observed_entities=3, feature_dim=2
        observations = np.array([[[
            [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],  # a0
            [[4.0, 4.0], [5.0, 5.0], [6.0, 6.0]],  # a1
            [[1.0, 1.0], [7.0, 7.0], [8.0, 8.0]],  # a2 (shares [1,1] with a0)
            [[9.0, 9.0], [0.0, 0.0], [0.0, 0.0]],  # a3 (with padding)
        ]], dtype=np.float32)

        # Edges: a1->a0, a2->a0, a0->a1, a3->a2
        edge_indices = np.array([[
            [1, 2, 0, 3],  # sources
            [0, 0, 1, 2],  # targets
        ]], dtype=int)

        alive_mask = np.array([[True, True, True, True]], dtype=bool)

        constructor = MagentVecObsDataConstructor(max_entities_perception=7, max_observed_entities=3, n_workers=0)

        result = constructor.process(
            observations=observations,
            states=None,
            edge_indices=edge_indices,
            alive_mask=alive_mask
        )

        # a0 receives from a1 and a2:
        # self: [1,1], [2,2], [3,3]
        # a1: [4,4], [5,5], [6,6]
        # a2: [1,1], [7,7], [8,8]
        # unique: [1,1], [2,2], [3,3], [4,4], [5,5], [6,6], [7,7], [8,8] → 8 total
        # truncated to 7: first 7
        expected_a0 = np.array([
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0],
            [6.0, 6.0],
            [7.0, 7.0],
        ], dtype=np.float32)

        # a1 receives from a0:
        # self: [4,4], [5,5], [6,6]
        # a0: [1,1], [2,2], [3,3]
        # unique: [4,4], [5,5], [6,6], [1,1], [2,2], [3,3] → 6 total
        # padded to 7: [4,4], [5,5], [6,6], [1,1], [2,2], [3,3], [0,0]
        expected_a1 = np.array([
            [4.0, 4.0],
            [5.0, 5.0],
            [6.0, 6.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [0.0, 0.0],
        ], dtype=np.float32)

        # a2 receives from a3:
        # self: [1,1], [7,7], [8,8]
        # a3: [9,9], [0,0], [0,0] → only [9,9] survives after zero removal
        # unique: [1,1], [7,7], [8,8], [9,9] → 4 total
        # padded to 7
        expected_a2 = np.array([
            [1.0, 1.0],
            [7.0, 7.0],
            [8.0, 8.0],
            [9.0, 9.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ], dtype=np.float32)

        # a3 receives from none:
        # self: [9,9], [0,0], [0,0] → only [9,9] survives after zero removal
        # padded to 7
        expected_a3 = np.array([
            [9.0, 9.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ], dtype=np.float32)

        np.testing.assert_array_equal(result[0, 0], expected_a0)
        np.testing.assert_array_equal(result[0, 1], expected_a1)
        np.testing.assert_array_equal(result[0, 2], expected_a2)
        np.testing.assert_array_equal(result[0, 3], expected_a3)


if __name__ == '__main__':
    unittest.main()