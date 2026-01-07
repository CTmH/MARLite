import numpy as np
import unittest
from marlite.util.self_supervised_data_constructor.magent_vec_obs_data_constructor import MagentVecObsDataConstructor


class TestMagentVecObsDataConstructor(unittest.TestCase):
    """Test suite for MagentVecObsDataConstructor."""

    def test_process_empty_edges_without_time_seq(self):
        """Test behavior when edge_indices is empty (no communication) without time sequence."""
        # Setup: batch_size=1, 2 agents, seq_len=1, max_observed_entities=2, feature_dim=3, max_entities_perception=4
        observations = np.array([[
            [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]],  # agent 0: two entities at time 0
            [[[7.0, 8.0, 9.0], [0.0, 0.0, 0.0]]],  # agent 1: one real + one zero-padded at time 0
        ]], dtype=np.float32)  # shape: (1, 2, 1, 2, 3)

        states = None
        # Empty edge_indices: List[List[np.ndarray]] where each time step has (2, 0) shape
        edge_indices = [[np.empty((2, 0), dtype=int)]]  # batch 0, time 0: (2, 0)
        alive_mask = np.array([[True, True]], dtype=bool)

        constructor = MagentVecObsDataConstructor(max_entities_perception=4, max_observed_entities=2, with_time_seq=False, n_workers=0)

        result, mask = constructor.process(
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

        # Check mask: first 2 entities should be True (real), last 2 should be False (padded)
        expected_mask_agent0 = np.array([True, True, False, False], dtype=bool)
        expected_mask_agent1 = np.array([True, False, False, False], dtype=bool)
        np.testing.assert_array_equal(mask[0, 0], expected_mask_agent0)
        np.testing.assert_array_equal(mask[0, 1], expected_mask_agent1)

    def test_process_empty_edges_with_time_seq(self):
        """Test behavior when edge_indices is empty (no communication) with time sequence."""
        # Setup: batch_size=1, 2 agents, seq_len=1, max_observed_entities=2, feature_dim=3, max_entities_perception=4
        observations = np.array([[
            [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]],  # agent 0: two entities at time 0
            [[[7.0, 8.0, 9.0], [0.0, 0.0, 0.0]]],  # agent 1: one real + one zero-padded at time 0
        ]], dtype=np.float32)  # shape: (1, 2, 1, 2, 3)

        states = None
        # Empty edge_indices: List[List[np.ndarray]] where each time step has (2, 0) shape
        edge_indices = [[np.empty((2, 0), dtype=int)]]  # batch 0, time 0: (2, 0)
        alive_mask = np.array([[True, True]], dtype=bool)

        constructor = MagentVecObsDataConstructor(max_entities_perception=4, max_observed_entities=2, with_time_seq=True, n_workers=0)

        result, mask = constructor.process(
            observations=observations,
            states=states,
            edge_indices=edge_indices,
            alive_mask=alive_mask
        )

        # Expected: each agent sees only its own 2 entities → padded to 4, with time dimension preserved
        expected_agent0 = np.array([[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]], dtype=np.float32)

        expected_agent1 = np.array([[
            [7.0, 8.0, 9.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]], dtype=np.float32)

        np.testing.assert_array_equal(result[0, 0], expected_agent0)
        np.testing.assert_array_equal(result[0, 1], expected_agent1)

        # Check mask: first 2 entities should be True (real), last 2 should be False (padded)
        expected_mask_agent0 = np.array([[True, True, False, False]], dtype=bool)
        expected_mask_agent1 = np.array([[True, False, False, False]], dtype=bool)
        np.testing.assert_array_equal(mask[0, 0], expected_mask_agent0)
        np.testing.assert_array_equal(mask[0, 1], expected_mask_agent1)

    def test_process_with_incoming_edges_without_time_seq(self):
        """Test with one incoming edge: agent1 -> agent0, without time sequence."""
        # batch_size=1, 2 agents, seq_len=1, max_observed_entities=2, feature_dim=2
        observations = np.array([[
            [[[1.0, 1.0], [2.0, 2.0]]],  # agent0: [e0, e1] at time 0
            [[[3.0, 3.0], [4.0, 4.0]]],  # agent1: [e2, e3] at time 0
        ]], dtype=np.float32)  # (1, 2, 1, 2, 2)

        # edge_indices: List[List[np.ndarray]] where each time step has (2, 1) shape with one edge from agent1 to agent0
        edge_indices = [[np.array([[1], [0]], dtype=int)]]  # batch 0, time 0: (2, 1), source=1, target=0

        alive_mask = np.array([[True, True]], dtype=bool)

        constructor = MagentVecObsDataConstructor(max_entities_perception=5, max_observed_entities=2, with_time_seq=False, n_workers=0)

        result, mask = constructor.process(
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

        # Check mask: agent0 has 4 real entities, agent1 has 2 real entities
        expected_mask_agent0 = np.array([True, True, True, True, False], dtype=bool)
        expected_mask_agent1 = np.array([True, True, False, False, False], dtype=bool)
        np.testing.assert_array_equal(mask[0, 0], expected_mask_agent0)
        np.testing.assert_array_equal(mask[0, 1], expected_mask_agent1)

    def test_process_with_incoming_edges_with_time_seq(self):
        """Test with different edges for each time step, with time sequence."""
        # batch_size=1, 2 agents, seq_len=2, max_observed_entities=2, feature_dim=2
        observations = np.array([[
            [[[1.0, 1.0], [2.0, 2.0]], [[1.1, 1.1], [2.1, 2.1]]],  # agent0: [e0, e1] at time 0 and [e0', e1'] at time 1
            [[[3.0, 3.0], [4.0, 4.0]], [[3.1, 3.1], [4.1, 4.1]]],  # agent1: [e2, e3] at time 0 and [e2', e3'] at time 1
        ]], dtype=np.float32)  # (1, 2, 2, 2, 2)

        # edge_indices: different edges for each time step
        # Time 0: one edge from agent1 to agent0 → [0, 1] means source=1, target=0
        # Time 1: different edge configuration, e.g., no edges
        edge_indices = [
            [  # batch 0
                np.array([[1], [0]], dtype=int),  # time 0: (2, 1), source=1, target=0
                np.empty((2, 0), dtype=int)       # time 1: (2, 0), no edges
            ]
        ]

        alive_mask = np.array([[True, True]], dtype=bool)

        constructor = MagentVecObsDataConstructor(max_entities_perception=5, max_observed_entities=2, with_time_seq=True, n_workers=0)

        result, mask = constructor.process(
            observations=observations,
            states=None,
            edge_indices=edge_indices,
            alive_mask=alive_mask
        )

        # agent0 sees at time 0: self([1,1],[2,2]) + agent1([3,3],[4,4]) → 4 entities → pad to 5
        # agent0 sees at time 1: only self([1.1,1.1],[2.1,2.1]) since no edges → 2 entities → pad to 5
        expected_agent0_t0 = np.array([
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [0.0, 0.0],
        ], dtype=np.float32)

        expected_agent0_t1 = np.array([
            [1.1, 1.1],
            [2.1, 2.1],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ], dtype=np.float32)

        # agent1 has no incoming edges at time 0 → only self, at time 1 → only self
        expected_agent1_t0 = np.array([
            [3.0, 3.0],
            [4.0, 4.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ], dtype=np.float32)

        expected_agent1_t1 = np.array([
            [3.1, 3.1],
            [4.1, 4.1],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ], dtype=np.float32)

        np.testing.assert_array_equal(result[0, 0, 0], expected_agent0_t0)
        np.testing.assert_array_equal(result[0, 0, 1], expected_agent0_t1)
        np.testing.assert_array_equal(result[0, 1, 0], expected_agent1_t0)
        np.testing.assert_array_equal(result[0, 1, 1], expected_agent1_t1)

        # Check mask: agent0 has 4 real entities at time 0, 2 real entities at time 1
        # agent1 has 2 real entities at each time step
        expected_mask_agent0_t0 = np.array([True, True, True, True, False], dtype=bool)
        expected_mask_agent0_t1 = np.array([True, True, False, False, False], dtype=bool)
        expected_mask_agent1_t0 = np.array([True, True, False, False, False], dtype=bool)
        expected_mask_agent1_t1 = np.array([True, True, False, False, False], dtype=bool)
        np.testing.assert_array_equal(mask[0, 0, 0], expected_mask_agent0_t0)
        np.testing.assert_array_equal(mask[0, 0, 1], expected_mask_agent0_t1)
        np.testing.assert_array_equal(mask[0, 1, 0], expected_mask_agent1_t0)
        np.testing.assert_array_equal(mask[0, 1, 1], expected_mask_agent1_t1)

    def test_process_with_self_loop_and_duplicates_without_time_seq(self):
        """Test self-loop and duplicate observations across agents without time sequence."""
        # batch_size=1, 2 agents, seq_len=1, max_observed_entities=2, feature_dim=2
        observations = np.array([[
            [[[1.0, 1.0], [2.0, 2.0]]],  # agent0 at time 0
            [[[1.0, 1.0], [3.0, 3.0]]],  # agent1 at time 0 — shares [1,1] with agent0
        ]], dtype=np.float32)

        # Self-loop on agent0 (0->0) + edge from agent1 to agent0 (1->0)
        edge_indices = [[np.array([[0, 1], [0, 0]], dtype=int)]]  # batch 0, time 0: (2, 2), edges: (0→0) and (1→0)

        alive_mask = np.array([[True, True]], dtype=bool)

        constructor = MagentVecObsDataConstructor(max_entities_perception=3, max_observed_entities=2, with_time_seq=False, n_workers=0)

        result, mask = constructor.process(
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

        # Check mask: agent0 has 3 real entities, agent1 has 2 real entities
        expected_mask_agent0 = np.array([True, True, True], dtype=bool)
        expected_mask_agent1 = np.array([True, True, False], dtype=bool)
        np.testing.assert_array_equal(mask[0, 0], expected_mask_agent0)
        np.testing.assert_array_equal(mask[0, 1], expected_mask_agent1)

    def test_process_with_zero_padding_and_truncation_without_time_seq(self):
        """Test removal of zero-padded rows and truncation to max_entities_perception without time sequence."""
        # batch_size=1, 1 agent, seq_len=1, max_observed_entities=4, but only first 2 are non-zero
        observations = np.array([[[[[1.0, 0.0],  # real
            [0.0, 1.0],  # real
            [0.0, 0.0],  # padding
            [0.0, 0.0],  # padding
        ]]]], dtype=np.float32)  # (1, 1, 1, 4, 2)

        edge_indices = [[np.empty((2, 0), dtype=int)]]  # no edges for time 0
        alive_mask = np.array([[True]])

        constructor = MagentVecObsDataConstructor(max_entities_perception=2, max_observed_entities=4, with_time_seq=False, n_workers=0)

        result, mask = constructor.process(
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

        # Check mask: both entities are real
        expected_mask = np.array([[[True, True]]], dtype=bool)
        np.testing.assert_array_equal(mask, expected_mask)

    def test_process_parallel_vs_sequential_consistency_without_time_seq(self):
        """Ensure n_workers=0 and n_workers=1 produce identical results without time sequence."""
        # Larger test case: batch_size=1, 3 agents, seq_len=1, max_observed_entities=2, feature_dim=2
        observations = np.array([[[[[1.0, 1.0], [2.0, 2.0]]],  # a0 at time 0
            [[[3.0, 3.0], [4.0, 4.0]]],  # a1 at time 0
            [[[5.0, 5.0], [6.0, 6.0]]],  # a2 at time 0
        ]], dtype=np.float32)

        # Edges: a1->a0, a2->a0, a0->a1
        edge_indices = [[np.array([
            [1, 2, 0],  # sources
            [0, 0, 1],  # targets → a0 gets from a1,a2; a1 gets from a0
        ], dtype=int)]]

        alive_mask = np.array([[True, True, True]], dtype=bool)

        constructor_seq = MagentVecObsDataConstructor(max_entities_perception=6, max_observed_entities=2, with_time_seq=False, n_workers=0)
        constructor_par = MagentVecObsDataConstructor(max_entities_perception=6, max_observed_entities=2, with_time_seq=False, n_workers=1)

        result_seq, mask_seq = constructor_seq.process(
            observations=observations,
            states=None,
            edge_indices=edge_indices,
            alive_mask=alive_mask
        )

        result_par, mask_par = constructor_par.process(
            observations=observations,
            states=None,
            edge_indices=edge_indices,
            alive_mask=alive_mask
        )

        np.testing.assert_array_equal(result_seq, result_par)
        np.testing.assert_array_equal(mask_seq, mask_par)

    def test_process_dead_agent_ignored_without_time_seq(self):
        """Dead agents contribute no observations and are skipped without time sequence."""
        observations = np.array([[[[[1.0, 1.0], [2.0, 2.0]]],  # a0 (alive) at time 0
            [[[3.0, 3.0], [4.0, 4.0]]],  # a1 (dead) at time 0
            [[[5.0, 5.0], [6.0, 6.0]]],  # a2 (alive) at time 0
        ]], dtype=np.float32)

        # Edges: a1->a0 (but a1 is dead, so ignored), a2->a0
        edge_indices = [[np.array([
            [1, 2],  # sources
            [0, 0],  # targets
        ], dtype=int)]]

        alive_mask = np.array([[True, False, True]], dtype=bool)  # a1 dead

        constructor = MagentVecObsDataConstructor(max_entities_perception=4, max_observed_entities=2, with_time_seq=False, n_workers=0)

        result, mask = constructor.process(
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

        # Check mask: a0 has 4 real entities, a1 is dead (all False), a2 has 2 real + 2 padded
        expected_mask_a0 = np.array([True, True, True, True], dtype=bool)
        expected_mask_a1 = np.array([False, False, False, False], dtype=bool)
        expected_mask_a2 = np.array([True, True, False, False], dtype=bool)
        np.testing.assert_array_equal(mask[0, 0], expected_mask_a0)
        np.testing.assert_array_equal(mask[0, 1], expected_mask_a1)
        np.testing.assert_array_equal(mask[0, 2], expected_mask_a2)

    def test_with_time_seq_parameter_behavior(self):
        """Test that with_time_seq parameter correctly controls output dimensions."""
        # batch_size=2, 3 agents, seq_len=3, max_observed_entities=2, feature_dim=2
        observations = np.array([[
            [  # agent 0
                [[1.0, 1.0], [2.0, 2.0]],  # time 0
                [[3.0, 3.0], [4.0, 4.0]],  # time 1
                [[5.0, 5.0], [6.0, 6.0]],  # time 2
            ],
            [  # agent 1
                [[7.0, 7.0], [8.0, 8.0]],  # time 0
                [[9.0, 9.0], [10.0, 10.0]],  # time 1
                [[11.0, 11.0], [12.0, 12.0]],  # time 2
            ],
            [  # agent 2
                [[13.0, 13.0], [14.0, 14.0]],  # time 0
                [[15.0, 15.0], [16.0, 16.0]],  # time 1
                [[17.0, 17.0], [18.0, 18.0]],  # time 2
            ]
        ], [
            [  # agent 0
                [[1.1, 1.1], [2.1, 2.1]],  # time 0
                [[3.1, 3.1], [4.1, 4.1]],  # time 1
                [[5.1, 5.1], [6.1, 6.1]],  # time 2
            ],
            [  # agent 1
                [[7.1, 7.1], [8.1, 8.1]],  # time 0
                [[9.1, 9.1], [10.1, 10.1]],  # time 1
                [[11.1, 11.1], [12.1, 12.1]],  # time 2
            ],
            [  # agent 2
                [[13.1, 13.1], [14.1, 14.1]],  # time 0
                [[15.1, 15.1], [16.1, 16.1]],  # time 1
                [[17.1, 17.1], [18.1, 18.1]],  # time 2
            ]
        ]], dtype=np.float32)  # (2, 3, 3, 2, 2)

        # edge_indices: no edges for any time step
        edge_indices = [
            [  # batch 0
                np.empty((2, 0), dtype=int),  # time 0
                np.empty((2, 0), dtype=int),  # time 1
                np.empty((2, 0), dtype=int),  # time 2
            ],
            [  # batch 1
                np.empty((2, 0), dtype=int),  # time 0
                np.empty((2, 0), dtype=int),  # time 1
                np.empty((2, 0), dtype=int),  # time 2
            ]
        ]
        alive_mask = np.array([[True, True, True], [True, True, True]], dtype=bool)

        # Test with_time_seq=False
        constructor_no_time = MagentVecObsDataConstructor(max_entities_perception=4, max_observed_entities=2, with_time_seq=False, n_workers=0)
        result_no_time, mask_no_time = constructor_no_time.process(
            observations=observations,
            states=None,
            edge_indices=edge_indices,
            alive_mask=alive_mask
        )

        # Expected shape without time: (2, 3, 4, 2) - only last time step
        expected_shape_no_time = (2, 3, 4, 2)
        self.assertEqual(result_no_time.shape, expected_shape_no_time)
        self.assertEqual(mask_no_time.shape, (2, 3, 4))

        # Test with_time_seq=True
        constructor_with_time = MagentVecObsDataConstructor(max_entities_perception=4, max_observed_entities=2, with_time_seq=True, n_workers=0)
        result_with_time, mask_with_time = constructor_with_time.process(
            observations=observations,
            states=None,
            edge_indices=edge_indices,
            alive_mask=alive_mask
        )

        # Expected shape with time: (2, 3, 3, 4, 2) - all time steps
        expected_shape_with_time = (2, 3, 3, 4, 2)
        self.assertEqual(result_with_time.shape, expected_shape_with_time)
        self.assertEqual(mask_with_time.shape, (2, 3, 3, 4))

        # Verify that when with_time_seq=False, we get the last time step from the full sequence
        # (This would be true if we processed the same data with time sequence and then took the last step)
        # Note: In with_time_seq=False mode, we take the last time step from input, so we need to compare appropriately


if __name__ == '__main__':
    unittest.main()