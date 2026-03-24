import numpy as np
import unittest
from marlite.util.self_supervised_data_constructor.magent_obs_data_constructor import (
    MagentImageObsDataConstructor,
)


class TestMagentImageObsDataConstructor(unittest.TestCase):
    """Test suite for MagentImageObsDataConstructor."""

    def test_process_empty_edges_without_time_seq(self):
        """Test behavior when edge_indices is empty (no communication) without time sequence."""
        # Setup: batch_size=1, seq_len=1, 2 agents, H=2, W=2, C=3, max_entities_perception=16
        # Create H,W,C array and fill some positions with valid feature vectors
        H, W, C = 2, 2, 3
        observations = np.zeros((1, 1, 2, H, W, C), dtype=np.float32)

        # Fill agent0's observation: set first pixel [0,0] to [1.0, 2.0, 3.0]
        observations[0, 0, 0, 0, 0, :] = [1.0, 2.0, 3.0]
        # Fill agent1's observation: set first pixel [0,0] to [4.0, 5.0, 6.0]
        observations[0, 0, 1, 0, 0, :] = [4.0, 5.0, 6.0]

        states = None
        # Empty edge_indices: List[List[np.ndarray]] where each time step has (2, 0) shape
        edge_indices = [[np.empty((2, 0), dtype=int)]]  # batch 0, time 0: (2, 0)
        # NEW DIMENSION: alive_mask is now (batch_size, seq_len, n_agents)
        alive_mask = np.array([[[True, True]]], dtype=bool)  # (1, 1, 2)

        constructor = MagentImageObsDataConstructor(
            max_entities_perception=16,
            with_time_seq=False,
            n_workers=0,
            channel_first=False,
        )

        result, mask = constructor.process(
            observations=observations,
            states=states,
            edge_indices=edge_indices,
            alive_mask=alive_mask,
        )

        # Expected: each agent sees only its own H*W=4 pixels → padded to 16
        # For agent0: [1,2,3] at position 0, zeros elsewhere
        expected_agent0 = np.zeros((16, 3), dtype=np.float32)
        expected_agent0[0, :] = [1.0, 2.0, 3.0]

        # For agent1: [4,5,6] at position 0, zeros elsewhere
        expected_agent1 = np.zeros((16, 3), dtype=np.float32)
        expected_agent1[0, :] = [4.0, 5.0, 6.0]

        np.testing.assert_array_equal(result[0, 0], expected_agent0)
        np.testing.assert_array_equal(result[0, 1], expected_agent1)

        # Check mask: first pixel should be True (real), rest should be False (padded)
        expected_mask_agent0 = np.array([True] + [False] * 15, dtype=bool)
        expected_mask_agent1 = np.array([True] + [False] * 15, dtype=bool)
        np.testing.assert_array_equal(mask[0, 0], expected_mask_agent0)
        np.testing.assert_array_equal(mask[0, 1], expected_mask_agent1)

    def test_process_empty_edges_with_time_seq(self):
        """Test behavior when edge_indices is empty (no communication) with time sequence."""
        # Setup: batch_size=1, seq_len=1, 2 agents, H=2, W=2, C=3, max_entities_perception=16
        H, W, C = 2, 2, 3
        observations = np.zeros((1, 1, 2, H, W, C), dtype=np.float32)

        # Fill agent0's observation: set first pixel [0,0] to [1.0, 2.0, 3.0]
        observations[0, 0, 0, 0, 0, :] = [1.0, 2.0, 3.0]
        # Fill agent1's observation: set first pixel [0,0] to [4.0, 5.0, 6.0]
        observations[0, 0, 1, 0, 0, :] = [4.0, 5.0, 6.0]

        states = None
        # Empty edge_indices: List[List[np.ndarray]] where each time step has (2, 0) shape
        edge_indices = [[np.empty((2, 0), dtype=int)]]  # batch 0, time 0: (2, 0)
        # NEW DIMENSION: alive_mask is now (batch_size, seq_len, n_agents)
        alive_mask = np.array([[[True, True]]], dtype=bool)  # (1, 1, 2)

        constructor = MagentImageObsDataConstructor(
            max_entities_perception=16,
            with_time_seq=True,
            n_workers=0,
            channel_first=False,
        )

        result, mask = constructor.process(
            observations=observations,
            states=states,
            edge_indices=edge_indices,
            alive_mask=alive_mask,
        )

        # Expected: each agent sees only its own H*W=4 pixels → padded to 16, with time dimension preserved
        expected_result_agent0 = np.zeros((16, 3), dtype=np.float32)
        expected_result_agent0[0, :] = [1.0, 2.0, 3.0]

        expected_result_agent1 = np.zeros((16, 3), dtype=np.float32)
        expected_result_agent1[0, :] = [4.0, 5.0, 6.0]

        np.testing.assert_array_equal(result[0][0][0], expected_result_agent0)
        np.testing.assert_array_equal(result[0][0][1], expected_result_agent1)

        # Check mask: first pixel should be True (real), rest should be False (padded)
        expected_mask_agent0 = np.array([True] + [False] * 15, dtype=bool)
        expected_mask_agent1 = np.array([True] + [False] * 15, dtype=bool)
        np.testing.assert_array_equal(mask[0][0][0], expected_mask_agent0)
        np.testing.assert_array_equal(mask[0][0][1], expected_mask_agent1)

    def test_process_channel_first_format(self):
        """Test channel_first=True conversion behavior."""
        # Setup: batch_size=1, seq_len=1, 2 agents, H=2, W=2, C=3, max_entities_perception=16
        # Create channel-first format: (batch_size, seq_len, n_agents, C, H, W)
        H, W, C = 2, 2, 3
        observations = np.zeros((1, 1, 2, C, H, W), dtype=np.float32)

        # Fill agent0's observation: set first channel [0] at position [0,0] to [1.0, 2.0, 3.0]
        # In channel-first, we need to set each channel separately
        observations[0, 0, 0, 0, 0, 0] = 1.0  # channel 0
        observations[0, 0, 0, 1, 0, 0] = 2.0  # channel 1
        observations[0, 0, 0, 2, 0, 0] = 3.0  # channel 2

        # Fill agent1's observation: set first channel [0] at position [0,0] to [4.0, 5.0, 6.0]
        observations[0, 0, 1, 0, 0, 0] = 4.0  # channel 0
        observations[0, 0, 1, 1, 0, 0] = 5.0  # channel 1
        observations[0, 0, 1, 2, 0, 0] = 6.0  # channel 2

        states = None
        edge_indices = [[np.empty((2, 0), dtype=int)]]  # batch 0, time 0: (2, 0)
        alive_mask = np.array([[[True, True]]], dtype=bool)  # (1, 1, 2)

        constructor = MagentImageObsDataConstructor(
            max_entities_perception=16,
            with_time_seq=False,
            n_workers=0,
            channel_first=True,
        )

        result, mask = constructor.process(
            observations=observations,
            states=states,
            edge_indices=edge_indices,
            alive_mask=alive_mask,
        )

        # Expected: same as channel-last case since conversion should happen
        expected_agent0 = np.zeros((16, 3), dtype=np.float32)
        expected_agent0[0, :] = [1.0, 2.0, 3.0]

        expected_agent1 = np.zeros((16, 3), dtype=np.float32)
        expected_agent1[0, :] = [4.0, 5.0, 6.0]

        np.testing.assert_array_equal(result[0, 0], expected_agent0)
        np.testing.assert_array_equal(result[0, 1], expected_agent1)

        # Check mask: first pixel should be True (real), rest should be False (padded)
        expected_mask_agent0 = np.array([True] + [False] * 15, dtype=bool)
        expected_mask_agent1 = np.array([True] + [False] * 15, dtype=bool)
        np.testing.assert_array_equal(mask[0, 0], expected_mask_agent0)
        np.testing.assert_array_equal(mask[0, 1], expected_mask_agent1)

    def test_process_with_incoming_edges_without_time_seq(self):
        """Test with one incoming edge: agent1 -> agent0, without time sequence."""
        # batch_size=1, seq_len=1, 2 agents, H=2, W=2, C=2, max_entities_perception=16
        H, W, C = 2, 2, 2
        observations = np.zeros((1, 1, 2, H, W, C), dtype=np.float32)

        # Fill agent0's observation: set [0,0] to [1.0, 1.0], [0,1] to [2.0, 2.0]
        observations[0, 0, 0, 0, 0, :] = [1.0, 1.0]
        observations[0, 0, 0, 0, 1, :] = [2.0, 2.0]

        # Fill agent1's observation: set [0,0] to [3.0, 3.0], [0,1] to [4.0, 4.0]
        observations[0, 0, 1, 0, 0, :] = [3.0, 3.0]
        observations[0, 0, 1, 0, 1, :] = [4.0, 4.0]

        # edge_indices: List[List[np.ndarray]] where each time step has (2, 1) shape with one edge from agent1 to agent0
        edge_indices = [
            [np.array([[1], [0]], dtype=int)],
            [np.array([[1], [0]], dtype=int)],
        ]  # batch 0, time 0: (2, 1), source=1, target=0

        # NEW DIMENSION: alive_mask is now (batch_size, seq_len, n_agents)
        alive_mask = np.array([[[True, True]]], dtype=bool)  # (1, 1, 2)

        constructor = MagentImageObsDataConstructor(
            max_entities_perception=16,
            with_time_seq=False,
            n_workers=0,
            channel_first=False,
        )

        result, mask = constructor.process(
            observations=observations,
            states=None,
            edge_indices=edge_indices,
            alive_mask=alive_mask,
        )

        # agent0 sees: self (4 pixels) + agent1 (4 pixels) → 8 entities → pad to 16
        expected_agent0 = np.zeros((16, 2), dtype=np.float32)
        # Self pixels: positions 0-3
        expected_agent0[0, :] = [1.0, 1.0]  # [0,0]
        expected_agent0[1, :] = [2.0, 2.0]  # [0,1]
        expected_agent0[2, :] = [3.0, 3.0]  # [1,0]
        expected_agent0[3, :] = [4.0, 4.0]  # [1,1]
        # Agent1 pixels: positions 4-7
        expected_agent0[4, :] = [0.0, 0.0]  # [0,0]
        expected_agent0[5, :] = [0.0, 0.0]  # [0,1]
        expected_agent0[6, :] = [0.0, 0.0]  # [1,0]
        expected_agent0[7, :] = [0.0, 0.0]  # [1,1]

        # agent1 has no incoming edges → only self (4 pixels)
        expected_agent1 = np.zeros((16, 2), dtype=np.float32)
        expected_agent1[0, :] = [3.0, 3.0]  # [0,0]
        expected_agent1[1, :] = [4.0, 4.0]  # [0,1]
        expected_agent1[2, :] = [0.0, 0.0]  # [1,0]
        expected_agent1[3, :] = [0.0, 0.0]  # [1,1]

        np.testing.assert_array_equal(result[0, 0], expected_agent0)
        np.testing.assert_array_equal(result[0, 1], expected_agent1)

        # Check mask: agent0 has 8 real entities, agent1 has 4 real entities
        expected_mask_agent0 = np.array([True] * 4 + [False] * 12, dtype=bool)
        expected_mask_agent1 = np.array([True] * 2 + [False] * 14, dtype=bool)
        np.testing.assert_array_equal(mask[0, 0], expected_mask_agent0)
        np.testing.assert_array_equal(mask[0, 1], expected_mask_agent1)

    def test_process_with_dead_agent_ignored_without_time_seq(self):
        """Dead agents contribute no observations and are skipped without time sequence."""
        # batch_size=1, seq_len=1, 3 agents, H=2, W=2, C=2, max_entities_perception=16
        H, W, C = 2, 2, 2
        observations = np.zeros((1, 1, 3, H, W, C), dtype=np.float32)

        # Fill agent0's observation: set [0,0] to [1.0, 1.0], [0,1] to [2.0, 2.0]
        observations[0, 0, 0, 0, 0, :] = [1.0, 1.0]
        observations[0, 0, 0, 0, 1, :] = [2.0, 2.0]

        # Fill agent1's observation: set [0,0] to [3.0, 3.0], [0,1] to [4.0, 4.0]
        observations[0, 0, 1, 0, 0, :] = [3.0, 3.0]
        observations[0, 0, 1, 0, 1, :] = [4.0, 4.0]

        # Fill agent2's observation: set [0,0] to [5.0, 5.0], [0,1] to [6.0, 6.0]
        observations[0, 0, 2, 0, 0, :] = [5.0, 5.0]
        observations[0, 0, 2, 0, 1, :] = [6.0, 6.0]

        # Edges: a1->a0 (but a1 is dead, so ignored), a2->a0
        edge_indices = [
            [
                np.array(
                    [
                        [1, 2],  # sources
                        [0, 0],  # targets
                    ],
                    dtype=int,
                )
            ]
        ]

        # NEW DIMENSION: alive_mask is now (batch_size, seq_len, n_agents)
        alive_mask = np.array([[[True, False, True]]], dtype=bool)  # a1 dead, (1, 1, 3)

        constructor = MagentImageObsDataConstructor(
            max_entities_perception=16,
            with_time_seq=False,
            n_workers=0,
            channel_first=False,
        )

        result, mask = constructor.process(
            observations=observations,
            states=None,
            edge_indices=edge_indices,
            alive_mask=alive_mask,
        )

        # a0 sees: self (4 pixels) + a2 (4 pixels) → 8 entities → exact fit
        expected_a0 = np.zeros((16, 2), dtype=np.float32)
        # Self pixels: positions 0-3
        expected_a0[0, :] = [1.0, 1.0]  # [0,0]
        expected_a0[1, :] = [2.0, 2.0]  # [0,1]
        expected_a0[2, :] = [5.0, 5.0]  # [1,0]
        expected_a0[3, :] = [6.0, 6.0]  # [1,1]

        # a1 is dead → all zeros
        expected_a1 = np.zeros((16, 2), dtype=np.float32)

        # a2 has no incoming edges → only self (4 pixels)
        expected_a2 = np.zeros((16, 2), dtype=np.float32)
        expected_a2[0, :] = [5.0, 5.0]  # [0,0]
        expected_a2[1, :] = [6.0, 6.0]  # [0,1]
        expected_a2[2, :] = [0.0, 0.0]  # [1,0]
        expected_a2[3, :] = [0.0, 0.0]  # [1,1]

        np.testing.assert_array_equal(result[0, 0], expected_a0)
        np.testing.assert_array_equal(result[0, 1], expected_a1)
        np.testing.assert_array_equal(result[0, 2], expected_a2)

        # Check mask: a0 has 8 real entities, a1 is dead (all False), a2 has 4 real + 12 padded
        expected_mask_a0 = np.array([True] * 4 + [False] * 12, dtype=bool)
        expected_mask_a1 = np.array([False] * 16, dtype=bool)
        expected_mask_a2 = np.array([True] * 2 + [False] * 14, dtype=bool)
        np.testing.assert_array_equal(mask[0, 0], expected_mask_a0)
        np.testing.assert_array_equal(mask[0, 1], expected_mask_a1)
        np.testing.assert_array_equal(mask[0, 2], expected_mask_a2)

    def test_with_time_seq_parameter_behavior(self):
        """Test that with_time_seq parameter correctly controls output dimensions."""
        # batch_size=2, seq_len=2, 2 agents, H=2, W=2, C=2, max_entities_perception=16
        H, W, C = 2, 2, 2
        observations = np.zeros((2, 2, 2, H, W, C), dtype=np.float32)

        # Fill some values for testing
        # Batch 0, time 0, agent 0: [0,0] = [1,1]
        observations[0, 0, 0, 0, 0, :] = [1.0, 1.0]
        # Batch 0, time 0, agent 1: [0,0] = [2,2]
        observations[0, 0, 1, 0, 0, :] = [2.0, 2.0]
        # Batch 0, time 1, agent 0: [0,0] = [3,3]
        observations[0, 1, 0, 0, 0, :] = [3.0, 3.0]
        # Batch 0, time 1, agent 1: [0,0] = [4,4]
        observations[0, 1, 1, 0, 0, :] = [4.0, 4.0]
        # Batch 1, time 0, agent 0: [0,0] = [5,5]
        observations[1, 0, 0, 0, 0, :] = [5.0, 5.0]
        # Batch 1, time 0, agent 1: [0,0] = [6,6]
        observations[1, 0, 1, 0, 0, :] = [6.0, 6.0]
        # Batch 1, time 1, agent 0: [0,0] = [7,7]
        observations[1, 1, 0, 0, 0, :] = [7.0, 7.0]
        # Batch 1, time 1, agent 1: [0,0] = [8,8]
        observations[1, 1, 1, 0, 0, :] = [8.0, 8.0]

        # edge_indices: no edges for any time step
        edge_indices = [
            [  # batch 0
                np.empty((2, 0), dtype=int),  # time 0
                np.empty((2, 0), dtype=int),  # time 1
            ],
            [  # batch 1
                np.empty((2, 0), dtype=int),  # time 0
                np.empty((2, 0), dtype=int),  # time 1
            ],
        ]

        # NEW DIMENSION: alive_mask is now (batch_size, seq_len, n_agents)
        alive_mask = np.array(
            [[[True, True], [True, True]], [[True, True], [True, True]]], dtype=bool
        )  # (2, 2, 2)

        # Test with_time_seq=False
        constructor_no_time = MagentImageObsDataConstructor(
            max_entities_perception=16,
            with_time_seq=False,
            n_workers=0,
            channel_first=False,
        )
        result_no_time, mask_no_time = constructor_no_time.process(
            observations=observations,
            states=None,
            edge_indices=edge_indices,
            alive_mask=alive_mask,
        )

        # Expected shape without time: (2, 2, 16, 2) - only last time step
        expected_shape_no_time = (2, 2, 16, 2)
        self.assertEqual(result_no_time.shape, expected_shape_no_time)
        self.assertEqual(mask_no_time.shape, (2, 2, 16))

        # Test with_time_seq=True
        constructor_with_time = MagentImageObsDataConstructor(
            max_entities_perception=16,
            with_time_seq=True,
            n_workers=0,
            channel_first=False,
        )
        result_with_time, mask_with_time = constructor_with_time.process(
            observations=observations,
            states=None,
            edge_indices=edge_indices,
            alive_mask=alive_mask,
        )

        # Expected shape with time: (2, 2, 2, 16, 2) - all time steps
        expected_shape_with_time = (2, 2, 2, 16, 2)
        self.assertEqual(result_with_time.shape, expected_shape_with_time)
        self.assertEqual(mask_with_time.shape, (2, 2, 2, 16))

    def test_exclude_features_single_index_without_time_seq(self):
        """Test exclude_features=[1] - should remove the second feature dimension."""
        # Setup: batch_size=1, seq_len=1, 2 agents, H=2, W=2, C=3, max_entities_perception=16
        H, W, C = 2, 2, 3
        observations = np.zeros((1, 1, 2, H, W, C), dtype=np.float32)

        # Fill agent0's observation: set [0,0] to [1.0, 2.0, 3.0]
        observations[0, 0, 0, 0, 0, :] = [1.0, 2.0, 3.0]
        # Fill agent1's observation: set [0,0] to [4.0, 5.0, 6.0]
        observations[0, 0, 1, 0, 0, :] = [4.0, 5.0, 6.0]

        states = None
        edge_indices = [[np.empty((2, 0), dtype=int)]]  # batch 0, time 0: (2, 0)
        alive_mask = np.array([[[True, True]]], dtype=bool)  # (1, 1, 2)

        # Test with exclude_features=[1] (remove second feature)
        constructor = MagentImageObsDataConstructor(
            max_entities_perception=16,
            with_time_seq=False,
            n_workers=0,
            exclude_features=[1],
            channel_first=False,
        )

        result, mask = constructor.process(
            observations=observations,
            states=states,
            edge_indices=edge_indices,
            alive_mask=alive_mask,
        )

        # Should have feature dimension 2 (3-1)
        self.assertEqual(result.shape[-1], 2)

        # Check that the remaining features are [0, 2] (first and third)
        # Expected agent0: [[1.0, 3.0], ...]
        expected_agent0 = np.zeros((16, 2), dtype=np.float32)
        expected_agent0[0, :] = [1.0, 3.0]

        # Expected agent1: [[4.0, 6.0], ...]
        expected_agent1 = np.zeros((16, 2), dtype=np.float32)
        expected_agent1[0, :] = [4.0, 6.0]

        np.testing.assert_array_equal(result[0, 0], expected_agent0)
        np.testing.assert_array_equal(result[0, 1], expected_agent1)

    def test_include_features_single_index_without_time_seq(self):
        """Test include_features=[1] - should keep only the second feature dimension."""
        # Setup: batch_size=1, seq_len=1, 2 agents, H=2, W=2, C=3, max_entities_perception=16
        H, W, C = 2, 2, 3
        observations = np.zeros((1, 1, 2, H, W, C), dtype=np.float32)

        # Fill agent0's observation: set [0,0] to [1.0, 2.0, 3.0]
        observations[0, 0, 0, 0, 0, :] = [1.0, 2.0, 3.0]
        # Fill agent1's observation: set [0,0] to [4.0, 5.0, 6.0]
        observations[0, 0, 1, 0, 0, :] = [4.0, 5.0, 6.0]

        states = None
        edge_indices = [[np.empty((2, 0), dtype=int)]]  # batch 0, time 0: (2, 0)
        alive_mask = np.array([[[True, True]]], dtype=bool)  # (1, 1, 2)

        # Test with include_features=[1] (keep only second feature)
        constructor = MagentImageObsDataConstructor(
            max_entities_perception=16,
            with_time_seq=False,
            n_workers=0,
            include_features=[1],
            channel_first=False,
        )

        result, mask = constructor.process(
            observations=observations,
            states=states,
            edge_indices=edge_indices,
            alive_mask=alive_mask,
        )

        # Should have feature dimension 1 (only index 1)
        self.assertEqual(result.shape[-1], 1)

        # Check that only feature index 1 is kept
        # Expected agent0: [[2.0], ...]
        expected_agent0 = np.zeros((16, 1), dtype=np.float32)
        expected_agent0[0, :] = [2.0]

        # Expected agent1: [[5.0], ...]
        expected_agent1 = np.zeros((16, 1), dtype=np.float32)
        expected_agent1[0, :] = [5.0]

        np.testing.assert_array_equal(result[0, 0], expected_agent0)
        np.testing.assert_array_equal(result[0, 1], expected_agent1)

    def test_entities_sorted_by_distance_then_angle(self):
        """Test that entities are sorted by distance (ascending) first, then angle (ascending) for image observations."""
        # batch_size=1, seq_len=1, 2 agents, H=2, W=2, C=3
        # Each pixel in HxW grid represents an entity with [rel_x, rel_y, value] features
        # Grid positions: [0,0], [0,1], [1,0], [1,1] correspond to different relative positions
        H, W, C = 2, 2, 3
        observations = np.zeros((1, 1, 2, H, W, C), dtype=np.float32)

        # Agent 0's observation:
        # [0,0] = [3.0, 0.0, 1.0] - dist=3
        # [0,1] = [0.0, 0.0, 0.0] - zero
        # [1,0] = [1.0, 0.0, 2.0] - dist=1
        # [1,1] = [0.0, 0.0, 0.0] - zero
        observations[0, 0, 0, 0, 0, :] = [3.0, 0.0, 1.0]
        observations[0, 0, 0, 1, 0, :] = [1.0, 0.0, 2.0]

        # Agent 1's observation:
        # [0,0] = [2.0, 0.0, 3.0] - dist=2
        # [0,1] = [0.0, 1.0, 4.0] - dist=1, angle=pi/2
        # [1,0] = [0.0, -1.0, 5.0] - dist=1, angle=-pi/2
        # [1,1] = [0.0, 0.0, 0.0] - zero
        observations[0, 0, 1, 0, 0, :] = [2.0, 0.0, 3.0]
        observations[0, 0, 1, 0, 1, :] = [0.0, 1.0, 4.0]
        observations[0, 0, 1, 1, 0, :] = [0.0, -1.0, 5.0]

        # Edge: agent1 -> agent0
        edge_indices = [[np.array([[1], [0]], dtype=int)]]
        alive_mask = np.array([[[True, True]]], dtype=bool)

        constructor = MagentImageObsDataConstructor(
            max_entities_perception=5,
            with_time_seq=False,
            n_workers=0,
            channel_first=False,
        )

        result, mask = constructor.process(
            observations=observations,
            states=None,
            edge_indices=edge_indices,
            alive_mask=alive_mask,
        )

        # After reshaping (H,W,C) -> (L,C) where L=H*W=4:
        # Agent 0's entities: [3,0,1], [0,0,0], [1,0,2], [0,0,0]
        # Agent 1's entities: [2,0,3], [0,1,4], [0,-1,5], [0,0,0]

        # Combined and sorted by distance, then angle:
        # 1. [0.0, -1.0, 5.0] - dist=1, angle=-pi/2
        # 2. [1.0, 0.0, 2.0] - dist=1, angle=0
        # 3. [0.0, 1.0, 4.0] - dist=1, angle=pi/2
        # 4. [2.0, 0.0, 3.0] - dist=2, angle=0
        # 5. [3.0, 0.0, 1.0] - dist=3, angle=0

        expected_agent0 = np.array(
            [
                [0.0, -1.0, 5.0],
                [1.0, 0.0, 2.0],
                [0.0, 1.0, 4.0],
                [2.0, 0.0, 3.0],
                [3.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        np.testing.assert_array_equal(result[0, 0], expected_agent0)

        # Check that all 5 entities are marked as real
        expected_mask_agent0 = np.array([True, True, True, True, True], dtype=bool)
        np.testing.assert_array_equal(mask[0, 0], expected_mask_agent0)


if __name__ == "__main__":
    unittest.main()
