import numpy as np
import unittest
from marlite.util.self_supervised_data_constructor.sumo_obs_data_constructor import SumoObsDataConstructor


class TestSumoObsDataConstructor(unittest.TestCase):
    """Test suite for SumoObsDataConstructor."""

    def test_process_empty_edges_without_time_seq(self):
        """Test behavior when edge_indices is empty (no communication) without time sequence."""
        # Setup: batch_size=1, seq_len=1, 2 agents, feature_dim=3, max_entities_perception=4
        observations = np.array([[
            [[1.0, 2.0, 3.0],  # agent 0 at time 0
             [4.0, 5.0, 6.0]],  # agent 1 at time 0
        ]], dtype=np.float32)  # shape: (1, 1, 2, 3)

        states = None
        # Empty edge_indices: List[List[np.ndarray]] where each time step has (2, 0) shape
        edge_indices = [[np.empty((2, 0), dtype=np.int64)]]  # batch 0, time 0: (2, 0)

        # Alive mask: (batch_size, seq_len, n_agents)
        alive_mask = np.array([[[True, True]]], dtype=bool)  # (1, 1, 2)

        constructor = SumoObsDataConstructor(max_entities_perception=4, with_time_seq=False, n_workers=0)

        result, mask = constructor.process(
            observations=observations,
            states=states,
            edge_indices=edge_indices,
            alive_mask=alive_mask
        )

        # Expected: each agent sees only its own observation (self-loop included in edge_indices)
        # Since edge_indices is empty, only self observation
        expected_agent0 = np.array([
            [1.0, 2.0, 3.0],  # self
            [0.0, 0.0, 0.0],  # padding
            [0.0, 0.0, 0.0],  # padding
            [0.0, 0.0, 0.0],  # padding
        ], dtype=np.float32)

        expected_agent1 = np.array([
            [4.0, 5.0, 6.0],  # self
            [0.0, 0.0, 0.0],  # padding
            [0.0, 0.0, 0.0],  # padding
            [0.0, 0.0, 0.0],  # padding
        ], dtype=np.float32)

        np.testing.assert_array_equal(result[0, 0], expected_agent0)
        np.testing.assert_array_equal(result[0, 1], expected_agent1)

        # Check mask: only first entity should be True (self)
        expected_mask_agent0 = np.array([True, False, False, False], dtype=bool)
        expected_mask_agent1 = np.array([True, False, False, False], dtype=bool)
        np.testing.assert_array_equal(mask[0, 0], expected_mask_agent0)
        np.testing.assert_array_equal(mask[0, 1], expected_mask_agent1)

    def test_process_with_incoming_edges_without_time_seq(self):
        """Test with one incoming edge: agent1 -> agent0, without time sequence."""
        # batch_size=1, seq_len=1, 2 agents, feature_dim=2
        observations = np.array([[
            [[1.0, 1.0],  # agent0 at time 0
             [2.0, 2.0]],  # agent1 at time 0
        ]], dtype=np.float32)  # (1, 1, 2, 2)

        # Edge indices: agent1 -> agent0 (plus self-loops)
        # edges: (0->0), (1->1), (1->0)
        edge_indices = [[np.array([[0, 1, 1],  # sources: 0, 1, 1
                                   [0, 1, 0]], dtype=np.int64)]]  # targets: 0, 1, 0

        # Alive mask: both agents alive
        alive_mask = np.array([[[True, True]]], dtype=bool)  # (1, 1, 2)

        constructor = SumoObsDataConstructor(max_entities_perception=3, with_time_seq=False, n_workers=0)

        result, mask = constructor.process(
            observations=observations,
            states=None,
            edge_indices=edge_indices,
            alive_mask=alive_mask
        )

        # agent0 sees: self (0) + agent1 (1) -> sorted: [0, 1]
        expected_agent0 = np.array([
            [1.0, 1.0],  # self (agent0)
            [2.0, 2.0],  # agent1
            [0.0, 0.0],  # padding
        ], dtype=np.float32)

        # agent1 sees: only self (no incoming edges except self-loop)
        expected_agent1 = np.array([
            [2.0, 2.0],  # self (agent1)
            [0.0, 0.0],  # padding
            [0.0, 0.0],  # padding
        ], dtype=np.float32)

        np.testing.assert_array_equal(result[0, 0], expected_agent0)
        np.testing.assert_array_equal(result[0, 1], expected_agent1)

        # Check mask
        expected_mask_agent0 = np.array([True, True, False], dtype=bool)
        expected_mask_agent1 = np.array([True, False, False], dtype=bool)
        np.testing.assert_array_equal(mask[0, 0], expected_mask_agent0)
        np.testing.assert_array_equal(mask[0, 1], expected_mask_agent1)

    def test_process_with_multiple_incoming_edges_without_time_seq(self):
        """Test with multiple incoming edges, sorting by agent index."""
        # batch_size=1, seq_len=1, 4 agents, feature_dim=2
        observations = np.array([[
            [[1.0, 1.0],  # agent0
             [2.0, 2.0],  # agent1
             [3.0, 3.0],  # agent2
             [4.0, 4.0]],  # agent3
        ]], dtype=np.float32)  # (1, 1, 4, 2)

        # Edge indices:
        # agent0 receives from: self, agent3, agent2, agent1 (but max_entities_perception=3)
        # edges: (0->0), (1->1), (2->2), (3->3) [self-loops]
        # edges: (3->0), (2->0), (1->0) [incoming to agent0]
        edge_indices = [[np.array([[0, 1, 2, 3, 3, 2, 1],  # sources
                                   [0, 1, 2, 3, 0, 0, 0]], dtype=np.int64)]]  # targets

        alive_mask = np.array([[[True, True, True, True]]], dtype=bool)

        constructor = SumoObsDataConstructor(max_entities_perception=3, with_time_seq=False, n_workers=0)

        result, mask = constructor.process(
            observations=observations,
            states=None,
            edge_indices=edge_indices,
            alive_mask=alive_mask
        )

        # agent0 sees: self (0) + incoming neighbors sorted: [1, 2, 3]
        # But max_entities_perception=3, so we keep: [0, 1, 2] (sorted, truncate)
        expected_agent0 = np.array([
            [1.0, 1.0],  # self (agent0)
            [2.0, 2.0],  # agent1
            [3.0, 3.0],  # agent2
            # agent3 is dropped due to truncation
        ], dtype=np.float32)

        # agent1, agent2, agent3 see only self
        expected_agent1 = np.array([
            [2.0, 2.0],  # self
            [0.0, 0.0],  # padding
            [0.0, 0.0],  # padding
        ], dtype=np.float32)

        np.testing.assert_array_equal(result[0, 0], expected_agent0)
        np.testing.assert_array_equal(result[0, 1], expected_agent1)

        # Check mask
        expected_mask_agent0 = np.array([True, True, True], dtype=bool)
        expected_mask_agent1 = np.array([True, False, False], dtype=bool)
        np.testing.assert_array_equal(mask[0, 0], expected_mask_agent0)
        np.testing.assert_array_equal(mask[0, 1], expected_mask_agent1)

    def test_process_with_time_seq(self):
        """Test with time sequence dimension."""
        # batch_size=1, seq_len=2, 2 agents, feature_dim=2
        observations = np.array([[
            [[1.0, 1.0],  # agent0 at time 0
             [2.0, 2.0]],  # agent1 at time 0
            [[3.0, 3.0],  # agent0 at time 1
             [4.0, 4.0]],  # agent1 at time 1
        ]], dtype=np.float32)  # (1, 2, 2, 2)

        # Edge indices: agent1 -> agent0 at both time steps
        edge_indices = [[
            np.array([[0, 1, 1],  # time 0: sources
                      [0, 1, 0]], dtype=np.int64),  # targets
            np.array([[0, 1, 1],  # time 1: sources
                      [0, 1, 0]], dtype=np.int64),  # targets
        ]]

        alive_mask = np.array([[[True, True], [True, True]]], dtype=bool)

        constructor = SumoObsDataConstructor(max_entities_perception=3, with_time_seq=True, n_workers=0)

        result, mask = constructor.process(
            observations=observations,
            states=None,
            edge_indices=edge_indices,
            alive_mask=alive_mask
        )

        # Check dimensions
        self.assertEqual(result.shape, (1, 2, 2, 3, 2))
        self.assertEqual(mask.shape, (1, 2, 2, 3))

        # agent0 at time 0: sees self + agent1
        expected_agent0_t0 = np.array([
            [1.0, 1.0],  # self
            [2.0, 2.0],  # agent1
            [0.0, 0.0],  # padding
        ], dtype=np.float32)

        # agent0 at time 1: sees self + agent1
        expected_agent0_t1 = np.array([
            [3.0, 3.0],  # self
            [4.0, 4.0],  # agent1
            [0.0, 0.0],  # padding
        ], dtype=np.float32)

        np.testing.assert_array_equal(result[0, 0, 0], expected_agent0_t0)
        np.testing.assert_array_equal(result[0, 1, 0], expected_agent0_t1)

    def test_process_dead_agent(self):
        """Test that dead agents are ignored."""
        # batch_size=1, seq_len=1, 3 agents, feature_dim=2
        observations = np.array([[
            [[1.0, 1.0],  # agent0 (alive)
             [2.0, 2.0],  # agent1 (dead)
             [3.0, 3.0]],  # agent2 (alive)
        ]], dtype=np.float32)

        # Edge indices: agent2 -> agent0, agent1 -> agent0 (but agent1 is dead)
        edge_indices = [[np.array([[0, 1, 2, 2, 1],  # sources
                                   [0, 1, 2, 0, 0]], dtype=np.int64)]]  # targets

        alive_mask = np.array([[[True, False, True]]], dtype=bool)  # agent1 dead

        constructor = SumoObsDataConstructor(max_entities_perception=3, with_time_seq=False, n_workers=0)

        result, mask = constructor.process(
            observations=observations,
            states=None,
            edge_indices=edge_indices,
            alive_mask=alive_mask
        )

        # agent0 sees: self + agent2 (agent1 is dead, so ignored)
        expected_agent0 = np.array([
            [1.0, 1.0],  # self
            [3.0, 3.0],  # agent2
            [0.0, 0.0],  # padding
        ], dtype=np.float32)

        # agent1 is dead: all zeros
        expected_agent1 = np.zeros((3, 2), dtype=np.float32)

        # agent2 sees only self
        expected_agent2 = np.array([
            [3.0, 3.0],  # self
            [0.0, 0.0],  # padding
            [0.0, 0.0],  # padding
        ], dtype=np.float32)

        np.testing.assert_array_equal(result[0, 0], expected_agent0)
        np.testing.assert_array_equal(result[0, 1], expected_agent1)
        np.testing.assert_array_equal(result[0, 2], expected_agent2)

        # Check mask
        expected_mask_agent0 = np.array([True, True, False], dtype=bool)
        expected_mask_agent1 = np.array([False, False, False], dtype=bool)
        expected_mask_agent2 = np.array([True, False, False], dtype=bool)
        np.testing.assert_array_equal(mask[0, 0], expected_mask_agent0)
        np.testing.assert_array_equal(mask[0, 1], expected_mask_agent1)
        np.testing.assert_array_equal(mask[0, 2], expected_mask_agent2)

    def test_exclude_features(self):
        """Test feature filtering with exclude_features."""
        # batch_size=1, seq_len=1, 2 agents, feature_dim=3
        observations = np.array([[
            [[1.0, 2.0, 3.0],  # agent0
             [4.0, 5.0, 6.0]],  # agent1
        ]], dtype=np.float32)

        # Edge indices: agent1 -> agent0
        edge_indices = [[np.array([[0, 1, 1],  # sources
                                   [0, 1, 0]], dtype=np.int64)]]  # targets

        alive_mask = np.array([[[True, True]]], dtype=bool)

        constructor = SumoObsDataConstructor(
            max_entities_perception=3,
            with_time_seq=False,
            n_workers=0,
            exclude_features=[1]  # remove second feature
        )

        result, mask = constructor.process(
            observations=observations,
            states=None,
            edge_indices=edge_indices,
            alive_mask=alive_mask
        )

        # Should have feature dimension 2 (3-1)
        self.assertEqual(result.shape[-1], 2)

        # agent0 sees: self + agent1, features [0, 2] (excluding index 1)
        expected_agent0 = np.array([
            [1.0, 3.0],  # self, features 0 and 2
            [4.0, 6.0],  # agent1, features 0 and 2
            [0.0, 0.0],  # padding
        ], dtype=np.float32)

        np.testing.assert_array_equal(result[0, 0], expected_agent0)

    def test_include_features(self):
        """Test feature filtering with include_features."""
        # batch_size=1, seq_len=1, 2 agents, feature_dim=3
        observations = np.array([[
            [[1.0, 2.0, 3.0],  # agent0
             [4.0, 5.0, 6.0]],  # agent1
        ]], dtype=np.float32)

        # Edge indices: agent1 -> agent0
        edge_indices = [[np.array([[0, 1, 1],  # sources
                                   [0, 1, 0]], dtype=np.int64)]]  # targets

        alive_mask = np.array([[[True, True]]], dtype=bool)

        constructor = SumoObsDataConstructor(
            max_entities_perception=3,
            with_time_seq=False,
            n_workers=0,
            include_features=[0, 2]  # keep only features 0 and 2
        )

        result, mask = constructor.process(
            observations=observations,
            states=None,
            edge_indices=edge_indices,
            alive_mask=alive_mask
        )

        # Should have feature dimension 2
        self.assertEqual(result.shape[-1], 2)

        # agent0 sees: self + agent1, features [0, 2]
        expected_agent0 = np.array([
            [1.0, 3.0],  # self, features 0 and 2
            [4.0, 6.0],  # agent1, features 0 and 2
            [0.0, 0.0],  # padding
        ], dtype=np.float32)

        np.testing.assert_array_equal(result[0, 0], expected_agent0)

    def test_parallel_vs_sequential_consistency(self):
        """Ensure n_workers=0 and n_workers=1 produce identical results."""
        # Larger test case
        np.random.seed(42)
        batch_size = 2
        seq_len = 3
        n_agents = 4
        feature_dim = 5

        observations = np.random.randn(batch_size, seq_len, n_agents, feature_dim).astype(np.float32)

        # Create random edge indices as List[List[np.ndarray]]
        edge_indices = []
        for b in range(batch_size):
            batch_edges = []
            for t in range(seq_len):
                # Create random edges (including self-loops)
                n_edges = np.random.randint(5, 15)
                sources = np.random.randint(0, n_agents, size=n_edges)
                targets = np.random.randint(0, n_agents, size=n_edges)
                batch_edges.append(np.stack([sources, targets], axis=0))
            edge_indices.append(batch_edges)

        alive_mask = np.ones((batch_size, seq_len, n_agents), dtype=bool)

        constructor_seq = SumoObsDataConstructor(max_entities_perception=6, with_time_seq=True, n_workers=0)
        constructor_par = SumoObsDataConstructor(max_entities_perception=6, with_time_seq=True, n_workers=1)

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

    def test_edge_indices_self_loop_handling(self):
        """Test that self-loops in edge_indices are handled correctly."""
        # batch_size=1, seq_len=1, 2 agents, feature_dim=2
        observations = np.array([[
            [[1.0, 1.0],  # agent0
             [2.0, 2.0]],  # agent1
        ]], dtype=np.float32)

        # Edge indices with only self-loops
        edge_indices = [[np.array([[0, 1],  # sources
                                   [0, 1]], dtype=np.int64)]]  # targets (self-loops only)

        alive_mask = np.array([[[True, True]]], dtype=bool)

        constructor = SumoObsDataConstructor(max_entities_perception=3, with_time_seq=False, n_workers=0)

        result, mask = constructor.process(
            observations=observations,
            states=None,
            edge_indices=edge_indices,
            alive_mask=alive_mask
        )

        # Each agent should see only itself
        expected_agent0 = np.array([
            [1.0, 1.0],  # self
            [0.0, 0.0],  # padding
            [0.0, 0.0],  # padding
        ], dtype=np.float32)

        expected_agent1 = np.array([
            [2.0, 2.0],  # self
            [0.0, 0.0],  # padding
            [0.0, 0.0],  # padding
        ], dtype=np.float32)

        np.testing.assert_array_equal(result[0, 0], expected_agent0)
        np.testing.assert_array_equal(result[0, 1], expected_agent1)


if __name__ == '__main__':
    unittest.main()