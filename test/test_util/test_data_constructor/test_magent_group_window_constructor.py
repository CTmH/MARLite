import numpy as np
import unittest
from marlite.util.self_supervised_data_constructor.magent_group_window_constructor import (
    MagentGroupWindowConstructor,
)


class TestMagentGroupWindowConstructor(unittest.TestCase):
    """Test suite for MagentGroupWindowConstructor (A2)."""

    def _make_data(self, channel_first=False):
        """Create synthetic MAgent state data with known agent positions.

        Places agents at specific grid positions in the BINARY_AGENT_ID channels
        and returns states, observations, grouping, and alive_mask.

        Agent layout on a 20x20 grid (consecutive binary encoding):
            agent 0 (binary_id=0):  at (5, 5),  team 0   (all binary bits = 0)
            agent 1 (binary_id=1):  at (6, 6),  team 0   (bit 0 = 1)
            agent 2 (binary_id=2):  at (15, 15), team 0  (bit 1 = 1)
            agent 3 (binary_id=3):  at (16, 16), team 0  (bits 0,1 = 1)

        Grouping: agents 0,1 → group 0; agents 2,3 → group 1.
        """
        B, T, N = 2, 2, 4
        if channel_first:
            states = np.zeros((B, T, 37, 20, 20), dtype=np.float16)
        else:
            states = np.zeros((B, T, 20, 20, 37), dtype=np.float16)

        for b in range(B):
            for t in range(T):
                if channel_first:
                    # agent 0 at (5,5): binary_id=0 (all zeros) — presence ch 1, no binary bits
                    states[b, t, 1, 5, 5] = 1
                    # agent 1 at (6,6): binary_id=1 — presence ch 1, bit 0 (ch 5)
                    states[b, t, 1, 6, 6] = 1
                    states[b, t, 5, 6, 6] = 1
                    # agent 2 at (15,15): binary_id=2 — presence ch 1, bit 1 (ch 6)
                    states[b, t, 1, 15, 15] = 1
                    states[b, t, 6, 15, 15] = 1
                    # agent 3 at (16,16): binary_id=3 — presence ch 1, bits 0,1 (ch 5,6)
                    states[b, t, 1, 16, 16] = 1
                    states[b, t, 5, 16, 16] = 1
                    states[b, t, 6, 16, 16] = 1
                else:
                    # agent 0 at (5,5): binary_id=0 (all zeros)
                    states[b, t, 5, 5, 1] = 1
                    # agent 1 at (6,6): binary_id=1
                    states[b, t, 6, 6, 1] = 1
                    states[b, t, 6, 6, 5] = 1
                    # agent 2 at (15,15): binary_id=2
                    states[b, t, 15, 15, 1] = 1
                    states[b, t, 15, 15, 6] = 1
                    # agent 3 at (16,16): binary_id=3
                    states[b, t, 16, 16, 1] = 1
                    states[b, t, 16, 16, 5] = 1
                    states[b, t, 16, 16, 6] = 1

        grouping = np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.int8)
        alive_mask = np.ones((B, T, N), dtype=bool)
        observations = np.zeros((B, T, N, 1), dtype=np.float16)
        return states, observations, grouping, alive_mask

    def test_output_shape_channel_first_false(self):
        """Output shape when channel_first=False: (B, n_groups, K, K, sel_C)."""
        states, obs, grouping, alive_mask = self._make_data(channel_first=False)
        constructor = MagentGroupWindowConstructor(
            n_groups=3,
            window_size=7,
            selected_channels=[0, 1, 2, 3, 4],
            binary_agent_id_dim=list(range(5, 15)),
            agent_presence_dim=[1, 3],
            channel_first=False,
        )
        result = constructor.process(obs, states, grouping, alive_mask)
        self.assertEqual(result.shape, (2, 3, 7, 7, 5))
        self.assertEqual(result.dtype, np.float16)

    def test_output_shape_channel_first_true(self):
        """Output shape when channel_first=True: (B, n_groups, sel_C, K, K)."""
        states, obs, grouping, alive_mask = self._make_data(channel_first=True)
        constructor = MagentGroupWindowConstructor(
            n_groups=3,
            window_size=7,
            selected_channels=[0, 1, 2, 3, 4],
            binary_agent_id_dim=list(range(5, 15)),
            agent_presence_dim=[1, 3],
            channel_first=True,
        )
        result = constructor.process(obs, states, grouping, alive_mask)
        self.assertEqual(result.shape, (2, 3, 5, 7, 7))
        self.assertEqual(result.dtype, np.float16)

    def test_group_content_nonzero_channel_first_false(self):
        """Group 0 and 1 have content; empty slot (group 2) is zero."""
        states, obs, grouping, alive_mask = self._make_data(channel_first=False)
        constructor = MagentGroupWindowConstructor(
            n_groups=3,
            window_size=7,
            selected_channels=[0, 1, 2, 3, 4],
            binary_agent_id_dim=list(range(5, 15)),
            agent_presence_dim=[1, 3],
            channel_first=False,
        )
        result = constructor.process(obs, states, grouping, alive_mask)

        # Group 0 (agents at 5,5 and 6,6) should have content in presence channel
        self.assertTrue(np.any(result[0, 0] != 0), "Group 0 should have non-zero content")
        # Group 1 (agents at 15,15 and 16,16) should have content
        self.assertTrue(np.any(result[0, 1] != 0), "Group 1 should have non-zero content")
        # Group 2 should be all zeros (no agents assigned)
        self.assertTrue(np.all(result[0, 2] == 0), "Empty group slot should be all zeros")

    def test_group_content_nonzero_channel_first_true(self):
        """Group content when channel_first=True."""
        states, obs, grouping, alive_mask = self._make_data(channel_first=True)
        constructor = MagentGroupWindowConstructor(
            n_groups=3,
            window_size=7,
            selected_channels=[0, 1, 2, 3, 4],
            binary_agent_id_dim=list(range(5, 15)),
            agent_presence_dim=[1, 3],
            channel_first=True,
        )
        result = constructor.process(obs, states, grouping, alive_mask)

        self.assertTrue(np.any(result[0, 0] != 0), "Group 0 should have non-zero content")
        self.assertTrue(np.any(result[0, 1] != 0), "Group 1 should have non-zero content")
        self.assertTrue(np.all(result[0, 2] == 0), "Empty group slot should be all zeros")

    def test_presence_channel_encodes_agent_locations(self):
        """The TEAM_0_PRESENCE channel (index 1) should show agents at their positions."""
        states, obs, grouping, alive_mask = self._make_data(channel_first=False)
        constructor = MagentGroupWindowConstructor(
            n_groups=1,
            window_size=7,
            selected_channels=[0, 1, 2, 3, 4],
            binary_agent_id_dim=list(range(5, 15)),
            agent_presence_dim=[1, 3],
            channel_first=False,
        )
        result = constructor.process(obs, states, grouping, alive_mask)

        # Group 0 centroid is (5.5, 5.5), window is 7×7 centered there
        # Agent 0 at (5,5), relative to window center: window coords ~(3,3)
        # Agent 1 at (6,6), relative to window center: window coords ~(4,4)
        # The presence channel (index 1) should have non-zero values at those positions
        group0 = result[0, 0]  # (K, K, sel_C)
        presence_ch = group0[:, :, 1]  # TEAM_0_PRESENCE is index 1

        # Both agents should be visible in the 7x7 window
        nonzero_count = np.sum(presence_ch > 0)
        self.assertGreaterEqual(nonzero_count, 2, "Expected at least 2 agent presence pixels")

    def test_all_dead_agents_zero_output(self):
        """When all agents are dead, output is all zeros."""
        states, obs, grouping, alive_mask = self._make_data(channel_first=False)
        # All dead
        alive_mask[:, :, :] = False

        constructor = MagentGroupWindowConstructor(
            n_groups=2,
            window_size=5,
            selected_channels=[0, 1, 2, 3, 4],
            binary_agent_id_dim=list(range(5, 15)),
            agent_presence_dim=[1, 3],
            channel_first=False,
        )
        result = constructor.process(obs, states, grouping, alive_mask)
        self.assertTrue(np.all(result == 0), "All dead agents should yield all zeros")

    def test_grouping_all_minus_one_zero_output(self):
        """When grouping is all -1 (no valid groups), output is all zeros."""
        states, obs, grouping, alive_mask = self._make_data(channel_first=False)
        grouping[:, :] = -1

        constructor = MagentGroupWindowConstructor(
            n_groups=2,
            window_size=5,
            selected_channels=[0, 1, 2, 3, 4],
            binary_agent_id_dim=list(range(5, 15)),
            agent_presence_dim=[1, 3],
            channel_first=False,
        )
        result = constructor.process(obs, states, grouping, alive_mask)
        self.assertTrue(np.all(result == 0), "All -1 grouping should yield all zeros")

    def test_fewer_groups_than_n_groups(self):
        """When fewer groups exist than n_groups, extra slots are zero."""
        states, obs, grouping, alive_mask = self._make_data(channel_first=False)
        # Default data has agents 0,1 in group 0 and 2,3 in group 1.
        # With n_groups=3, group 2 should be zero.

        constructor = MagentGroupWindowConstructor(
            n_groups=3,
            window_size=7,
            selected_channels=[0, 1, 2, 3, 4],
            binary_agent_id_dim=list(range(5, 15)),
            agent_presence_dim=[1, 3],
            channel_first=False,
        )
        result = constructor.process(obs, states, grouping, alive_mask)

        self.assertTrue(np.any(result[0, 0] != 0), "Group 0 should have content")
        self.assertTrue(np.any(result[0, 1] != 0), "Group 1 should have content")
        self.assertTrue(np.all(result[0, 2] == 0), "Group 2 should be zeros (no agents)")

    def test_partially_dead_agents(self):
        """Dead agents are excluded from group centroid computation."""
        states, obs, grouping, alive_mask = self._make_data(channel_first=False)
        # Kill agents 1 and 3
        alive_mask[:, :, 1] = False
        alive_mask[:, :, 3] = False
        # Remove dead agents from state (their presence pixels persist otherwise)
        states[:, :, 6, 6, :] = 0   # agent 1
        states[:, :, 16, 16, :] = 0  # agent 3

        constructor = MagentGroupWindowConstructor(
            n_groups=2,
            window_size=7,
            selected_channels=[0, 1, 2, 3, 4],
            binary_agent_id_dim=list(range(5, 15)),
            agent_presence_dim=[1, 3],
            channel_first=False,
        )
        result = constructor.process(obs, states, grouping, alive_mask)

        # Group 0 centroid should now be at (5,5) (only agent 0 alive)
        # Group 1 centroid should now be at (15,15) (only agent 2 alive)
        self.assertTrue(np.any(result[0, 0] != 0), "Group 0 should have agent 0's content")
        self.assertTrue(np.any(result[0, 1] != 0), "Group 1 should have agent 2's content")

    def test_selected_channels_subsets_state(self):
        """selected_channels controls which state channels appear in output."""
        states, obs, grouping, alive_mask = self._make_data(channel_first=False)
        # Only select the obstacle channel (0) and TEAM_0_PRESENCE (1)
        constructor = MagentGroupWindowConstructor(
            n_groups=1,
            window_size=7,
            selected_channels=[0, 1],
            binary_agent_id_dim=list(range(5, 15)),
            agent_presence_dim=[1, 3],
            channel_first=False,
        )
        result = constructor.process(obs, states, grouping, alive_mask)
        # Last dim should be 2 (selected_channels count)
        self.assertEqual(result.shape, (2, 1, 7, 7, 2))

    def test_window_out_of_bounds_padding(self):
        """When centroid is near map edge, window is zero-padded."""
        states, obs, grouping, alive_mask = self._make_data(channel_first=False)
        # agent 0 is at (5,5) — near top-left corner
        # Window of 15 will extend beyond map edges
        grouping = np.array([[0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int8)

        constructor = MagentGroupWindowConstructor(
            n_groups=1,
            window_size=15,
            selected_channels=[0, 1, 2, 3, 4],
            binary_agent_id_dim=list(range(5, 15)),
            agent_presence_dim=[1, 3],
            channel_first=False,
        )
        result = constructor.process(obs, states, grouping, alive_mask)
        # Shape should be correct despite OOB
        self.assertEqual(result.shape, (2, 1, 15, 15, 5))
        # Should have some non-zero content (agent pixels) and some zero (OOB padding)
        self.assertTrue(np.any(result[0, 0] != 0), "Should have agent content")
        self.assertTrue(np.any(result[0, 0] == 0), "Should have OOB zero padding")


if __name__ == "__main__":
    unittest.main()
