import numpy as np
import unittest
from marlite.util.self_supervised_data_constructor.magent_group_features_constructor import (
    MagentGroupFeaturesConstructor,
)


class TestMagentGroupFeaturesConstructor(unittest.TestCase):
    """Test suite for MagentGroupFeaturesConstructor (D)."""

    def _make_data(self, channel_first=False):
        """Create synthetic MAgent state data with known agent positions and HP.

        Agent layout on a 20x20 grid (consecutive binary encoding):
            agent 0 (binary_id=0): at (5, 5), HP=10,  team 0 (our team)   (all binary bits = 0)
            agent 1 (binary_id=1): at (6, 6), HP=5,   team 0 (our team)   (bit 0 = 1)
            agent 2 (binary_id=2): at (15, 15), HP=8,  team 1 (enemy)     (bit 1 = 1)
            agent 3 (binary_id=3): at (16, 16), HP=3,  team 1 (enemy)     (bits 0,1 = 1)
            obstacle: at (10, 10)

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
                    # agent 0: binary_id=0, presence ch 1, HP ch 2
                    states[b, t, 1, 5, 5] = 1
                    states[b, t, 2, 5, 5] = 10
                    # agent 1: binary_id=1, presence ch 1, HP ch 2, bit 0 (ch 5)
                    states[b, t, 1, 6, 6] = 1
                    states[b, t, 2, 6, 6] = 5
                    states[b, t, 5, 6, 6] = 1
                    # enemy 2: binary_id=2, presence ch 3, HP ch 4, bit 1 (ch 6)
                    states[b, t, 3, 15, 15] = 1
                    states[b, t, 4, 15, 15] = 8
                    states[b, t, 6, 15, 15] = 1
                    # enemy 3: binary_id=3, presence ch 3, HP ch 4, bits 0,1 (ch 5,6)
                    states[b, t, 3, 16, 16] = 1
                    states[b, t, 4, 16, 16] = 3
                    states[b, t, 5, 16, 16] = 1
                    states[b, t, 6, 16, 16] = 1
                    # obstacle
                    states[b, t, 0, 10, 10] = 1
                else:
                    # agent 0: binary_id=0
                    states[b, t, 5, 5, 1] = 1
                    states[b, t, 5, 5, 2] = 10
                    # agent 1: binary_id=1
                    states[b, t, 6, 6, 1] = 1
                    states[b, t, 6, 6, 2] = 5
                    states[b, t, 6, 6, 5] = 1
                    # enemy 2: binary_id=2
                    states[b, t, 15, 15, 3] = 1
                    states[b, t, 15, 15, 4] = 8
                    states[b, t, 15, 15, 6] = 1
                    # enemy 3: binary_id=3
                    states[b, t, 16, 16, 3] = 1
                    states[b, t, 16, 16, 4] = 3
                    states[b, t, 16, 16, 5] = 1
                    states[b, t, 16, 16, 6] = 1
                    # obstacle
                    states[b, t, 10, 10, 0] = 1

        grouping = np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.int8)
        alive_mask = np.ones((B, T, N), dtype=bool)
        observations = np.zeros((B, T, N, 1), dtype=np.float16)
        return states, observations, grouping, alive_mask

    def test_output_shape(self):
        """Output shape: (B, n_groups, N_FEATURES)."""
        states, obs, grouping, alive_mask = self._make_data(channel_first=False)
        constructor = MagentGroupFeaturesConstructor(
            n_groups=3,
            window_size=7,
            binary_agent_id_dim=list(range(5, 15)),
            agent_presence_dim=[1, 3],
            our_team_presence_dim=1,
            our_team_hp_dim=2,
            enemy_team_presence_dim=3,
            enemy_team_hp_dim=4,
            obstacle_dim=0,
            channel_first=False,
        )
        result, _ = constructor.process(obs, states, grouping, alive_mask)
        self.assertEqual(result.shape, (2, 3, 18))
        self.assertEqual(result.dtype, np.float16)

    def test_our_team_features_group0_channel_first_false(self):
        """Group 0 (our team) features: density=2/49, HP mean=7.5."""
        states, obs, grouping, alive_mask = self._make_data(channel_first=False)
        constructor = MagentGroupFeaturesConstructor(
            n_groups=2,
            window_size=7,
            binary_agent_id_dim=list(range(5, 15)),
            agent_presence_dim=[1, 3],
            our_team_presence_dim=1,
            our_team_hp_dim=2,
            enemy_team_presence_dim=3,
            enemy_team_hp_dim=4,
            obstacle_dim=0,
            channel_first=False,
        )
        result, _ = constructor.process(obs, states, grouping, alive_mask)

        f = result[0, 0]
        # Our density: 2 agents / 49 cells (only TEAM_0 agents in window)
        self.assertAlmostEqual(float(f[0]), 2.0 / 49.0, places=2)
        # Our HP mean: (10 + 5) / 2 = 7.5
        self.assertAlmostEqual(float(f[1]), 7.5, places=1)
        # Our total HP: 15
        self.assertAlmostEqual(float(f[3]), 15.0, places=1)

    def test_our_team_features_group0_channel_first_true(self):
        """Group 0 features when channel_first=True should match channel_first=False."""
        states, obs, grouping, alive_mask = self._make_data(channel_first=True)
        constructor = MagentGroupFeaturesConstructor(
            n_groups=2,
            window_size=7,
            binary_agent_id_dim=list(range(5, 15)),
            agent_presence_dim=[1, 3],
            our_team_presence_dim=1,
            our_team_hp_dim=2,
            enemy_team_presence_dim=3,
            enemy_team_hp_dim=4,
            obstacle_dim=0,
            channel_first=True,
        )
        result, _ = constructor.process(obs, states, grouping, alive_mask)

        f = result[0, 0]
        self.assertAlmostEqual(float(f[0]), 2.0 / 49.0, places=2)
        self.assertAlmostEqual(float(f[1]), 7.5, places=1)

    def test_enemy_features_group1_channel_first_false(self):
        """Group 1 (enemy team from our perspective) features."""
        states, obs, grouping, alive_mask = self._make_data(channel_first=False)
        constructor = MagentGroupFeaturesConstructor(
            n_groups=2,
            window_size=7,
            binary_agent_id_dim=list(range(5, 15)),
            agent_presence_dim=[1, 3],
            our_team_presence_dim=1,
            our_team_hp_dim=2,
            enemy_team_presence_dim=3,
            enemy_team_hp_dim=4,
            obstacle_dim=0,
            channel_first=False,
        )
        result, _ = constructor.process(obs, states, grouping, alive_mask)

        f = result[0, 1]
        # Enemy density: 2 enemies / 49 cells
        self.assertAlmostEqual(float(f[5]), 2.0 / 49.0, places=2)
        # Enemy HP mean: (8 + 3) / 2 = 5.5
        self.assertAlmostEqual(float(f[6]), 5.5, places=1)
        # Enemy total HP: 11
        self.assertAlmostEqual(float(f[8]), 11.0, places=1)

    def test_obstacle_feature_group0(self):
        """Obstacle density should be captured."""
        states, obs, grouping, alive_mask = self._make_data(channel_first=False)
        # Place all agents in group 0 so the window includes the obstacle at (10,10)
        grouping[:, :] = 0

        constructor = MagentGroupFeaturesConstructor(
            n_groups=1,
            window_size=15,  # large enough to include obstacle
            binary_agent_id_dim=list(range(5, 15)),
            agent_presence_dim=[1, 3],
            our_team_presence_dim=1,
            our_team_hp_dim=2,
            enemy_team_presence_dim=3,
            enemy_team_hp_dim=4,
            obstacle_dim=0,
            channel_first=False,
        )
        result, _ = constructor.process(obs, states, grouping, alive_mask)

        f = result[0, 0]
        self.assertGreater(float(f[10]), 0.0, "Obstacle density should be > 0")

    def test_all_dead_agents_zero_output(self):
        """When all agents are dead, output is all zeros."""
        states, obs, grouping, alive_mask = self._make_data(channel_first=False)
        alive_mask[:, :, :] = False

        constructor = MagentGroupFeaturesConstructor(
            n_groups=2,
            window_size=7,
            binary_agent_id_dim=list(range(5, 15)),
            agent_presence_dim=[1, 3],
            our_team_presence_dim=1,
            our_team_hp_dim=2,
            enemy_team_presence_dim=3,
            enemy_team_hp_dim=4,
            obstacle_dim=0,
            channel_first=False,
        )
        result, _ = constructor.process(obs, states, grouping, alive_mask)
        self.assertTrue(np.all(result == 0), "All dead agents → all zeros")

    def test_grouping_all_minus_one_zero_output(self):
        """When grouping is all -1, output is all zeros."""
        states, obs, grouping, alive_mask = self._make_data(channel_first=False)
        grouping[:, :] = -1

        constructor = MagentGroupFeaturesConstructor(
            n_groups=2,
            window_size=7,
            binary_agent_id_dim=list(range(5, 15)),
            agent_presence_dim=[1, 3],
            our_team_presence_dim=1,
            our_team_hp_dim=2,
            enemy_team_presence_dim=3,
            enemy_team_hp_dim=4,
            obstacle_dim=0,
            channel_first=False,
        )
        result, _ = constructor.process(obs, states, grouping, alive_mask)
        self.assertTrue(np.all(result == 0), "All -1 grouping → all zeros")

    def test_fewer_groups_than_n_groups(self):
        """Extra group slots are zero when fewer groups exist."""
        states, obs, grouping, alive_mask = self._make_data(channel_first=False)
        # All agents in group 0
        grouping[:, :] = 0

        constructor = MagentGroupFeaturesConstructor(
            n_groups=3,
            window_size=7,
            binary_agent_id_dim=list(range(5, 15)),
            agent_presence_dim=[1, 3],
            our_team_presence_dim=1,
            our_team_hp_dim=2,
            enemy_team_presence_dim=3,
            enemy_team_hp_dim=4,
            obstacle_dim=0,
            channel_first=False,
        )
        result, _ = constructor.process(obs, states, grouping, alive_mask)

        self.assertFalse(np.all(result[0, 0] == 0), "Group 0 should have content")
        self.assertTrue(np.all(result[0, 1] == 0), "Group 1 should be zeros")
        self.assertTrue(np.all(result[0, 2] == 0), "Group 2 should be zeros")

    def test_partially_dead_agents(self):
        """Dead agents are excluded from group features by zeroing them in state."""
        states, obs, grouping, alive_mask = self._make_data(channel_first=False)
        # Kill agents 1 and 3
        alive_mask[:, :, 1] = False
        alive_mask[:, :, 3] = False

        # Also remove dead agents from the state (presence, HP, binary bits)
        # Agent 1 at (6,6): zero all channels
        states[:, :, 6, 6, :] = 0
        # Agent 3 at (16,16): zero all channels
        states[:, :, 16, 16, :] = 0

        constructor = MagentGroupFeaturesConstructor(
            n_groups=2,
            window_size=7,
            binary_agent_id_dim=list(range(5, 15)),
            agent_presence_dim=[1, 3],
            our_team_presence_dim=1,
            our_team_hp_dim=2,
            enemy_team_presence_dim=3,
            enemy_team_hp_dim=4,
            obstacle_dim=0,
            channel_first=False,
        )
        result, _ = constructor.process(obs, states, grouping, alive_mask)

        f_g0 = result[0, 0]
        f_g1 = result[0, 1]

        # Group 0: only agent 0 alive (binary_id=0, HP=10), our density = 1/49
        self.assertAlmostEqual(float(f_g0[0]), 1.0 / 49.0, places=2)
        self.assertAlmostEqual(float(f_g0[1]), 10.0, places=1)

        # Group 1: only agent 2 alive (binary_id=2, HP=8), enemy density = 1/49
        self.assertAlmostEqual(float(f_g1[5]), 1.0 / 49.0, places=2)
        self.assertAlmostEqual(float(f_g1[6]), 8.0, places=1)

    def test_features_in_valid_range(self):
        """All features should be within reasonable ranges."""
        states, obs, grouping, alive_mask = self._make_data(channel_first=False)
        constructor = MagentGroupFeaturesConstructor(
            n_groups=2,
            window_size=7,
            binary_agent_id_dim=list(range(5, 15)),
            agent_presence_dim=[1, 3],
            our_team_presence_dim=1,
            our_team_hp_dim=2,
            enemy_team_presence_dim=3,
            enemy_team_hp_dim=4,
            obstacle_dim=0,
            channel_first=False,
        )
        result, _ = constructor.process(obs, states, grouping, alive_mask)

        for g in range(2):
            f = result[0, g]
            # Densities should be in [0, 1]
            for i in [0, 5, 10, 11]:
                self.assertTrue(0.0 <= float(f[i]) <= 1.0,
                                f"Feature {i} ({float(f[i])}) should be in [0, 1]")
            # Nearest distances should be in [0, 1] (normalized by K)
            for i in [12, 13]:
                self.assertTrue(0.0 <= float(f[i]) <= 1.0,
                                f"Feature {i} ({float(f[i])}) should be in [0, 1]")
            # Border proximity should be in [0, 1]
            self.assertTrue(0.0 <= float(f[14]) <= 1.0)
            # Ratio should be in [0, 1]
            self.assertTrue(0.0 <= float(f[15]) <= 1.0)

    def test_group_size_feature_matches_alive_count(self):
        """Feature 11 (group_size) should count all alive agents in group normalized by K²."""
        states, obs, grouping, alive_mask = self._make_data(channel_first=False)
        constructor = MagentGroupFeaturesConstructor(
            n_groups=1,
            window_size=7,
            binary_agent_id_dim=list(range(5, 15)),
            agent_presence_dim=[1, 3],
            our_team_presence_dim=1,
            our_team_hp_dim=2,
            enemy_team_presence_dim=3,
            enemy_team_hp_dim=4,
            obstacle_dim=0,
            channel_first=False,
        )
        # All 4 agents in group 0 (2 ours + 2 enemies)
        grouping[:, :] = 0

        result, _ = constructor.process(obs, states, grouping, alive_mask)
        f = result[0, 0]
        # Group size = all alive agents = 4 / 49
        self.assertAlmostEqual(float(f[11]), 4.0 / 49.0, places=2)

    def test_border_proximity_near_top_left(self):
        """Border proximity for agents near (5,5) — close to top-left corner."""
        states, obs, grouping, alive_mask = self._make_data(channel_first=False)
        grouping[:, :] = 0

        constructor = MagentGroupFeaturesConstructor(
            n_groups=1,
            window_size=7,
            binary_agent_id_dim=list(range(5, 15)),
            agent_presence_dim=[1, 3],
            our_team_presence_dim=1,
            our_team_hp_dim=2,
            enemy_team_presence_dim=3,
            enemy_team_hp_dim=4,
            obstacle_dim=0,
            channel_first=False,
        )
        result, _ = constructor.process(obs, states, grouping, alive_mask)
        f = result[0, 0]
        # Centroid is ~(8, 8) on a 20x20 map → border distance = 8
        # Normalized: 8/20 = 0.4
        self.assertLess(float(f[14]), 0.5, "Should be close to border")
        self.assertGreater(float(f[14]), 0.0, "Should not be at exact border")


    def test_parallel_vs_sequential_consistency(self):
        """Ensure n_workers=0 and n_workers=2 produce identical feature vectors."""
        for channel_first in (False, True):
            states, obs, grouping, alive_mask = self._make_data(channel_first=channel_first)
            constructor_seq = MagentGroupFeaturesConstructor(
                n_groups=3,
                window_size=7,
                binary_agent_id_dim=list(range(5, 15)),
                agent_presence_dim=[1, 3],
                our_team_presence_dim=1,
                our_team_hp_dim=2,
                enemy_team_presence_dim=3,
                enemy_team_hp_dim=4,
                obstacle_dim=0,
                channel_first=channel_first,
                n_workers=0,
            )
            constructor_par = MagentGroupFeaturesConstructor(
                n_groups=3,
                window_size=7,
                binary_agent_id_dim=list(range(5, 15)),
                agent_presence_dim=[1, 3],
                our_team_presence_dim=1,
                our_team_hp_dim=2,
                enemy_team_presence_dim=3,
                enemy_team_hp_dim=4,
                obstacle_dim=0,
                channel_first=channel_first,
                n_workers=2,
            )

            result_seq, mask_seq = constructor_seq.process(
                obs, states, grouping, alive_mask
            )
            result_par, mask_par = constructor_par.process(
                obs, states, grouping, alive_mask
            )

            np.testing.assert_array_equal(result_seq, result_par)
            np.testing.assert_array_equal(mask_seq, mask_par)


if __name__ == "__main__":
    unittest.main()
