import numpy as np
import unittest
from marlite.util.self_supervised_data_constructor.magent_vec_state_group_features_constructor import (
    MagentVecStateGroupFeaturesConstructor,
)


class TestMagentVecStateGroupFeaturesConstructor(unittest.TestCase):

    FEATURE_DIM = 62
    N_OUR = 4
    N_EN = 3
    K = N_OUR  # normalization constant

    def _make_data(self):
        """State: (B, T, N_all, F) with N_all = N_OUR + N_EN.

        Our agents (team=0):
            agent 0: HP=10, at (0.25, 0.25), 1 nearby obstacle
            agent 1: HP=5,  at (0.30, 0.30), no obstacles
            agent 2: HP=2,  at (0.70, 0.50)
            agent 3: HP=0,  at (0.50, 0.50)  (dead)

        Enemy agents (team=1):
            agent 4: HP=8,  at (0.35, 0.35)  ← close to agents 0,1
            agent 5: HP=3,  at (0.38, 0.40), 1 nearby obstacle
            agent 6: HP=0,  at (0.60, 0.60)  (dead)

        grouping: (B, N_OUR) — agents 0,1 in group 0; agent 2 in group 1; agent 3 unassigned.
        """
        N_ALL = self.N_OUR + self.N_EN
        B, T = 2, 2
        states = np.zeros((B, T, N_ALL, self.FEATURE_DIM), dtype=np.float16)

        for b in range(B):
            for t in range(T):
                # Our agents
                states[b, t, 0, 0] = 10; states[b, t, 0, 1] = 0
                states[b, t, 0, 3] = 0.25; states[b, t, 0, 4] = 0.25
                states[b, t, 0, 28] = 1  # obstacle

                states[b, t, 1, 0] = 5; states[b, t, 1, 1] = 0
                states[b, t, 1, 3] = 0.30; states[b, t, 1, 4] = 0.30

                states[b, t, 2, 0] = 2; states[b, t, 2, 1] = 0
                states[b, t, 2, 3] = 0.70; states[b, t, 2, 4] = 0.50

                states[b, t, 3, 0] = 0; states[b, t, 3, 1] = 0
                states[b, t, 3, 3] = 0.50; states[b, t, 3, 4] = 0.50

                # Enemy agents
                states[b, t, 4, 0] = 8; states[b, t, 4, 1] = 1
                states[b, t, 4, 3] = 0.35; states[b, t, 4, 4] = 0.35

                states[b, t, 5, 0] = 3; states[b, t, 5, 1] = 1
                states[b, t, 5, 3] = 0.38; states[b, t, 5, 4] = 0.40
                states[b, t, 5, 31] = 1  # obstacle

                states[b, t, 6, 0] = 0; states[b, t, 6, 1] = 1
                states[b, t, 6, 3] = 0.60; states[b, t, 6, 4] = 0.60

        grouping = np.array([[0, 0, 1, -1], [0, 0, 1, -1]], dtype=np.int8)
        alive_mask = np.ones((B, T, self.N_OUR), dtype=bool)
        observations = np.zeros((B, T, self.N_OUR, 1), dtype=np.float16)
        return states, observations, grouping, alive_mask

    def _make_ctor(self, n_groups=3, **kwargs):
        d = dict(
            n_groups=n_groups,
            observation_range=0.3,
            coord_dims=(3, 4),
            hp_dim=0,
            team_dim=1,
            my_team=0,
            enemy_team=1,
            action_dim=21,
            n_offsets=12,
        )
        d.update(kwargs)
        return MagentVecStateGroupFeaturesConstructor(**d)

    # ── basic ─────────────────────────────────────────────────────────────

    def test_N_FEATURES(self):
        self.assertEqual(MagentVecStateGroupFeaturesConstructor.N_FEATURES, 17)

    def test_output_shape(self):
        s, o, g, a = self._make_data()
        c = self._make_ctor(n_groups=3)
        r, pm = c.process(o, s, g, a)
        self.assertEqual(r.shape, (2, 3, 17))
        self.assertEqual(r.dtype, np.float16)
        self.assertEqual(pm.shape, (2, 3))
        self.assertEqual(pm.dtype, bool)

    # ── our features ──────────────────────────────────────────────────────

    def test_our_team_features(self):
        s, o, g, a = self._make_data()
        c = self._make_ctor(n_groups=2)
        r, pm = c.process(o, s, g, a)

        f = r[0, 0]
        self.assertAlmostEqual(float(f[0]), 2.0 / self.K, places=3)  # density
        self.assertAlmostEqual(float(f[1]), 7.5, places=1)            # HP mean
        self.assertAlmostEqual(float(f[3]), 15.0, places=1)           # HP total

    def test_dead_excluded_by_hp(self):
        s, o, g, a = self._make_data()
        g[:, 3] = 0  # dead agent 3 into group 0
        c = self._make_ctor(n_groups=2)
        r, pm = c.process(o, s, g, a)
        f = r[0, 0]
        self.assertAlmostEqual(float(f[0]), 2.0 / self.K, places=3)

    # ── enemy features via observation range ──────────────────────────────

    def test_enemies_within_range(self):
        """Enemies 4,5 at (0.35,0.35), (0.38,0.40) are visible from agents 0,1."""
        s, o, g, a = self._make_data()
        c = self._make_ctor(n_groups=2, observation_range=0.3)
        r, pm = c.process(o, s, g, a)

        f = r[0, 0]
        # agents 0,1 see enemies 4,5 → dedup → 2 enemies
        # en density = 2 / K = 2 / 4
        self.assertAlmostEqual(float(f[5]), 2.0 / self.K, places=3)

    def test_enemies_out_of_range(self):
        s, o, g, a = self._make_data()
        c = self._make_ctor(n_groups=2, observation_range=0.02)
        r, pm = c.process(o, s, g, a)
        f = r[0, 0]
        self.assertAlmostEqual(float(f[5]), 0.0, places=3)

    def test_hp_zero_enemies_excluded(self):
        s, o, g, a = self._make_data()
        # enemy 6 has HP=0 → not counted
        c = self._make_ctor(n_groups=2, observation_range=0.6)
        r, pm = c.process(o, s, g, a)
        f = r[0, 0]
        self.assertAlmostEqual(float(f[5]), 2.0 / self.K, places=3)

    def test_enemy_deduplication(self):
        s, o, g, a = self._make_data()
        # agents 0,1 both see enemies 4,5 → dedup → 2
        c = self._make_ctor(n_groups=2, observation_range=0.5)
        r, pm = c.process(o, s, g, a)
        f = r[0, 0]
        self.assertAlmostEqual(float(f[5]), 2.0 / self.K, places=3)

    # ── obstacle density ──────────────────────────────────────────────────

    def test_obstacle_density(self):
        s, o, g, a = self._make_data()
        c = self._make_ctor(n_groups=2, observation_range=0.3)
        r, pm = c.process(o, s, g, a)
        # agent 0 has 1 obstacle / 12 offsets; agent 1 has 0
        f = r[0, 0]
        self.assertAlmostEqual(float(f[10]), 1.0 / 24.0, places=3)

    # ── border proximity ──────────────────────────────────────────────────

    def test_border_proximity(self):
        s, o, g, a = self._make_data()
        c = self._make_ctor(n_groups=2, observation_range=0.3)
        r, pm = c.process(o, s, g, a)
        f = r[0, 0]
        # centroid (0.275,0.275) → min(0.275,0.725,0.275,0.725)=0.275
        self.assertAlmostEqual(float(f[13]), 0.275, places=2)

    def test_border_proximity_near_edge(self):
        s, o, g, a = self._make_data()
        s[:, :, 0, 3] = 0.01; s[:, :, 0, 4] = 0.50
        s[:, :, 1, 0] = 0; s[:, :, 2, 0] = 0  # kill 1,2
        c = self._make_ctor(n_groups=1, observation_range=0.3)
        r, pm = c.process(o, s, g, a)
        self.assertLess(float(r[0, 0, 13]), 0.02)

    # ── zero-fill: dead / -1 / extra slots ────────────────────────────────

    def test_all_dead_zero(self):
        s, o, g, a = self._make_data()
        for i in range(self.N_OUR):
            s[:, :, i, 0] = 0
        c = self._make_ctor(n_groups=2, observation_range=0.3)
        r, pm = c.process(o, s, g, a)
        self.assertTrue(np.all(r == 0))
        self.assertFalse(pm.any())

    def test_grouping_all_minus_one_zero(self):
        s, o, g, a = self._make_data()
        g[:, :] = -1
        c = self._make_ctor(n_groups=2, observation_range=0.3)
        r, pm = c.process(o, s, g, a)
        self.assertTrue(np.all(r == 0))
        self.assertFalse(pm.any())

    def test_extra_group_slots_zero(self):
        s, o, g, a = self._make_data()
        g[:, :] = 0  # all in group 0
        c = self._make_ctor(n_groups=3, observation_range=0.3)
        r, pm = c.process(o, s, g, a)
        self.assertFalse(np.all(r[0, 0] == 0))
        self.assertTrue(np.all(r[0, 1] == 0))
        self.assertTrue(np.all(r[0, 2] == 0))
        self.assertTrue(pm[0, 0])
        self.assertFalse(pm[0, 1])
        self.assertFalse(pm[0, 2])

    # ── misc ──────────────────────────────────────────────────────────────

    def test_features_in_valid_range(self):
        s, o, g, a = self._make_data()
        c = self._make_ctor(n_groups=2, observation_range=0.3)
        r, pm = c.process(o, s, g, a)
        for gid in range(2):
            f = r[0, gid]
            for i in [0, 5, 10]:
                v = float(f[i])
                self.assertTrue(0.0 <= v <= 1.0, f"feat {i}={v}")
            for i in [11, 12]:
                v = float(f[i])
                self.assertTrue(0.0 <= v <= 1.0, f"feat {i}={v}")
            self.assertTrue(0.0 <= float(f[13]) <= 1.0)
            self.assertTrue(0.0 <= float(f[14]) <= 1.0)

    def test_single_agent_no_dispersion(self):
        s, o, g, a = self._make_data()
        s[:, :, 1, 0] = 0  # kill agent 1
        c = self._make_ctor(n_groups=2, observation_range=0.3)
        r, pm = c.process(o, s, g, a)
        self.assertAlmostEqual(float(r[0, 0, 4]), 0.0, places=3)

    def test_partially_dead(self):
        s, o, g, a = self._make_data()
        s[:, :, 1, 0] = 0; s[:, :, 5, 0] = 0
        g[:, 1] = -1
        c = self._make_ctor(n_groups=2, observation_range=0.3)
        r, pm = c.process(o, s, g, a)
        self.assertAlmostEqual(float(r[0, 0, 0]), 1.0 / self.K, places=3)
        self.assertAlmostEqual(float(r[0, 0, 1]), 10.0, places=1)
        self.assertAlmostEqual(float(r[0, 1, 0]), 1.0 / self.K, places=3)
        self.assertAlmostEqual(float(r[0, 1, 1]), 2.0, places=1)

    def test_all_dead_enemies_zero_ratio(self):
        s, o, g, a = self._make_data()
        s[:, :, 0, 0] = 0; s[:, :, 1, 0] = 0; s[:, :, 2, 0] = 0
        g[:, :] = 0
        c = self._make_ctor(n_groups=1, observation_range=0.5)
        r, pm = c.process(o, s, g, a)
        # all our agents dead → no group matches → array stays zero
        self.assertTrue(np.all(r == 0))
        self.assertFalse(pm.any())

    def test_action_dim_n_offsets(self):
        s, o, g, a = self._make_data()
        c = self._make_ctor(n_groups=1, observation_range=0.3,
                            action_dim=13, n_offsets=8)
        r, pm = c.process(o, s, g, a)
        self.assertEqual(r.shape, (2, 1, 17))

    def test_observation_range_is_chebyshev(self):
        """Observation uses square region (Chebyshev distance).
        range=0.1, agents 0,1 at (0.25,0.25),(0.30,0.30).
        enemy 4 at (0.35,0.33): visible to both (dx<=0.1,dy<=0.1).
        enemy 5 at (0.42,0.38): dx=0.12>0.1 for agent 1, dx=0.17>0.1 for agent 0 → invisible.
        With Euclidean: enemy 4 dist≈0.128>0.1 to agent 0 → would be invisible."""
        s, o, g, a = self._make_data()
        s[:, :, 4, 3] = 0.35; s[:, :, 4, 4] = 0.33
        s[:, :, 5, 3] = 0.42; s[:, :, 5, 4] = 0.38
        c = self._make_ctor(n_groups=2, observation_range=0.1)
        r, pm = c.process(o, s, g, a)

        f = r[0, 0]
        self.assertAlmostEqual(float(f[5]), 1.0 / self.K, places=3,
            msg="Only enemy 4 visible (square range)")


if __name__ == "__main__":
    unittest.main()
