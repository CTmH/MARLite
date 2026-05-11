import numpy as np
from typing import Tuple
from concurrent.futures import ProcessPoolExecutor
from multiprocessing.shared_memory import SharedMemory


class MagentVecStateGroupFeaturesConstructor:
    """Group semantic features constructor for MAgent vector-state format.

    Works with states of shape (B, N_all, F). State is filtered to our-team
    agents to match grouping dimensions. Enemy features are computed from
    deduplicated enemies within each group member's observation range.

    Feature vector (N_FEATURES=17):
        0:  Our team agent count density (count / total_our_team_count)
        1:  Our team health mean
        2:  Our team health std
        3:  Our team total health
        4:  Our team spatial dispersion (mean L2 distance, normalized)
        5:  Enemy agent count density (within observation range, deduplicated)
        6:  Enemy health mean (within observation range, deduplicated)
        7:  Enemy health std (within observation range, deduplicated)
        8:  Enemy total health (within observation range, deduplicated)
        9:  Enemy spatial dispersion (within observation range, deduplicated)
        10: Obstacle density (from nearby obstacle channels)
        11: Distance from centroid to nearest enemy (observed, deduplicated)
        12: Distance from centroid to nearest own agent
        13: Border proximity (from normalized coords: 0=at edge, 1=center)
        14: Enemy-to-our agent count ratio (observed, deduplicated)
        15: Our team health spatial distribution (HP-weighted position variance)
        16: Enemy team health spatial distribution (observed, deduplicated)

    Invalid groups (all our-team agents dead or not assigned) are zero-filled.
    """

    N_FEATURES = 17

    def __init__(
        self,
        n_groups: int,
        observation_range: float,
        coord_dims: Tuple[int, int] = (3, 4),
        hp_dim: int = 0,
        team_dim: int = 1,
        my_team: int = 0,
        enemy_team: int = 1,
        action_dim: int = 21,
        n_offsets: int = 12,
        n_workers: int = 0,
    ):
        self.n_groups = n_groups
        self.observation_range = observation_range
        self.coord_dims = coord_dims
        self.hp_dim = hp_dim
        self.team_dim = team_dim
        self.my_team = my_team
        self.enemy_team = enemy_team
        self.action_dim = action_dim
        self.n_offsets = n_offsets
        self.n_workers = n_workers
        self._nearby_start = 5 + action_dim

    def process(
        self,
        observations: np.ndarray,
        states: np.ndarray,
        grouping: np.ndarray,
        alive_mask: np.ndarray,
    ) -> np.ndarray:
        state_last = states[:, -1]
        batch_size = state_last.shape[0]

        if self.n_workers > 1 and batch_size > 1:
            return self._process_parallel(state_last, grouping)

        result = np.zeros(
            (batch_size, self.n_groups, self.N_FEATURES), dtype=np.float16
        )
        padding_mask = np.ones((batch_size, self.n_groups), dtype=bool)
        for b in range(batch_size):
            r, m = self._process_single(state_last[b], grouping[b])
            result[b] = r
            padding_mask[b] = m
        return result, padding_mask

    def _process_parallel(self, state_last, grouping):
        batch_size = state_last.shape[0]
        n_workers = min(batch_size, self.n_workers)

        shm = SharedMemory(create=True, size=state_last.nbytes)
        try:
            shm_array = np.ndarray(
                state_last.shape, dtype=state_last.dtype, buffer=shm.buf
            )
            np.copyto(shm_array, state_last)

            sample_ids = list(range(batch_size))
            chunk_size = (batch_size + n_workers - 1) // n_workers
            chunks = [
                sample_ids[i : i + chunk_size]
                for i in range(0, batch_size, chunk_size)
            ]
            args = [
                (shm.name, state_last.shape, state_last.dtype, chunk, grouping, self)
                for chunk in chunks
            ]

            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                chunk_results = list(executor.map(_process_chunk_worker, args))

            result = np.zeros(
                (batch_size, self.n_groups, self.N_FEATURES), dtype=np.float16
            )
            padding_mask = np.ones((batch_size, self.n_groups), dtype=bool)
            for chunk, chunk_result in zip(chunks, chunk_results):
                for b, (r, m) in zip(chunk, chunk_result):
                    result[b] = r
                    padding_mask[b] = m
            return result, padding_mask
        finally:
            shm.close()
            shm.unlink()

    def _process_single(self, st_all, grp):
        teams = st_all[:, self.team_dim]
        our_mask = teams == self.my_team
        en_mask = teams == self.enemy_team

        our_st = st_all[our_mask]
        en_st = st_all[en_mask]

        K = len(our_st)
        if K == 0:
            return (
                np.zeros((self.n_groups, self.N_FEATURES), dtype=np.float16),
                np.zeros(self.n_groups, dtype=bool),
            )

        our_hps = our_st[:, self.hp_dim]
        our_alive = our_hps > 0

        en_hps = en_st[:, self.hp_dim]
        en_alive = en_hps > 0

        our_coords = our_st[:, self.coord_dims]
        en_coords_all = en_st[:, self.coord_dims]

        result = np.zeros((self.n_groups, self.N_FEATURES), dtype=np.float16)
        padding_mask = np.zeros(self.n_groups, dtype=bool)

        for gid in range(self.n_groups):
            gmask = (grp == gid) & our_alive
            if not gmask.any():
                continue

            padding_mask[gid] = True

            our_in_g = our_coords[gmask]
            our_hps_g = our_hps[gmask]
            our_cy, our_cx = our_in_g.mean(axis=0)

            seen_en = self._collect_observed_enemies(
                our_in_g, en_coords_all, en_alive
            )
            en_coords = en_coords_all[seen_en]
            en_hps_g = en_hps[seen_en]

            result[gid] = self._compute_features(
                our_in_g, our_hps_g, our_cy, our_cx,
                en_coords, en_hps_g, our_st, gmask, K,
            )

        return result, padding_mask

    def _collect_observed_enemies(self, our_coords, en_coords_all, en_alive):
        seen = set()
        for our_pos in our_coords:
            dy = np.abs(en_coords_all[:, 0] - our_pos[0])
            dx = np.abs(en_coords_all[:, 1] - our_pos[1])
            visible = np.where(
                (dy <= self.observation_range) & (dx <= self.observation_range) & en_alive
            )[0]
            seen.update(visible.tolist())
        return np.array(sorted(seen), dtype=int)

    def _compute_features(
        self, our_coords, our_hps, our_cy, our_cx,
        en_coords, en_hps, our_st, gmask, K,
    ):
        our_count = len(our_coords)
        en_count = len(en_coords)

        feat_our_density = our_count / K
        feat_our_hp_mean = our_hps.mean() if our_count > 0 else 0.0
        feat_our_hp_std = our_hps.std() if our_count > 0 else 0.0
        feat_our_total_hp = our_hps.sum() if our_count > 0 else 0.0

        if our_count > 1:
            dists = np.sqrt(((our_coords - [our_cy, our_cx]) ** 2).sum(axis=1))
            feat_our_dispersion = dists.mean() / K
        else:
            feat_our_dispersion = 1.0 if our_count == 0 else 0.0

        feat_en_density = en_count / K
        feat_en_hp_mean = en_hps.mean() if en_count > 0 else 0.0
        feat_en_hp_std = en_hps.std() if en_count > 0 else 0.0
        feat_en_total_hp = en_hps.sum() if en_count > 0 else 0.0

        if en_count > 1:
            en_cy, en_cx = en_coords.mean(axis=0)
            dists = np.sqrt(((en_coords - [en_cy, en_cx]) ** 2).sum(axis=1))
            feat_en_dispersion = dists.mean() / K
        else:
            feat_en_dispersion = 1.0 if en_count == 0 else 0.0

        # Obstacle density
        if our_count > 0:
            obstacle_hits = 0
            total_probes = 0
            our_indices = np.where(gmask)[0]
            for idx in our_indices:
                for i in range(self.n_offsets):
                    bias = self._nearby_start + i * 3 + 2
                    obstacle_hits += our_st[idx, bias]
                    total_probes += 1
            feat_obs_density = (
                obstacle_hits / total_probes if total_probes > 0 else 0.0
            )
        else:
            feat_obs_density = 0.0

        # Nearest enemy distance
        if our_count > 0 and en_count > 0:
            en_dists = np.sqrt(((en_coords - [our_cy, our_cx]) ** 2).sum(axis=1))
            feat_nearest_en = en_dists.min() / K
        else:
            feat_nearest_en = 1.0

        # Nearest our agent distance
        if our_count > 1:
            group_dists = np.sqrt(
                ((our_coords - [our_cy, our_cx]) ** 2).sum(axis=1)
            )
            feat_nearest_our = group_dists.min() / K
        else:
            feat_nearest_our = 1.0

        # Border proximity
        if our_count > 0:
            feat_border_prox = min(our_cy, 1.0 - our_cy, our_cx, 1.0 - our_cx)
        else:
            feat_border_prox = 0.5

        # Ratio
        ratio = en_count / our_count if our_count > 0 else 0.0
        feat_ratio = min(ratio, 5.0) / 5.0

        # Our HP-weighted spatial
        if our_count > 1 and our_hps.sum() > 0:
            hp_sum = our_hps.sum()
            w_cy = (our_hps * our_coords[:, 0]).sum() / hp_sum
            w_cx = (our_hps * our_coords[:, 1]).sum() / hp_sum
            wy_var = (our_hps * (our_coords[:, 0] - w_cy) ** 2).sum() / hp_sum
            wx_var = (our_hps * (our_coords[:, 1] - w_cx) ** 2).sum() / hp_sum
            feat_our_hp_spatial = np.sqrt(wy_var + wx_var) / K
        else:
            feat_our_hp_spatial = 1.0

        # Enemy HP-weighted spatial
        if en_count > 1 and en_hps.sum() > 0:
            hp_sum = en_hps.sum()
            w_cy = (en_hps * en_coords[:, 0]).sum() / hp_sum
            w_cx = (en_hps * en_coords[:, 1]).sum() / hp_sum
            wy_var = (en_hps * (en_coords[:, 0] - w_cy) ** 2).sum() / hp_sum
            wx_var = (en_hps * (en_coords[:, 1] - w_cx) ** 2).sum() / hp_sum
            feat_en_hp_spatial = np.sqrt(wy_var + wx_var) / K
        else:
            feat_en_hp_spatial = 1.0

        return np.array(
            [
                feat_our_density,
                feat_our_hp_mean,
                feat_our_hp_std,
                feat_our_total_hp,
                feat_our_dispersion,
                feat_en_density,
                feat_en_hp_mean,
                feat_en_hp_std,
                feat_en_total_hp,
                feat_en_dispersion,
                feat_obs_density,
                feat_nearest_en,
                feat_nearest_our,
                feat_border_prox,
                feat_ratio,
                feat_our_hp_spatial,
                feat_en_hp_spatial,
            ],
            dtype=np.float16,
        )


def _process_chunk_worker(args):
    shm_name, shape, dtype, chunk_ids, grouping, constructor = args

    shm = SharedMemory(name=shm_name)
    try:
        state_last = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        result = np.zeros(
            (len(chunk_ids), constructor.n_groups, constructor.N_FEATURES),
            dtype=np.float16,
        )
        padding_mask = np.ones((len(chunk_ids), constructor.n_groups), dtype=bool)
        for i, b in enumerate(chunk_ids):
            r, m = constructor._process_single(state_last[b], grouping[b])
            result[i] = r
            padding_mask[i] = m
        return result, padding_mask
    finally:
        shm.close()
