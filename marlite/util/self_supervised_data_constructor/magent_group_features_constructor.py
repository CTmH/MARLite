import numpy as np
from typing import Optional, List


class MagentGroupFeaturesConstructor:
    """D: Group semantic features constructor for MAgent environment.

    Extracts a fixed-size feature vector for each group based on a window
    crop centered on the group's spatial centroid. Features describe the
    semantic composition of the group's region: agent presence, health,
    obstacles, spatial dispersion, etc.

    State channels in MAgent (reference from MAgentWrapper):
        0:  OBSTACLE
        1:  TEAM_0_PRESENCE
        2:  TEAM_0_HP
        3:  TEAM_1_PRESENCE
        4:  TEAM_1_HP
        5-14:  BINARY_AGENT_ID
        15-35: ONE_HOT_ACTION (or 15-27 for pursuit)
        36:    LAST_REWARD (or 28 for pursuit)

    Feature vector (n_features dimension):
        0:  Our team agent count density (count / window_area)
        1:  Our team health mean
        2:  Our team health std
        3:  Our team total health
        4:  Our team spatial dispersion (mean L2 distance to centroid, normalized by window_size)
        5:  Enemy agent count density
        6:  Enemy health mean
        7:  Enemy health std
        8:  Enemy total health
        9:  Enemy spatial dispersion
        10: Obstacle density
        11: Group alive agent count (normalized by max possible)
        12: Distance from centroid to nearest enemy agent (normalized by window_size)
        13: Distance from centroid to nearest our agent (normalized by window_size)
        14: Border proximity (min distance from centroid to map edge, normalized)
        15: Enemy-to-our agent count ratio (clamped to [0, 5])
        16: Our team health spatial distribution (HP-weighted position variance)
        17: Enemy team health spatial distribution
    """

    N_FEATURES = 18

    def __init__(
        self,
        n_groups: int,
        window_size: int,
        binary_agent_id_dim: List[int],
        agent_presence_dim: List[int],
        our_team_presence_dim: int,
        our_team_hp_dim: int,
        enemy_team_presence_dim: int,
        enemy_team_hp_dim: int,
        obstacle_dim: int,
        channel_first: bool = False,
    ):
        self.n_groups = n_groups
        self.window_size = window_size
        self.binary_agent_id_dim = binary_agent_id_dim
        self.agent_presence_dim = agent_presence_dim
        self.our_team_presence_dim = our_team_presence_dim
        self.our_team_hp_dim = our_team_hp_dim
        self.enemy_team_presence_dim = enemy_team_presence_dim
        self.enemy_team_hp_dim = enemy_team_hp_dim
        self.obstacle_dim = obstacle_dim
        self.channel_first = channel_first

    def process(
        self,
        observations: np.ndarray,
        states: np.ndarray,
        grouping: np.ndarray,
        alive_mask: np.ndarray,
    ) -> np.ndarray:
        """Process states to produce group semantic feature vectors.

        Args:
            observations: (B, T, N, C, H, W) or (B, T, N, H, W, C). Not used directly.
            states: (B, T, C, H, W) if channel_first else (B, T, H, W, C).
            grouping: (B, N) zone indices from GroupBuilder. Values are group IDs
                or -1 for dead agents.
            alive_mask: (B, T, N) boolean mask of alive agents.

        Returns:
            (B, n_groups, N_FEATURES) float feature array.
        """
        if self.channel_first:
            state_last = states[:, -1]  # (B, C, H, W)
        else:
            state_last = states[:, -1]  # (B, H, W, C)

        batch_size = state_last.shape[0]
        alive_last = alive_mask[:, -1]
        K = self.window_size
        K_sq = K * K

        result = np.zeros((batch_size, self.n_groups, self.N_FEATURES), dtype=np.float16)

        for b in range(batch_size):
            st = state_last[b]
            grp = grouping[b]
            alv = alive_last[b]

            if self.channel_first:
                C, H_map, W_map = st.shape
            else:
                H_map, W_map, C = st.shape

            # Extract agent positions from state
            agent_positions = self._extract_positions(st)

            # Collect per-group positions (only alive agents with known positions)
            group_positions = {}
            for agent_idx in range(len(grp)):
                gid = int(grp[agent_idx])
                if gid < 0 or not alv[agent_idx]:
                    continue
                if agent_idx in agent_positions:
                    if gid not in group_positions:
                        group_positions[gid] = []
                    group_positions[gid].append(agent_positions[agent_idx])

            for gid, positions in group_positions.items():
                if gid >= self.n_groups or gid < 0 or len(positions) == 0:
                    continue

                # Group centroid
                ys = np.array([p[0] for p in positions], dtype=np.float64)
                xs = np.array([p[1] for p in positions], dtype=np.float64)
                cy = np.mean(ys)
                cx = np.mean(xs)

                # Crop window
                y_start = int(np.round(cy - K // 2))
                x_start = int(np.round(cx - K // 2))
                y_end = y_start + K
                x_end = x_start + K

                # Get window data from state
                window = self._crop_window(st, y_start, x_start, y_end, x_end,
                                            H_map, W_map, self.channel_first)

                # Extract per-channel data from window
                features = self._compute_features(
                    window, ys, xs, cy, cx, H_map, W_map, K, self.channel_first
                )
                result[b, gid] = features

        return result

    def _crop_window(self, st, y_start, x_start, y_end, x_end, H, W, channel_first):
        """Crop a K×K window from state with zero-padding for out-of-bounds."""
        pad_top = max(0, -y_start)
        pad_left = max(0, -x_start)
        pad_bottom = max(0, y_end - H)
        pad_right = max(0, x_end - W)

        y_s = max(0, y_start)
        x_s = max(0, x_start)
        y_e = min(H, y_end)
        x_e = min(W, x_end)

        if channel_first:
            C = st.shape[0]
            crop = st[:, y_s:y_e, x_s:x_e]
            if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
                h_c = y_e - y_s
                w_c = x_e - x_s
                padded = np.zeros((C, h_c + pad_top + pad_bottom, w_c + pad_left + pad_right),
                                  dtype=st.dtype)
                padded[:, pad_top:pad_top + h_c, pad_left:pad_left + w_c] = crop
                return padded
            return crop
        else:
            C = st.shape[2]
            crop = st[y_s:y_e, x_s:x_e]
            if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
                h_c = y_e - y_s
                w_c = x_e - x_s
                padded = np.zeros((h_c + pad_top + pad_bottom, w_c + pad_left + pad_right, C),
                                  dtype=st.dtype)
                padded[pad_top:pad_top + h_c, pad_left:pad_left + w_c] = crop
                return padded
            return crop

    def _compute_features(self, window, our_ys, our_xs, cy, cx, H_map, W_map, K, channel_first):
        """Compute all semantic features from the cropped window.

        Args:
            window: (C, K, K) or (K, K, C) cropped state region.
            our_ys, our_xs: arrays of our agents' global positions.
            cy, cx: group centroid in global coordinates.
            H_map, W_map: full state grid dimensions.
            K: window size.
            channel_first: if True, window shape is (C, K, K).

        Returns:
            np.ndarray of shape (N_FEATURES,).
        """
        if channel_first:
            # window: (C, K, K)
            our_pres = window[self.our_team_presence_dim]       # (K, K)
            our_hp   = window[self.our_team_hp_dim]              # (K, K)
            en_pres  = window[self.enemy_team_presence_dim]      # (K, K)
            en_hp    = window[self.enemy_team_hp_dim]            # (K, K)
            obstacle = window[self.obstacle_dim]                 # (K, K)
        else:
            # window: (K, K, C)
            our_pres = window[:, :, self.our_team_presence_dim]
            our_hp   = window[:, :, self.our_team_hp_dim]
            en_pres  = window[:, :, self.enemy_team_presence_dim]
            en_hp    = window[:, :, self.enemy_team_hp_dim]
            obstacle = window[:, :, self.obstacle_dim]

        # Masks
        our_mask = our_pres > 0
        en_mask = en_pres > 0
        obs_mask = obstacle > 0

        our_count = our_mask.sum()
        en_count = en_mask.sum()
        obs_count = obs_mask.sum()
        K_sq = K * K

        # Our team features
        feat_our_density = our_count / K_sq
        feat_our_hp_mean = our_hp[our_mask].mean() if our_count > 0 else 0.0
        feat_our_hp_std = our_hp[our_mask].std() if our_count > 0 else 0.0
        feat_our_total_hp = our_hp[our_mask].sum() if our_count > 0 else 0.0

        # Our team spatial dispersion (in window coordinates)
        if our_count > 0:
            our_rows, our_cols = np.where(our_mask)
            our_wy = our_rows.astype(np.float64)
            our_wx = our_cols.astype(np.float64)
            ow_cy = our_wy.mean()
            ow_cx = our_wx.mean()
            mean_dist = np.sqrt((our_wy - ow_cy) ** 2 + (our_wx - ow_cx) ** 2).mean()
            feat_our_dispersion = mean_dist / K
        else:
            feat_our_dispersion = 1.0  # max dispersion when no agents

        # Enemy features
        feat_en_density = en_count / K_sq
        feat_en_hp_mean = en_hp[en_mask].mean() if en_count > 0 else 0.0
        feat_en_hp_std = en_hp[en_mask].std() if en_count > 0 else 0.0
        feat_en_total_hp = en_hp[en_mask].sum() if en_count > 0 else 0.0

        # Enemy spatial dispersion
        if en_count > 0:
            en_rows, en_cols = np.where(en_mask)
            en_wy = en_rows.astype(np.float64)
            en_wx = en_cols.astype(np.float64)
            enw_cy = en_wy.mean()
            enw_cx = en_wx.mean()
            mean_dist = np.sqrt((en_wy - enw_cy) ** 2 + (en_wx - enw_cx) ** 2).mean()
            feat_en_dispersion = mean_dist / K
        else:
            feat_en_dispersion = 1.0

        # Obstacle density
        feat_obs_density = obs_count / K_sq

        # Group size
        feat_group_size = len(our_ys) / (K_sq)  # normalized by window area

        # Nearest enemy distance from centroid
        if en_count > 0:
            en_rows, en_cols = np.where(en_mask)
            en_global_y = en_rows.astype(np.float64)  # these are window-local coords
            en_global_x = en_cols.astype(np.float64)
            # Window top-left in global coords:
            y_start = int(np.round(cy - K // 2))
            x_start = int(np.round(cx - K // 2))
            en_gy = en_global_y + y_start
            en_gx = en_global_x + x_start
            dists = np.sqrt((en_gy - cy) ** 2 + (en_gx - cx) ** 2)
            feat_nearest_en = dists.min() / K
        else:
            feat_nearest_en = 1.0  # far

        # Nearest our agent distance from centroid
        if len(our_ys) > 0:
            dists = np.sqrt((our_ys - cy) ** 2 + (our_xs - cx) ** 2)
            feat_nearest_our = dists.min() / K
        else:
            feat_nearest_our = 1.0

        # Border proximity
        dist_top = cy
        dist_bottom = H_map - 1 - cy
        dist_left = cx
        dist_right = W_map - 1 - cx
        border_dist = min(dist_top, dist_bottom, dist_left, dist_right)
        feat_border_prox = border_dist / max(H_map, W_map)

        # Enemy-to-our ratio
        if our_count > 0:
            ratio = en_count / our_count
        else:
            ratio = 5.0 if en_count > 0 else 0.0
        feat_ratio = min(ratio, 5.0) / 5.0

        # Our team health-weighted position variance
        if our_count > 0:
            hp_weights = our_hp[our_mask]
            hp_sum = hp_weights.sum()
            if hp_sum > 0:
                wy = our_rows.astype(np.float64)
                wx = our_cols.astype(np.float64)
                w_cy = (hp_weights * wy).sum() / hp_sum
                w_cx = (hp_weights * wx).sum() / hp_sum
                wy_var = (hp_weights * (wy - w_cy) ** 2).sum() / hp_sum
                wx_var = (hp_weights * (wx - w_cx) ** 2).sum() / hp_sum
                feat_our_hp_spatial = np.sqrt(wy_var + wx_var) / K
            else:
                feat_our_hp_spatial = 1.0
        else:
            feat_our_hp_spatial = 1.0

        # Enemy team health-weighted position variance
        if en_count > 0:
            hp_weights = en_hp[en_mask]
            hp_sum = hp_weights.sum()
            if hp_sum > 0:
                en_rows2, en_cols2 = np.where(en_mask)
                wy = en_rows2.astype(np.float64)
                wx = en_cols2.astype(np.float64)
                w_cy = (hp_weights * wy).sum() / hp_sum
                w_cx = (hp_weights * wx).sum() / hp_sum
                wy_var = (hp_weights * (wy - w_cy) ** 2).sum() / hp_sum
                wx_var = (hp_weights * (wx - w_cx) ** 2).sum() / hp_sum
                feat_en_hp_spatial = np.sqrt(wy_var + wx_var) / K
            else:
                feat_en_hp_spatial = 1.0
        else:
            feat_en_hp_spatial = 1.0

        return np.array([
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
            feat_group_size,
            feat_nearest_en,
            feat_nearest_our,
            feat_border_prox,
            feat_ratio,
            feat_our_hp_spatial,
            feat_en_hp_spatial,
        ], dtype=np.float16)

    def _extract_positions(self, state_single: np.ndarray) -> dict:
        """Extract agent positions from a single state grid.

        Args:
            state_single: (C, H, W) if channel_first else (H, W, C).

        Returns:
            dict mapping agent_idx -> (y, x).
        """
        if self.channel_first:
            C, H, W_ = state_single.shape
            state_hwc = np.transpose(state_single, (1, 2, 0))
        else:
            H, W_, C = state_single.shape
            state_hwc = state_single

        presence = np.any(state_hwc[:, :, self.agent_presence_dim] > 0, axis=-1)
        ys, xs = np.where(presence)
        if len(ys) == 0:
            return {}

        # Decode binary agent IDs
        binary_ids = state_hwc[:, :, self.binary_agent_id_dim]  # (H, W, n_bits)
        binary_ids = binary_ids[ys, xs]  # (n_agents, n_bits)
        powers = 2 ** np.arange(len(self.binary_agent_id_dim), dtype=binary_ids.dtype)
        agent_ids = np.dot(binary_ids, powers).astype(int)
        return {int(aid.item()): (int(y.item()), int(x.item()))
                for aid, y, x in zip(agent_ids, ys, xs)}
