import numpy as np
from typing import Optional, List


class MagentGroupWindowConstructor:
    """A2: Group-centric fixed window crop from global state (MAgent).

    For each group identified by the GroupBuilder, crops a K×K window from the
    global state centered on the group's spatial centroid. This provides the
    "ground truth" of what the group region looks like, serving as a
    reconstruction target for the group consensus vector.

    State channels in MAgent (reference from MAgentWrapper):
        0:  OBSTACLE
        1:  TEAM_0_PRESENCE
        2:  TEAM_0_HP
        3:  TEAM_1_PRESENCE
        4:  TEAM_1_HP
        5-14:  BINARY_AGENT_ID
        15-35: ONE_HOT_ACTION (or 15-27 for pursuit)
        36:    LAST_REWARD (or 28 for pursuit)
    """

    def __init__(
        self,
        n_groups: int,
        window_size: int,
        selected_channels: List[int],
        binary_agent_id_dim: List[int],
        agent_presence_dim: List[int],
        channel_first: bool = False,
        n_workers: int = 0,
    ):
        self.n_groups = n_groups
        self.window_size = window_size
        self.selected_channels = selected_channels
        self.binary_agent_id_dim = binary_agent_id_dim
        self.agent_presence_dim = agent_presence_dim
        self.channel_first = channel_first
        self.n_workers = n_workers

    def process(
        self,
        observations: np.ndarray,
        states: np.ndarray,
        grouping: np.ndarray,
        alive_mask: np.ndarray,
    ) -> np.ndarray:
        """Process states to produce group-centric window crops.

        Args:
            observations: (B, T, N, C, H, W) or (B, T, N, H, W, C), depending on channel_first.
                Not directly used; accepted for interface consistency.
            states: (B, T, C, H, W) if channel_first else (B, T, H, W, C).
            grouping: (B, N) zone indices from GroupBuilder.  Values are group IDs
                (0, 1, ...) or -1 for dead agents.
            alive_mask: (B, T, N) boolean mask of alive agents.

        Returns:
            If channel_first: (B, n_groups, selected_channels, K, K)
            Else: (B, n_groups, K, K, selected_channels)
        """
        if self.channel_first:
            # states: (B, T, C, H, W)
            state_last = states[:, -1]  # (B, C, H, W)
        else:
            # states: (B, T, H, W, C)
            state_last = states[:, -1]  # (B, H, W, C)

        batch_size = state_last.shape[0]
        alive_last = alive_mask[:, -1]  # (B, N)
        K = self.window_size

        if self.channel_first:
            out_shape = (batch_size, self.n_groups, len(self.selected_channels), K, K)
        else:
            out_shape = (batch_size, self.n_groups, K, K, len(self.selected_channels))
        result = np.zeros(out_shape, dtype=np.float16)
        padding_mask = np.zeros((batch_size, self.n_groups), dtype=bool)

        for b in range(batch_size):
            st = state_last[b]
            grp = grouping[b]        # (N,)
            alv = alive_last[b]      # (N,)

            if self.channel_first:
                # st: (C, H, W)
                H, W = st.shape[1], st.shape[2]
            else:
                # st: (H, W, C)
                H, W = st.shape[0], st.shape[1]

            # Collect agent positions: map agent idx -> (y, x)
            agent_positions = self._extract_positions(st)

            # Build per-group position list (only alive agents that have positions)
            group_positions = {}
            for agent_idx in range(len(grp)):
                gid = int(grp[agent_idx])
                if gid < 0 or not alv[agent_idx]:
                    continue
                if agent_idx in agent_positions:
                    if gid not in group_positions:
                        group_positions[gid] = []
                    group_positions[gid].append(agent_positions[agent_idx])

            # Process each group
            processed = set()
            for gid in sorted(group_positions.keys()):
                if gid >= self.n_groups or gid < 0:
                    continue
                positions = group_positions[gid]
                if len(positions) == 0:
                    continue

                # Centroid
                ys = [p[0] for p in positions]
                xs = [p[1] for p in positions]
                cy = int(np.round(np.mean(ys)))
                cx = int(np.round(np.mean(xs)))

                # Crop window
                y_start = cy - K // 2
                x_start = cx - K // 2
                y_end = y_start + K
                x_end = x_start + K

                # Pad if out of bounds
                pad_top = max(0, -y_start)
                pad_left = max(0, -x_start)
                pad_bottom = max(0, y_end - H)
                pad_right = max(0, x_end - W)

                y_start_clamped = max(0, y_start)
                x_start_clamped = max(0, x_start)
                y_end_clamped = min(H, y_end)
                x_end_clamped = min(W, x_end)

                if self.channel_first:
                    # st: (C, H, W)
                    crop = st[:, y_start_clamped:y_end_clamped, x_start_clamped:x_end_clamped]
                    crop = crop[self.selected_channels]  # (sel_C, h_crop, w_crop)
                    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
                        padded = np.zeros((len(self.selected_channels), K, K), dtype=st.dtype)
                        h_crop = y_end_clamped - y_start_clamped
                        w_crop = x_end_clamped - x_start_clamped
                        padded[:, pad_top:pad_top + h_crop, pad_left:pad_left + w_crop] = crop
                        crop = padded
                    # Ensure final size
                    if crop.shape[1] != K or crop.shape[2] != K:
                        final_crop = np.zeros((len(self.selected_channels), K, K), dtype=st.dtype)
                        h = min(K, crop.shape[1])
                        w = min(K, crop.shape[2])
                        final_crop[:, :h, :w] = crop[:, :h, :w]
                        crop = final_crop
                else:
                    # st: (H, W, C)
                    crop = st[y_start_clamped:y_end_clamped, x_start_clamped:x_end_clamped]
                    crop = crop[..., self.selected_channels]  # (h_crop, w_crop, sel_C)
                    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
                        padded = np.zeros((K, K, len(self.selected_channels)), dtype=st.dtype)
                        h_crop = y_end_clamped - y_start_clamped
                        w_crop = x_end_clamped - x_start_clamped
                        padded[pad_top:pad_top + h_crop, pad_left:pad_left + w_crop] = crop
                        crop = padded
                    if crop.shape[0] != K or crop.shape[1] != K:
                        final_crop = np.zeros((K, K, len(self.selected_channels)), dtype=st.dtype)
                        h = min(K, crop.shape[0])
                        w = min(K, crop.shape[1])
                        final_crop[:h, :w] = crop[:h, :w]
                        crop = final_crop

                result[b, gid] = crop
                padding_mask[b, gid] = True
                processed.add(gid)

            # Unoccupied group slots remain zero-padded (already zeros)

        return result, padding_mask

    def _extract_positions(self, state_single: np.ndarray) -> dict:
        """Extract agent positions from a single state grid.

        Args:
            state_single: (C, H, W) if channel_first else (H, W, C).

        Returns:
            dict mapping agent_idx -> (y, x) position.
        """
        if self.channel_first:
            C, H, W_ = state_single.shape
            state_hwc = np.transpose(state_single, (1, 2, 0))  # (H, W, C)
        else:
            H, W_, C = state_single.shape
            state_hwc = state_single

        # Agent presence: any channel in agent_presence_dim > 0
        presence = np.any(state_hwc[:, :, self.agent_presence_dim] > 0, axis=-1)

        # Get coordinates of all present agents
        ys, xs = np.where(presence)

        # Decode binary agent IDs
        binary_ids = state_hwc[:, :, self.binary_agent_id_dim]  # (H, W, n_bits)
        binary_ids = binary_ids[ys, xs]  # (n_agents, n_bits)
        powers = 2 ** np.arange(len(self.binary_agent_id_dim), dtype=binary_ids.dtype)
        agent_ids = np.dot(binary_ids, powers).astype(int)

        return {int(aid.item()): (int(y.item()), int(x.item()))
                for aid, y, x in zip(agent_ids, ys, xs)}
