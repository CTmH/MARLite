import numpy as np
from typing import Union, List, Optional
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist
from numpy import ndarray
from marlite.algorithm.group_builder.group_builder import GroupBuilder
from marlite.algorithm.graph_builder.graph_util import binary_to_decimal


class MAgentLabelPropagationGroupBuilder(GroupBuilder):
    def __init__(
        self,
        binary_agent_id_dim: list,
        agent_presence_dim: list,
        comm_distance: float,
        distance_metric: str = "cityblock",
        n_workers: int = 0,
        valid_node_list: Union[list, None] = None,
        n_groups: Optional[int] = None,
        channel_first: bool = False,
        dtype=np.int16,
    ):
        super().__init__(dtype=dtype)
        self.binary_agent_id_dim = binary_agent_id_dim
        self.agent_presence_dim = agent_presence_dim
        self.comm_distance = comm_distance
        self.distance_metric = distance_metric
        self.n_workers = n_workers
        self.valid_node_list = valid_node_list
        self.n_groups = n_groups
        self.channel_first = channel_first

    @staticmethod
    def _merge_excess_groups(coords, labels, n_groups):
        """Iteratively merge spatially closest groups until count ≤ n_groups.

        Each merge combines two groups into one, updating the centroid of
        the surviving group to reflect the new union.  Final labels are
        re-mapped to consecutive integers starting from 0.

        Args:
            coords:   (n_alive, 2)  agent positions  (y, x)
            labels:   (n_alive,)    group labels from connected_components
            n_groups: int           maximum allowed groups

        Returns:
            remapped labels  (n_alive,)  dtype int8, consecutive from 0.
        """
        # ── Compute per-group centroids ──────────────────────────────────
        unique_groups = np.unique(labels)
        centroids = {}
        for g in unique_groups:
            mask = (labels == g)
            centroids[g] = coords[mask].mean(axis=0)

        current_labels = labels.copy()
        current_n = len(unique_groups)

        while current_n > n_groups:
            # Find the two groups with closest centroids
            min_dist = float('inf')
            merge_a, merge_b = -1, -1
            group_ids = sorted(centroids.keys())
            for i in range(len(group_ids)):
                for j in range(i + 1, len(group_ids)):
                    gi, gj = group_ids[i], group_ids[j]
                    dist = np.linalg.norm(centroids[gi] - centroids[gj])
                    if dist < min_dist:
                        min_dist = dist
                        merge_a, merge_b = gi, gj

            # Merge group b into group a
            current_labels[current_labels == merge_b] = merge_a
            mask_a = (current_labels == merge_a)
            centroids[merge_a] = coords[mask_a].mean(axis=0)
            del centroids[merge_b]
            current_n -= 1

        # Re-label to consecutive IDs starting from 0
        unique = sorted(np.unique(current_labels))
        remap = {old: new for new, old in enumerate(unique)}
        remapped = np.array([remap[l] for l in current_labels], dtype=np.int8)

        return remapped

    @staticmethod
    def _process_sample(
        sample_state,
        binary_agent_id_dim: list,
        agent_presence_dim: list,
        comm_distance: float,
        distance_metric: str,
        valid_node_list: Union[list, None] = None,
        n_groups: Optional[int] = None,
    ):
        binary_agent_id = sample_state[:, :, binary_agent_id_dim]
        agent_positions = np.apply_along_axis(
            binary_to_decimal, -1, binary_agent_id
        ).astype(np.int64)
        agent_presence = sample_state[:, :, agent_presence_dim]
        agent_presence = agent_presence.astype(np.int64)
        agent_presence = agent_presence.sum(axis=-1)
        agent_positions = (
            agent_positions * agent_presence + agent_presence - np.ones_like(agent_presence)
        )

        valid_elements = {
            val: (i, j)
            for i, row in enumerate(agent_positions)
            for j, val in enumerate(row)
            if val >= 0
        }
        sorted_ids = sorted(valid_elements.keys())
        sorted_ids = np.array(sorted_ids, dtype=np.int64)
        coords = np.array([valid_elements[k] for k in sorted_ids], dtype=np.int64)

        if valid_node_list is None:
            valid_node_list = sorted_ids.tolist()

        n = len(valid_node_list)

        if len(coords) > 0:
            distances = cdist(coords, coords, metric=distance_metric)
            adj_matrix = (distances <= comm_distance).astype(np.int64)
            np.fill_diagonal(adj_matrix, 0)
            n_components, labels = connected_components(
                adj_matrix, directed=False, return_labels=True
            )

            # Merge excess groups when actual count exceeds capacity
            if n_groups is not None and n_components > n_groups:
                labels = MAgentLabelPropagationGroupBuilder._merge_excess_groups(
                    coords, labels, n_groups
                )

            full_labels = np.full(n, -1, dtype=np.int8)
            for i, node_id in enumerate(sorted_ids):
                if node_id in valid_node_list:
                    mapped_idx = valid_node_list.index(node_id)
                    full_labels[mapped_idx] = labels[i]
        else:
            full_labels = np.full(n, -1, dtype=np.int8)

        return full_labels

    def forward(self, states: ndarray) -> ndarray:
        if self.channel_first:
            states = np.transpose(states, (0, 2, 3, 1))
        bs = states.shape[0]

        results = []
        for b in range(bs):
            labels = self._process_sample(
                states[b],
                self.binary_agent_id_dim,
                self.agent_presence_dim,
                self.comm_distance,
                self.distance_metric,
                self.valid_node_list,
                self.n_groups,
            )
            results.append(labels)

        zone_indices = np.stack(results, axis=0)
        return zone_indices.astype(self.dtype)

    def reset(self):
        return self
