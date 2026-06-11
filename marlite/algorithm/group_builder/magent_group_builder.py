import numpy as np
import torch
from typing import Union, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist
from scipy.cluster.vq import kmeans2
from copy import deepcopy
from numpy import ndarray
from marlite.algorithm.group_builder.group_builder import GroupBuilder
from marlite.algorithm.graph_builder.graph_util import binary_to_decimal


class MAgentLabelPropagationGroupBuilder(GroupBuilder):
    def __init__(
        self,
        binary_agent_id_dim: List[int],
        agent_presence_dim: List[int],
        comm_distance: float,
        distance_metric: str = "cityblock",
        n_workers: int = 0,
        valid_node_list: Union[List[int], None] = None,
        n_groups: Optional[int] = None,
        update_interval: int = 1,
        channel_first: bool = False,
        dtype: str = 'int16',
    ):
        super().__init__(dtype=dtype)
        self.binary_agent_id_dim = binary_agent_id_dim
        self.agent_presence_dim = agent_presence_dim
        self.comm_distance = comm_distance
        self.distance_metric = distance_metric
        self.n_workers = n_workers
        self.valid_node_list = valid_node_list
        self.n_groups = n_groups
        self.update_interval = update_interval
        self.channel_first = channel_first
        self.step_counter = 0
        self.cached_labels = None

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
        remapped = np.array([remap[l] for l in current_labels], dtype=np.int64)

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

            full_labels = np.full(n, -1, dtype=np.int64)
            for i, node_id in enumerate(sorted_ids):
                if node_id in valid_node_list:
                    mapped_idx = valid_node_list.index(node_id)
                    full_labels[mapped_idx] = labels[i]
        else:
            full_labels = np.full(n, -1, dtype=np.int64)

        return full_labels

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        states = states.cpu().numpy()
        if self.channel_first:
            states = np.transpose(states, (0, 2, 3, 1))
        bs = states.shape[0]

        if not self.training:
            self.step_counter += 1
            if (
                self.step_counter % self.update_interval != 0
                and self.cached_labels is not None
            ):
                return torch.from_numpy(deepcopy(self.cached_labels).astype(self.dtype))

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

        group_indices = np.stack(results, axis=0)

        if not self.training:
            self.cached_labels = group_indices

        return torch.from_numpy(group_indices.astype(self.dtype))

    def reset(self):
        self.step_counter = 0
        self.cached_labels = None
        return self


class MAgentVecLPGroupBuilder(GroupBuilder):
    def __init__(
        self,
        coord_dims: Tuple[int, int],
        hp_dim: int,
        team_dim: int,
        selected_teams: List[int],
        comm_distance: float,
        distance_metric: str = "euclidean",
        n_groups: Optional[int] = None,
        update_interval: int = 1,
        n_workers: int = 0,
    ):
        super().__init__()
        self.coord_dims = coord_dims
        self.hp_dim = hp_dim
        self.team_dim = team_dim
        self.selected_teams = selected_teams
        self.comm_distance = comm_distance
        self.distance_metric = distance_metric
        self.n_groups = n_groups
        self.update_interval = update_interval
        self.n_workers = n_workers
        self.step_counter = 0
        self.cached_labels = None

    @staticmethod
    def _process_sample(
        sample_state: ndarray,
        coord_dims: Tuple[int, int],
        hp_dim: int,
        team_dim: int,
        selected_teams: List[int],
        comm_distance: float,
        distance_metric: str,
        n_groups: Optional[int] = None,
    ) -> ndarray:
        coords = sample_state[:, coord_dims]
        hps = sample_state[:, hp_dim]
        teams = sample_state[:, team_dim]

        candidate_mask = np.isin(teams, selected_teams)
        n_candidates = candidate_mask.sum()

        candidate_coords = coords[candidate_mask]
        candidate_hps = hps[candidate_mask]

        valid_mask = candidate_hps > 0
        valid_local_ids = np.where(valid_mask)[0]

        if len(valid_local_ids) == 0:
            return np.full(n_candidates, -1, dtype=np.int64)

        valid_coords = candidate_coords[valid_mask]
        distances = cdist(valid_coords, valid_coords, metric=distance_metric)
        adj_matrix = (distances <= comm_distance).astype(np.int64)
        np.fill_diagonal(adj_matrix, 0)

        n_components, comp_labels = connected_components(
            adj_matrix, directed=False, return_labels=True
        )

        if n_groups is not None and n_components > n_groups:
            comp_labels = MAgentLabelPropagationGroupBuilder._merge_excess_groups(
                valid_coords, comp_labels, n_groups
            )

        full_labels = np.full(n_candidates, -1, dtype=np.int64)
        full_labels[valid_local_ids] = comp_labels
        return full_labels

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        states = states.cpu().numpy()
        bs = states.shape[0]

        if not self.training:
            self.step_counter += 1
            if (
                self.step_counter % self.update_interval != 0
                and self.cached_labels is not None
            ):
                # ── Return cached labels with dead-agent correction ──────
                # state:  (bs, N_all, F)   includes BOTH ally + enemy teams
                # result: (bs, n_cand)      only selected-team agents
                # Mask must align selected-team subset to avoid cross-team indexing
                # NOTE: bool indexing always returns 1D because numpy cannot
                # guarantee equal True counts per row at compile time.
                # Here every row has exactly n_cand True entries, so the
                # reshape(bs, -1) → (bs, n_cand) is safe and restores shape.
                result = deepcopy(self.cached_labels)                   # (bs, n_cand)
                teams = states[:, :, self.team_dim]                     # (bs, N_all)
                hps   = states[:, :, self.hp_dim]                       # (bs, N_all)
                cand_mask = np.isin(teams, self.selected_teams)         # (bs, N_all) bool
                cand_hps  = hps[cand_mask].reshape(bs, -1)              # hps[cand_mask]: flat (bs*n_cand,) → reshape → (bs, n_cand)
                result[cand_hps <= 0] = -1                              # dead → -1, alive unchanged
                return torch.from_numpy(result)
            self.cached_labels = None

        n_workers = min(bs, self.n_workers)

        if n_workers > 1:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                results = list(
                    executor.map(
                        self._process_sample,
                        [states[b] for b in range(bs)],
                        [self.coord_dims] * bs,
                        [self.hp_dim] * bs,
                        [self.team_dim] * bs,
                        [self.selected_teams] * bs,
                        [self.comm_distance] * bs,
                        [self.distance_metric] * bs,
                        [self.n_groups] * bs,
                    )
                )
        else:
            results = [
                self._process_sample(
                    states[b],
                    self.coord_dims,
                    self.hp_dim,
                    self.team_dim,
                    self.selected_teams,
                    self.comm_distance,
                    self.distance_metric,
                    self.n_groups,
                )
                for b in range(bs)
            ]

        labels = np.stack(results, axis=0)

        if not self.training:
            self.cached_labels = labels

        return torch.from_numpy(labels.astype(self.dtype))

    def reset(self):
        self.step_counter = 0
        self.cached_labels = None
        return self


class MagentKMeansGroupBuilder(GroupBuilder):
    def __init__(
        self,
        binary_agent_id_dim: List[int],
        agent_presence_dim: List[int],
        n_groups: int = 4,
        min_group_size: int = 1,
        n_workers: int = 0,
        valid_node_list: Union[List[int], None] = None,
        update_interval: int = 1,
        channel_first: bool = False,
        dtype: str = 'int16',
    ):
        super().__init__(dtype=dtype)
        self.binary_agent_id_dim = binary_agent_id_dim
        self.agent_presence_dim = agent_presence_dim
        self.n_groups = n_groups
        self.min_group_size = min_group_size
        self.n_workers = n_workers
        self.valid_node_list = valid_node_list
        self.update_interval = update_interval
        self.channel_first = channel_first
        self.step_counter = 0
        self.cached_labels = None

    @staticmethod
    def _compute_n_clusters(n_alive, n_groups, min_group_size):
        if n_alive == 0:
            return 0
        max_clusters = max(1, n_alive // min_group_size)
        return min(n_groups, max_clusters)

    @staticmethod
    def _kmeans_cluster(coords, n_clusters, seed=None):
        if coords.shape[0] <= n_clusters:
            return np.arange(coords.shape[0], dtype=np.int64) % n_clusters
        rng = np.random.default_rng(seed)
        idx = rng.choice(coords.shape[0], n_clusters, replace=False)
        init = coords[idx].astype(np.float64)
        _, labels = kmeans2(
            coords.astype(np.float64), init, iter=100, minit="matrix"
        )
        return labels.astype(np.int64)

    @staticmethod
    def _process_sample(
        sample_state,
        binary_agent_id_dim,
        agent_presence_dim,
        valid_node_list,
        n_groups,
        min_group_size,
        seed=None,
    ):
        binary_agent_id = sample_state[:, :, binary_agent_id_dim]
        agent_positions = np.apply_along_axis(
            binary_to_decimal, -1, binary_agent_id
        ).astype(np.int64)
        agent_presence = sample_state[:, :, agent_presence_dim]
        agent_presence = agent_presence.astype(np.int64)
        agent_presence = agent_presence.sum(axis=-1)
        agent_positions = (
            agent_positions * agent_presence
            + agent_presence
            - np.ones_like(agent_presence)
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
        n_alive = len(coords)

        if n_alive == 0:
            return np.full(n, -1, dtype=np.int64)

        n_clusters = MagentKMeansGroupBuilder._compute_n_clusters(
            n_alive, n_groups, min_group_size
        )

        cluster_labels = MagentKMeansGroupBuilder._kmeans_cluster(
            coords, n_clusters, seed
        )

        full_labels = np.full(n, -1, dtype=np.int64)
        for i, node_id in enumerate(sorted_ids):
            if node_id in valid_node_list:
                mapped_idx = valid_node_list.index(node_id)
                full_labels[mapped_idx] = cluster_labels[i]

        return full_labels

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        states = states.cpu().numpy()
        if self.channel_first:
            states = np.transpose(states, (0, 2, 3, 1))
        bs = states.shape[0]

        if not self.training:
            self.step_counter += 1
            if (
                self.step_counter % self.update_interval != 0
                and self.cached_labels is not None
            ):
                return torch.from_numpy(deepcopy(self.cached_labels).astype(self.dtype))

        rng = np.random.default_rng()

        results = []
        for b in range(bs):
            labels = self._process_sample(
                states[b],
                self.binary_agent_id_dim,
                self.agent_presence_dim,
                self.valid_node_list,
                self.n_groups,
                self.min_group_size,
                seed=rng.integers(0, 2**31),
            )
            results.append(labels)

        group_indices = np.stack(results, axis=0)

        if not self.training:
            self.cached_labels = group_indices

        return torch.from_numpy(group_indices.astype(self.dtype))

    def reset(self):
        self.step_counter = 0
        self.cached_labels = None
        return self


class MagentVecKMeansGroupBuilder(GroupBuilder):
    def __init__(
        self,
        coord_dims: Tuple[int, int],
        hp_dim: int,
        team_dim: int,
        selected_teams: List[int],
        n_groups: int = 4,
        min_group_size: int = 1,
        update_interval: int = 1,
        n_workers: int = 0,
        dtype: str = 'int16',
    ):
        super().__init__(dtype=dtype)
        self.coord_dims = coord_dims
        self.hp_dim = hp_dim
        self.team_dim = team_dim
        self.selected_teams = selected_teams
        self.n_groups = n_groups
        self.min_group_size = min_group_size
        self.update_interval = update_interval
        self.n_workers = n_workers
        self.step_counter = 0
        self.cached_labels = None

    @staticmethod
    def _process_sample(
        sample_state,
        coord_dims,
        hp_dim,
        team_dim,
        selected_teams,
        n_groups,
        min_group_size,
        seed=None,
    ):
        coords = sample_state[:, coord_dims]
        hps = sample_state[:, hp_dim]
        teams = sample_state[:, team_dim]

        candidate_mask = np.isin(teams, selected_teams)
        n_candidates = candidate_mask.sum()

        candidate_coords = coords[candidate_mask]
        candidate_hps = hps[candidate_mask]

        valid_mask = candidate_hps > 0
        valid_local_ids = np.where(valid_mask)[0]
        n_alive = len(valid_local_ids)

        if n_alive == 0:
            return np.full(n_candidates, -1, dtype=np.int64)

        n_clusters = MagentKMeansGroupBuilder._compute_n_clusters(
            n_alive, n_groups, min_group_size
        )

        valid_coords = candidate_coords[valid_mask]
        cluster_labels = MagentKMeansGroupBuilder._kmeans_cluster(
            valid_coords, n_clusters, seed
        )

        full_labels = np.full(n_candidates, -1, dtype=np.int64)
        full_labels[valid_local_ids] = cluster_labels
        return full_labels

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        states = states.cpu().numpy()
        bs = states.shape[0]

        if not self.training:
            self.step_counter += 1
            if (
                self.step_counter % self.update_interval != 0
                and self.cached_labels is not None
            ):
                # ── Return cached labels with dead-agent correction ──────
                # state:  (bs, N_all, F)   includes BOTH ally + enemy teams
                # result: (bs, n_cand)      only selected-team agents
                # Mask must align selected-team subset to avoid cross-team indexing
                # NOTE: bool indexing always returns 1D because numpy cannot
                # guarantee equal True counts per row at compile time.
                # Here every row has exactly n_cand True entries, so the
                # reshape(bs, -1) → (bs, n_cand) is safe and restores shape.
                result = deepcopy(self.cached_labels)                   # (bs, n_cand)
                teams = states[:, :, self.team_dim]                     # (bs, N_all)
                hps   = states[:, :, self.hp_dim]                       # (bs, N_all)
                cand_mask = np.isin(teams, self.selected_teams)         # (bs, N_all) bool
                cand_hps  = hps[cand_mask].reshape(bs, -1)              # hps[cand_mask]: flat (bs*n_cand,) → reshape → (bs, n_cand)
                result[cand_hps <= 0] = -1                              # dead → -1, alive unchanged
                return torch.from_numpy(result)
            self.cached_labels = None

        rng = np.random.default_rng()

        if self.n_workers > 1:
            n_workers = min(bs, self.n_workers)
            seeds = rng.integers(0, 2**31, size=bs).tolist()
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                results = list(
                    executor.map(
                        self._process_sample,
                        [states[b] for b in range(bs)],
                        [self.coord_dims] * bs,
                        [self.hp_dim] * bs,
                        [self.team_dim] * bs,
                        [self.selected_teams] * bs,
                        [self.n_groups] * bs,
                        [self.min_group_size] * bs,
                        seeds,
                    )
                )
        else:
            results = [
                self._process_sample(
                    states[b],
                    self.coord_dims,
                    self.hp_dim,
                    self.team_dim,
                    self.selected_teams,
                    self.n_groups,
                    self.min_group_size,
                    seed=rng.integers(0, 2**31),
                )
                for b in range(bs)
            ]

        labels = np.stack(results, axis=0)

        if not self.training:
            self.cached_labels = labels

        return torch.from_numpy(labels.astype(self.dtype))

    def reset(self):
        self.step_counter = 0
        self.cached_labels = None
        return self
