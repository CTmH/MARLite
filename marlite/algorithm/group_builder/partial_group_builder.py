import numpy as np
import torch
from typing import List
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor

from marlite.algorithm.group_builder.group_builder import GroupBuilder
from marlite.algorithm.graph_builder.graph_util import (
    extract_agent_positions_batch,
    build_partial_groups,
)


class PartialGroupMAgentBuilder(GroupBuilder):
    """
    Group builder for MAgent grid states that assigns each valid agent a
    group label derived from the community (subgraph) partition of the full
    communication graph.

    Mirrors :class:`marlite.algorithm.graph_builder.partial_graph_builder.PartialGraphMAgentBuilder`:
    the full communication graph is built from ``valid_node_list + target_node_list``,
    partitioned into ``n_groups`` communities via greedy modularity detection, and
    the community index of each valid node is returned as the group label.

    Agents that are not present in the state, or that are excluded from any
    community, receive group label ``-1``. When the actual number of communities
    detected is fewer than ``n_groups``, the labels are still assigned as
    ``0, 1, 2, ...`` (smaller group numbers first).
    """

    def __init__(
            self,
            binary_agent_id_dim: List[int],
            agent_presence_dim: List[int],
            comm_distance: int,
            valid_node_list: List[int],
            target_node_list: List[int],
            distance_metric: str = 'cityblock',
            n_workers: int = 8,
            n_groups: int = 2,
            update_interval: int = 1,
            channel_first: bool = False,
            dtype: str = 'int16'):
        super().__init__(dtype=dtype)
        self.binary_agent_id_dim = binary_agent_id_dim
        self.agent_presence_dim = agent_presence_dim
        self.comm_distance = comm_distance
        self.distance_metric = distance_metric
        self.n_workers = n_workers
        self.n_groups = n_groups
        self.valid_node_list = valid_node_list
        self.target_node_list = target_node_list

        self.update_interval = update_interval
        self.channel_first = channel_first
        self.step_counter = 0
        self.cached_labels = None

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        states = states.cpu().numpy()
        if self.channel_first:
            states = np.transpose(states, (0, 2, 3, 1))

        if not self.training:
            if (self.step_counter % self.update_interval != 0
                and self.cached_labels is not None):
                self.step_counter += 1
                return torch.from_numpy(deepcopy(self.cached_labels).astype(self.dtype))

        batched_coords_with_id = extract_agent_positions_batch(
            states, self.binary_agent_id_dim, self.agent_presence_dim
        )

        bs = states.shape[0]
        use_multi_process = bs > 1 and self.n_workers > 1
        if use_multi_process:
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                results = list(executor.map(
                    self._process_sample,
                    batched_coords_with_id
                ))
        else:
            results = []
            for coords in batched_coords_with_id:
                result = self._process_sample(coords)
                results.append(result)

        group_indices = np.stack(results, axis=0)

        if not self.training:
            self.cached_labels = group_indices

        self.step_counter += 1
        return torch.from_numpy(group_indices.astype(self.dtype))

    def _process_sample(self, coords_with_id: np.ndarray) -> np.ndarray:
        """Process a single sample to produce group labels."""
        return build_partial_groups(
            coords_with_id=coords_with_id,
            comm_distance=self.comm_distance,
            distance_metric=self.distance_metric,
            n_groups=self.n_groups,
            valid_node_list=self.valid_node_list,
            target_node_list=self.target_node_list,
        )

    def reset(self):
        self.step_counter = 0
        self.cached_labels = None
        return self


class PartialGroupVectorStateBuilder(GroupBuilder):
    """
    Group builder for MAgent2 vector states with shape (batch_size, num_agents,
    feature_dim) that assigns each valid agent a group label derived from the
    community (subgraph) partition of the full communication graph.

    Mirrors :class:`marlite.algorithm.graph_builder.partial_graph_builder.PartialGraphVectorStateBuilder`:
    the full communication graph is built from ``valid_node_list + target_node_list``,
    partitioned into ``n_groups`` communities via greedy modularity detection, and
    the community index of each valid node is returned as the group label.

    Agents with ``hp <= 0`` (or not in the state), or that are excluded from any
    community, receive group label ``-1``. When the actual number of communities
    detected is fewer than ``n_groups``, the labels are still assigned as
    ``0, 1, 2, ...`` (smaller group numbers first).
    """

    def __init__(
            self,
            coord_dim: List[int],
            hp_dim: int,
            comm_distance: int,
            valid_node_list: List[int],
            target_node_list: List[int],
            distance_metric: str = 'cityblock',
            n_workers: int = 8,
            n_groups: int = 2,
            update_interval: int = 1,
            dtype: str = 'int16'):
        super().__init__(dtype=dtype)
        self.coord_dim = coord_dim
        self.hp_dim = hp_dim
        self.comm_distance = comm_distance
        self.distance_metric = distance_metric
        self.n_workers = n_workers
        self.n_groups = n_groups
        self.valid_node_list = valid_node_list
        self.target_node_list = target_node_list

        self.update_interval = update_interval
        self.step_counter = 0
        self.cached_labels = None

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        states = states.cpu().numpy()

        if not self.training:
            if (self.step_counter % self.update_interval != 0
                and self.cached_labels is not None):
                self.step_counter += 1
                return torch.from_numpy(deepcopy(self.cached_labels).astype(self.dtype))

        batched_coords_with_id = []
        bs = states.shape[0]
        for state in states:
            coords_with_id = []
            for i, col in enumerate(state):
                if col[self.hp_dim] > 0:
                    coords_with_id.append([i, col[self.coord_dim[0]], col[self.coord_dim[1]]])
            batched_coords_with_id.append(np.array(coords_with_id))

        use_multi_process = bs > 1 and self.n_workers > 1
        if use_multi_process:
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                results = list(executor.map(
                    self._process_sample,
                    batched_coords_with_id
                ))
        else:
            results = []
            for coords in batched_coords_with_id:
                result = self._process_sample(coords)
                results.append(result)

        group_indices = np.stack(results, axis=0)

        if not self.training:
            self.cached_labels = group_indices

        self.step_counter += 1
        return torch.from_numpy(group_indices.astype(self.dtype))

    def _process_sample(self, coords_with_id: np.ndarray) -> np.ndarray:
        """Process a single sample to produce group labels."""
        return build_partial_groups(
            coords_with_id=coords_with_id,
            comm_distance=self.comm_distance,
            distance_metric=self.distance_metric,
            n_groups=self.n_groups,
            valid_node_list=self.valid_node_list,
            target_node_list=self.target_node_list,
        )

    def reset(self):
        self.step_counter = 0
        self.cached_labels = None
        return self
