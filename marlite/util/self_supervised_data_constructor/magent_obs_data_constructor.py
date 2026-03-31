import numpy as np
from typing import Optional, Tuple, List, Type, Any
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory
import multiprocessing as mp
import sys
import os
from marlite.util.self_supervised_data_constructor.self_supervised_data_constructor import (
    SelfSupervisedDataConstructor,
)


def remove_duplicates_numba(entities):
    """
    Remove duplicate rows from a 2D array.
    Pure numpy implementation using np.unique for O(n log n) complexity.
    """
    if entities.shape[0] == 0:
        return entities

    return np.unique(entities, axis=0)


def get_alive_neighbors_numba(edge_indices_t, alive_agents, agent_idx):
    """
    Get neighbors for a specific agent using vectorized numpy operations.
    O(m+n) complexity using np.isin for lookup.
    """
    source_agents = edge_indices_t[0]
    target_agents = edge_indices_t[1]

    alive_source_mask = np.isin(source_agents, alive_agents)
    alive_target_mask = np.isin(target_agents, alive_agents)
    valid_mask = alive_source_mask & alive_target_mask

    filtered_sources = source_agents[valid_mask]
    filtered_targets = target_agents[valid_mask]

    mask = filtered_targets == agent_idx
    in_edge_sources = filtered_sources[mask]

    if len(in_edge_sources) == 0:
        return np.empty(0, dtype=source_agents.dtype)

    unique_sources = np.unique(in_edge_sources)
    return unique_sources[unique_sources != agent_idx]


def sort_entities_by_distance_angle_numba(entities):
    """
    Sort entities by distance first, then by angle.
    Pure numpy implementation using np.lexsort for O(n log n) complexity.
    Assumes entities have at least 2 features representing relative x, y coordinates.

    Args:
        entities: (N, feature_dim) array where first 2 columns are [rel_x, rel_y]

    Returns:
        Sorted entities array
    """
    if entities.shape[0] <= 1:
        return entities

    distances = np.sqrt(entities[:, 0] ** 2 + entities[:, 1] ** 2)
    angles = np.arctan2(entities[:, 1], entities[:, 0])

    order = np.lexsort((angles, distances))
    return entities[order]


def pad_or_truncate_numba(entities, max_entities_perception):
    """
    Pad or truncate entities to fixed size.
    """
    num_entities = entities.shape[0]
    feature_dim = entities.shape[1] if entities.ndim > 1 else 0

    if num_entities == 0:
        result = np.zeros((max_entities_perception, feature_dim), dtype=entities.dtype)
        mask = np.zeros(max_entities_perception, dtype=bool)
        return result, mask

    if num_entities > max_entities_perception:
        result = np.empty((max_entities_perception, feature_dim), dtype=entities.dtype)
        result[:max_entities_perception, :] = entities[:max_entities_perception, :]
        mask = np.ones(max_entities_perception, dtype=bool)
    else:
        result = np.zeros((max_entities_perception, feature_dim), dtype=entities.dtype)
        result[:num_entities, :] = entities[:num_entities, :]
        mask = np.zeros(max_entities_perception, dtype=bool)
        mask[:num_entities] = True

    return result, mask


def _process_single_agent_numba(
    observations_np, edge_indices_np, alive_mask_np, agent_idx, max_entities_perception
):
    """
    Process a single agent across all time steps.
    """
    seq_len = observations_np.shape[0]
    max_observed_entities = observations_np.shape[2]
    feature_dim = observations_np.shape[3]

    result = np.zeros(
        (seq_len, max_entities_perception, feature_dim), dtype=observations_np.dtype
    )
    mask = np.zeros((seq_len, max_entities_perception), dtype=bool)

    for t in range(seq_len):
        alive_agents_at_t = np.where(alive_mask_np[t, :])[0]
        edge_indices_t = edge_indices_np[t]

        self_obs = np.empty(
            (max_observed_entities, feature_dim), dtype=observations_np.dtype
        )
        self_obs[:, :] = observations_np[t, agent_idx, :, :]

        neighbor_agents = get_alive_neighbors_numba(
            edge_indices_t, alive_agents_at_t, agent_idx
        )

        total_entities = max_observed_entities * (1 + len(neighbor_agents))

        all_entities = np.empty(
            (total_entities, feature_dim), dtype=observations_np.dtype
        )

        entity_idx = 0
        all_entities[entity_idx : entity_idx + max_observed_entities, :] = self_obs[
            :, :
        ]
        entity_idx += max_observed_entities

        for neighbor_idx in neighbor_agents:
            all_entities[entity_idx : entity_idx + max_observed_entities, :] = (
                observations_np[t, neighbor_idx, :, :]
            )
            entity_idx += max_observed_entities

        non_zero_mask = np.sum(np.abs(all_entities), axis=1) > 0
        clean_entities = all_entities[non_zero_mask]

        unique_entities = remove_duplicates_numba(clean_entities)
        sorted_entities = sort_entities_by_distance_angle_numba(unique_entities)
        final_entities, final_mask = pad_or_truncate_numba(
            sorted_entities, max_entities_perception
        )

        result[t, :max_entities_perception, :] = final_entities[
            :max_entities_perception, :
        ]
        mask[t, :max_entities_perception] = final_mask[:max_entities_perception]

    return result, mask


def _is_linux():
    return sys.platform.startswith("linux")


def _create_shared_memory_arrays(observations, alive_mask):
    shm_obs = shared_memory.SharedMemory(create=True, size=observations.nbytes)
    shm_alive = shared_memory.SharedMemory(create=True, size=alive_mask.nbytes)

    obs_arr = np.ndarray(
        observations.shape, dtype=observations.dtype, buffer=shm_obs.buf
    )
    obs_arr[:] = observations[:]

    alive_arr = np.ndarray(
        alive_mask.shape, dtype=alive_mask.dtype, buffer=shm_alive.buf
    )
    alive_arr[:] = alive_mask[:]

    return shm_obs, shm_alive, obs_arr, alive_arr


def _cleanup_shared_memory(shm_obs, shm_alive):
    shm_obs.close()
    shm_obs.unlink()
    shm_alive.close()
    shm_alive.unlink()


def _worker_init(
    shm_obs_name, obs_shape, obs_dtype, shm_alive_name, alive_shape, alive_dtype
):
    global _worker_obs_shm, _worker_alive_shm, _worker_obs, _worker_alive
    _worker_obs_shm = shared_memory.SharedMemory(name=shm_obs_name)
    _worker_alive_shm = shared_memory.SharedMemory(name=shm_alive_name)
    _worker_obs = np.ndarray(obs_shape, dtype=obs_dtype, buffer=_worker_obs_shm.buf)
    _worker_alive = np.ndarray(
        alive_shape, dtype=alive_dtype, buffer=_worker_alive_shm.buf
    )


def _worker_process_batch(args):
    batch_indices, edge_indices_list, max_entities_perception = args
    results = []
    for idx in batch_indices:
        obs = _worker_obs[idx]
        alive = _worker_alive[idx]
        edges = edge_indices_list[idx]
        result, mask = _process_single_sample(
            (obs, edges, alive, max_entities_perception)
        )
        results.append((result, mask))
    return results


def _process_single_sample(args):
    """
    Process a single sample (batch element) sequentially.
    """
    observations_np, edge_indices_np, alive_mask_np, max_entities_perception = args

    seq_len, n_agents, max_observed_entities, feature_dim = observations_np.shape

    result = np.zeros(
        (seq_len, n_agents, max_entities_perception, feature_dim),
        dtype=observations_np.dtype,
    )
    mask = np.zeros((seq_len, n_agents, max_entities_perception), dtype=bool)

    for t in range(seq_len):
        for agent_idx in range(n_agents):
            if alive_mask_np[t, agent_idx]:
                agent_result, agent_mask = _process_single_agent_numba(
                    observations_np,
                    edge_indices_np,
                    alive_mask_np,
                    agent_idx,
                    max_entities_perception,
                )
                result[t, agent_idx, :, :] = agent_result[t, :, :]
                mask[t, agent_idx, :] = agent_mask[t, :]

    return result, mask


class MagentObsDataConstructor(SelfSupervisedDataConstructor):
    """
    Abstract base class for Magent observation data constructors.
    Encapsulates common logic for processing observations with communication graphs.
    Subclasses must implement preprocessing to unify input formats.
    """

    def __init__(
        self,
        max_entities_perception: int,
        with_time_seq: bool = False,
        n_workers: int = 0,
        include_features: Optional[List[int]] = None,
        exclude_features: Optional[List[int]] = None,
    ):
        """
        Initialize the base Magent observation data constructor.

        Args:
            max_entities_perception: Maximum number of entities each agent can perceive
            with_time_seq: Whether to include time sequence dimension in output
            n_workers: Number of worker processes for parallel processing
            include_features: List of feature dimension indices to keep (takes precedence over exclude_features)
            exclude_features: List of feature dimension indices to filter out
        """
        super().__init__(n_workers=n_workers)
        self.with_time_seq = with_time_seq
        self.max_entities_perception = max_entities_perception
        self.exclude_features = exclude_features if exclude_features is not None else []
        self.include_features = include_features if include_features is not None else []

    def process(
        self,
        observations: np.ndarray,
        states: Optional[np.ndarray],
        edge_indices: List[List[np.ndarray]],
        alive_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Main entry point for processing observations.
        Handles time sequence logic, parallelization, and feature filtering.
        Delegates core per-agent processing to subclasses via preprocess_observations().

        Args:
            observations: Raw observations (format depends on subclass)
            states: Optional array, not used in this implementation
            edge_indices: List of shape (batch_size, seq_len) containing np.ndarray of shape (2, edge_num)
            alive_mask: Array of shape (batch_size, seq_len, n_agents)

        Returns:
            A tuple containing processed result and mask arrays
        """
        processed_observations = self.preprocess_observations(observations)

        if len(processed_observations.shape) != 5:
            raise ValueError(
                f"Expected preprocessed observations to have 5 dimensions "
                f"(batch_size, seq_len, n_agents, max_observed_entities, feature_dim), "
                f"got {len(processed_observations.shape)}"
            )

        batch_size, seq_len, n_agents, max_observed_entities, feature_dim = (
            processed_observations.shape
        )

        if not self.with_time_seq:
            processed_observations = processed_observations[:, [-1], :, :, :]
            edge_indices = [
                [batch_edge_indices[-1]] for batch_edge_indices in edge_indices
            ]
            seq_len = 1

        result = np.zeros(
            (batch_size, seq_len, n_agents, self.max_entities_perception, feature_dim),
            dtype=processed_observations.dtype,
        )
        mask = np.zeros(
            (batch_size, seq_len, n_agents, self.max_entities_perception), dtype=bool
        )

        if self.n_workers > 1:
            shm_obs, shm_alive, _, _ = _create_shared_memory_arrays(
                processed_observations, alive_mask
            )

            try:
                n_workers = min(self.n_workers, batch_size)
                base_samples = batch_size // n_workers
                extra_samples = batch_size % n_workers

                worker_args = []
                for worker_id in range(n_workers):
                    start_idx = worker_id * base_samples + min(worker_id, extra_samples)
                    end_idx = (
                        start_idx
                        + base_samples
                        + (1 if worker_id < extra_samples else 0)
                    )
                    if start_idx >= batch_size:
                        break
                    batch_indices = list(range(start_idx, end_idx))
                    worker_args.append(
                        (batch_indices, edge_indices, self.max_entities_perception)
                    )

                mp_ctx = mp.get_context("fork") if _is_linux() else None

                with ProcessPoolExecutor(
                    max_workers=n_workers,
                    initializer=_worker_init,
                    initargs=(
                        shm_obs.name,
                        processed_observations.shape,
                        processed_observations.dtype,
                        shm_alive.name,
                        alive_mask.shape,
                        alive_mask.dtype,
                    ),
                    mp_context=mp_ctx,
                ) as executor:
                    results = list(executor.map(_worker_process_batch, worker_args))

                batch_idx = 0
                for worker_result_list in results:
                    for processed_sample, mask_batch in worker_result_list:
                        result[batch_idx] = processed_sample
                        mask[batch_idx] = mask_batch
                        batch_idx += 1
            finally:
                _cleanup_shared_memory(shm_obs, shm_alive)
        else:
            for batch_idx in range(batch_size):
                processed_sample, mask_batch = _process_single_sample(
                    (
                        processed_observations[batch_idx],
                        edge_indices[batch_idx],
                        alive_mask[batch_idx],
                        self.max_entities_perception,
                    )
                )
                result[batch_idx] = processed_sample
                mask[batch_idx] = mask_batch

        if not self.with_time_seq:
            result = result.squeeze(1)
            mask = mask.squeeze(1)

        if self.include_features:
            valid_include_features = [
                f for f in self.include_features if 0 <= f < result.shape[-1]
            ]
            result = result[..., valid_include_features]
        elif self.exclude_features:
            remaining_features = [
                f for f in range(result.shape[-1]) if f not in self.exclude_features
            ]
            result = result[..., remaining_features]

        return result, mask

    def preprocess_observations(self, observations: np.ndarray) -> np.ndarray:
        """
        Preprocess raw observations into unified format (B, T, N, L, F).
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement preprocess_observations")


class MagentVecObsDataConstructor(MagentObsDataConstructor):
    """
    Magent environment's vector observation data constructor for self-supervised learning.
    Handles vector observations with shape (batch_size, seq_len, n_agents, max_observed_entities, feature_dim).
    """

    def __init__(
        self,
        max_entities_perception: int,
        with_time_seq: bool = False,
        n_workers: int = 0,
        include_features: Optional[List[int]] = None,
        exclude_features: Optional[List[int]] = None,
    ):
        super().__init__(
            max_entities_perception,
            with_time_seq,
            n_workers,
            include_features,
            exclude_features,
        )

    def preprocess_observations(self, observations: np.ndarray) -> np.ndarray:
        if len(observations.shape) != 5:
            raise ValueError(
                f"Expected vector observations to have 5 dimensions "
                f"(batch_size, seq_len, n_agents, max_observed_entities, feature_dim), "
                f"got {len(observations.shape)}"
            )
        return observations


class MagentImageObsDataConstructor(MagentObsDataConstructor):
    """
    Magent environment's image observation data constructor for self-supervised learning.
    Handles image observations with shape (batch_size, seq_len, n_agents, H, W, C) or
    (batch_size, seq_len, n_agents, C, H, W) when channel_first=True.
    """

    def __init__(
        self,
        max_entities_perception: int,
        with_time_seq: bool = False,
        n_workers: int = 0,
        include_features: Optional[List[int]] = None,
        exclude_features: Optional[List[int]] = None,
        channel_first: bool = False,
    ):
        super().__init__(
            max_entities_perception,
            with_time_seq,
            n_workers,
            include_features,
            exclude_features,
        )
        self.channel_first = channel_first

    def preprocess_observations(self, observations: np.ndarray) -> np.ndarray:
        if len(observations.shape) != 6:
            raise ValueError(
                f"Expected image observations to have 6 dimensions "
                f"(batch_size, seq_len, n_agents, H, W, C) or (batch_size, seq_len, n_agents, C, H, W), got {len(observations.shape)}"
            )
        if self.channel_first:
            observations = np.transpose(observations, (0, 1, 2, 4, 5, 3))

        batch_size, seq_len, n_agents, H, W, C = observations.shape
        L = H * W
        reshaped_observations = observations.reshape(
            batch_size, seq_len, n_agents, L, C
        )

        return reshaped_observations
