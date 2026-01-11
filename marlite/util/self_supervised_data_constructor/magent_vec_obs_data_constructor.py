import numpy as np
from math import ceil
from typing import Optional, Tuple, List
from concurrent.futures import ProcessPoolExecutor
from marlite.util.self_supervised_data_constructor.self_supervised_data_constructor import SelfSupervisedDataConstructor
import numba
from numba import jit

@jit(nopython=True, cache=True)
def _remove_duplicates_numba(entities):
    """
    Numba-accelerated function to remove duplicate rows from a 2D array.
    """
    if entities.shape[0] == 0:
        return entities

    # Use a simple approach to find unique rows
    n_rows = entities.shape[0]
    n_cols = entities.shape[1]

    # Track which rows are unique
    is_unique = np.ones(n_rows, dtype=numba.boolean)

    for i in range(n_rows):
        if is_unique[i]:  # Only check if this row hasn't been marked as duplicate yet
            for j in range(i + 1, n_rows):
                if is_unique[j]:  # Only compare with rows that haven't been marked as duplicate
                    equal = True
                    for k in range(n_cols):
                        if entities[i, k] != entities[j, k]:
                            equal = False
                            break
                    if equal:
                        is_unique[j] = False  # Mark as duplicate

    # Count unique rows
    unique_count = 0
    for i in range(n_rows):
        if is_unique[i]:
            unique_count += 1

    # Create result array
    unique_entities = np.empty((unique_count, n_cols), dtype=entities.dtype)
    out_idx = 0
    for i in range(n_rows):
        if is_unique[i]:
            for j in range(n_cols):
                unique_entities[out_idx, j] = entities[i, j]
            out_idx += 1

    return unique_entities

@jit(nopython=True, cache=True)
def _get_alive_neighbors_numba(edge_indices_t, alive_agents, agent_idx):
    """
    Numba-accelerated function to get neighbors for a specific agent.
    """
    source_agents = edge_indices_t[0]
    target_agents = edge_indices_t[1]

    # Create mask for edges where both source and target are alive
    alive_source_mask = np.zeros(len(source_agents), dtype=numba.boolean)
    alive_target_mask = np.zeros(len(target_agents), dtype=numba.boolean)

    for i in range(len(alive_agents)):
        for j in range(len(source_agents)):
            if source_agents[j] == alive_agents[i]:
                alive_source_mask[j] = True
        for j in range(len(target_agents)):
            if target_agents[j] == alive_agents[i]:
                alive_target_mask[j] = True

    valid_edge_mask = alive_source_mask & alive_target_mask

    # Count valid edges
    valid_count = 0
    for i in range(len(valid_edge_mask)):
        if valid_edge_mask[i]:
            valid_count += 1

    # Filter the edge indices to only include valid edges
    filtered_sources = np.empty(valid_count, dtype=source_agents.dtype)
    filtered_targets = np.empty(valid_count, dtype=target_agents.dtype)

    idx = 0
    for i in range(len(valid_edge_mask)):
        if valid_edge_mask[i]:
            filtered_sources[idx] = source_agents[i]
            filtered_targets[idx] = target_agents[i]
            idx += 1

    # Find all edges where this agent is the TARGET (i.e., j -> agent_idx)
    in_edge_count = 0
    for i in range(len(filtered_targets)):
        if filtered_targets[i] == agent_idx:
            in_edge_count += 1

    in_edge_sources = np.empty(in_edge_count, dtype=source_agents.dtype)

    idx = 0
    for i in range(len(filtered_targets)):
        if filtered_targets[i] == agent_idx:
            in_edge_sources[idx] = filtered_sources[i]
            idx += 1

    # Remove duplicates and exclude self
    if in_edge_count == 0:
        return np.empty(0, dtype=source_agents.dtype)

    # Sort and remove duplicates
    sorted_sources = np.sort(in_edge_sources)
    unique_sources = np.empty(len(sorted_sources), dtype=sorted_sources.dtype)
    unique_count = 0
    prev_val = -1
    for val in sorted_sources:
        if val != prev_val and val != agent_idx:
            unique_sources[unique_count] = val
            unique_count += 1
            prev_val = val

    return unique_sources[:unique_count]

@jit(nopython=True, cache=True)
def _pad_or_truncate_numba(entities, max_entities_perception):
    """
    Numba-accelerated function to pad or truncate entities to fixed size.
    """
    num_entities = entities.shape[0]
    feature_dim = entities.shape[1] if entities.ndim > 1 else 0

    if num_entities == 0:
        result = np.zeros((max_entities_perception, feature_dim), dtype=entities.dtype)
        mask = np.zeros(max_entities_perception, dtype=numba.boolean)
        return result, mask

    if num_entities > max_entities_perception:
        result = np.empty((max_entities_perception, feature_dim), dtype=entities.dtype)
        for i in range(max_entities_perception):
            for j in range(feature_dim):
                result[i, j] = entities[i, j]
        mask = np.ones(max_entities_perception, dtype=numba.boolean)
    else:
        result = np.zeros((max_entities_perception, feature_dim), dtype=entities.dtype)
        for i in range(num_entities):
            for j in range(feature_dim):
                result[i, j] = entities[i, j]
        mask = np.zeros(max_entities_perception, dtype=numba.boolean)
        for i in range(num_entities):
            mask[i] = True

    return result, mask

@jit(nopython=True, cache=True)
def _process_single_agent_numba(observations_np, edge_indices_np, alive_agents, agent_idx,
                                max_observed_entities, max_entities_perception, feature_dim, seq_len):
    """
    Numba-accelerated function to process a single agent across all time steps.
    """
    # Prepare result for this agent
    result = np.zeros((seq_len, max_entities_perception, feature_dim), dtype=observations_np.dtype)
    mask = np.zeros((seq_len, max_entities_perception), dtype=numba.boolean)

    # Process each time step
    for t in range(seq_len):
        # Get the edge indices for this specific time step
        edge_indices_t = edge_indices_np[t]  # (2, edge_num)

        # Step 1: Get ALL observations from this agent at time t (max_observed_entities entities)
        # Shape: (max_observed_entities, feature_dim)
        self_obs = np.empty((max_observed_entities, feature_dim), dtype=observations_np.dtype)
        for k in range(max_observed_entities):
            for f in range(feature_dim):
                self_obs[k, f] = observations_np[agent_idx, t, k, f]

        # Step 2: Get observations from agents that send TO this agent (in-edges only)
        neighbor_agents = _get_alive_neighbors_numba(edge_indices_t, alive_agents, agent_idx)

        # Calculate total number of entities we'll have
        total_entities = max_observed_entities * (1 + len(neighbor_agents))

        # Collect all entity vectors: self + neighbors' full observations at time t
        all_entities = np.empty((total_entities, feature_dim), dtype=observations_np.dtype)

        # Add self's max_observed_entities entities at time t
        entity_idx = 0
        for k in range(max_observed_entities):
            for f in range(feature_dim):
                all_entities[entity_idx, f] = self_obs[k, f]
            entity_idx += 1

        # Add each neighbor's max_observed_entities entities at time t
        for neighbor_idx in neighbor_agents:
            for k in range(max_observed_entities):
                for f in range(feature_dim):
                    all_entities[entity_idx, f] = observations_np[neighbor_idx, t, k, f]
                entity_idx += 1

        # Step 3: Remove zero-filled vectors (padding)
        non_zero_mask = np.zeros(total_entities, dtype=numba.boolean)
        for i in range(total_entities):
            is_nonzero = False
            for f in range(feature_dim):
                if all_entities[i, f] != 0:
                    is_nonzero = True
                    break
            non_zero_mask[i] = is_nonzero

        # Count non-zero entities
        nonzero_count = 0
        for i in range(len(non_zero_mask)):
            if non_zero_mask[i]:
                nonzero_count += 1

        # Create clean entities array
        clean_entities = np.empty((nonzero_count, feature_dim), dtype=observations_np.dtype)
        idx = 0
        for i in range(len(non_zero_mask)):
            if non_zero_mask[i]:
                for f in range(feature_dim):
                    clean_entities[idx, f] = all_entities[i, f]
                idx += 1

        # Step 4: Remove duplicates (exact row matches)
        unique_entities = _remove_duplicates_numba(clean_entities)

        # Step 5: Truncate or pad to max_entities_perception
        final_entities, final_mask = _pad_or_truncate_numba(unique_entities, max_entities_perception)

        # Store results
        for i in range(max_entities_perception):
            for f in range(feature_dim):
                result[t, i, f] = final_entities[i, f]
            mask[t, i] = final_mask[i]

    return result, mask


class MagentVecObsDataConstructor(SelfSupervisedDataConstructor):
    """
    Magent environment's vector observation data constructor for self-supervised learning.
    """

    def __init__(self, max_entities_perception: int, max_observed_entities: int, with_time_seq: bool = False, n_workers: int = 0):
        """
        Initialize the Magent environment's vector observation data constructor.

        Args:
            max_entities_perception: Maximum number of entities each agent can perceive
            max_observed_entities: Maximum number of observed entities per agent
            with_time_seq: Whether to include time sequence dimension in output
            n_workers: Number of worker processes for parallel processing
        """
        super().__init__(n_workers=n_workers)
        self.with_time_seq = with_time_seq
        self.max_entities_perception = max_entities_perception
        self.max_observed_entities = max_observed_entities

    def process(self, observations: np.ndarray, states: Optional[np.ndarray],
                edge_indices: List[List[np.ndarray]], alive_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process the input data to construct self-supervised learning data.

        Args:
            observations: Array of shape (batch_size, n_agents, seq_len, max_observed_entities, feature_dim)
            states: Optional array, not used in this implementation
            edge_indices: List of shape (batch_size, seq_len) containing np.ndarray of shape (2, edge_num) for each time step
            alive_mask: Array of shape (batch_size, n_agents)

        Returns:
            A tuple containing:
            - Processed array of shape (batch_size, n_agents, max_entities_perception, feature_dim) if with_time_seq=False
              or (batch_size, n_agents, seq_len, max_entities_perception, feature_dim) if with_time_seq=True
            - Mask array of shape (batch_size, n_agents, max_entities_perception) if with_time_seq=False
              or (batch_size, n_agents, seq_len, max_entities_perception) if with_time_seq=True indicating padding
        """
        # Validate observations dimensions
        if len(observations.shape) != 5:
            raise ValueError(f"Expected observations to have 5 dimensions (batch_size, n_agents, seq_len, max_observed_entities, feature_dim), got {len(observations.shape)}")

        if not self.with_time_seq:
            observations = observations[:, :, [-1], :, :]
            # For edge_indices, take only the last time step for each batch
            edge_indices = [[batch_edge_indices[-1]] for batch_edge_indices in edge_indices]

        batch_size, n_agents, seq_len, max_observed_entities, feature_dim = observations.shape

        # Prepare result array with time sequence
        result = np.zeros((batch_size, n_agents, seq_len, self.max_entities_perception, feature_dim),
                            dtype=observations.dtype)
        # Prepare mask array with time sequence
        mask = np.zeros((batch_size, n_agents, seq_len, self.max_entities_perception), dtype=bool)

        if self.n_workers > 1:
            # Divide batch among workers using balanced assignment
            base_samples = batch_size // self.n_workers
            extra_samples = batch_size % self.n_workers

            # Create list of batch indices for each worker
            worker_args = []
            for worker_id in range(self.n_workers):
                # Calculate start and end indices for this worker
                start_idx = worker_id * base_samples + min(worker_id, extra_samples)
                end_idx = start_idx + base_samples + (1 if worker_id < extra_samples else 0)

                if start_idx >= batch_size:
                    break

                # Prepare arguments for this worker's tasks
                worker_batch_args = []
                for i in range(start_idx, end_idx):
                    args = (
                        observations[i],
                        edge_indices[i],
                        alive_mask[i],
                        self.max_observed_entities,
                        self.max_entities_perception,
                        feature_dim,
                        seq_len
                    )
                    worker_batch_args.append(args)

                worker_args.append(worker_batch_args)

            # Use thread pool for parallel processing
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                # Execute in parallel - each worker processes multiple batches
                results = list(executor.map(self._process_multiple_samples, worker_args))

                # Collect results from each worker and place them in correct positions
                batch_idx = 0
                for worker_result_list in results:
                    for processed_sample, mask_batch in worker_result_list:
                        result[batch_idx] = processed_sample
                        mask[batch_idx] = mask_batch
                        batch_idx += 1
        else:
            # Process sequentially
            for batch_idx in range(batch_size):
                processed_sample, mask_batch = self._process_single_sample((
                    observations[batch_idx],
                    edge_indices[batch_idx],
                    alive_mask[batch_idx],
                    self.max_observed_entities,
                    self.max_entities_perception,
                    feature_dim,
                    seq_len
                ))

                result[batch_idx] = processed_sample
                mask[batch_idx] = mask_batch

        if not self.with_time_seq:
            result = result.squeeze(2)
            mask = mask.squeeze(2)

        return result, mask

    def _process_multiple_samples(self, batch_args_list):
        """
        Process multiple samples by a single worker (used for parallel processing).

        Args:
            batch_args_list: List of tuples, each containing arguments for _process_single_batch

        Returns:
            A list of tuples, each containing:
            - Processed numpy array of shape (n_agents, seq_len, max_entities_perception, feature_dim)
            - Mask array of shape (n_agents, seq_len, max_entities_perception) indicating padding
        """
        results = []
        for args in batch_args_list:
            processed_batch, mask_batch = self._process_single_sample(args)
            results.append((processed_batch, mask_batch))
        return results

    def _process_single_sample(self, args):
        """
        Process a single sample of data (used for parallel processing).

        Args:
            args: Tuple containing (observations, edge_indices, alive_mask, max_observed_entities,
                    max_entities_perception, feature_dim, seq_len)

        Returns:
            A tuple containing:
            - Processed numpy array of shape (n_agents, seq_len, max_entities_perception, feature_dim)
            - Mask array of shape (n_agents, seq_len, max_entities_perception) indicating padding
        """
        observations_np, edge_indices_np, alive_mask_np, max_observed_entities, max_entities_perception, feature_dim, seq_len = args

        n_agents = observations_np.shape[0]

        # Prepare result for this batch - always include time dimension for processing
        result = np.zeros((n_agents, seq_len, max_entities_perception, feature_dim), dtype=observations_np.dtype)
        mask = np.zeros((n_agents, seq_len, max_entities_perception), dtype=bool)

        # Convert alive mask to boolean indices
        alive_agents = np.where(alive_mask_np)[0]

        # Process each alive agent
        for agent_idx in alive_agents:
            agent_result, agent_mask = _process_single_agent_numba(
                observations_np, edge_indices_np, alive_agents, agent_idx,
                max_observed_entities, max_entities_perception, feature_dim, seq_len
            )

            # Copy results back to main arrays
            for t in range(seq_len):
                for i in range(max_entities_perception):
                    for f in range(feature_dim):
                        result[agent_idx, t, i, f] = agent_result[t, i, f]
                    mask[agent_idx, t, i] = agent_mask[t, i]

        return result, mask