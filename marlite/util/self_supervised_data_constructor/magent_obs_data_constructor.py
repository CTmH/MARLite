import numpy as np
from typing import Optional, Tuple, List, Type, Any
from concurrent.futures import ProcessPoolExecutor
from marlite.util.self_supervised_data_constructor.self_supervised_data_constructor import SelfSupervisedDataConstructor
import numba
from numba import jit

@jit(nopython=True, cache=True)
def remove_duplicates_numba(entities):
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
            unique_entities[out_idx, :] = entities[i, :]
            out_idx += 1

    return unique_entities

@jit(nopython=True, cache=True)
def get_alive_neighbors_numba(edge_indices_t, alive_agents, agent_idx):
    """
    Numba-accelerated function to get neighbors for a specific agent.
    """
    source_agents = edge_indices_t[0]
    target_agents = edge_indices_t[1]

    # Create mask for edges where both source and target are alive
    alive_source_mask = np.zeros(len(source_agents), dtype=numba.boolean)
    alive_target_mask = np.zeros(len(target_agents), dtype=numba.boolean)

    for i in range(len(alive_agents)):
        alive_agent = alive_agents[i]
        for j in range(len(source_agents)):
            if source_agents[j] == alive_agent:
                alive_source_mask[j] = True
        for j in range(len(target_agents)):
            if target_agents[j] == alive_agent:
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
def pad_or_truncate_numba(entities, max_entities_perception):
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
        result[:max_entities_perception, :] = entities[:max_entities_perception, :]
        mask = np.ones(max_entities_perception, dtype=numba.boolean)
    else:
        result = np.zeros((max_entities_perception, feature_dim), dtype=entities.dtype)
        result[:num_entities, :] = entities[:num_entities, :]
        mask = np.zeros(max_entities_perception, dtype=numba.boolean)
        mask[:num_entities] = True

    return result, mask

@jit(nopython=True, cache=True)
def _process_single_agent_numba(observations_np, edge_indices_np, alive_mask_np,
                                         agent_idx, max_entities_perception):
    """
    Numba-accelerated function to process a single agent across all time steps.
    Updated for new alive_mask dimension: (seq_len, n_agents) and observations dimension: (seq_len, n_agents, max_observed_entities, feature_dim)
    """
    # Get dimensions from observations_np
    seq_len = observations_np.shape[0]
    max_observed_entities = observations_np.shape[2]
    feature_dim = observations_np.shape[3]

    # Prepare result for this agent
    result = np.zeros((seq_len, max_entities_perception, feature_dim), dtype=observations_np.dtype)
    mask = np.zeros((seq_len, max_entities_perception), dtype=numba.boolean)

    # Process each time step
    for t in range(seq_len):
        # Get alive agents at time t
        # NEW DIMENSION: alive_mask_np is (seq_len, n_agents), so alive_mask_np[t, :] gives us all agents at time t
        alive_agents_at_t = np.where(alive_mask_np[t, :])[0]  # Get all agents alive at time t

        # Get the edge indices for this specific time step
        edge_indices_t = edge_indices_np[t]  # (2, edge_num)

        # Step 1: Get ALL observations from this agent at time t (max_observed_entities entities)
        # Shape: (max_observed_entities, feature_dim)
        self_obs = np.empty((max_observed_entities, feature_dim), dtype=observations_np.dtype)
        self_obs[:, :] = observations_np[t, agent_idx, :, :]

        # Step 2: Get observations from agents that send TO this agent (in-edges only)
        neighbor_agents = get_alive_neighbors_numba(edge_indices_t, alive_agents_at_t, agent_idx)

        # Calculate total number of entities we'll have
        total_entities = max_observed_entities * (1 + len(neighbor_agents))

        # Collect all entity vectors: self + neighbors' full observations at time t
        all_entities = np.empty((total_entities, feature_dim), dtype=observations_np.dtype)

        # Add self's max_observed_entities entities at time t
        entity_idx = 0
        all_entities[entity_idx:entity_idx + max_observed_entities, :] = self_obs[:, :]
        entity_idx += max_observed_entities

        # Add each neighbor's max_observed_entities entities at time t
        for neighbor_idx in neighbor_agents:
            all_entities[entity_idx:entity_idx + max_observed_entities, :] = observations_np[t, neighbor_idx, :, :]
            entity_idx += max_observed_entities

        # Step 3: Remove zero-filled vectors (padding)
        non_zero_mask = np.sum(np.abs(all_entities), axis=1) > 0
        #non_zero_mask = np.zeros(total_entities, dtype=numba.boolean)
        #for i in range(total_entities):
        #    is_nonzero = False
        #    for f in range(feature_dim):
        #        if all_entities[i, f] != 0:
        #            is_nonzero = True
        #            break
        #    non_zero_mask[i] = is_nonzero

        # Count non-zero entities
        nonzero_count = np.sum(non_zero_mask)
        #nonzero_count = 0
        #for i in range(len(non_zero_mask)):
        #    if non_zero_mask[i]:
        #        nonzero_count += 1

        # Create clean entities array
        clean_entities = all_entities[non_zero_mask]
        #clean_entities = np.empty((nonzero_count, feature_dim), dtype=observations_np.dtype)
        #idx = 0
        #for i in range(len(non_zero_mask)):
        #    if non_zero_mask[i]:
        #        clean_entities[idx, :] = all_entities[i, :]
        #        idx += 1

        # Step 4: Remove duplicates (exact row matches)
        unique_entities = remove_duplicates_numba(clean_entities)

        # Step 5: Truncate or pad to max_entities_perception
        final_entities, final_mask = pad_or_truncate_numba(unique_entities, max_entities_perception)

        # Store results using vectorized assignment
        result[t, :max_entities_perception, :] = final_entities[:max_entities_perception, :]
        mask[t, :max_entities_perception] = final_mask[:max_entities_perception]

    return result, mask

class MagentObsDataConstructor(SelfSupervisedDataConstructor):
    """
    Abstract base class for Magent observation data constructors.
    Encapsulates common logic for processing observations with communication graphs.
    Subclasses must implement preprocessing to unify input formats.
    """

    def __init__(self, max_entities_perception: int, with_time_seq: bool = False, n_workers: int = 0,
                 include_features: Optional[List[int]] = None, exclude_features: Optional[List[int]] = None):
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

    def process(self, observations: np.ndarray, states: Optional[np.ndarray],
                edge_indices: List[List[np.ndarray]], alive_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
        # Preprocess raw observations into unified format: (B, T, N, L, F)
        processed_observations = self.preprocess_observations(observations)

        # Validate unified format
        if len(processed_observations.shape) != 5:
            raise ValueError(f"Expected preprocessed observations to have 5 dimensions "
                           f"(batch_size, seq_len, n_agents, max_observed_entities, feature_dim), "
                           f"got {len(processed_observations.shape)}")

        # Extract dimensions from preprocessed observations
        batch_size, seq_len, n_agents, max_observed_entities, feature_dim = processed_observations.shape

        # Handle with_time_seq flag: slice to last timestep if needed
        if not self.with_time_seq:
            processed_observations = processed_observations[:, [-1], :, :, :]
            # For edge_indices, take only the last time step for each batch
            edge_indices = [[batch_edge_indices[-1]] for batch_edge_indices in edge_indices]
            # Update seq_len after slicing
            seq_len = 1

        # Prepare result and mask arrays
        result = np.zeros((batch_size, seq_len, n_agents, self.max_entities_perception, feature_dim),
                          dtype=processed_observations.dtype)
        mask = np.zeros((batch_size, seq_len, n_agents, self.max_entities_perception), dtype=bool)

        # Parallel or sequential processing
        if self.n_workers > 1:
            # Divide batch among workers
            base_samples = batch_size // self.n_workers
            extra_samples = batch_size % self.n_workers

            worker_args = []
            for worker_id in range(self.n_workers):
                start_idx = worker_id * base_samples + min(worker_id, extra_samples)
                end_idx = start_idx + base_samples + (1 if worker_id < extra_samples else 0)
                if start_idx >= batch_size:
                    break

                worker_batch_args = []
                for i in range(start_idx, end_idx):
                    args = (
                        processed_observations[i],
                        edge_indices[i],
                        alive_mask[i],
                        self.max_entities_perception
                    )
                    worker_batch_args.append(args)
                worker_args.append(worker_batch_args)

            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                results = list(executor.map(self._process_multiple_samples, worker_args))

                batch_idx = 0
                for worker_result_list in results:
                    for processed_sample, mask_batch in worker_result_list:
                        result[batch_idx] = processed_sample
                        mask[batch_idx] = mask_batch
                        batch_idx += 1
        else:
            for batch_idx in range(batch_size):
                processed_sample, mask_batch = self._process_single_sample((
                    processed_observations[batch_idx],
                    edge_indices[batch_idx],
                    alive_mask[batch_idx],
                    self.max_entities_perception
                ))
                result[batch_idx] = processed_sample
                mask[batch_idx] = mask_batch

        # Remove time dimension if not requested
        if not self.with_time_seq:
            result = result.squeeze(1)
            mask = mask.squeeze(1)

        # Apply feature filtering
        if self.include_features:
            valid_include_features = [f for f in self.include_features if 0 <= f < result.shape[-1]]
            result = result[..., valid_include_features]
        elif self.exclude_features:
            remaining_features = [f for f in range(result.shape[-1]) if f not in self.exclude_features]
            result = result[..., remaining_features]

        return result, mask

    def preprocess_observations(self, observations: np.ndarray) -> np.ndarray:
        """
        Preprocess raw observations into unified format (B, T, N, L, F).
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement preprocess_observations")

    def _process_multiple_samples(self, batch_args_list):
        """
        Process multiple samples in parallel (used by ProcessPoolExecutor).
        """
        results = []
        for args in batch_args_list:
            processed_batch, mask_batch = self._process_single_sample(args)
            results.append((processed_batch, mask_batch))
        return results

    def _process_single_sample(self, args):
        """
        Process a single sample (batch element) sequentially.
        """
        observations_np, edge_indices_np, alive_mask_np, max_entities_perception = args

        seq_len, n_agents, max_observed_entities, feature_dim = observations_np.shape

        result = np.zeros((seq_len, n_agents, max_entities_perception, feature_dim), dtype=observations_np.dtype)
        mask = np.zeros((seq_len, n_agents, max_entities_perception), dtype=bool)

        for t in range(seq_len):
            for agent_idx in range(n_agents):
                if alive_mask_np[t, agent_idx]:
                    # Use the shared numba function from vec obs constructor
                    agent_result, agent_mask = _process_single_agent_numba(
                        observations_np, edge_indices_np, alive_mask_np,
                        agent_idx, max_entities_perception
                    )
                    result[t, agent_idx, :, :] = agent_result[t, :, :]
                    mask[t, agent_idx, :] = agent_mask[t, :]

        return result, mask


class MagentVecObsDataConstructor(MagentObsDataConstructor):
    """
    Magent environment's vector observation data constructor for self-supervised learning.
    Handles vector observations with shape (batch_size, seq_len, n_agents, max_observed_entities, feature_dim).
    """

    def __init__(self, max_entities_perception: int, with_time_seq: bool = False, n_workers: int = 0,
                 include_features: Optional[List[int]] = None, exclude_features: Optional[List[int]] = None):
        """
        Initialize the Magent environment's vector observation data constructor.

        Args:
            max_entities_perception: Maximum number of entities each agent can perceive
            with_time_seq: Whether to include time sequence dimension in output
            n_workers: Number of worker processes for parallel processing
            include_features: List of feature dimension indices to keep (takes precedence over exclude_features)
            exclude_features: List of feature dimension indices to filter out
        """
        super().__init__(max_entities_perception, with_time_seq, n_workers, include_features, exclude_features)

    def preprocess_observations(self, observations: np.ndarray) -> np.ndarray:
        """
        Vector observations are already in the expected format (B, T, N, L, F).
        Just validate and return.
        """
        if len(observations.shape) != 5:
            raise ValueError(f"Expected vector observations to have 5 dimensions "
                           f"(batch_size, seq_len, n_agents, max_observed_entities, feature_dim), "
                           f"got {len(observations.shape)}")
        return observations


class MagentImageObsDataConstructor(MagentObsDataConstructor):
    """
    Magent environment's image observation data constructor for self-supervised learning.
    Handles image observations with shape (batch_size, seq_len, n_agents, H, W, C) or
    (batch_size, seq_len, n_agents, C, H, W) when channel_first=True.
    """

    def __init__(self, max_entities_perception: int, with_time_seq: bool = False, n_workers: int = 0,
                 include_features: Optional[List[int]] = None, exclude_features: Optional[List[int]] = None,
                 channel_first: bool = False):
        """
        Initialize the Magent environment's image observation data constructor.

        Args:
            max_entities_perception: Maximum number of entities each agent can perceive
            with_time_seq: Whether to include time sequence dimension in output
            n_workers: Number of worker processes for parallel processing
            include_features: List of feature dimension indices to keep (takes precedence over exclude_features)
            exclude_features: List of feature dimension indices to filter out
            channel_first: Whether input observations are in channel-first format (C, H, W) vs channel-last (H, W, C)
        """
        super().__init__(max_entities_perception, with_time_seq, n_workers, include_features, exclude_features)
        self.channel_first = channel_first

    def preprocess_observations(self, observations: np.ndarray) -> np.ndarray:
        """
        Preprocess image observations into unified format (B, T, N, L, F):
        - Handle channel_first conversion
        - Reshape (H, W, C) to (L, C) where L = H * W
        """
        # Validate input format
        if len(observations.shape) != 6:
            raise ValueError(f"Expected image observations to have 6 dimensions "
                           f"(batch_size, seq_len, n_agents, H, W, C) or (batch_size, seq_len, n_agents, C, H, W), got {len(observations.shape)}")
        # Handle channel_first format by converting to channel_last
        if self.channel_first:
            # Input shape: (batch_size, seq_len, n_agents, C, H, W)
            # Convert to: (batch_size, seq_len, n_agents, H, W, C)
            observations = np.transpose(observations, (0, 1, 2, 4, 5, 3))

        # Extract spatial dimensions
        batch_size, seq_len, n_agents, H, W, C = observations.shape

        # Reshape (H, W, C) to (L, C or F) where L = H * W
        L = H * W
        reshaped_observations = observations.reshape(batch_size, seq_len, n_agents, L, C)

        return reshaped_observations
