import numpy as np
from typing import Optional, Tuple, List
from concurrent.futures import ProcessPoolExecutor
from marlite.util.self_supervised_data_constructor.self_supervised_data_constructor import SelfSupervisedDataConstructor
import numba
from numba import jit
import sys
import multiprocessing as mp
from multiprocessing import shared_memory


@jit(nopython=True, cache=True)
def get_incoming_neighbors_numba(edge_indices_t, alive_agents, agent_idx):
    """
    Numba-accelerated function to get incoming neighbors for a specific agent.
    In SUMO, edge_indices is a directed graph (source -> target).
    We want agents that send information TO this agent (target == agent_idx).

    Args:
        edge_indices_t: Edge indices for a single time step, shape (2, edge_num)
        alive_agents: Array of alive agent indices at this time step
        agent_idx: The target agent index

    Returns:
        Array of source agent indices that have edges pointing to agent_idx (excluding self)
        and are alive
    """
    source_agents = edge_indices_t[0]
    target_agents = edge_indices_t[1]

    # Create a set of alive agents for fast lookup
    max_agent_idx = 0
    for i in range(len(source_agents)):
        if source_agents[i] > max_agent_idx:
            max_agent_idx = source_agents[i]
    for i in range(len(target_agents)):
        if target_agents[i] > max_agent_idx:
            max_agent_idx = target_agents[i]

    is_alive = np.zeros(max_agent_idx + 1, dtype=numba.boolean)
    for i in range(len(alive_agents)):
        if alive_agents[i] <= max_agent_idx:
            is_alive[alive_agents[i]] = True

    # Find all edges where this agent is the TARGET (i.e., source -> agent_idx)
    # and source != agent_idx (exclude self-loop) and source is alive
    in_edge_count = 0
    for i in range(len(target_agents)):
        if (target_agents[i] == agent_idx and
            source_agents[i] != agent_idx and
            source_agents[i] < len(is_alive) and
            is_alive[source_agents[i]]):
            in_edge_count += 1

    if in_edge_count == 0:
        return np.empty(0, dtype=source_agents.dtype)

    in_edge_sources = np.empty(in_edge_count, dtype=source_agents.dtype)

    idx = 0
    for i in range(len(target_agents)):
        if (target_agents[i] == agent_idx and
            source_agents[i] != agent_idx and
            source_agents[i] < len(is_alive) and
            is_alive[source_agents[i]]):
            in_edge_sources[idx] = source_agents[i]
            idx += 1

    return in_edge_sources


@jit(nopython=True, cache=True)
def sort_and_truncate_neighbors_numba(neighbors, max_neighbors):
    """
    Numba-accelerated function to sort neighbors by index and truncate to max_neighbors.

    Args:
        neighbors: Array of neighbor indices
        max_neighbors: Maximum number of neighbors to keep

    Returns:
        Sorted and truncated array of neighbor indices
    """
    if len(neighbors) == 0:
        return np.empty(0, dtype=neighbors.dtype)

    # Sort neighbors by index (ascending order)
    sorted_neighbors = np.sort(neighbors)

    # Remove duplicates
    unique_neighbors = np.empty(len(sorted_neighbors), dtype=sorted_neighbors.dtype)
    unique_count = 0
    prev_val = -1
    for val in sorted_neighbors:
        if val != prev_val:
            unique_neighbors[unique_count] = val
            unique_count += 1
            prev_val = val

    # Truncate to max_neighbors
    if unique_count > max_neighbors:
        return unique_neighbors[:max_neighbors]
    else:
        return unique_neighbors[:unique_count]


@jit(nopython=True, cache=True)
def gather_observations_numba(observations_t, agent_idx, neighbor_indices, max_entities_perception):
    """
    Numba-accelerated function to gather observations for a specific agent at time t.

    Args:
        observations_t: Observations at time t, shape (n_agents, feature_dim)
        agent_idx: The target agent index
        neighbor_indices: Array of neighbor indices (already sorted and truncated)
        max_entities_perception: Maximum number of entities to perceive

    Returns:
        Tuple of (gathered_observations, mask)
        - gathered_observations: shape (max_entities_perception, feature_dim)
        - mask: shape (max_entities_perception) indicating valid entries
    """
    n_agents, feature_dim = observations_t.shape

    # Initialize result arrays
    result = np.zeros((max_entities_perception, feature_dim), dtype=observations_t.dtype)
    mask = np.zeros(max_entities_perception, dtype=numba.boolean)

    # Always include self observation first (since edge_indices includes self_loop)
    result[0, :] = observations_t[agent_idx, :]
    mask[0] = True

    # Add neighbor observations
    result_idx = 1
    for neighbor_idx in neighbor_indices:
        if result_idx >= max_entities_perception:
            break

        # Skip self (already included and should not be in neighbor_indices)
        if neighbor_idx == agent_idx:
            continue

        result[result_idx, :] = observations_t[neighbor_idx, :]
        mask[result_idx] = True
        result_idx += 1

    return result, mask


@jit(nopython=True, cache=True)
def process_single_agent_sumo_numba(observations_np, edge_indices_np, alive_mask_np,
                                    agent_idx, max_entities_perception):
    """
    Numba-accelerated function to process a single agent across all time steps for SUMO.

    Args:
        observations_np: Observations array, shape (seq_len, n_agents, feature_dim)
        edge_indices_np: Edge indices array, shape (seq_len, 2, edge_num)
        alive_mask_np: Alive mask array, shape (seq_len, n_agents)
        agent_idx: The target agent index
        max_entities_perception: Maximum number of entities to perceive

    Returns:
        Tuple of (agent_result, agent_mask)
        - agent_result: shape (seq_len, max_entities_perception, feature_dim)
        - agent_mask: shape (seq_len, max_entities_perception) indicating valid entries
    """
    seq_len, n_agents, feature_dim = observations_np.shape

    # Initialize result arrays
    agent_result = np.zeros((seq_len, max_entities_perception, feature_dim), dtype=observations_np.dtype)
    agent_mask = np.zeros((seq_len, max_entities_perception), dtype=numba.boolean)

    # Process each time step
    for t in range(seq_len):
        # Check if agent is alive at time t
        if not alive_mask_np[t, agent_idx]:
            # Agent is not alive, skip processing (result and mask remain zeros)
            continue

        # Get alive agents at time t
        alive_agents_at_t = np.where(alive_mask_np[t, :])[0]

        # Get edge indices for this time step
        edge_indices_t = edge_indices_np[t]  # shape (2, edge_num)

        # Get incoming neighbors (agents that send information TO this agent)
        # Only include neighbors that are alive
        incoming_neighbors = get_incoming_neighbors_numba(edge_indices_t, alive_agents_at_t, agent_idx)

        # Sort neighbors by index, remove duplicates, and truncate
        # We can have at most max_entities_perception - 1 neighbors (since self takes one slot)
        sorted_neighbors = sort_and_truncate_neighbors_numba(
            incoming_neighbors, max_entities_perception - 1
        )

        # Gather observations for this agent at time t
        observations_t = observations_np[t]  # shape (n_agents, feature_dim)
        gathered_obs, gathered_mask = gather_observations_numba(
            observations_t, agent_idx, sorted_neighbors, max_entities_perception
        )

        # Store results
        agent_result[t, :, :] = gathered_obs
        agent_mask[t, :] = gathered_mask

    return agent_result, agent_mask


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


def _sumo_worker_init(
    shm_obs_name, obs_shape, obs_dtype, shm_alive_name, alive_shape, alive_dtype
):
    global _worker_obs_shm, _worker_alive_shm, _worker_obs, _worker_alive
    _worker_obs_shm = shared_memory.SharedMemory(name=shm_obs_name)
    _worker_alive_shm = shared_memory.SharedMemory(name=shm_alive_name)
    _worker_obs = np.ndarray(obs_shape, dtype=obs_dtype, buffer=_worker_obs_shm.buf)
    _worker_alive = np.ndarray(
        alive_shape, dtype=alive_dtype, buffer=_worker_alive_shm.buf
    )


def _sumo_worker_process_batch(args):
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

    seq_len, n_agents, feature_dim = observations_np.shape

    # Convert edge_indices from list of arrays to numpy array for numba
    edge_indices_array = np.zeros((seq_len, 2, 0), dtype=np.int64)
    if seq_len > 0:
        # Find max edge_num across all time steps
        max_edge_num = 0
        for t in range(seq_len):
            if len(edge_indices_np[t].shape) == 2:
                max_edge_num = max(max_edge_num, edge_indices_np[t].shape[1])

        edge_indices_array = np.zeros((seq_len, 2, max_edge_num), dtype=np.int64)
        for t in range(seq_len):
            edge_array_t = edge_indices_np[t]
            if len(edge_array_t.shape) == 2 and edge_array_t.shape[1] > 0:
                edge_indices_array[t, :, :edge_array_t.shape[1]] = edge_array_t

    result = np.zeros((seq_len, n_agents, max_entities_perception, feature_dim), dtype=observations_np.dtype)
    mask = np.zeros((seq_len, n_agents, max_entities_perception), dtype=bool)

    for agent_idx in range(n_agents):
        # Process each agent using numba-accelerated function
        agent_result, agent_mask = process_single_agent_sumo_numba(
            observations_np, edge_indices_array, alive_mask_np,
            agent_idx, max_entities_perception
        )
        result[:, agent_idx, :, :] = agent_result
        mask[:, agent_idx, :] = agent_mask

    return result, mask


class SumoObsDataConstructor(SelfSupervisedDataConstructor):
    """
    SUMO environment's observation data constructor for self-supervised learning.
    Handles observations with shape (batch_size, seq_len, n_agents, feature_dim).

    In SUMO traffic simulation environment:
    - Agents (traffic lights) don't die, but we still implement alive_mask logic for consistency
    - edge_indices represents a directed graph (source -> target)
    - edge_indices already includes self_loop, so no need to add self-connection
    - Information flows from source agents to target agents
    """

    def __init__(self, max_entities_perception: int, with_time_seq: bool = False, n_workers: int = 0,
                 include_features: Optional[List[int]] = None, exclude_features: Optional[List[int]] = None):
        """
        Initialize the SUMO environment's observation data constructor.

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
        Process the input data to construct self-supervised learning data for SUMO.

        Args:
            observations: Array of shape (batch_size, seq_len, n_agents, feature_dim)
            states: Optional array, not used in this implementation
            edge_indices: List of shape (batch_size, seq_len) containing np.ndarray of shape (2, edge_num)
            alive_mask: Array of shape (batch_size, seq_len, n_agents)

        Returns:
            A tuple containing:
            - Processed array of shape (batch_size, seq_len, n_agents, max_entities_perception, feature_dim)
            - Mask array of shape (batch_size, seq_len, n_agents, max_entities_perception) indicating padding
            - Dimension of seq_len will not present in output if with_time_seq is False.
        """
        # Validate input dimensions
        if len(observations.shape) != 4:
            raise ValueError(f"Expected observations to have 4 dimensions "
                           f"(batch_size, seq_len, n_agents, feature_dim), "
                           f"got {len(observations.shape)}")

        batch_size, seq_len, n_agents, feature_dim = observations.shape

        # Handle with_time_seq flag: slice to last timestep if needed
        if not self.with_time_seq:
            observations = observations[:, [-1], :, :]
            # For edge_indices, take only the last time step for each batch
            edge_indices = [[batch_edge_indices[-1]] for batch_edge_indices in edge_indices]
            alive_mask = alive_mask[:, [-1], :]
            seq_len = 1

        # Prepare result and mask arrays
        result = np.zeros((batch_size, seq_len, n_agents, self.max_entities_perception, feature_dim),
                          dtype=observations.dtype)
        mask = np.zeros((batch_size, seq_len, n_agents, self.max_entities_perception), dtype=bool)

        # Parallel or sequential processing
        if self.n_workers > 1:
            n_workers = min(self.n_workers, batch_size)
            if n_workers >= 1:
                shm_obs, shm_alive, _, _ = _create_shared_memory_arrays(
                    observations, alive_mask
                )
                try:
                    # Divide batch among workers
                    base_samples = batch_size // n_workers
                    extra_samples = batch_size % n_workers

                    worker_args = []
                    for worker_id in range(n_workers):
                        start_idx = worker_id * base_samples + min(worker_id, extra_samples)
                        end_idx = start_idx + base_samples + (1 if worker_id < extra_samples else 0)
                        if start_idx >= batch_size:
                            break
                        batch_indices = list(range(start_idx, end_idx))
                        worker_args.append(
                            (batch_indices, edge_indices, self.max_entities_perception)
                        )

                    mp_ctx = mp.get_context("fork") if _is_linux() else None

                    with ProcessPoolExecutor(
                        max_workers=n_workers,
                        initializer=_sumo_worker_init,
                        initargs=(
                            shm_obs.name,
                            observations.shape,
                            observations.dtype,
                            shm_alive.name,
                            alive_mask.shape,
                            alive_mask.dtype,
                        ),
                        mp_context=mp_ctx,
                    ) as executor:
                        results = list(executor.map(_sumo_worker_process_batch, worker_args))

                    batch_idx = 0
                    for worker_result_list in results:
                        for processed_sample, mask_batch in worker_result_list:
                            result[batch_idx] = processed_sample
                            mask[batch_idx] = mask_batch
                            batch_idx += 1
                finally:
                    _cleanup_shared_memory(shm_obs, shm_alive)
            # else: batch_size == 0, nothing to process; result/mask stay zeros
        else:
            for batch_idx in range(batch_size):
                processed_sample, mask_batch = _process_single_sample((
                    observations[batch_idx],
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