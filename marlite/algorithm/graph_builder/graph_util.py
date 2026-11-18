import numpy as np
from numba import jit
from typing import Tuple, List
from scipy.spatial.distance import cdist
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

@jit(nopython=True, cache=True)
def binary_to_decimal_numba(binary_array):
    """
    Numba-accelerated function to convert binary array to decimal number.
    """
    decimal_number = 0
    length = len(binary_array)
    for i in range(length):
        decimal_number += binary_array[i] * (2 ** i)
    return decimal_number


def extract_agent_positions_batch(states: np.ndarray,
                                binary_agent_id_dim: list,
                                agent_presence_dim: list) -> List[np.ndarray]:
    """Extract agent positions from batched states using vectorized numpy operations.

    This function efficiently extracts agent positions from batched state tensors
    using numpy's vectorized operations, avoiding explicit loops for better performance.

    Args:
        states: Input states with shape (batch_size, height, width, channels)
        binary_agent_id_dim: List of channel indices that contain binary agent ID
        agent_presence_dim: List of channel indices that indicate agent presence

    Returns:
        List of agent positions arrays with shape (n_agents, 3) where each entry contains
        [agent_id, y_coord, x_coord]. The length of list is batch_size. The n_agents
        dimension is number of present agents.
    """
    batch_size, height, width, channels = states.shape

    # Extract presence channels and determine if agent is present at each position
    # Shape: (batch_size, height, width)
    presence_data = states[:, :, :, agent_presence_dim]
    agent_present = np.any(presence_data > 0, axis=-1)  # Agent present if any presence channel > 0

    # Extract binary ID channels
    # Shape: (batch_size, height, width, len(binary_agent_id_dim))
    binary_id_data = states[:, :, :, binary_agent_id_dim]

    # Create coordinate grids
    # Shape: (height, width)
    y_coords, x_coords = np.mgrid[0:height, 0:width]

    # Tile coordinates for batch processing
    # Shape: (batch_size, height, width)
    y_grid = np.tile(y_coords[None, :, :], (batch_size, 1, 1))
    x_grid = np.tile(x_coords[None, :, :], (batch_size, 1, 1))

    # Reshape data for processing
    # Flatten spatial dimensions for easier indexing
    flat_presence = agent_present.reshape(batch_size, -1)  # (batch_size, height*width)
    flat_binary_ids = binary_id_data.reshape(batch_size, -1, len(binary_agent_id_dim))  # (batch_size, h*w, n_bits)
    flat_y = y_grid.reshape(batch_size, -1)  # (batch_size, h*w)
    flat_x = x_grid.reshape(batch_size, -1)  # (batch_size, h*w)

    # Convert binary IDs to decimal using matrix multiplication
    # Create powers of 2 for binary to decimal conversion
    powers_of_2 = 2 ** np.arange(len(binary_agent_id_dim))  # small-endian
    powers_of_2 = powers_of_2.astype(flat_binary_ids.dtype)

    # Matrix multiply to convert binary to decimal
    # Shape: (batch_size, height*width)
    agent_ids_flat = np.einsum('bij,j->bi', flat_binary_ids, powers_of_2)

    # Filter only positions where agents are present
    # Use boolean indexing to get only valid agent positions
    agent_positions = []
    max_agents = 0

    for b in range(batch_size):
        # Get indices where agents are present
        present_mask = flat_presence[b]
        if np.any(present_mask):
            # Extract data for present agents
            valid_ids = agent_ids_flat[b][present_mask]
            valid_y = flat_y[b][present_mask]
            valid_x = flat_x[b][present_mask]

            # Stack into (n_valid_agents, 3) array [id, y, x]
            positions = np.stack([valid_ids, valid_y, valid_x], axis=1)
            agent_positions.append(positions.astype(int))
            max_agents = max(max_agents, len(valid_ids))
        else:
            # No agents present in this batch item
            agent_positions.append(np.zeros((0, 3), dtype=int))

    return agent_positions

def binary_to_decimal(binary_list):
    # Use the numba-optimized version
    return binary_to_decimal_numba(np.array(binary_list, dtype=np.int64))

def build_communication_graph(
    coords_with_id: np.ndarray,
    comm_distance: float,
    distance_metric: str,
    valid_node_list: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build communication graph based on agent coordinates with IDs and communication distance.

    Args:
        coords_with_id: Array of agent positions with shape (n_agents, 3), each row is [agent_id, y_coord, x_coord]
        comm_distance: Communication distance threshold
        distance_metric: Distance metric for calculating distances between agents
        valid_node_list: List of valid node IDs to include in the result graph

    Returns:
        Tuple of (adjacency_matrix, edge_index)
        - adjacency_matrix: Binary matrix of shape (max_node_id+1, max_node_id+1), zero-padded for missing nodes
        - edge_index: Edge index array of shape (2, num_edges), using original agent IDs
    """
    if len(coords_with_id) == 0 or not valid_node_list:
        max_node_id = max(valid_node_list) if valid_node_list else 0
        adj_matrix = np.zeros((max_node_id + 1, max_node_id + 1), dtype=np.int8)
        edge_index = np.zeros((2, 0), dtype=np.int64)
        return adj_matrix, edge_index

    max_node_id = max(valid_node_list)

    # Filter coords: only keep rows where agent_id is in valid_node_list
    mask = np.isin(coords_with_id[:, 0].astype(int), np.array(valid_node_list, dtype=int))
    filtered_coords_with_id = coords_with_id[mask]

    # Extract agent IDs and coordinates
    agent_ids = filtered_coords_with_id[:, 0].astype(int)
    coords = filtered_coords_with_id[:, 1:3]  # shape: (num_valid_agents, 2)

    if len(coords) == 0:
        adj_matrix = np.zeros((max_node_id + 1, max_node_id + 1), dtype=np.int8)
        edge_index = np.zeros((2, 0), dtype=np.int64)
        return adj_matrix, edge_index

    # Calculate pairwise distance matrix
    distances = cdist(coords, coords, metric=distance_metric)
    # Create upper triangular mask, excluding diagonal (no self-loops)
    mask = (distances <= comm_distance) & np.triu(np.ones_like(distances, dtype=bool), k=1)
    rows, cols = np.where(mask)

    # Map local indices to original agent IDs
    node_ids = agent_ids  # local index i corresponds to agent id = node_ids[i]
    edge_index = np.vstack([node_ids[rows], node_ids[cols]]).astype(int)

    # Build adjacency matrix with actual distances (symmetric)
    adj_matrix = np.zeros((max_node_id + 1, max_node_id + 1), dtype=np.float32)
    # Fill both directions (symmetric matrix)
    adj_matrix[edge_index[0], edge_index[1]] = distances[rows, cols]
    adj_matrix[edge_index[1], edge_index[0]] = distances[rows, cols]

    return adj_matrix, edge_index

def build_partial_graph(
    coords_with_id: np.ndarray,
    comm_distance: int,
    distance_metric: str,
    n_subgraphs: int,
    valid_node_list: List[int],
    target_node_list: List[int]
):
    """
    Process a single batch item to build partial graph with community detection.

    Args:
        coords_with_id: Array of agent positions with shape (n_agents, 3),
                        each row is [agent_id, y_coord, x_coord].
        comm_distance: Communication distance threshold.
        distance_metric: Distance metric for calculating distances between agents.
        n_subgraphs: Number of subgraphs for community detection.
        valid_node_list: List of valid node IDs to include in the result graph.
        target_node_list: List of target node IDs to include in the initial graph.

    Returns:
        Tuple of (adjacency_matrix, edge_index)
    """
    # Build full communication graph (with distances in adj_matrix)
    adj_matrix_full, edge_index_full = build_communication_graph(
        coords_with_id=coords_with_id,
        comm_distance=comm_distance,
        distance_metric=distance_metric,
        valid_node_list=valid_node_list + target_node_list
    )
    max_node_id = max(valid_node_list)

    if edge_index_full.shape[1] <= 0:
        return np.zeros((max_node_id + 1, max_node_id + 1), dtype=np.float32), np.zeros((2, 0), dtype=np.int64)

    # Extract valid indices used in the graph
    valid_indices = np.array(valid_node_list + target_node_list)

    # Build networkx graph using existing distances from adj_matrix_full
    G = nx.Graph()
    edges_with_weight = []
    for u, v in edge_index_full.T.astype(int):
        dist = adj_matrix_full[u, v]
        if dist > 0:  # Should always be true, but safe check
            edges_with_weight.append((u, v, 1.0 / dist)) # shorter distance → higher weight

    G.add_weighted_edges_from(edges_with_weight)

    # Run greedy modularity community detection
    communities = list(greedy_modularity_communities(
        G=G,
        best_n=n_subgraphs,
        weight='weight'
    ))

    # Map nodes to their community
    node_community_map = {}
    for comm_id, comm in enumerate(communities):
        for node in comm:
            node_community_map[node] = comm_id

    # Filter edges: only keep those within same community AND both endpoints in valid_node_list
    filtered_edges = []
    valid_node_set = set(valid_node_list)
    for u, v in edge_index_full.T.astype(int):
        if (u in valid_node_set and v in valid_node_set and
            node_community_map.get(u) == node_community_map.get(v)):
            filtered_edges.append([u, v])

    # Rebuild binary or distance-based adjacency matrix from filtered edges
    adj_matrix = np.zeros((max_node_id + 1, max_node_id + 1), dtype=np.float32)
    edge_index = np.array(filtered_edges, dtype=int).T if filtered_edges else np.zeros((2, 0), dtype=int)

    if edge_index.size > 0:
        # Fill symmetric adjacency matrix with actual distances from original graph
        adj_matrix[edge_index[0], edge_index[1]] = adj_matrix_full[edge_index[0], edge_index[1]]
        adj_matrix[edge_index[1], edge_index[0]] = adj_matrix_full[edge_index[1], edge_index[0]]

    return adj_matrix, edge_index

def build_partial_groups(
    coords_with_id: np.ndarray,
    comm_distance: int,
    distance_metric: str,
    n_groups: int,
    valid_node_list: List[int],
    target_node_list: List[int],
) -> np.ndarray:
    """
    Build group labels based on community (subgraph) detection.

    Mirrors the subgraph partition produced by ``build_partial_graph``: the
    full communication graph is built from ``valid_node_list + target_node_list``
    and then partitioned into communities via ``greedy_modularity_communities``.
    Each community's node set is treated as a single communication group, and
    the returned label for a valid node is the index of its community.

    Only ``valid_node_list`` agents receive output labels. Agents that are not
    present in the state, or that are excluded from any community, are labeled
    ``-1``. The returned labels are always a consecutive prefix
    ``0, 1, ..., n_actual-1`` of the non-negative integers, where
    ``n_actual`` is the number of *non-empty* communities. Communities that
    contain only target nodes are skipped, so when the actual number of
    non-empty communities is fewer than ``n_groups`` the labels are still
    ``0, 1, 2, ...`` (smaller group numbers first).

    Isolated agents (no other agent within ``comm_distance``) are each placed
    in their own community, so they receive distinct labels.

    Args:
        coords_with_id: Array of agent positions with shape (n_agents, 3),
                        each row is [agent_id, y_coord, x_coord].
        comm_distance: Communication distance threshold.
        distance_metric: Distance metric for calculating distances between agents.
        n_groups: Target number of groups/communities for greedy modularity detection.
        valid_node_list: List of valid node IDs to include in the output labels.
        target_node_list: List of target node IDs to include in the community detection
                         but not in the output group labels.

    Returns:
        group_labels: Array of shape (len(valid_node_list),) with community IDs.
                      ``-1`` indicates the agent is not present or has no group assignment.
    """
    n_valid = len(valid_node_list)
    full_labels = np.full(n_valid, -1, dtype=np.int64)

    # Build full communication graph (valid + target nodes)
    adj_matrix_full, edge_index_full = build_communication_graph(
        coords_with_id=coords_with_id,
        comm_distance=comm_distance,
        distance_metric=distance_metric,
        valid_node_list=valid_node_list + target_node_list
    )

    valid_node_set = set(valid_node_list)
    valid_node_to_idx = {node_id: idx for idx, node_id in enumerate(valid_node_list)}

    if n_groups <= 1:
        return full_labels

    # Build networkx graph. We always seed it with the present nodes so that
    # isolated agents are still partitioned (each becomes its own community).
    all_node_set = set(valid_node_list) | set(target_node_list)
    present_nodes = {n for n in all_node_set if n in valid_node_set or
                     n in set(target_node_list)}
    # Use the adj matrix to determine which nodes are actually present:
    # a node is present if any of its rows/cols in the full adj matrix has
    # an entry, OR if it appears in coords_with_id.
    present_in_state = set(coords_with_id[:, 0].astype(int).tolist()) if len(coords_with_id) > 0 else set()
    present_nodes = present_in_state & all_node_set

    if not present_nodes:
        return full_labels

    G = nx.Graph()
    G.add_nodes_from(present_nodes)

    for u, v in edge_index_full.T.astype(int):
        if u in present_nodes and v in present_nodes:
            dist = adj_matrix_full[u, v]
            if dist > 0:
                G.add_edge(u, v, weight=1.0 / dist)  # shorter distance → higher weight

    # Clamp best_n to the number of nodes in the graph (networkx constraint).
    best_n = min(n_groups, G.number_of_nodes())
    if best_n <= 1:
        # Cannot form more than one group; assign each present valid node to
        # its own community so the result is still well-defined.
        for node in present_nodes:
            if node in valid_node_set:
                # Will be remapped below to consecutive labels
                full_labels[valid_node_to_idx[node]] = node  # temporary unique id
        # Remap to consecutive 0, 1, 2, ...
        used = sorted({int(v) for v in full_labels if v >= 0})
        remap = {old: new for new, old in enumerate(used)}
        for i in range(len(full_labels)):
            if full_labels[i] >= 0:
                full_labels[i] = remap[int(full_labels[i])]
        return full_labels

    # Run greedy modularity community detection
    communities = list(greedy_modularity_communities(
        G=G,
        best_n=best_n,
        weight='weight'
    ))

    # Assign labels 0, 1, 2, ... in the order communities are returned.
    for comm_id, comm in enumerate(communities):
        for node in comm:
            if node in valid_node_set:
                full_labels[valid_node_to_idx[node]] = comm_id

    # Remap labels to be consecutive from 0, skipping communities that have
    # no valid node (they only contain target nodes). This guarantees that
    # the returned label set is always {0, 1, ..., n_actual-1} where
    # n_actual is the number of *non-empty* communities — the
    # "smaller group numbers first" rule.
    used_comm_ids = []
    seen = set()
    for label in full_labels:
        if label >= 0 and int(label) not in seen:
            used_comm_ids.append(int(label))
            seen.add(int(label))
    remap = {old: new for new, old in enumerate(used_comm_ids)}
    for i in range(len(full_labels)):
        if full_labels[i] >= 0:
            full_labels[i] = remap[int(full_labels[i])]

    return full_labels