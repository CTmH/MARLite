import numpy as np
import networkx as nx
from typing import Tuple, List
from networkx.algorithms.community import greedy_modularity_communities
from scipy.spatial.distance import cdist

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
'''
    # Pad all agent position arrays to the same length
    padded_positions = []
    for pos in agent_positions:
        if len(pos) < max_agents:
            padding = np.zeros((max_agents - len(pos), 3))
            pos = np.vstack([pos, padding])
        padded_positions.append(pos)

    # Stack into final array
    result = np.stack(padded_positions)  # (batch_size, max_agents, 3)

    return result
'''

def binary_to_decimal(binary_list):
    decimal_number = 0
    length = len(binary_list)
    y = [2**i  for i in range(length)]
    y = np.array(y)
    decimal_number = np.dot(binary_list, y)  # dot product to get decimal number.
    return int(decimal_number)


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


'''
def build_graph_from_map_2d(A: np.array, threshold: int, distance_metric: str = 'cityblock'):
    """
    Build the adjacency matrix and edge_index for a graph.
    A: 2D array representing elements on the map. Negative numbers are obstacles, non-negative numbers are valid elements.
    threshold: Connection threshold. An edge exists between two nodes only if their distance is less than or equal to k.
    """
    # Collect valid element coordinates (filtering obstacles)
    valid_elements = {k: v for k, v in [(val, (i, j)) for i, row in enumerate(A) for j, val in enumerate(row) if val >= 0]}
    sorted_ids = sorted(valid_elements.keys())
    sorted_ids = np.array(sorted_ids, dtype=np.int64)
    coords = np.array([valid_elements[k] for k in sorted_ids], dtype=np.int64)

    # Vectorized Manhattan distance calculation
    distances = cdist(coords, coords, metric=distance_metric)

    # Generate valid edge pairs within threshold (upper triangular matrix)
    mask = (distances <= threshold) & (np.triu(np.ones_like(distances, dtype=bool), k=1))
    rows, cols = np.where(mask)

    # Build adjacency matrix
    max_id = A.max() if A.size > 0 else -1
    n = max_id + 1 if max_id >= 0 else 0
    adj_matrix = np.zeros((n, n), dtype=np.int64)
    adj_matrix[sorted_ids[rows], sorted_ids[cols]] = 1
    adj_matrix[sorted_ids[cols], sorted_ids[rows]] = 1  # Symmetric connections

    # Generate edge_index in COO format
    edge_indices = np.vstack([sorted_ids[rows], sorted_ids[cols]]).astype(np.int64)

    return adj_matrix, edge_indices

def build_graph_from_state_with_binary_agent_id(
    state,
    binary_agent_id_dim: list,
    agent_presence_dim: list,
    threshold: int,
    distance_metric: str = 'cityblock'
):
    """
    Construct a graph from observations.

    Args:
        state (Tensor): Observations tensor of shape (Batch Size, Agent Number, Time Step, Feature Dimensions).

    Returns:
        Data: A PyG Data object representing the graph.
    """
    binary_agent_id = state[:, :, binary_agent_id_dim]
    agent_positions = np.apply_along_axis(binary_to_decimal, -1, binary_agent_id).astype(np.int64)
    agent_presence = state[:, :, agent_presence_dim]  # Get the state with agent presence.
    agent_presence = agent_presence.astype(np.int64)
    agent_presence = agent_presence.sum(axis=-1)  # Sum across the last dimension to get a single value per team.
    agent_positions = agent_positions + agent_presence - 1  # Adjust agent positions based on presence, -1 means no agent is present.

    return build_graph_from_map_2d(agent_positions, threshold, distance_metric)

def filter_edge_index(edge_index: list, node_ids: list):
    node_ids_set = set(node_ids)
    edge_index = np.array(edge_index).astype(np.int64)  # Convert to numpy array for easier manipulation.
    edge_index = edge_index.T  # Transpose to make it easier to filter. Each row is an edge.
    filtered_edge_index = []
    for edge in edge_index:
        if edge[0] in node_ids_set and edge[1] in node_ids_set:
            filtered_edge_index.append(edge)
    filtered_edge_index = np.array(filtered_edge_index).T.astype(np.int64)  # Convert back to original shape. Each column is an edge.
    return filtered_edge_index

def build_team_graph_batch(states, team_indices, binary_agent_id_dim,
                          agent_presence_dim, comm_distance, distance_metric: str = 'cityblock'):
    adj_batch = []
    edge_batch = []
    for state in states:
        adj, edges = build_graph_from_state_with_binary_agent_id(
            state, binary_agent_id_dim,
            agent_presence_dim, comm_distance,
            distance_metric
        )
        team_adj = adj[team_indices][:, team_indices]
        team_edges = filter_edge_index(edges, team_indices)
        adj_batch.append(team_adj)
        edge_batch.append(team_edges)
    return np.stack(adj_batch), edge_batch

'''