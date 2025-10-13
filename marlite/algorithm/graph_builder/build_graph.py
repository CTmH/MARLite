import numpy as np
from scipy.spatial.distance import cdist

def binary_to_decimal(binary_list):
    decimal_number = 0
    length = len(binary_list)
    y = [2**i  for i in range(length)]
    y = np.array(y)
    decimal_number = np.dot(binary_list, y)  # dot product to get decimal number.
    return int(decimal_number)

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