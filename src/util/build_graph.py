import numpy as np

def binary_to_decimal(binary_list):
    decimal_number = 0
    length = len(binary_list)
    y = [2**i  for i in range(length)]
    y = np.array(y)
    decimal_number = np.dot(binary_list, y)  # dot product to get decimal number. 
    return decimal_number

def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def build_graph_from_map_2d(A: np.array, threshold: int, metric = manhattan_distance):
    """
    Build the adjacency matrix and edge_index for a graph.
    A: 2D array representing elements on the map. Negative numbers are obstacles, non-negative numbers are valid elements.
    threshold: Connection threshold. An edge exists between two nodes only if their distance is less than or equal to k.
    metric: Function to calculate distance. Default is Manhattan distance.
    """
    # Find all valid elements and their positions
    elements = {i: None for i in range(A.max() + 1)}
    for i in range(len(A)):
        for j in range(len(A[0])):
            if A[i][j] >= 0:
                elements[A[i][j]] = (i, j)
    
    # Initialize adjacency matrix and edge_index
    n = len(elements)
    adj_matrix = np.zeros((n, n), dtype=np.int64)
    edge_index = []
    
    # Calculate Manhattan distance and build adjacency matrix and edge_index
    for i in range(n):
        for j in range(i + 1, n):
            if elements[i] is not None and elements[j] is not None:  # Ensure both are valid elements (not obstacles)
                dist = manhattan_distance(elements[i], elements[j])
                if dist <= threshold:
                    adj_matrix[i][j] = 1
                    adj_matrix[j][i] = 1
                    edge_index.append([i, j])
    
    # Convert edge_index to torch tensor and make it two-dimensional
    edge_index = np.array(edge_index, dtype=np.int64).T
    
    return adj_matrix, edge_index

def build_graph_from_state_with_binary_agent_id(
    state,
    binary_agent_id_dim: list,
    agent_presence_dim: list,
    threshold: int,
    metric = manhattan_distance
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
    
    return build_graph_from_map_2d(agent_positions, threshold, metric)

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