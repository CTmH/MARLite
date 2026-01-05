import numpy as np
from typing import Optional
from concurrent.futures import ProcessPoolExecutor
from marlite.util.self_supervised_data_constructor.self_supervised_data_constructor import SelfSupervisedDataConstructor


class MagentVecObsDataConstructor(SelfSupervisedDataConstructor):
    """
    Magent environment's vector observation data constructor for self-supervised learning.
    """

    def __init__(self, max_entities_perception: int, max_observed_entities: int, n_workers: int = 0):
        """
        Initialize the Magent environment's vector observation data constructor.

        Args:
            max_entities_perception: Maximum number of entities each agent can perceive
            n_workers: Number of worker processes for parallel processing
        """
        super().__init__(n_workers=n_workers)
        self.max_entities_perception = max_entities_perception
        self.max_observed_entities = max_observed_entities

    def process(self, observations: np.ndarray, states: Optional[np.ndarray],
                edge_indices: np.ndarray, alive_mask: np.ndarray) -> np.ndarray:
        """
        Process the input data to construct self-supervised learning data.

        Args:
            observations: Array of shape (batch_size, n_agents, max_observed_entities, feature_dim)
            states: Optional array, not used in this implementation
            edge_indices: Array of shape (batch_size, 2, edge_num)
            alive_mask: Array of shape (batch_size, n_agents)
            max_observed_entities: Number of entities the agent can see (max entities to perceive from communication)

        Returns:
            Processed array of shape (batch_size, n_agents, max_entities_perception, feature_dim)
        """
        batch_size, n_agents, _, feature_dim = observations.shape

        # Prepare result array
        result = np.zeros((batch_size, n_agents, self.max_entities_perception, feature_dim),
                         dtype=observations.dtype)

        if self.n_workers > 0:
            # Use process pool for parallel processing
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                # Prepare arguments for each batch
                args_list = [
                    (
                        observations[i],
                        edge_indices[i],
                        alive_mask[i],
                        self.max_observed_entities,
                        self.max_entities_perception,
                        feature_dim
                    )
                    for i in range(batch_size)
                ]

                # Execute in parallel
                results = list(executor.map(self._process_single_batch, args_list))

                # Collect results
                for i, processed_batch in enumerate(results):
                    result[i] = processed_batch
        else:
            # Process sequentially
            for batch_idx in range(batch_size):
                result[batch_idx] = self._process_single_batch((
                    observations[batch_idx],
                    edge_indices[batch_idx],
                    alive_mask[batch_idx],
                    self.max_observed_entities,
                    self.max_entities_perception,
                    feature_dim
                ))

        return result

    def _process_single_batch(self, args):
        """
        Process a single batch of data (used for parallel processing).

        Args:
            args: Tuple containing (observations, edge_indices, alive_mask, max_observed_entities,
                    max_entities_perception, feature_dim)

        Returns:
            Processed numpy array of shape (n_agents, max_entities_perception, feature_dim)
        """
        observations_np, edge_indices_np, alive_mask_np, max_observed_entities, max_entities_perception, feature_dim = args

        n_agents = observations_np.shape[0]

        # Prepare result for this batch
        result = np.zeros((n_agents, max_entities_perception, feature_dim), dtype=observations_np.dtype)

        # Convert alive mask to boolean indices
        alive_agents = np.where(alive_mask_np)[0]

        # Filter edges to only include those between alive agents
        source_agents = edge_indices_np[0]  # (edge_num,)
        target_agents = edge_indices_np[1]  # (edge_num,)

        # Create mask for edges where both source and target are alive
        alive_source_mask = np.isin(source_agents, alive_agents)
        alive_target_mask = np.isin(target_agents, alive_agents)
        valid_edge_mask = alive_source_mask & alive_target_mask

        # Filter the edge indices to only include valid edges
        filtered_edge_indices = edge_indices_np[:, valid_edge_mask]  # (2, num_valid_edges)

        # Process each alive agent
        for agent_idx in alive_agents:
            # Step 1: Get ALL observations from this agent (max_observed_entities entities)
            # Shape: (max_observed_entities, feature_dim)
            self_obs = observations_np[agent_idx]  # (max_observed_entities, feature_dim)

            # Step 2: Get observations from agents that send TO this agent (in-edges only)
            # Use filtered edge indices to ensure only alive agents are considered
            target_edges = filtered_edge_indices[1]  # (num_valid_edges,)

            # Find all edges where this agent is the TARGET (i.e., j -> agent_idx)
            in_edge_mask = target_edges == agent_idx

            # Get the SOURCE agents of those in-edges (i.e., all j such that j -> agent_idx)
            in_edge_sources = filtered_edge_indices[0][in_edge_mask]  # (num_in_edges,)

            # Remove duplicates and exclude self (though self-loop j==i would be harmless, but not typical)
            neighbor_agents = np.unique(in_edge_sources)
            # Optional: remove self if self-loop exists (rare in comms graphs)
            neighbor_agents = neighbor_agents[neighbor_agents != agent_idx]

            # Collect all entity vectors: self + neighbors' full observations
            all_entity_vectors = []

            # Add self's max_observed_entities entities
            for k in range(max_observed_entities):
                all_entity_vectors.append(self_obs[k])  # (feature_dim,)

            # Add each neighbor's max_observed_entities entities
            for neighbor_idx in neighbor_agents:
                neighbor_obs = observations_np[neighbor_idx]  # (max_observed_entities, feature_dim)
                for k in range(max_observed_entities):
                    all_entity_vectors.append(neighbor_obs[k])

            # Convert to array: (total_entities, feature_dim)
            if all_entity_vectors:
                all_entities = np.stack(all_entity_vectors, axis=0)  # (N, feature_dim)
            else:
                # fallback: only self observation
                all_entities = self_obs  # (max_observed_entities, feature_dim)

            # Step 3: Remove zero-filled vectors (padding)
            non_zero_mask = ~np.all(all_entities == 0, axis=1)
            clean_entities = all_entities[non_zero_mask]

            # Step 4: Remove duplicates (exact row matches)
            if clean_entities.shape[0] > 0:
                _, unique_indices = np.unique(clean_entities, axis=0, return_index=True)
                unique_entities = clean_entities[np.sort(unique_indices)]
            else:
                unique_entities = clean_entities

            # Step 5: Truncate or pad to max_entities_perception
            num_unique = unique_entities.shape[0]
            if num_unique > max_entities_perception:
                final_entities = unique_entities[:max_entities_perception]
            else:
                final_entities = np.pad(
                    unique_entities,
                    ((0, max_entities_perception - num_unique), (0, 0)),
                    mode='constant',
                    constant_values=0
                )

            result[agent_idx] = final_entities

        return result