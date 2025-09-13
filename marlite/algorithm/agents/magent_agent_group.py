import numpy as np
from typing import Dict, Any, List
from marlite.algorithm.agents.agent_group import AgentGroup

class MagentPreyAgentGroup(AgentGroup):
    '''
    agents: Dict[(agent_name(str), model_name(str))]
    model_configs: Dict[model_name(str), ModelConfig]
    '''
    def __init__(self, agents: Dict[str, str]) -> None:
        self.agents = agents

    def act(
            self,
            observations: Dict[str, np.ndarray],
            state: np.ndarray,
            avail_actions: Dict[str, Any],
            traj_padding_mask: np.ndarray,
            alive_agents: List[str],
            epsilon: float = .0
        ) -> Dict[str, Any]:
        """
        Select actions based on Q-values and exploration with action masking.

        Args:
            observations (dict): Dictionary mapping agent IDs to observation arrays.
                Each observation array should have shape compatible with the agent's observation space.
            state (numpy array): Global state information for generating communication graph.
            avail_actions (dict): Dictionary mapping agent IDs to either action masks (numpy arrays)
                                or action spaces (gymnasium.spaces.Space). Each mask is a 1D array where 1
                                indicates available actions, and 0 indicates unavailable actions.
            traj_padding_mask (numpy array): Padding mask for trajectory processing.
                This is used to handle variable-length trajectories by indicating which positions
                contain valid data vs padding.
            alive_agents (list): List of agent IDs that are currently alive/active in the environment.
                Only these agents will have their actions returned in the output.
            epsilon (float): Exploration rate.
                - 0.0: Always choose optimal actions (greedy)
                - 1.0: Always choose random actions (pure exploration)
                - Values between 0.0 and 1.0: Mix of exploration and exploitation

        Returns:
            dict: Selected actions for each agent, with action mask applied, and edge indices.
                - 'actions': Dictionary mapping only alive agents to their selected actions
                - 'all_actions': Dictionary mapping all agents to their selected actions (including dead ones)
        """
        obstacle_and_other_team_presence = {key: value[-1,:,:,0] + value[-1,:,:,3] for key, value in observations.items()} # value: (T*obs_len*obs_len*F)

        # (num_agents, n, n)
        o_tensor = np.stack(list(obstacle_and_other_team_presence.values()))
        batch_size, n, m = o_tensor.shape
        # Centre Coord
        cx = n // 2
        cy = m // 2

        offsets = np.array([(-1,-1), (-1,0), (-1,1),
                        (0,-1),  (0,0),  (0,1),
                        (1,-1),  (1,0),  (1,1)])

        # Get all agent coordinates (batch_size, m, 2)
        points = [np.argwhere(o > 0) for o in o_tensor]

        # Vectorized calculation of Manhattan distance sum
        grid_coords = offsets + (cx, cy)
        dist_sums = np.zeros((batch_size, 9))

        for i in range(batch_size):
            if len(points[i]) > 0:
                diff = np.abs(grid_coords[:, None] - points[i][None]) # grid_coords shape changes to (9, 1, 2), points shape changes to (1, N, 2)
                # Calculate the distance from each grid point to all targets
                dist_sums[i] = np.sum(np.sum(diff, axis=-1), axis=-1)

        # Find the index of the position with the maximum distance sum
        max_indices = np.argmax(dist_sums, axis=1)

        # Construct action dictionary
        all_actions = {agent: max_indices[i]
                for i, agent in enumerate(obstacle_and_other_team_presence.keys())}

        actual_actions = {agent: all_actions[agent] for agent in alive_agents}

        return {'actions': actual_actions, 'all_actions': all_actions}