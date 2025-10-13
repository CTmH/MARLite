import numpy as np
from typing import Dict, Any, List
from marlite.algorithm.agents.agent_group import AgentGroup

class MAgentPreyAgentGroup(AgentGroup):
    '''
    agents: Dict[(agent_name(str), model_name(str))]
    model_configs: Dict[model_name(str), ModelConfig]
    '''
    def __init__(self, agents: Dict[str, str]) -> None:
       self.agents = list(agents.keys())

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


class MAgentBattleAgentGroup(AgentGroup):
    '''
    Agent group for the battle environment that handles enemy agent actions.
    Implements attack_8 (8 directions) and move_12 (12 positions within Manhattan distance 3)
    with obstacle avoidance.
    '''
    def __init__(self, agents: Dict[str, str]) -> None:
        self.agents = list(agents.keys())

        # Initialize constant class attributes
        self.do_nothing_action = 0
        self.move_start_idx = 0
        self.attack_start_idx = 12  # 1-based after 12 move actions and do_nothing
        self.max_manhattan_dist = 2
        self.obs_size = 13  # 13x13 observation grid

        # Pre-compute attack offsets (attack_8): from left to right, top to bottom
        self.attack_offsets = np.array([
            (-1, -1), (-1, 0), (-1, 1),  # top row
            (0, -1),           (0, 1),   # middle row (excluding center)
            (1, -1),  (1, 0),  (1, 1)    # bottom row
        ])

        # Pre-compute move offsets (move_12): all positions within Manhattan distance 3
        self.move_offsets = []
        for dx in range(-self.max_manhattan_dist, self.max_manhattan_dist + 1):
            for dy in range(-self.max_manhattan_dist, self.max_manhattan_dist + 1):
                if abs(dx) + abs(dy) <= self.max_manhattan_dist:
                    self.move_offsets.append((dx, dy))

        # Convert to numpy array and sort by y (top to bottom), then x (left to right)
        self.move_offsets = np.array(self.move_offsets)
        sorted_indices = np.lexsort((self.move_offsets[:, 0], self.move_offsets[:, 1]))
        self.move_offsets = self.move_offsets[sorted_indices]

        # Validate move offsets count
        if len(self.move_offsets) != 13:
            raise ValueError(f"Expected 13 move offsets, got {len(self.move_offsets)}")

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
        Select actions for battle environment agents with obstacle avoidance.

        Args:
            observations (dict): Dictionary mapping agent IDs to observation arrays.
                Shape: (T*obs_len*obs_len*F) where F=5 channels
            state (numpy array): Global state information.
            avail_actions (dict): Dictionary mapping agent IDs to action masks or spaces.
            traj_padding_mask (numpy array): Padding mask for trajectory processing.
            alive_agents (list): List of agent IDs that are currently alive.
            epsilon (float): Exploration rate.

        Returns:
            dict: Selected actions for each agent with obstacle avoidance.
                - 'actions': Dictionary mapping only alive agents to their selected actions
                - 'all_actions': Dictionary mapping all agents to their selected actions
        """
        # Extract obstacle map and other team presence from observations
        # Channel 0: obstacle/off the map, Channel 3: other team presence
        obstacle_map = {key: value[-1, :, :, 0] for key, value in observations.items()}
        other_team_presence = {key: value[-1, :, :, 3] for key, value in observations.items()}

        # Combine obstacle and other team presence for pathfinding
        combined_map = {key: obs + other for key, (obs, other) in
                       zip(obstacle_map.keys(), zip(obstacle_map.values(), other_team_presence.values()))}

        # Get tensor of combined maps
        o_tensor = np.stack(list(combined_map.values()))
        batch_size, n, m = o_tensor.shape

        # Center coordinates (agent's current position)
        cx = n // 2
        cy = m // 2

        # For each agent, find the best action considering obstacles
        all_actions = {}

        for i, agent in enumerate(observations.keys()):
            # Get the local observation grid for this agent
            local_grid = o_tensor[i]

            # Find nearby enemies (other team presence > 0)
            enemy_positions = np.argwhere(other_team_presence[agent] > 0)

            # Default action is do_nothing (action 0)
            best_action = self.do_nothing_action
            max_enemy_value = 0

            # Calculate distances to enemies from each attack position
            agent_pos = np.array([cx, cy])

            # Check if we can attack any enemies
            if len(enemy_positions) > 0:

                for j, offset in enumerate(self.attack_offsets):
                    attack_pos = agent_pos + offset

                    # Check if attack position is valid (within bounds)
                    if (0 <= attack_pos[0] < n and 0 <= attack_pos[1] < m):
                        # Check if there's an enemy at this position
                        enemy_at_pos = np.any(np.all(enemy_positions == attack_pos, axis=1))

                        if enemy_at_pos:
                            # This attack action would hit an enemy
                            # In battle environment, attacking an opponent gives reward
                            if 1 > max_enemy_value:
                                max_enemy_value = 1
                                best_action = self.attack_start_idx + j  # attack actions start at index 13

            # If no profitable attack, consider movement
            if max_enemy_value == 0 and len(self.move_offsets) > 0:
                # Find the move that maximizes distance from obstacles/enemies
                # or moves toward enemies if they're nearby

                best_move_value = -1

                for j, offset in enumerate(self.move_offsets):
                    move_pos = agent_pos + offset

                    # Check if move position is valid (within bounds and not blocked)
                    if (0 <= move_pos[0] < n and 0 <= move_pos[1] < m):
                        # Check if the position is blocked by obstacle or another agent
                        if local_grid[move_pos[0], move_pos[1]] == 0:  # Not blocked
                            # Calculate value based on proximity to enemies
                            move_value = 0

                            # Look for enemies near the target position
                            for enemy_pos in enemy_positions:
                                manhattan_dist = abs(move_pos[0] - enemy_pos[0]) + abs(move_pos[1] - enemy_pos[1])
                                if manhattan_dist <= 2:  # Enemy is close
                                    move_value += 1 / (manhattan_dist + 1)  # Closer enemies are more attractive

                            if move_value > best_move_value:
                                best_move_value = move_value
                                best_action = self.move_start_idx + j  # move actions start at index 1

            all_actions[agent] = best_action

        # Return actions only for alive agents
        actual_actions = {agent: all_actions[agent] for agent in alive_agents}

        return {'actions': actual_actions, 'all_actions': all_actions}