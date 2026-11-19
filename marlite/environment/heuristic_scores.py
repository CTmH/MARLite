"""Memoryless native battle scores, independent of trainers and configuration.

Movement uses native indices 0..12 (center at 6); attacks use 13..20.
The geometric scoring rules are preserved from the previous implementation.
"""

from typing import Dict, Any, List
import numpy as np


class BattleScores:
    def __init__(self, strategy="advanced", temperature=1.0, top_k=8):
        self.strategy, self.temperature, self.top_k = strategy, temperature, top_k
        self.do_nothing_action = 6
        self.move_start_idx, self.attack_start_idx = 0, 13
        self.max_manhattan_dist, self.obs_size, self._total_actions = 2, 13, 21
        self.attack_offsets = np.array(
            [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        )
        offsets = np.array(
            [(x, y) for x in range(-2, 3) for y in range(-2, 3) if abs(x) + abs(y) <= 2]
        )
        self.move_offsets = offsets[np.lexsort((offsets[:, 0], offsets[:, 1]))]

    def basic_strategy(
        self, observations: Dict[str, np.ndarray], alive_agents: List[str]
    ) -> Dict[str, Any]:
        """
        Basic strategy: Attack when possible, otherwise move toward enemies.
        This is the original strategy from the current implementation.

        Args:
            observations (dict): Dictionary mapping agent IDs to observation arrays.
            alive_agents (list): List of agent IDs that are currently alive.

        Returns:
            dict: Selected actions for each agent.
        """
        # Extract obstacle map and other team presence from observations
        obstacle_map = {key: value[-1, :, :, 0] for key, value in observations.items()}
        other_team_presence = {key: value[-1, :, :, 3] for key, value in observations.items()}

        # Combine obstacle and other team presence for pathfinding
        combined_map = {
            key: obs + other
            for key, (obs, other) in zip(
                obstacle_map.keys(), zip(obstacle_map.values(), other_team_presence.values())
            )
        }

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
                    if 0 <= attack_pos[0] < n and 0 <= attack_pos[1] < m:
                        # Check if there's an enemy at this position
                        enemy_at_pos = np.any(np.all(enemy_positions == attack_pos, axis=1))

                        if enemy_at_pos:
                            # This attack action would hit an enemy
                            if 1 > max_enemy_value:
                                max_enemy_value = 1
                                best_action = (
                                    self.attack_start_idx + j
                                )  # attack actions start at index 13

            # If no profitable attack, consider movement
            if max_enemy_value == 0 and len(self.move_offsets) > 0:
                # Find the move that maximizes distance from obstacles/enemies
                # or moves toward enemies if they're nearby

                best_move_value = -1

                for j, offset in enumerate(self.move_offsets):
                    move_pos = agent_pos + offset

                    # Check if move position is valid (within bounds and not blocked)
                    if 0 <= move_pos[0] < n and 0 <= move_pos[1] < m:
                        # Check if the position is blocked by obstacle or another agent
                        if local_grid[move_pos[0], move_pos[1]] == 0:  # Not blocked
                            # Calculate value based on proximity to enemies
                            move_value = 0

                            # Look for enemies near the target position
                            for enemy_pos in enemy_positions:
                                manhattan_dist = abs(move_pos[0] - enemy_pos[0]) + abs(
                                    move_pos[1] - enemy_pos[1]
                                )
                                if manhattan_dist <= 2:  # Enemy is close
                                    move_value += 1 / (
                                        manhattan_dist + 1
                                    )  # Closer enemies are more attractive

                            if move_value > best_move_value:
                                best_move_value = move_value
                                best_action = (
                                    self.move_start_idx + j
                                )  # move actions start at index 1

            all_actions[agent] = best_action

        return all_actions

    def advanced_strategy(
        self, observations: Dict[str, np.ndarray], alive_agents: List[str]
    ) -> Dict[str, Any]:
        """
        Advanced strategy: Tactical decision making with HP management,
        coordinated attacks, and strategic positioning.

        Args:
            observations (dict): Dictionary mapping agent IDs to observation arrays.
            alive_agents (list): List of agent IDs that are currently alive.

        Returns:
            dict: Selected actions for each agent.
        """
        # Extract all relevant observation channels
        obstacle_map = {key: value[-1, :, :, 0] for key, value in observations.items()}
        my_team_presence = {key: value[-1, :, :, 1] for key, value in observations.items()}
        my_team_hp = {key: value[-1, :, :, 2] for key, value in observations.items()}
        other_team_presence = {key: value[-1, :, :, 3] for key, value in observations.items()}
        other_team_hp = {key: value[-1, :, :, 4] for key, value in observations.items()}

        # Get tensor of combined maps for pathfinding
        combined_map = {
            key: obs + other
            for key, (obs, other) in zip(
                obstacle_map.keys(), zip(obstacle_map.values(), other_team_presence.values())
            )
        }
        o_tensor = np.stack(list(combined_map.values()))
        batch_size, n, m = o_tensor.shape

        # Center coordinates (agent's current position)
        cx = n // 2
        cy = m // 2

        all_actions = {}

        for i, agent in enumerate(observations.keys()):
            local_grid = o_tensor[i]
            agent_pos = np.array([cx, cy])

            # Get agent's current HP (from center position)
            current_hp = my_team_hp[agent][cx, cy]

            # Find nearby enemies and allies
            enemy_positions = np.argwhere(other_team_presence[agent] > 0)
            ally_positions = np.argwhere(my_team_presence[agent] > 0)

            # Remove self from ally positions
            ally_positions = ally_positions[~np.all(ally_positions == agent_pos, axis=1)]

            # Calculate tactical metrics
            enemy_count = len(enemy_positions)
            ally_count = len(ally_positions)
            enemy_hp_sum = (
                np.sum(other_team_hp[agent][other_team_presence[agent] > 0])
                if enemy_count > 0
                else 0
            )
            ally_hp_sum = (
                np.sum(my_team_hp[agent][my_team_presence[agent] > 0]) if ally_count > 0 else 0
            )

            # Strategy decision tree
            best_action = self.do_nothing_action

            # 1. HP Management: Retreat if low HP and outnumbered
            if current_hp < 3 and (
                enemy_count > ally_count + 1 or enemy_hp_sum > ally_hp_sum * 1.5
            ):
                # Find safest retreat position (away from enemies)
                best_retreat_value = -float("inf")

                for j, offset in enumerate(self.move_offsets):
                    move_pos = agent_pos + offset

                    if (
                        0 <= move_pos[0] < n
                        and 0 <= move_pos[1] < m
                        and local_grid[move_pos[0], move_pos[1]] == 0
                    ):
                        # Calculate safety score (distance from enemies)
                        safety_score = 0
                        for enemy_pos in enemy_positions:
                            dist = abs(move_pos[0] - enemy_pos[0]) + abs(move_pos[1] - enemy_pos[1])
                            safety_score += dist  # Higher distance = safer

                        # Prefer positions near allies for protection
                        ally_proximity = 0
                        for ally_pos in ally_positions:
                            dist = abs(move_pos[0] - ally_pos[0]) + abs(move_pos[1] - ally_pos[1])
                            if dist <= 2:
                                ally_proximity += 1 / (dist + 1)

                        total_value = safety_score + ally_proximity * 5

                        if total_value > best_retreat_value:
                            best_retreat_value = total_value
                            best_action = self.move_start_idx + j

            # 2. Coordinated Attack: Attack if we have numerical advantage
            elif enemy_count > 0 and (ally_count >= enemy_count or current_hp > 5):
                best_attack_value = -1

                for j, offset in enumerate(self.attack_offsets):
                    attack_pos = agent_pos + offset

                    if 0 <= attack_pos[0] < n and 0 <= attack_pos[1] < m:
                        enemy_at_pos = np.any(np.all(enemy_positions == attack_pos, axis=1))

                        if enemy_at_pos:
                            # Calculate attack value based on enemy HP and ally support
                            enemy_hp = other_team_hp[agent][attack_pos[0], attack_pos[1]]
                            attack_value = 10 - enemy_hp  # Prefer low HP enemies

                            # Bonus for having allies nearby to support
                            ally_support = 0
                            for ally_pos in ally_positions:
                                dist = abs(attack_pos[0] - ally_pos[0]) + abs(
                                    attack_pos[1] - ally_pos[1]
                                )
                                if dist <= 2:
                                    ally_support += 1

                            attack_value += ally_support * 2

                            if attack_value > best_attack_value:
                                best_attack_value = attack_value
                                best_action = self.attack_start_idx + j

            # 3. Strategic Positioning: Move to better tactical positions
            if best_action == self.do_nothing_action and len(self.move_offsets) > 0:
                best_position_value = -float("inf")

                for j, offset in enumerate(self.move_offsets):
                    move_pos = agent_pos + offset

                    if (
                        0 <= move_pos[0] < n
                        and 0 <= move_pos[1] < m
                        and local_grid[move_pos[0], move_pos[1]] == 0
                    ):
                        position_value = 0

                        # Value based on enemy proximity (but not too close)
                        for enemy_pos in enemy_positions:
                            dist = abs(move_pos[0] - enemy_pos[0]) + abs(move_pos[1] - enemy_pos[1])
                            if 2 <= dist <= 4:  # Optimal attack range
                                position_value += 3
                            elif dist < 2:  # Too close
                                position_value -= 2

                        # Value based on ally support
                        for ally_pos in ally_positions:
                            dist = abs(move_pos[0] - ally_pos[0]) + abs(move_pos[1] - ally_pos[1])
                            if dist <= 2:  # Good for coordination
                                position_value += 2

                        # Value based on map control (center positions)
                        center_dist = abs(move_pos[0] - cx) + abs(move_pos[1] - cy)
                        position_value += (n - center_dist) * 0.5  # Prefer center

                        if position_value > best_position_value:
                            best_position_value = position_value
                            best_action = self.move_start_idx + j

            all_actions[agent] = best_action

        return all_actions

    def probability_strategy(
        self, observations: Dict[str, np.ndarray], alive_agents: List[str]
    ) -> Dict[str, int]:
        """
        Probability-based strategy: Computes action values for all 21 actions
        using the advanced strategy's scoring logic (without phase gating),
        then samples from a softmax distribution over the top-K feasible actions.

        Unlike the deterministic advanced strategy which uses conditional
        phase gating (retreat when low HP, attack when advantage, etc.),
        this strategy evaluates all actions simultaneously and samples
        proportionally to their scores, providing smoother exploration.

        Args:
            observations (dict): Dictionary mapping agent IDs to observation arrays.
            alive_agents (list): List of agent IDs that are currently alive.

        Returns:
            tuple: (actual_actions dict, all_actions dict).
        """
        # Extract observation channels: obstacle, team presence, and HP
        obstacle_map = {key: value[-1, :, :, 0] for key, value in observations.items()}
        my_team_presence = {key: value[-1, :, :, 1] for key, value in observations.items()}
        other_team_presence = {key: value[-1, :, :, 3] for key, value in observations.items()}
        other_team_hp = {key: value[-1, :, :, 4] for key, value in observations.items()}

        # Combine obstacle and enemy presence for pathfinding
        combined_map = {
            key: obs + other
            for key, (obs, other) in zip(
                obstacle_map.keys(), zip(obstacle_map.values(), other_team_presence.values())
            )
        }
        o_tensor = np.stack(list(combined_map.values()))
        batch_size, n, m = o_tensor.shape

        cx = n // 2
        cy = m // 2

        all_actions = {}

        for i, agent in enumerate(observations.keys()):
            local_grid = o_tensor[i]
            agent_pos = np.array([cx, cy])

            enemy_positions = np.argwhere(other_team_presence[agent] > 0)
            ally_positions = np.argwhere(my_team_presence[agent] > 0)
            ally_positions = ally_positions[~np.all(ally_positions == agent_pos, axis=1)]

            # Initialize all 21 actions with -inf (infeasible by default)
            action_values = np.full(self._total_actions, -np.inf)

            # Score attack actions (indices 13-20): value based on enemy HP and ally support
            for j, offset in enumerate(self.attack_offsets):
                attack_pos = agent_pos + offset
                action_idx = self.attack_start_idx + j

                if 0 <= attack_pos[0] < n and 0 <= attack_pos[1] < m:
                    enemy_at_pos = np.any(np.all(enemy_positions == attack_pos, axis=1))
                    if enemy_at_pos:
                        enemy_hp = other_team_hp[agent][attack_pos[0], attack_pos[1]]
                        attack_value = 10 - enemy_hp

                        ally_support = 0
                        for ally_pos in ally_positions:
                            dist = abs(attack_pos[0] - ally_pos[0]) + abs(
                                attack_pos[1] - ally_pos[1]
                            )
                            if dist <= 2:
                                ally_support += 1
                        attack_value += ally_support * 2

                        action_values[action_idx] = attack_value

            # Score move actions (indices 1-12): combine retreat safety and offensive positioning
            for j, offset in enumerate(self.move_offsets):
                move_pos = agent_pos + offset
                action_idx = self.move_start_idx + j

                if (
                    0 <= move_pos[0] < n
                    and 0 <= move_pos[1] < m
                    and local_grid[move_pos[0], move_pos[1]] == 0
                ):
                    # Retreat component: maximize distance from enemies, stay near allies
                    safety_score = 0
                    for enemy_pos in enemy_positions:
                        dist = abs(move_pos[0] - enemy_pos[0]) + abs(move_pos[1] - enemy_pos[1])
                        safety_score += dist

                    ally_proximity = 0
                    for ally_pos in ally_positions:
                        dist = abs(move_pos[0] - ally_pos[0]) + abs(move_pos[1] - ally_pos[1])
                        if dist <= 2:
                            ally_proximity += 1 / (dist + 1)

                    retreat_value = safety_score + ally_proximity * 5

                    # Position component: stay at optimal attack range, near allies, control center
                    position_value = 0
                    for enemy_pos in enemy_positions:
                        dist = abs(move_pos[0] - enemy_pos[0]) + abs(move_pos[1] - enemy_pos[1])
                        if 2 <= dist <= 4:
                            position_value += 3
                        elif dist < 2:
                            position_value -= 2

                    for ally_pos in ally_positions:
                        dist = abs(move_pos[0] - ally_pos[0]) + abs(move_pos[1] - ally_pos[1])
                        if dist <= 2:
                            position_value += 2

                    center_dist = abs(move_pos[0] - cx) + abs(move_pos[1] - cy)
                    position_value += (n - center_dist) * 0.5

                    action_values[action_idx] = max(retreat_value, position_value)

            # Do nothing (action 0): baseline value for fallback
            action_values[self.do_nothing_action] = 0.0

            # Select top-K feasible actions and apply softmax with temperature
            feasible_mask = np.isfinite(action_values)
            feasible_indices = np.where(feasible_mask)[0]
            feasible_values = action_values[feasible_indices]

            sorted_order = np.argsort(-feasible_values)
            top_k = min(self.top_k, len(feasible_indices))
            top_indices = feasible_indices[sorted_order[:top_k]]
            top_values = action_values[top_indices]

            exp_values = np.exp((top_values - np.max(top_values)) / self.temperature)
            probs = exp_values / np.sum(exp_values)

            chosen_action = int(self.rng.choice(top_indices, p=probs))
            all_actions[agent] = chosen_action

        actual_actions = {agent: all_actions[agent] for agent in alive_agents}
        return actual_actions, all_actions
