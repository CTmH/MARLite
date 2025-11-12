# marlite/environment/magent_wrapper.py
from typing import Dict, Any, Tuple, List
import numpy as np
from collections import deque
from pettingzoo.utils import BaseParallelWrapper
from marlite.algorithm.agents.agent_group_config import AgentGroupConfig
from marlite.util.env_util import obs_preprocess, precompute_manhattan_offsets, ensure_all_agents_present
from marlite.algorithm.graph_builder.graph_util import binary_to_decimal


class MAgentWrapper(BaseParallelWrapper):
    """Base wrapper class for environments with opponent agents.

    This class provides a common interface for handling opponent agents in multi-agent
    environments, including observation history management and action concatenation.
    """

    # Channel indices for state tensor
    OBSTACLE_CHANNEL = 0
    TEAM_0_PRESENCE_CHANNEL = 1
    TEAM_0_HP_CHANNEL = 2
    TEAM_1_PRESENCE_CHANNEL = 3
    TEAM_1_HP_CHANNEL = 4
    BINARY_AGENT_ID_START = 5
    BINARY_AGENT_ID_END = 14  # 5-14: 10 channels
    ONE_HOT_ACTION_START = 15
    ONE_HOT_ACTION_END = 35   # 15-35: 21 channels
    LAST_REWARD_CHANNEL = 36

    def __init__(self, env, opponent_agent_group_config: Dict[str, Any], opp_obs_queue_len: int, channel_first: bool = False, vector_state: bool = False):
        """Initialize the wrapper with opponent agent configuration.

        Args:
            env: The base environment to wrap
            opponent_agent_group_config: Configuration for opponent agents
            opp_obs_queue_len: Length of observation history queue for opponents
            channel_first: Whether to transpose observations to channel-first format
            vector_state: If True, state() will output a matrix of shape (n_agent, feature_dim)
        """
        self.opponent_agent_group_config = opponent_agent_group_config
        self.opponent_agent_group_config = AgentGroupConfig(**self.opponent_agent_group_config)
        self.opponent_agent_group = self.opponent_agent_group_config.get_agent_group()
        self.opp_obs_queue_len = opp_obs_queue_len
        self.channel_first = channel_first
        self.vector_state = vector_state
        super().__init__(env=env)

        self.possible_agents = []
        self.observation_spaces = {}
        self.action_spaces = {}

        self.opponent_observations = {}
        self.opponent_actions = {}
        self.possible_opponent_agents = []
        self.opponent_observation_history = deque(maxlen=self.opp_obs_queue_len)  # Queue to store opponent's observations

        self._n_env_possible_agents = len(self.env.possible_agents)
        self.manhattan_offsets = precompute_manhattan_offsets(2)

        self.opponent_avail_actions = {}
        self.default_opponent_obs = {}

    @property
    def agents(self) -> List[str]:
        """Get the list of alive agents in the environment."""
        env_agents = set(self.env.agents)
        agents = [agent for agent in self.possible_agents if agent in env_agents]
        return agents

    @property
    def opponent_agents(self) -> List[str]:
        """Get the list of alive agents in the environment."""
        env_agents = set(self.env.agents)
        opponent_agents = [agent for agent in self.possible_opponent_agents if agent in env_agents]
        return opponent_agents

    @property
    def num_agents(self) -> int:
        """Get the number of agents in the environment."""
        return len(self.agents)

    @property
    def max_num_agents(self) -> int:
        """Get the maximum number of possible agents in the environment."""
        return len(self.possible_agents)

    def _concat_action_dict(self, agent_actions: Dict[str, Any], opponent_actions: Dict[str, Any]) -> Dict[str, Any]:
        """Concatenate agent actions with opponent actions.

        Args:
            agent_actions: Actions from the agent side
            opponent_actions: Actions from the opponent side

        Returns:
            Combined dictionary of all actions
        """
        combined_actions = agent_actions | opponent_actions
        combined_actions = {agent: combined_actions.get(agent, 0) for agent in self.env.agents}
        return combined_actions

    def step(self, actions: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        """Execute one step in the environment with opponent actions.

        Args:
            actions: Actions from the agent side

        Returns:
            Tuple containing:
                - Observations for agents
                - Rewards for agents
                - Terminations for agents
                - Truncations for agents
                - Additional info
        """
        opp_obs = list(self.opponent_observation_history)
        opp_obs, traj_padding_mask = obs_preprocess(opp_obs, self.possible_opponent_agents, self.opp_obs_queue_len)
        alive_opponent = self.opponent_agents
        self.opponent_actions = self.opponent_agent_group.act(opp_obs,
                                                              self.env.state(),
                                                              self.opponent_avail_actions,
                                                              traj_padding_mask,
                                                              alive_opponent,
                                                              epsilon=0.0)
        combined_actions = self._concat_action_dict(actions, self.opponent_actions)  # Combine actions with opponent's actions
        observations, rewards, terminations, truncations, infos = self.env.step(combined_actions)

        self.opponent_observations = {agent: observations[agent] for agent in observations.keys() if agent in self.possible_opponent_agents}
        self.opponent_observations = ensure_all_agents_present(self.opponent_observations, self.default_opponent_obs)
        self.opponent_observation_history.append(self.opponent_observations)

        agent_observations = {}
        possible_agents_set = set(self.possible_agents)
        for agent in observations.keys():
            if agent in possible_agents_set:
                obs = observations[agent].astype(np.int8)
                if self.channel_first:
                    # (H, W, C) -> (C, H, W)
                    obs = np.transpose(obs, (2, 0, 1))
                agent_observations[agent] = obs
        agent_rewards = {agent: rewards[agent] for agent in rewards.keys() if agent in possible_agents_set}
        agent_terminations = {agent: terminations[agent] for agent in terminations.keys() if agent in possible_agents_set}
        agent_truncations = {agent: truncations[agent] for agent in truncations.keys() if agent in possible_agents_set}
        agent_infos = {agent: infos[agent] for agent in infos.keys() if agent in possible_agents_set}

        return agent_observations, agent_rewards, agent_terminations, agent_truncations, agent_infos

    def reset(self, seed: int = None, options: Dict[str, Any] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment and initialize opponent observation history.

        Args:
            seed: Random seed for reproducibility
            options: Additional reset options

        Returns:
            Tuple containing:
                - Initial observations for agents
                - Initial info dictionary
        """
        observations, info = self.env.reset(seed=seed, options=options)
        self.opponent_observations = {agent: observations[agent] for agent in self.possible_opponent_agents}
        self.opponent_observation_history.clear()
        self.opponent_observation_history.append(self.opponent_observations)
        agent_observations = {}
        for agent in self.possible_agents:
            obs = observations[agent].astype(np.int8)
            if self.channel_first:
                # (H, W, C) -> (C, H, W)
                obs = np.transpose(obs, (2, 0, 1))
            agent_observations[agent] = obs
        agent_info = {agent: info[agent] for agent in self.possible_agents} # For compatibility with other environments
        return agent_observations, agent_info

    def state(self) -> np.ndarray:
        """Get the global state of the environment.

        Returns:
            The environment state as a numpy array
            If vector_state is True, returns a matrix of shape (n_agent, feature_dim)
            Otherwise, returns the original state of shape (L, L, 36)
        """
        state = self.env.state()

        if not self.vector_state:
            if self.channel_first:
                return np.transpose(state.astype(np.int8), (2, 0, 1))
            return state.astype(np.int8)

        # Find all agent positions
        agent_positions = []
        agent_ids = []
        agent_team = []

        # Check team 0 agents
        team_0_presence = state[:, :, self.TEAM_0_PRESENCE_CHANNEL]
        team_0_positions = np.argwhere(team_0_presence > 0)

        for pos in team_0_positions:
            y, x = pos
            binary_id = state[y, x, self.BINARY_AGENT_ID_START:self.BINARY_AGENT_ID_END+1]
            agent_id = binary_to_decimal(binary_id)
            agent_positions.append((y, x))
            agent_ids.append(agent_id)
            agent_team.append(0)

        # Check team 1 agents
        team_1_presence = state[:, :, self.TEAM_1_PRESENCE_CHANNEL]
        team_1_positions = np.argwhere(team_1_presence > 0)

        for pos in team_1_positions:
            y, x = pos
            binary_id = state[y, x, self.BINARY_AGENT_ID_START:self.BINARY_AGENT_ID_END+1]
            agent_id = binary_to_decimal(binary_id)
            agent_positions.append((y, x))
            agent_ids.append(agent_id)
            agent_team.append(1)

        action_space = self.ONE_HOT_ACTION_END - self.ONE_HOT_ACTION_START + 1
        feature_dim = 1 + 1 + 1 + 2 + action_space + len(self.manhattan_offsets) * 3 # hp + team + last_reward + coords + action + nearby
        if not agent_ids:
            # No agents found, return empty matrix

            return np.zeros((self._n_env_possible_agents, feature_dim), dtype=np.int8)

        # Create feature matrix
        feature_matrix = np.zeros((self._n_env_possible_agents, feature_dim), dtype=np.float16)

        map_size = state.shape[:2]
        # Fill feature matrix
        for agent_id, team, (y, x) in zip(agent_ids, agent_team, agent_positions):

            # HP (1 feature)
            if state[y, x, self.TEAM_0_PRESENCE_CHANNEL] > 0:
                feature_matrix[agent_id, 0] = state[y, x, self.TEAM_0_HP_CHANNEL]
            else:
                feature_matrix[agent_id, 0] = state[y, x, self.TEAM_1_HP_CHANNEL]

            # Team (1 feature)
            feature_matrix[agent_id, 1] = team

            # Last reward (1 feature)
            feature_matrix[agent_id, 2] = state[y, x, self.LAST_REWARD_CHANNEL]

            # Position (2 features)
            feature_matrix[agent_id, 3] = y / map_size[0]
            feature_matrix[agent_id, 4] = x / map_size[1]

            nearby_idx_start = 5 + action_space
            # One-hot action (21 features)
            feature_matrix[agent_id, 5:nearby_idx_start] = state[y, x, self.ONE_HOT_ACTION_START:self.ONE_HOT_ACTION_END+1]

            # Nearby features (12 features)
            for i, (dx, dy) in enumerate(self.manhattan_offsets):
                bias = nearby_idx_start + i * 3
                ny, nx = y + dy, x + dx
                if 0 <= ny < state.shape[0] and 0 <= nx < state.shape[1]:
                    # Check if position has obstacle or any agent
                    feature_matrix[agent_id, bias] = state[ny, nx, self.TEAM_0_PRESENCE_CHANNEL]
                    feature_matrix[agent_id, bias + 1] = state[ny, nx, self.TEAM_1_PRESENCE_CHANNEL]
                    feature_matrix[agent_id, bias + 2] = state[ny, nx, self.OBSTACLE_CHANNEL]
                else:
                    # Out of bounds - treat as obstacle
                    feature_matrix[agent_id, bias] = 0
                    feature_matrix[agent_id, bias + 1] = 0
                    feature_matrix[agent_id, bias + 2] = 1

        return feature_matrix

class AdversarialPursuitPredator(MAgentWrapper):
    """Wrapper for predator agents in adversarial pursuit environment.

    This class configures the environment for predator agents in an adversarial pursuit
    scenario where predators try to catch prey.
    """

    # Channel indices for state tensor
    ONE_HOT_ACTION_END = 27   # 15-27: 13 channels
    LAST_REWARD_CHANNEL = 28

    def __init__(self, env, opponent_agent_group_config: Dict[str, Any], opp_obs_queue_len: int, channel_first: bool = False, vector_state: bool = False):
        """Initialize the predator wrapper.

        Args:
            env: The base environment to wrap
            opponent_agent_group_config: Configuration for prey agents
            opp_obs_queue_len: Length of observation history queue for prey
            channel_first: Whether to transpose observations to channel-first format
        """
        super().__init__(env, opponent_agent_group_config, opp_obs_queue_len, channel_first, vector_state)

        self.possible_agents = [agent for agent in self.env.possible_agents if agent.startswith('predator_')]
        self.observation_spaces = {agent: self.env.observation_space(agent) for agent in self.possible_agents}
        self.action_spaces = {agent: self.env.action_space(agent) for agent in self.possible_agents}
        self.possible_opponent_agents = [agent for agent in self.env.possible_agents if agent.startswith('prey_')]
        self.opponent_avail_actions = {agent: self.env.action_spaces[agent] for agent in self.possible_opponent_agents}
        self.default_opponent_obs = {agent: np.zeros(self.env.observation_space(agent).shape, dtype=np.int8) for agent in self.possible_opponent_agents}


class AdversarialPursuitPrey(MAgentWrapper):
    """Wrapper for prey agents in adversarial pursuit environment.

    This class configures the environment for prey agents in an adversarial pursuit
    scenario where prey try to avoid predators.
    """

    # Channel indices for state tensor
    ONE_HOT_ACTION_END = 27   # 15-27: 13 channels
    LAST_REWARD_CHANNEL = 28

    def __init__(self, env, opponent_agent_group_config: Dict[str, Any], opp_obs_queue_len: int, channel_first: bool = False, vector_state: bool = False):
        """Initialize the prey wrapper.

        Args:
            env: The base environment to wrap
            opponent_agent_group_config: Configuration for predator agents
            opp_obs_queue_len: Length of observation history queue for predators
            channel_first: Whether to transpose observations to channel-first format
        """
        super().__init__(env, opponent_agent_group_config, opp_obs_queue_len, channel_first, vector_state)

        self.possible_agents = [agent for agent in self.env.possible_agents if agent.startswith('prey_')]
        self.observation_spaces = {agent: self.env.observation_space(agent) for agent in self.possible_agents}
        self.action_spaces = {agent: self.env.action_space(agent) for agent in self.possible_agents}
        self.possible_opponent_agents = [agent for agent in self.env.possible_agents if agent.startswith('predator_')]
        self.opponent_avail_actions = {agent: self.env.action_spaces[agent] for agent in self.possible_opponent_agents}
        self.default_opponent_obs = {agent: np.zeros(self.env.observation_space(agent).shape, dtype=np.int8) for agent in self.possible_opponent_agents}


class BattleWrapper(MAgentWrapper):
    """Wrapper for battle environment with red and blue teams.

    This class configures the environment for a battle scenario between two teams
    (red and blue) where each team has multiple agents.
    """

    def __init__(self, env, opponent_agent_group_config: Dict[str, Any], opp_obs_queue_len: int, channel_first: bool = False, vector_state: bool = False):
        """Initialize the battle wrapper.

        Args:
            env: The base environment to wrap
            opponent_agent_group_config: Configuration for blue team agents
            opp_obs_queue_len: Length of observation history queue for blue team
            channel_first: Whether to transpose observations to channel-first format
        """
        super().__init__(env, opponent_agent_group_config, opp_obs_queue_len, channel_first, vector_state)

        self.possible_agents = [agent for agent in self.env.possible_agents if agent.startswith('red_')]
        self.observation_spaces = {agent: self.env.observation_space(agent) for agent in self.possible_agents}
        self.action_spaces = {agent: self.env.action_space(agent) for agent in self.possible_agents}
        self.possible_opponent_agents = [agent for agent in self.env.possible_agents if agent.startswith('blue_')]
        self.opponent_avail_actions = {agent: self.env.action_spaces[agent] for agent in self.possible_opponent_agents}
        for agent in self.possible_opponent_agents:
            temp_var = self.env.observation_space(agent)
        self.default_opponent_obs = {agent: np.zeros(self.env.observation_space(agent).shape, dtype=np.int8) for agent in self.possible_opponent_agents}