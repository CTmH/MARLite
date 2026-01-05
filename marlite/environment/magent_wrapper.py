from typing import Dict, Any, Tuple, List
import numpy as np
from collections import deque
from pettingzoo.utils import BaseParallelWrapper, ParallelEnv
from marlite.algorithm.agents.agent_group_config import AgentGroupConfig
from marlite.util.env_util import obs_preprocess, precompute_manhattan_offsets, ensure_all_agents_present
from marlite.algorithm.graph_builder.graph_util import binary_to_decimal


class MAgentWrapper(BaseParallelWrapper):
    """Base wrapper class for environments with opponent agents.

    This wrapper provides a common interface for handling opponent agents in multi-agent
    environments. It supports:
    - Observation history management for opponent agents
    - Action concatenation between agent and opponent actions
    - Vectorized observation processing (closest agents by Manhattan distance)
    - Channel-first format conversion
    - Vectorized state representation

    The wrapper processes observations to extract relevant information about obstacles,
    team agents, and opponent agents, and provides flexible configuration for different
    environment requirements.
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

    OBSERVATION_OBSTACLE_CHANNEL = 0
    OBSERVATION_TEAM_0_PRESENCE_CHANNEL = 1
    OBSERVATION_TEAM_1_PRESENCE_CHANNEL = 4
    #OBSERVATION_AGENT_POSITION_Y = 31
    #OBSERVATION_AGENT_POSITION_X = 32

    def __init__(
            self,
            env: ParallelEnv,
            opponent_agent_group_config: Dict[str, Any],
            opp_obs_queue_len: int,
            channel_first: bool = False,
            vector_state: bool = False,
            vector_observation: bool = False,
            max_vector_observation_records: int = 8
            ):
        """Initialize the wrapper with opponent agent configuration.

        Args:
            env: The base environment to wrap
            opponent_agent_group_config: Configuration for opponent agents
                (should contain parameters needed to initialize AgentGroupConfig)
            opp_obs_queue_len: Length of observation history queue for opponents
            channel_first: Whether to transpose observations to channel-first format
                (H, W, C) -> (C, H, W)
            vector_state: If True, state() will output a matrix of shape (n_agent, feature_dim)
                instead of the raw state tensor
            vector_observation: If True, observations will be processed to extract
                the closest max_vector_observation_records agents by Manhattan distance
            max_vector_observation_records: Maximum number of agents to include in vectorized observations
        """
        self.opponent_agent_group_config = opponent_agent_group_config
        self.opponent_agent_group_config = AgentGroupConfig(**self.opponent_agent_group_config)
        self.opponent_agent_group = self.opponent_agent_group_config.get_agent_group()
        self.opp_obs_queue_len = opp_obs_queue_len
        self.channel_first = channel_first
        self.vector_state = vector_state
        super().__init__(env=env)

        # New attributes for vectorized observation
        self.vector_agent_observation = vector_observation
        self.max_vector_observation_records = max_vector_observation_records

        self.possible_agents = []
        self._possible_agents_set = set(self.possible_agents)
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
        """Get the list of alive agents in the environment.

        Returns:
            List of agent identifiers that are currently alive in the environment
        """
        agents = [agent for agent in self.env.agents if agent in self._possible_agents_set]
        return agents

    @property
    def opponent_agents(self) -> List[str]:
        """Get the list of alive opponent agents in the environment.

        Returns:
            List of opponent agent identifiers that are currently alive in the environment
        """
        env_agents = set(self.env.agents)
        opponent_agents = [agent for agent in self.possible_opponent_agents if agent in env_agents]
        return opponent_agents

    @property
    def num_agents(self) -> int:
        """Get the number of agents in the environment.

        Returns:
            Number of currently alive agents in the environment
        """
        return len(self.agents)

    @property
    def max_num_agents(self) -> int:
        """Get the maximum number of possible agents in the environment.

        Returns:
            Maximum number of agents that could be present in the environment
        """
        return len(self.possible_agents)

    def _concat_action_dict(self, agent_actions: Dict[str, Any], opponent_actions: Dict[str, Any]) -> Dict[str, Any]:
        """Concatenate agent actions with opponent actions.

        Combines actions from both agent and opponent sides into a single dictionary
        that can be passed to the underlying environment's step() method.

        Args:
            agent_actions: Actions from the agent side (keys are agent names)
            opponent_actions: Actions from the opponent side (keys are opponent agent names)

        Returns:
            Combined dictionary of all actions, with keys being all agents in the environment
        """
        combined_actions = agent_actions | opponent_actions
        combined_actions = {agent: combined_actions.get(agent, 0) for agent in self.env.agents}
        return combined_actions

    def _process_observations(self, observations: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Process observations according to wrapper configuration.

        Applies the configured processing to observations:
        - Vectorized observation: extracts closest agents by Manhattan distance
        - Channel-first format: transposes (H, W, C) to (C, H, W)
        - Type conversion: converts to float16 for memory efficiency

        Args:
            observations: Dictionary mapping agent names to their observation tensors

        Returns:
            Processed observations dictionary with same keys as input
        """

        processed_observations = {}
        if self.vector_agent_observation:
            for agent in observations.keys():
                # (H, W, C) -> (n_units, n_channels)
                processed_observations[agent] = self._vectorize_observation(observations[agent]).astype(np.float16)
        elif self.channel_first:
            for agent in observations.keys():
                # (H, W, C) -> (C, H, W)
                processed_observations[agent] = np.transpose(observations[agent], (2, 0, 1)).astype(np.float16)
        else:
            for agent in observations.keys():
                processed_observations[agent] = observations[agent].astype(np.float16)

        return processed_observations

    def _vectorize_observation(self, observation: np.ndarray) -> np.ndarray:
        """Process observation to extract closest agents by Manhattan distance.

        Converts a grid-based observation into a vectorized representation containing
        only the most relevant nearby entities (obstacles and agents), sorted by
        Manhattan distance from the agent's position (center of the observation).

        Args:
            observation: The agent observation tensor of shape (H, W, C)

        Returns:
            Matrix of shape (max_vector_observation_records, C) containing the closest
            entities' full observation vectors, padded with zeros if fewer entities exist
        """
        # Get observation dimensions
        H, W, C = observation.shape
        center_y, center_x = H // 2, W // 2  # Agent is at center

        # Find all positions of interest: obstacles and agents
        positions = []
        distances = []

        # Check obstacles
        obstacle_map = observation[:, :, self.OBSERVATION_OBSTACLE_CHANNEL]
        obstacle_positions = np.argwhere(obstacle_map > 0)
        for y, x in obstacle_positions:
            dist = abs(y - center_y) + abs(x - center_x)
            positions.append((y, x))
            distances.append(dist)

        # Check team 0 agents
        team_0_presence = observation[:, :, self.OBSERVATION_TEAM_0_PRESENCE_CHANNEL]
        team_0_positions = np.argwhere(team_0_presence > 0)
        for y, x in team_0_positions:
            dist = abs(y - center_y) + abs(x - center_x)
            positions.append((y, x))
            distances.append(dist)

        # Check team 1 agents
        team_1_presence = observation[:, :, self.OBSERVATION_TEAM_1_PRESENCE_CHANNEL]
        team_1_positions = np.argwhere(team_1_presence > 0)
        for y, x in team_1_positions:
            dist = abs(y - center_y) + abs(x - center_x)
            positions.append((y, x))
            distances.append(dist)

        # Sort positions by distance
        if positions:
            # Convert to numpy arrays for sorting
            positions = np.array(positions)
            distances = np.array(distances)

            # Sort by distance
            sorted_indices = np.argsort(distances)
            sorted_positions = positions[sorted_indices]
        else:
            sorted_positions = np.array([]).reshape(0, 2)

        # Create result matrix
        max_records = self.max_vector_observation_records
        result = np.zeros((max_records, C), dtype=np.float16)

        # Fill with closest entities (obstacles and agents)
        n_entities = min(len(sorted_positions), max_records)
        for i in range(n_entities):
            y, x = sorted_positions[i]
            result[i] = observation[y, x]

        return result

    def _vectorize_state(self, state: np.ndarray) -> np.ndarray:
        """Get the vector representation of the environment state.

        Converts the raw state tensor into a matrix where each row represents
        features for a single agent, enabling easier processing by neural networks.

        Args:
            state: The raw environment state tensor of shape (L, L, C)

        Returns:
            A matrix of shape (n_agent, feature_dim) containing vectorized features
            for each agent, where feature_dim includes:
            - HP (1 feature)
            - Team (1 feature)
            - Last reward (1 feature)
            - Normalized coordinates (2 features)
            - One-hot action (21 features)
            - Nearby features (12 features: 4 offsets × 3 channels each)
        """
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
            return np.zeros((self._n_env_possible_agents, feature_dim), dtype=np.float16)

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

    def step(self, actions: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        """Execute one step in the environment with opponent actions.

        This method handles the complete step cycle including:
        - Processing opponent observation history
        - Generating opponent actions using the opponent agent group
        - Combining agent and opponent actions
        - Executing the step in the underlying environment
        - Processing observations for the agent side only

        Args:
            actions: Actions from the agent side (keys are agent names)

        Returns:
            Tuple containing:
                - Observations for agents (dictionary mapping agent names to observations)
                - Rewards for agents (dictionary mapping agent names to rewards)
                - Terminations for agents (dictionary mapping agent names to termination flags)
                - Truncations for agents (dictionary mapping agent names to truncation flags)
                - Additional info (dictionary mapping agent names to info dictionaries)
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

        agent_observations = {agent: observations[agent] for agent in observations.keys() if agent in self._possible_agents_set}
        agent_rewards = {agent: rewards[agent] for agent in rewards.keys() if agent in self._possible_agents_set}
        agent_terminations = {agent: terminations[agent] for agent in terminations.keys() if agent in self._possible_agents_set}
        agent_truncations = {agent: truncations[agent] for agent in truncations.keys() if agent in self._possible_agents_set}
        agent_infos = {agent: infos[agent] for agent in infos.keys() if agent in self._possible_agents_set}

        agent_observations = self._process_observations(agent_observations)

        return agent_observations, agent_rewards, agent_terminations, agent_truncations, agent_infos

    def reset(self, seed: int = None, options: Dict[str, Any] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment and initialize opponent observation history.

        This method resets the underlying environment and initializes the opponent
        observation history queue with the initial opponent observations.

        Args:
            seed: Random seed for reproducibility
            options: Additional reset options to pass to the underlying environment

        Returns:
            Tuple containing:
                - Initial observations for agents (dictionary mapping agent names to observations)
                - Initial info dictionary (dictionary mapping agent names to info dictionaries)
        """
        observations, infos = self.env.reset(seed=seed, options=options)
        self.opponent_observations = {agent: observations[agent] for agent in self.possible_opponent_agents}
        self.opponent_observation_history.clear()
        self.opponent_observation_history.append(self.opponent_observations)
        agent_observations = {agent: observations[agent] for agent in observations.keys() if agent in self._possible_agents_set}
        agent_infos = {agent: infos[agent] for agent in infos.keys() if agent in self._possible_agents_set}

        agent_observations = self._process_observations(agent_observations)

        return agent_observations, agent_infos

    def state(self) -> np.ndarray:
        """Get the global state of the environment.

        Returns the environment state in the configured format:
        - If vector_state is False: returns the raw state tensor (L, L, C)
        - If vector_state is True: returns a vectorized matrix (n_agent, feature_dim)
        - If channel_first is True: transposes the raw state to (C, L, L)

        Returns:
            The environment state as a numpy array in the configured format
        """
        ret_state = None
        state = self.env.state()

        if not self.vector_state:
            if self.channel_first:
                ret_state = np.transpose(state.astype(np.float16), (2, 0, 1))
            else:
                ret_state = state.astype(np.float16)
        else:
            ret_state = self._vectorize_state(state)

        return ret_state


class AdversarialPursuitPredator(MAgentWrapper):
    """Wrapper for predator agents in adversarial pursuit environment.

    This class configures the environment for predator agents in an adversarial pursuit
    scenario where predators try to catch prey. It filters the possible agents to only
    include predator agents and sets up the opponent agents as prey.

    The wrapper inherits all functionality from MAgentWrapper and specializes it
    for the predator role in the adversarial pursuit environment.
    """

    # Channel indices for state tensor
    ONE_HOT_ACTION_END = 27   # 15-27: 13 channels
    LAST_REWARD_CHANNEL = 28

    def __init__(
            self,
            env: ParallelEnv,
            opponent_agent_group_config: Dict[str, Any],
            opp_obs_queue_len: int,
            channel_first: bool = False,
            vector_state: bool = False,
            vector_observation: bool = False,
            max_vector_observation_records: int = 8
            ):
        """Initialize the predator wrapper.

        Args:
            env: The base environment to wrap
            opponent_agent_group_config: Configuration for prey agents
            opp_obs_queue_len: Length of observation history queue for prey
            channel_first: Whether to transpose observations to channel-first format
            vector_state: If True, state() will output vectorized state representation
            vector_observation: If True, observations will be vectorized
            max_vector_observation_records: Maximum number of agents in vectorized observations
        """
        super().__init__(
            env,
            opponent_agent_group_config,
            opp_obs_queue_len,
            channel_first,
            vector_state,
            vector_observation,
            max_vector_observation_records)

        self.possible_agents = [agent for agent in self.env.possible_agents if agent.startswith('predator_')]
        self.observation_spaces = {agent: self.env.observation_space(agent) for agent in self.possible_agents}
        self.action_spaces = {agent: self.env.action_space(agent) for agent in self.possible_agents}
        self.possible_opponent_agents = [agent for agent in self.env.possible_agents if agent.startswith('prey_')]
        self.opponent_avail_actions = {agent: self.env.action_spaces[agent] for agent in self.possible_opponent_agents}
        self.default_opponent_obs = {agent: np.zeros(self.env.observation_space(agent).shape, dtype=np.float16) for agent in self.possible_opponent_agents}
        self._possible_agents_set = set(self.possible_agents)


class AdversarialPursuitPrey(MAgentWrapper):
    """Wrapper for prey agents in adversarial pursuit environment.

    This class configures the environment for prey agents in an adversarial pursuit
    scenario where prey try to avoid predators. It filters the possible agents to only
    include prey agents and sets up the opponent agents as predators.

    The wrapper inherits all functionality from MAgentWrapper and specializes it
    for the prey role in the adversarial pursuit environment.
    """

    # Channel indices for state tensor
    ONE_HOT_ACTION_END = 27   # 15-27: 13 channels
    LAST_REWARD_CHANNEL = 28

    def __init__(
            self,
            env: ParallelEnv,
            opponent_agent_group_config: Dict[str, Any],
            opp_obs_queue_len: int,
            channel_first: bool = False,
            vector_state: bool = False,
            vector_observation: bool = False,
            max_vector_observation_records: int = 8
            ):
        """Initialize the prey wrapper.

        Args:
            env: The base environment to wrap
            opponent_agent_group_config: Configuration for predator agents
            opp_obs_queue_len: Length of observation history queue for predators
            channel_first: Whether to transpose observations to channel-first format
            vector_state: If True, state() will output vectorized state representation
            vector_observation: If True, observations will be vectorized
            max_vector_observation_records: Maximum number of agents in vectorized observations
        """
        super().__init__(
            env,
            opponent_agent_group_config,
            opp_obs_queue_len,
            channel_first,
            vector_state,
            vector_observation,
            max_vector_observation_records)

        self.possible_agents = [agent for agent in self.env.possible_agents if agent.startswith('prey_')]
        self.observation_spaces = {agent: self.env.observation_space(agent) for agent in self.possible_agents}
        self.action_spaces = {agent: self.env.action_space(agent) for agent in self.possible_agents}
        self.possible_opponent_agents = [agent for agent in self.env.possible_agents if agent.startswith('predator_')]
        self.opponent_avail_actions = {agent: self.env.action_spaces[agent] for agent in self.possible_opponent_agents}
        self.default_opponent_obs = {agent: np.zeros(self.env.observation_space(agent).shape, dtype=np.float16) for agent in self.possible_opponent_agents}
        self._possible_agents_set = set(self.possible_agents)


class BattleWrapper(MAgentWrapper):
    """Wrapper for battle environment with red and blue teams.

    This class configures the environment for a battle scenario between two teams
    (red and blue) where each team has multiple agents. It filters the possible agents
    to only include red team agents and sets up the opponent agents as blue team agents.

    The wrapper inherits all functionality from MAgentWrapper and specializes it
    for the battle environment with team-based opposition.
    """

    def __init__(
            self,
            env: ParallelEnv,
            opponent_agent_group_config: Dict[str, Any],
            opp_obs_queue_len: int,
            channel_first: bool = False,
            vector_state: bool = False,
            vector_observation: bool = False,
            max_vector_observation_records: int = 8
            ):
        """Initialize the battle wrapper.

        Args:
            env: The base environment to wrap
            opponent_agent_group_config: Configuration for blue team agents
            opp_obs_queue_len: Length of observation history queue for blue team
            channel_first: Whether to transpose observations to channel-first format
            vector_state: If True, state() will output vectorized state representation
            vector_observation: If True, observations will be vectorized
            max_vector_observation_records: Maximum number of agents in vectorized observations
        """
        super().__init__(
            env,
            opponent_agent_group_config,
            opp_obs_queue_len,
            channel_first,
            vector_state,
            vector_observation,
            max_vector_observation_records)

        self.possible_agents = [agent for agent in self.env.possible_agents if agent.startswith('red_')]
        self.observation_spaces = {agent: self.env.observation_space(agent) for agent in self.possible_agents}
        self.action_spaces = {agent: self.env.action_space(agent) for agent in self.possible_agents}
        self.possible_opponent_agents = [agent for agent in self.env.possible_agents if agent.startswith('blue_')]
        self.opponent_avail_actions = {agent: self.env.action_spaces[agent] for agent in self.possible_opponent_agents}
        for agent in self.possible_opponent_agents:
            temp_var = self.env.observation_space(agent)
        self.default_opponent_obs = {agent: np.zeros(self.env.observation_space(agent).shape, dtype=np.float16) for agent in self.possible_opponent_agents}
        self._possible_agents_set = set(self.possible_agents)