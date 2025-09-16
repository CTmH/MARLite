from typing import Dict, Any
import numpy as np
from collections import deque
from pettingzoo.utils import BaseParallelWrapper
from marlite.algorithm.agents.agent_group_config import AgentGroupConfig
from marlite.util.env_util import obs_preprocess

# NOT COMPLETED
class BattleWrapper(BaseParallelWrapper):
    def __init__(self, env, opponent_agent_group_config: Dict[str, Any], opp_obs_queue_len: int, channel_first: bool = False):
        self.opponent_agent_group_config = opponent_agent_group_config
        self.opponent_agent_group_config = AgentGroupConfig(**self.opponent_agent_group_config)
        self.opponent_agent_group = self.opponent_agent_group_config.get_agent_group()
        self.opp_obs_queue_len = opp_obs_queue_len
        self.channel_first = channel_first
        super().__init__(env=env)

        self.agents = [f'red_{i}' for i in range(80)]
        self.possible_agents = self.agents[:]
        self.observation_spaces = {agent: self.env.observation_space(agent) for agent in self.agents}
        self.action_spaces = {agent: self.env.action_space(agent) for agent in self.agents}

        self.opponent_observations = None
        self.opponent_actions = None
        self.opponent_agents = [f'blue_{i}' for i in range(80)]
        self.opponent_observation_history = deque(maxlen=self.opp_obs_queue_len)  # Queue to store opponent's observationss

    @property
    def num_agents(self) -> int:
        """Get the number of agents in the environment."""
        return len(self.agents)

    @property
    def max_num_agents(self) -> int:
        """Get the number of agents in the environment."""
        return len(self.possible_agents)

    def step(self, actions: Dict) -> tuple:
        opponent_avail_actions = {agent: self.env.action_spaces[agent] for agent in self.opponent_agents}
        opp_obs = list(self.opponent_observation_history)
        opp_obs, traj_padding_mask = obs_preprocess(opp_obs, self.opponent_agents, self.opp_obs_queue_len)
        alive_opponent = [agent for agent in self.opponent_agents if agent in set(self.env.agents)]
        self.opponent_actions = self.opponent_agent_group.act(opp_obs,
                                                              self.env.state(),
                                                              opponent_avail_actions,
                                                              traj_padding_mask,
                                                              alive_opponent,
                                                              epsilon=0.0)
        actions = actions | self.opponent_actions  # Combine actions with opponent's actions
        observations, rewards, terminations, truncations, infos = self.env.step(actions)

        self.opponent_observations = {agent: observations[agent] for agent in self.opponent_agents}
        self.opponent_observation_history.append(self.opponent_observations)

        agent_observations = {}
        for agent in self.agents:
            obs = observations[agent].astype(np.int8)
            if self.channel_first:
                # (H, W, C) -> (C, H, W)
                obs = np.transpose(obs, (2, 0, 1))
            agent_observations[agent] = obs
        agent_rewards = {agent: rewards[agent] for agent in self.agents}
        agent_terminations = {agent: terminations[agent] for agent in self.agents}
        agent_truncations = {agent: truncations[agent] for agent in self.agents}
        agent_infos = {agent: infos[agent] for agent in self.agents}

        return agent_observations, agent_rewards, agent_terminations, agent_truncations, agent_infos

    def reset(self, seed: int = None, options: Dict = None) -> tuple:
        observations, info = self.env.reset(seed=seed, options=options)   # Magent2 environment reset does not return info
        self.opponent_observations = {agent: observations[agent] for agent in self.opponent_agents}
        self.opponent_observation_history.clear()
        self.opponent_observation_history.append(self.opponent_observations)
        agent_observations = {}
        for agent in self.agents:
            obs = observations[agent].astype(np.int8)
            if self.channel_first:
                # (H, W, C) -> (C, H, W)
                obs = np.transpose(obs, (2, 0, 1))
            agent_observations[agent] = obs
        agent_info = {agent: info[agent] for agent in self.agents} # For compatibility with other environments
        return agent_observations, agent_info

    def state(self) -> np.ndarray:
        if self.channel_first:
            return np.transpose(self.env.state().astype(np.int8), (2, 0, 1))
        return self.env.state().astype(np.int8)