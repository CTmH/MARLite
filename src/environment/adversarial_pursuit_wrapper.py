from typing import Dict, Any
import numpy as np
from collections import deque
from pettingzoo.utils import BaseParallelWrapper
from ..algorithm.agents.agent_group_config import AgentGroupConfig

class AdversarialPursuitPredator(BaseParallelWrapper):
    def __init__(self, env, opponent_agent_group_config: Dict[str, Any], opp_obs_queue_len: int):
        self.opponent_agent_group_config = opponent_agent_group_config
        self.opponent_agent_group_config = AgentGroupConfig(**self.opponent_agent_group_config)
        self.opponent_agent_group = self.opponent_agent_group_config.get_agent_group()
        self.opp_obs_queue_len = opp_obs_queue_len
        super().__init__(env=env)

        self.agents = [f'predator_{i}' for i in range(25)]
        self.possible_agents = self.agents[:]
        self.observation_spaces = {agent: self.env.observation_space(agent) for agent in self.agents}
        self.action_spaces = {agent: self.env.action_space(agent) for agent in self.agents}

        self.opponent_observations = None
        self.opponent_actions = None
        self.opponent_agents = [f'prey_{i}' for i in range(50)]
        self.opponent_observation_history = deque(maxlen=self.opp_obs_queue_len)  # Queue to store opponent's observationss

        self.binary_agent_id_dim = [i for i in range(5,15)]
        self.agent_presence_dim = [1, 3]
        self.prey_presence_dim = [1]
        self.predator_presence_dim = [3]

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
        opp_obs_dict = {agent: np.array([opp_obs[i][agent] for i in range(len(opp_obs))]) for agent in self.opponent_agents}
        self.opponent_actions = self.opponent_agent_group.act(opp_obs_dict, opponent_avail_actions, epsilon=0.0)
        actions = actions | self.opponent_actions  # Combine actions with opponent's actions
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        #observations = self._filter_observations(observations)

        self.opponent_observations = {agent: observations[agent] for agent in self.opponent_agents}
        self.opponent_observation_history.append(self.opponent_observations)

        agent_observations = {agent: observations[agent].astype(np.int8) for agent in self.agents}
        agent_rewards = {agent: rewards[agent] for agent in self.agents}
        agent_terminations = {agent: terminations[agent] for agent in self.agents}
        agent_truncations = {agent: truncations[agent] for agent in self.agents}
        agent_infos = {agent: infos[agent] for agent in self.agents}

        return agent_observations, agent_rewards, agent_terminations, agent_truncations, agent_infos

    def reset(self, seed: int = None, options: Dict = None) -> tuple:
        observations, info = self.env.reset(seed=seed, options=options)  # Magent2 environment reset does not return info
        self.opponent_observations = {agent: observations[agent] for agent in self.opponent_agents}
        self.opponent_observation_history.clear()
        self.opponent_observation_history.append(self.opponent_observations)
        agent_observations = {agent: observations[agent].astype(np.int8) for agent in self.agents}
        agent_info = {agent: info[agent] for agent in self.agents} # For compatibility with other environments
        return agent_observations, agent_info

    def state(self) -> np.ndarray:
        return self.env.state().astype(np.int8)

    def _filter_observations(self, observations):
        # Filter out extra features
        filtered_observations = {key: value[:,:,:5] for key, value in observations.items()}
        return filtered_observations

class AdversarialPursuitPrey(BaseParallelWrapper):
    def __init__(self, env, opponent_agent_group_config: Dict[str, Any], opp_obs_queue_len: int):
        self.opponent_agent_group_config = opponent_agent_group_config
        self.opponent_agent_group_config = AgentGroupConfig(**self.opponent_agent_group_config)
        self.opponent_agent_group = self.opponent_agent_group_config.get_agent_group()
        self.opp_obs_queue_len = opp_obs_queue_len
        super().__init__(env=env)

        self.agents = [f'prey_{i}' for i in range(50)]
        self.possible_agents = self.agents[:]
        self.observation_spaces = {agent: self.env.observation_space(agent) for agent in self.agents}
        self.action_spaces = {agent: self.env.action_space(agent) for agent in self.agents}

        self.opponent_observations = None
        self.opponent_actions = None
        self.opponent_agents = [f'predator_{i}' for i in range(25)]
        self.opponent_observation_history = deque(maxlen=self.opp_obs_queue_len)  # Queue to store opponent's observations

        self.binary_agent_id_dim = [i for i in range(5,15)]
        self.agent_presence_dim = [1, 3]
        self.prey_presence_dim = [1]
        self.predator_presence_dim = [3]

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
        opp_obs_dict = {agent: np.array([opp_obs[i][agent] for i in range(len(opp_obs))]) for agent in self.opponent_agents}
        self.opponent_actions = self.opponent_agent_group.act(opp_obs_dict, opponent_avail_actions, epsilon=0.0)
        actions = self.opponent_actions | actions  # Combine actions with opponent's actions
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        #observations = self._filter_observations(observations)

        self.opponent_observations = {agent: observations[agent] for agent in self.opponent_agents}
        self.opponent_observation_history.append(self.opponent_observations)

        agent_observations = {agent: observations[agent].astype(np.int8) for agent in self.agents}
        agent_rewards = {agent: rewards[agent] for agent in self.agents}
        agent_terminations = {agent: terminations[agent] for agent in self.agents}
        agent_truncations = {agent: truncations[agent] for agent in self.agents}
        agent_infos = {agent: infos[agent] for agent in self.agents}

        return agent_observations, agent_rewards, agent_terminations, agent_truncations, agent_infos

    def reset(self, seed: int = None, options: Dict = None) -> tuple:
        observations, info = self.env.reset(seed=seed, options=options)  # Magent2 environment reset does not return info
        self.opponent_observations = {agent: observations[agent] for agent in self.opponent_agents}
        self.opponent_observation_history.clear()
        self.opponent_observation_history.append(self.opponent_observations)
        agent_observations = {agent: observations[agent].astype(np.int8) for agent in self.agents}
        agent_info = {agent: info[agent] for agent in self.agents} # For compatibility with other environments
        return agent_observations, agent_info

    def state(self) -> np.ndarray:
        return self.env.state().astype(np.int8)

    def _filter_observations(self, observations):
        # Filter out extra features
        filtered_observations = {key: value[:,:,:5] for key, value in observations.items()}
        return filtered_observations