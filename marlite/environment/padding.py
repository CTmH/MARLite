"""Make heterogeneous flat observations/actions explicit at the environment edge."""

import numpy as np
from gymnasium import spaces
from pettingzoo.utils import BaseParallelWrapper


class PadAgentView(BaseParallelWrapper):
    """Flatten/pad local observations and expose static plus dynamic action masks.

    The optional feature-validity vector distinguishes padding from observed zero.
    No action is remapped: padded action indices remain illegal for that agent.
    This wrapper does not add global state or an agent's partners to its view.
    """

    def __init__(self, env, include_observation_mask=True):
        super().__init__(env)
        self.possible_agents = tuple(env.possible_agents)
        self.include_observation_mask = include_observation_mask
        self.widths, self.actions = {}, {}
        for agent in self.possible_agents:
            observation = env.observation_space(agent)
            if isinstance(observation, spaces.Dict):
                observation = observation["observation"]
            if not isinstance(observation, spaces.Box):
                raise ValueError("Padding requires numeric Box local observations")
            action = env.action_space(agent)
            if not isinstance(action, spaces.Discrete) or action.start != 0:
                raise ValueError("Padding requires zero-based discrete actions")
            self.widths[agent], self.actions[agent] = int(np.prod(observation.shape)), int(action.n)
        if not self.possible_agents:
            raise ValueError("Padding needs at least one possible agent")
        self.width, self.n_actions = max(self.widths.values()), max(self.actions.values())
        output_width = self.width * (2 if include_observation_mask else 1)
        self.observation_spaces = {
            agent: spaces.Dict(
                {
                    "observation": spaces.Box(-np.inf, np.inf, (output_width,), dtype=np.float32),
                    "action_mask": spaces.MultiBinary(self.n_actions),
                }
            )
            for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: spaces.Discrete(self.n_actions) for agent in self.possible_agents
        }
        self._last_masks = {}

    @property
    def agents(self):
        return self.env.agents

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _pad(self, observations):
        padded = {}
        for agent, value in observations.items():
            dynamic = value.get("action_mask") if isinstance(value, dict) else None
            array = np.asarray(
                value["observation"] if isinstance(value, dict) else value, dtype=np.float32
            ).reshape(-1)
            if array.size != self.widths[agent]:
                raise ValueError("An agent's observation shape changed within an episode")
            features = np.zeros(self.width, dtype=np.float32)
            features[: array.size] = array
            if self.include_observation_mask:
                validity = np.zeros(self.width, dtype=np.float32)
                validity[: array.size] = 1
                features = np.concatenate((features, validity))
            available = np.zeros(self.n_actions, dtype=np.int8)
            available[: self.actions[agent]] = 1
            if dynamic is not None:
                dynamic = np.asarray(dynamic, dtype=bool)
                if dynamic.shape != (self.actions[agent],):
                    raise ValueError("Native action mask does not match its discrete space")
                available[: self.actions[agent]] &= dynamic
            self._last_masks[agent] = available
            padded[agent] = {"observation": features, "action_mask": available.copy()}
        return padded

    def reset(self, seed=None, options=None):
        observations, infos = self.env.reset(seed=seed, options=options)
        self._last_masks = {}
        return self._pad(observations), infos

    def step(self, actions):
        if set(actions) != set(self.env.agents):
            raise ValueError("Supply exactly one action for each live controlled agent")
        for agent, action in actions.items():
            if not 0 <= action < self.actions[agent] or not self._last_masks[agent][action]:
                raise ValueError("Padded or unavailable action cannot reach the native environment")
        observations, rewards, terminated, truncated, infos = self.env.step(actions)
        return self._pad(observations), rewards, terminated, truncated, infos
