import numpy as np
from typing import Dict, Any, List
from marlite.algorithm.agents.agent_group import AgentGroup

class RandomAgentGroup(AgentGroup):
    '''
    agents: Dict[(agent_name(str), model_name(str))]
    model_configs: Dict[model_name(str), ModelConfig]
    '''
    def __init__(self, agents: Dict[str, str]) -> None:
        self.agents = agents

    def forward(self, observations: Dict[str, np.ndarray]) -> Dict[str, Any]:
        return {'q_val': None}

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
        if isinstance(next(iter(avail_actions.values())), np.ndarray):
            action_masks = np.array([avail_actions[agent_id] for agent_id in self.agents])
            mask_probs = action_masks / np.sum(action_masks, axis=1, keepdims=True)
            random_actions = np.array([
                np.random.choice(len(probs), p=probs)
                for probs in mask_probs
            ]).astype(np.int64)
        else:
            random_actions = {agent: avail_actions[agent].sample() for agent in avail_actions.keys()}
        actions = {agent: random_actions[agent] for agent in alive_agents}

        return {'actions': actions, 'all_actions': random_actions}

    def set_agent_group_params(self, model_params: Dict[str, dict], feature_extractor_params: Dict[str, dict]) -> 'AgentGroup':
        return self