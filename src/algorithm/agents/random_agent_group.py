import numpy as np
import torch
from typing import Dict, Any, Type
from .agent_group import AgentGroup

class RandomAgentGroup(AgentGroup):
    '''
    agents: Dict[(agent_name(str), model_name(str))]
    model_configs: Dict[model_name(str), ModelConfig]
    '''
    def __init__(self, agents: Dict[str, str]) -> None:
        self.agents = agents

    def forward(self, observations: Dict[str, np.ndarray]) -> Dict[str, Any]:
        return {'q_val': None}

    def act(self, observations, avail_actions, epsilon=0.0) -> Dict[str, Any]:
        random_actions = {agent: avail_actions[agent].sample() for agent in avail_actions.keys()}
        return {'actions': random_actions}

    def set_agent_group_params(self, model_params: Dict[str, dict], feature_extractor_params: Dict[str, dict]) -> Type[AgentGroup]:
        return self
    
    def get_agent_group_params(self) -> Type[AgentGroup]:
        return self
    
    def zero_grad(self) -> Type[AgentGroup]:
        return self
    
    def step(self) -> Type[AgentGroup]:
        return self