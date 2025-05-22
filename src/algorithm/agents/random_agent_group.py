import numpy as np
import torch
from torch.optim import Optimizer as TorchOptimizer
from copy import deepcopy
from typing import Dict
from .agent_group import AgentGroup

class RandomAgentGroup(AgentGroup):
    '''
    agents: Dict[(agent_name(str), model_name(str))]
    model_configs: Dict[model_name(str), ModelConfig]
    '''
    def __init__(self, agents: Dict[str, str]) -> None:
        self.agents = agents

    def forward(self, observations: Dict[str, np.ndarray], eval_mode=True) -> torch.Tensor:
        return self

    def act(self, observations: Dict[str, np.ndarray], avail_actions: Dict, epsilon: int) -> np.ndarray:
        random_actions = {agent: avail_actions[agent].sample() for agent in avail_actions.keys()}
        return random_actions

    def set_agent_group_params(self, model_params: Dict[str, dict], feature_extractor_params: Dict[str, dict]):
        return self
    
    def get_agent_group_params(self):
        return self
    
    def zero_grad(self):
        return self
    
    def step(self):
        return self