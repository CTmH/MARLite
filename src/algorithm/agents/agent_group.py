import numpy as np
from typing import Dict
import torch

class AgentGroup(object):

    def forward(self, observations):
        raise NotImplementedError

    def forward(self, observations: Dict[str, np.ndarray], eval_mode=True) -> torch.Tensor:
        raise NotImplementedError

    def act(observations: Dict[str, np.ndarray], avail_actions: Dict, epsilon: int) -> np.ndarray:
        raise NotImplementedError

    def set_agent_group_params(self, model_params: Dict[str, dict], feature_extractor_params: Dict[str, dict]):
        raise NotImplementedError
    
    def get_agent_group_params(self):
        raise NotImplementedError
    
    def zero_grad(self):
        raise NotImplementedError
    
    def step(self):
        raise NotImplementedError
    
    def to_device(self, device):
        raise NotImplementedError
    
    def eval(self):
        raise NotImplementedError
    
    def train(self):
        raise NotImplementedError
    
    def share_memory(self):
        raise NotImplementedError
    
    def save_params(self, path: str):
        raise NotImplementedError
    
    def load_params(self, path: str):
        raise NotImplementedError
    
    def reset(self):
        raise NotImplementedError