import numpy as np
from typing import Dict, Type, List, Tuple, Any
import torch

class AgentGroup(object):

    def forward(self, observations: Dict[str, np.ndarray]) -> Dict[str, Any]:
        raise NotImplementedError
    
    def forward(self, observations: Dict[str, np.ndarray], states: np.ndarray, edge_indices: Type[List[np.ndarray] | None] = None) -> Dict[str, Any]:
        raise NotImplementedError

    def act(observations: Dict[str, np.ndarray], avail_actions: Dict, epsilon: float) -> Dict[str, Any]:
        raise NotImplementedError
    
    def act(self, observations: Dict[str, np.ndarray], state: np.ndarray, avail_actions: Dict, epsilon: float) -> Dict[str, Any]:
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