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

    def set_model_params(self, model_params: Dict[str, dict], feature_extractor_params: Dict[str, dict]):
        raise NotImplementedError
    
    def get_model_params(self):
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