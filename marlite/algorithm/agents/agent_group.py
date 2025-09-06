from abc import abstractmethod
import numpy as np
from typing import Dict, List, Any

class AgentGroup(object):

    @abstractmethod
    def forward(self, observations: Dict[str, np.ndarray]) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def forward(self, observations: Dict[str, np.ndarray], states: np.ndarray, edge_indices: List[np.ndarray] | None = None) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def act(self, observations: Dict[str, np.ndarray], avail_actions: Dict[str, Any], epsilon: float) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def act(self, observations: Dict[str, np.ndarray], state: np.ndarray, avail_actions: Dict[str, Any], epsilon: float) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def set_agent_group_params(self, model_params: Dict[str, dict], feature_extractor_params: Dict[str, dict]):
        raise NotImplementedError

    @abstractmethod
    def get_agent_group_params(self) -> Dict[str, dict]:
        raise NotImplementedError

    @abstractmethod
    def zero_grad(self) -> 'AgentGroup':
        raise NotImplementedError

    @abstractmethod
    def step(self) -> 'AgentGroup':
        raise NotImplementedError

    @abstractmethod
    def to_device(self, device) -> 'AgentGroup':
        raise NotImplementedError

    @abstractmethod
    def eval(self) -> 'AgentGroup':
        raise NotImplementedError

    @abstractmethod
    def train(self) -> 'AgentGroup':
        raise NotImplementedError

    @abstractmethod
    def share_memory(self) -> 'AgentGroup':
        raise NotImplementedError

    @abstractmethod
    def wrap_data_parallel(self) -> 'AgentGroup':
        raise NotImplementedError

    @abstractmethod
    def unwrap_data_parallel(self) -> 'AgentGroup':
        raise NotImplementedError

    @abstractmethod
    def save_params(self, path: str) -> 'AgentGroup':
        raise NotImplementedError

    @abstractmethod
    def load_params(self, path: str) -> 'AgentGroup':
        raise NotImplementedError

    @abstractmethod
    def compile_models(self) -> 'AgentGroup':
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> 'AgentGroup':
        raise NotImplementedError