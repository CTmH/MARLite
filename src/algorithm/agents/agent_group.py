import numpy as np
import torch
from torch.optim import Optimizer as TorchOptimizer
from copy import deepcopy
from typing import Dict
from ..model.model_config import ModelConfig
from ..model import RNNModel

class AgentGroup:
    '''
    agents: Dict[(agent_name(str), model_name(str))]
    model_configs: Dict[model_name(str), ModelConfig]
    '''
    def __init__(self, 
                agents: Dict[str, str], 
                model_configs: Dict[str, ModelConfig],
                feature_extractors: Dict[str, ModelConfig],
                optim: TorchOptimizer = torch.optim.Adam,
                lr: float = 1e-3,
                device: str = 'cpu') -> None:
        self.device = device
        self.agents = agents
        self.models = {model_name: config.get_model().to(device=self.device) for model_name, config in model_configs.items()}
        self.feature_extractors = {model_name: config.get_model().to(device=self.device) for model_name, config in feature_extractors.items()}
        params_to_optimize = [{'params': model.parameters()} for model in self.models.values()]
        params_to_optimize += [{'params': extractor.parameters()} for extractor in self.feature_extractors.values()]
        self.optimizer = optim(params_to_optimize, lr=lr)

        # Initialize model_to_agent dictionary and model_to_agent_indices dictionary
        self.model_to_agents = {model_name:[] for model_name in model_configs.keys()}
        self.model_to_agent_indices = {model_name:[] for model_name in model_configs.keys()}
        for i, (agent_name, model_name) in enumerate(self.agents.items()):
            assert model_name in self.model_to_agents.keys(), f"Model {model_name} not found in model_configs"
            self.model_to_agents[model_name].append(agent_name)
            self.model_to_agent_indices[model_name].append(i)

        # Initialize hidden states if the models are RNNModels
        self.hidden_states = {}
        self.init_hidden_states()
        

    def get_q_values(self, observations: Dict[str, np.ndarray], eval_mode=True) -> torch.Tensor:
        raise NotImplementedError

    def act(observations: Dict[str, np.ndarray], avail_actions: Dict, epsilon: int) -> np.ndarray:
        raise NotImplementedError
   
    def init_hidden_states(self):
        self.hidden_states = {agent_name:
                        self.models[model_name].init_hidden() if isinstance(self.models[model_name], RNNModel) else None \
                        for agent_name, model_name in self.agents.items()}
        return self

    def set_model_params(self, model_params: Dict[str, dict], feature_extractor_params: Dict[str, dict]):
        for (model_name, model), (_, fe) in zip(self.models.items(), self.feature_extractors.items()):
            model.load_state_dict(model_params[model_name])
            fe.load_state_dict(feature_extractor_params[model_name])
        return self
    
    def get_model_params(self):
        model_params = {model_name:deepcopy(model.state_dict()) for model_name, model in self.models.items()}
        feature_extractor_params = {model_name:deepcopy(fe.state_dict()) for model_name, fe in self.feature_extractors.items()}
        return model_params, feature_extractor_params
    
    def zero_grad(self):
        self.optimizer.zero_grad()
        return self
    
    def step(self):
        self.optimizer.step()
        return self
