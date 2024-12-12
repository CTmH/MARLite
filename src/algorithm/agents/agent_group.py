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
                optim: TorchOptimizer = torch.optim.Adam,
                device: str = 'cpu') -> None:
        self.device = device
        self.agents = agents
        self.models = {model_name: config.get_model().to(device=self.device) for model_name, config in model_configs.items()}
        self.optimizers = {model_name: optim(model.parameters()) for model_name, model in self.models.items()}

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
        

    def get_q_values(self, observations: Dict[str, np.ndarray], eval_mode=False) -> torch.Tensor:
        raise NotImplementedError

    def act(observations: Dict[str, np.ndarray], avail_actions: Dict, epsilon: int) -> np.ndarray:
        raise NotImplementedError
   
    def init_hidden_states(self):
        self.hidden_states = {agent_name:
                        self.models[model_name].init_hidden() if isinstance(self.models[model_name], RNNModel) else None \
                        for agent_name, model_name in self.agents.items()}
        return self

    def set_model_params(self, params: Dict[str, dict]):
        for ag_name, model in self.models.items():
            model.load_state_dict(params[ag_name])
        return self
    
    def get_model_params(self):
        params = {agent_name:deepcopy(model.state_dict()) for agent_name, model in self.models.items()}
        return params
    
    def zero_grad(self):
        for _, opt in self.optimizers.items():
            opt.zero_grad()
        return self
    
    def step(self):
        for _, opt in self.optimizers.items():
            # Assuming loss is computed in Learner
            # loss.backward() should be called before this function to compute gradients.
            opt.step()
