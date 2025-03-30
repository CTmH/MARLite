import numpy as np
import torch
from torch.optim import Optimizer
from copy import deepcopy
from typing import Dict, Tuple
from torch import Tensor
from torch import bernoulli, ones, argmax, stack
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv  # Example GNN layers
from ..model.model_config import ModelConfig
from ..model import RNNModel
from .agent_group import AgentGroup
from src.util.optimizer_config import OptimizerConfig

class GNNAgentGroup(AgentGroup):
    def __init__(self, 
                agent_model_dict: Dict[str, str], 
                model_configs: Dict[str, ModelConfig],
                feature_extractors_configs: Dict[str, ModelConfig],
                graph_model_config: ModelConfig,
                optimizer_config: OptimizerConfig,
                device = 'cpu') -> None:
        super().__init__()
        self.device = device
        self.agent_model_dict = agent_model_dict
        self.models = {model_name: config.get_model() for model_name, config in model_configs.items()}  # Model for hidden state prediction
        self.feature_extractors = {model_name: config.get_model() for model_name, config in feature_extractors_configs.items()}
        self.graph_model = graph_model_config.get_model()  # Graph model for message passing
        params_to_optimize = [{'params': model.parameters()} for model in self.models.values()]
        params_to_optimize += [{'params': extractor.parameters()} for extractor in self.feature_extractors.values()]
        params_to_optimize += [{'params': self.graph_model.parameters()}]
        self.optimizer = optimizer_config.get_optimizer(params_to_optimize)

        # Initialize model_to_agent dictionary and model_to_agent_indices dictionary
        self.model_to_agents = {model_name:[] for model_name in model_configs.keys()}
        self.model_to_agent_indices = {model_name:[] for model_name in model_configs.keys()}
        for i, (agent_name, model_name) in enumerate(self.agent_model_dict.items()):
            assert model_name in self.model_to_agents.keys(), f"Model {model_name} not found in model_configs"
            self.model_to_agents[model_name].append(agent_name)
            self.model_to_agent_indices[model_name].append(i)

        # Initialize hidden states if the models are RNNModels
        self.hidden_states = {}
        self.init_hidden_states()

    def forward(self, observations, edge_index: list) -> Tensor:
        msg = [None for _ in range(len(self.agent_model_dict))]
        for (model_name, model), (_, fe) in zip(self.models.items(), 
                                                self.feature_extractors.items()):
            selected_agents = self.model_to_agents[model_name]
            idx = self.model_to_agent_indices[model_name]
            # observation shape: (Batch Size, Agent Number, Time Step, Feature Dimensions) (B, N, T, F)
            obs = observations[:,idx]
            obs = torch.Tensor(obs)
            # (B, N, T, (obs_shape)) -> (B*N*T, (obs_shape))
            bs = obs.shape[0]
            n_agents = len(selected_agents)
            ts = obs.shape[2]
            obs_shape = list(obs.shape[3:])
            obs = obs.reshape(bs*n_agents*ts, *obs_shape)
            obs = obs.to(self.device)
            obs_vectorized = fe(obs) # (B*N*T, (obs_shape)) -> (B*N*T, F)
            obs_vectorized = obs_vectorized.reshape(bs*n_agents, ts, -1) # (B*N*T, F) -> (B*N, T, F)
            if isinstance(model, RNNModel):
                h = [model.init_hidden() for _ in range(obs_vectorized.shape[0])]
                h = torch.stack(h).to(self.device)
                h = h.permute(1, 0, 2)
                msg_selected, _ = model(obs_vectorized, h)
                msg_selected = msg_selected[:,-1,:] # get the last output 
            # TODO: Add code for handling other types of models (e.g., CNNs)
            msg_selected = msg_selected.reshape(bs, n_agents, -1) # (B, N, Action Space)
            msg_selected = msg_selected.permute(1, 0, 2)  # (N, B, Action Space)

            for i, m in zip(idx, msg_selected):
                msg[i] = m
        
        msg = torch.stack(msg).to(self.device) # (N, B, Action Space)
        msg = msg.permute(1, 0, 2)  # (B, N, Action Space)

        # Communication between agents using the graph model. 
        q_val = []
        for m, e in zip(msg, edge_index):
            e = Tensor(e).type(torch.int64).to(self.device)
            q = self.graph_model(m, e) # (B, N, Action Space)
            q_val.append(q)
        q_val = torch.stack(q_val).to(self.device) # (B, N, Action Space)
        return q_val
    
    def act(self, observations, edge_index, avail_actions, epsilon=0.0, eval_mode=True):
        """
        Select actions based on Q-values and exploration.

        Args:
            observations (list of Tensor): List of observation tensors.
            avail_actions (dict): Dictionary mapping agent IDs to available action distributions.
            epsilon (float): Exploration rate.
            eval_mode (bool): Whether to set models to evaluation mode.

        Returns:
            numpy array: Selected actions for each agent.
        """
        if eval_mode:
            self.eval()  # Set models to evaluation mode
        else:
            self.train()  # Set models to training mode
        self.init_hidden_states()
        obs = [observations[ag] for ag in self.agent_model_dict.keys()]
        obs = np.stack(obs)
        obs = np.expand_dims(obs, axis=0)
        edge_index = np.expand_dims(edge_index, axis=0)
        q_values = self.forward(obs, edge_index)
        q_values = q_values.detach().cpu().numpy().squeeze()
        random_choices = np.random.binomial(1, epsilon, len(self.agent_model_dict)).astype(np.int64)
        random_actions = [avail_actions[key].sample() for key in avail_actions.keys()]
        random_actions = np.array(random_actions).astype(np.int64)

        actions = random_choices * random_actions \
        + (1 - random_choices) * np.argmax(q_values, axis=-1).astype(np.int64)
        actions = actions.astype(np.int64).tolist()

        actions = {agent_id: action for agent_id, action in zip(self.agent_model_dict.keys(), actions)}
        
        return actions
    
    def init_hidden_states(self):
        self.hidden_states = {agent_name:
                        self.models[model_name].init_hidden() if isinstance(self.models[model_name], RNNModel) else None \
                        for agent_name, model_name in self.agent_model_dict.items()}
        return self
    
    def set_model_params(self, model_params: Dict[str, dict], feature_extractor_params: Dict[str, dict]):
        for (model_name, model), (_, fe) in zip(self.models.items(), self.feature_extractors.items()):
            model.load_state_dict(model_params[model_name])
            fe.load_state_dict(feature_extractor_params[model_name])
        return self
    
    def set_comm_model_params(self, graph_model_params: dict):
        self.graph_model.load_state_dict(graph_model_params)
        return self
    
    def get_model_params(self):
        model_params = {model_name:deepcopy(model.state_dict()) for model_name, model in self.models.items()}
        feature_extractor_params = {model_name:deepcopy(fe.state_dict()) for model_name, fe in self.feature_extractors.items()}
        return model_params, feature_extractor_params
    
    def get_comm_model_params(self):
        comm_model_params = deepcopy(self.graph_model.state_dict())
        return comm_model_params
    
    def zero_grad(self):
        self.optimizer.zero_grad()
        return self
    
    def step(self):
        self.optimizer.step()
        return self
    
    def to(self, device: str):
        for (_, model), (_, fe) in zip(self.models.items(), self.feature_extractors.items()):
            model.to(device)
            fe.to(device)
        self.graph_model.to(device)
        self.device = device
        return self
    
    def eval(self):
        for (_, model), (_, fe) in zip(self.models.items(), self.feature_extractors.items()):
            model.eval()
            fe.eval()
        self.graph_model.eval()
        return self
    
    def train(self):
        for (_, model), (_, fe) in zip(self.models.items(), self.feature_extractors.items()):
            model.train()
            fe.train()
        self.graph_model.train()
        return self