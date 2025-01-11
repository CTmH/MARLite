import numpy as np
import torch
import logging
from torch.optim import Optimizer
from copy import deepcopy
from typing import Dict
from torch import Tensor
from torch import bernoulli, ones, argmax, stack
from ..model.model_config import ModelConfig
from ..model import RNNModel
from .agent_group import AgentGroup
from ..model import RNNModel
from src.util.optimizer_config import OptimizerConfig

class QMIXAgentGroup(AgentGroup):
    def __init__(self, 
                agent_model_dict: Dict[str, str], 
                model_configs: Dict[str, ModelConfig],
                feature_extractors_configs: Dict[str, ModelConfig],
                optimizer_config: OptimizerConfig,
                device = 'cpu') -> None:
        super().__init__()
        self.device = device
        self.agent_model_dict = agent_model_dict
        self.models = {model_name: config.get_model() for model_name, config in model_configs.items()}
        self.feature_extractors = {model_name: config.get_model() for model_name, config in feature_extractors_configs.items()}
        params_to_optimize = [{'params': model.parameters()} for model in self.models.values()]
        params_to_optimize += [{'params': extractor.parameters()} for extractor in self.feature_extractors.values()]
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

    def foward(self, observations):
        q_val = [None for _ in range(len(self.agent_model_dict))]
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
                q_selected, _ = model(obs_vectorized, h)
                q_selected = q_selected[:,-1,:] # get the last output 
            # TODO: Add code for handling other types of models (e.g., CNNs)
            q_selected = q_selected.reshape(bs, n_agents, -1) # (B, N, Action Space)
            q_selected = q_selected.permute(1, 0, 2)  # (N, B, Action Space)

            for i, q in zip(idx, q_selected):
                q_val[i] = q
        
        q_val = torch.stack(q_val).to(self.device) # (N, B, Action Space)
        q_val = q_val.permute(1, 0, 2)  # (B, N, Action Space)

        return q_val


    def get_q_values(self, observations) -> Tensor:
        """
        Get the Q-values for the given observations.

        Returns:
            Tensor: Concatenated Q-value tensor across all agents.
        """

        q_values = [None for _ in range(len(self.agent_model_dict))]

        for (model_name, model), (_, fe) in zip(self.models.items(), self.feature_extractors.items()):
            selected_agents = self.model_to_agents[model_name]
            idx = self.model_to_agent_indices[model_name]
            obs = [Tensor(observations[ag]) for ag in selected_agents]
            selected_hidden_states = [self.hidden_states[ag] for ag in selected_agents]
            obs = stack(obs).to(device=self.device)
            
            if isinstance(model, RNNModel):
                selected_hidden_states = stack(selected_hidden_states).to(device=self.device)  # N, (D * \text{num\_layers}, H_{out})
                selected_hidden_states = selected_hidden_states.permute(1, 0, 2) # (D * \text{num\_layers}, N, H_{out})
                bs = obs.shape[0]
                ts = obs.shape[1]
                obs_shape = obs.shape[2:]
                obs = obs.reshape(bs*ts, *obs_shape)
                feature = fe(obs)
                feature = feature.reshape(bs, ts, -1)
                qv, hs = model(feature, selected_hidden_states)
                # qv shape: torch.Size([2, 1, 5]) (N：batch size, L: seq length, D * H_{out})
                # hs shape: torch.Size([1, 2, 128]) (D * \text{num\_layers}, N, H_{out})
                qv = qv[:,-1,:] # get the last output (N, D * H_{out})
                hs = hs.permute(1, 0, 2) # (N, D * \text{num\_layers}, H_{out})
            else:
                # TODO: Add code for handling other types of models (e.g., CNNs)
                qv = model(obs)
                hs = [None for _ in range(len(idx))]

            for i, ag, q, h in zip(idx, selected_agents, qv, hs):
                self.hidden_states[ag] = h
                q_values[i] = q

        q_values = stack(q_values)

        return q_values

    def act(self, observations, avail_actions, epsilon=0.0, eval_mode=True):
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
        q_values = self.get_q_values(observations)
        q_values = q_values.detach().to('cpu')
        random_choices = bernoulli(epsilon * ones(len(self.agent_model_dict)))

        random_actions = [avail_actions[key].sample() for key in avail_actions.keys()]
        random_actions = Tensor(random_actions).to(device='cpu')

        actions = random_choices * random_actions \
            + (1 - random_choices) * argmax(q_values, axis=-1)
        actions = actions.detach().to('cpu').numpy()
        actions = actions.astype(int).tolist()

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
    
    def to(self, device: str):
        for (_, model), (_, fe) in zip(self.models.items(), self.feature_extractors.items()):
            model.to(device)
            fe.to(device)
        self.device = device
        return self
    
    def eval(self):
        for (_, model), (_, fe) in zip(self.models.items(), self.feature_extractors.items()):
            model.eval()
            fe.eval()
        return self
    
    def train(self):
        for (_, model), (_, fe) in zip(self.models.items(), self.feature_extractors.items()):
            model.train()
            fe.train()
        return self