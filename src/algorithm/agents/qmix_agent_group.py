import numpy as np
import torch
import os
from torch.optim import Optimizer
from copy import deepcopy
from typing import Dict
from torch import Tensor
from torch import bernoulli, ones, argmax, stack
from ..model.model_config import ModelConfig
from ..model import TimeSeqModel, RNNModel
from .agent_group import AgentGroup
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
        self.params_to_optimize = [{'params': model.parameters()} for model in self.models.values()]
        self.params_to_optimize += [{'params': extractor.parameters()} for extractor in self.feature_extractors.values()]
        self.optimizer = optimizer_config.get_optimizer(self.params_to_optimize)

        # Initialize model_to_agent dictionary and model_to_agent_indices dictionary
        self.model_to_agents = {model_name:[] for model_name in model_configs.keys()}
        self.model_to_agent_indices = {model_name:[] for model_name in model_configs.keys()}
        for i, (agent_name, model_name) in enumerate(self.agent_model_dict.items()):
            assert model_name in self.model_to_agents.keys(), f"Model {model_name} not found in model_configs"
            self.model_to_agents[model_name].append(agent_name)
            self.model_to_agent_indices[model_name].append(i)

    def forward(self, observations) -> Tensor:
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
            if isinstance(model, TimeSeqModel):
                # (B, N, T, *(obs_shape)) -> (B*N*T, *(obs_shape))
                obs = obs.reshape(bs*n_agents*ts, *obs_shape).to(self.device)
                obs_vectorized = fe(obs) # (B*N*T, (obs_shape)) -> (B*N*T, F)
                obs_vectorized = obs_vectorized.reshape(bs*n_agents, ts, -1) # (B*N*T, F) -> (B*N, T, F)
                # cudnn RNN backward can only be called in training mode
                if isinstance(model, RNNModel):
                    model.train()
            else:
                obs = obs[:,:,-1, :] # (B, N, T, *(obs_shape)) -> (B, N, *(obs_shape))
                obs = obs.reshape(bs*n_agents, *obs_shape).to(self.device) # (B, N, *(obs_shape)) -> (B*N, *(obs_shape))
                obs_vectorized = fe(obs) # (B*N, *(obs_shape)) -> (B*N, F)
            q_selected = model(obs_vectorized)
            q_selected = q_selected.reshape(bs, n_agents, -1) # (B, N, Action Space)
            q_selected = q_selected.permute(1, 0, 2)  # (N, B, Action Space)

            for i, q in zip(idx, q_selected):
                q_val[i] = q
        
        q_val = torch.stack(q_val).to(self.device) # (N, B, Action Space)
        q_val = q_val.permute(1, 0, 2)  # (B, N, Action Space)

        return q_val

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
        obs = [observations[ag] for ag in self.agent_model_dict.keys()]
        obs = np.stack(obs)
        obs = np.expand_dims(obs, axis=0)
        q_values = self.forward(obs)
        q_values = q_values.detach().cpu().numpy().squeeze()
        random_choices = np.random.binomial(1, epsilon, len(self.agent_model_dict)).astype(np.int64)
        random_actions = [avail_actions[key].sample() for key in avail_actions.keys()]
        random_actions = np.array(random_actions).astype(np.int64)

        actions = random_choices * random_actions \
        + (1 - random_choices) * np.argmax(q_values, axis=-1).astype(np.int64)
        actions = actions.astype(np.int64).tolist()

        actions = {agent_id: action for agent_id, action in zip(self.agent_model_dict.keys(), actions)}
        
        return actions
    
    def set_agent_group_params(self, params: Dict[str, dict]):
        feature_extractor_params = params.get("feature_extractor", {})
        model_params = params.get("model", {})
        for (model_name, fe), (_, model) in zip(
                self.feature_extractors.items(),
                self.models.items()
        ):
            fe.load_state_dict(feature_extractor_params[model_name])
            model.load_state_dict(model_params[model_name])

        return self
    
    def get_agent_group_params(self):
        feature_extractor_params = {
            model_name: deepcopy(fe.state_dict()) 
            for model_name, fe in self.feature_extractors.items()
        }
        model_params = {
            model_name: deepcopy(model.state_dict()) 
            for model_name, model in self.models.items()
        }
        params = {
            "feature_extractor": feature_extractor_params,
            "model": model_params,
        }
        return params

    def zero_grad(self):
        self.optimizer.zero_grad()
        return self
    
    def step(self):
        for p in self.params_to_optimize:
            torch.nn.utils.clip_grad_norm_(
                p['params'],
                max_norm=5.0
            )
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
    
    def save_params(self, path: str):
        os.makedirs(path, exist_ok=True)
        for (model_name, model), (_, fe) in zip(
            self.models.items(),
            self.feature_extractors.items()):
            model_dir = os.path.join(path, model_name)
            os.makedirs(model_dir, exist_ok=True)
            torch.save(fe.state_dict(), os.path.join(model_dir, 'feature_extractor.pth'))
            torch.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))
        return self
    
    def load_params(self, path: str):
        for (model_name, model), (_, fe) in zip(
            self.models.items(),
            self.feature_extractors.items()):
            model_dir = os.path.join(path, model_name)
            fe.load_state_dict(torch.load(os.path.join(model_dir, 'feature_extractor.pth')))
            model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pth')))
        return self