import numpy as np
import torch
import os
from copy import deepcopy
from typing import Dict, Any, List
from torch.nn import DataParallel
from marlite.algorithm.model.model_config import ModelConfig
from marlite.algorithm.model import TimeSeqModel, RNNModel, Conv1DModel, AttentionModel
from marlite.algorithm.agents.agent_group import AgentGroup
from marlite.util.optimizer_config import OptimizerConfig
from marlite.util.lr_scheduler_config import LRSchedulerConfig

class QMIXAgentGroup(AgentGroup):
    def __init__(self,
                agent_model_dict: Dict[str, str],
                model_configs: Dict[str, ModelConfig],
                feature_extractors_configs: Dict[str, ModelConfig],
                optimizer_config: OptimizerConfig,
                lr_scheduler_config: LRSchedulerConfig=None,
                device = 'cpu') -> None:
        super().__init__()
        self.device = device
        self.agent_model_dict = agent_model_dict
        self.models = {model_name: config.get_model() for model_name, config in model_configs.items()}
        self.feature_extractors = {model_name: config.get_model() for model_name, config in feature_extractors_configs.items()}
        self.params_to_optimize = [{'params': model.parameters()} for model in self.models.values()]
        self.params_to_optimize += [{'params': extractor.parameters()} for extractor in self.feature_extractors.values()]
        self.optimizer = optimizer_config.get_optimizer(self.params_to_optimize)
        self.lr_scheduler = None
        if lr_scheduler_config:
            self.lr_scheduler = lr_scheduler_config.get_lr_scheduler(self.optimizer)

        # Initialize model_to_agent dictionary and model_to_agent_indices dictionary
        self.model_to_agents = {model_name:[] for model_name in model_configs.keys()}
        self.model_to_agent_indices = {model_name:[] for model_name in model_configs.keys()}
        for i, (agent_name, model_name) in enumerate(self.agent_model_dict.items()):
            assert model_name in self.model_to_agents.keys(), f"Model {model_name} not found in model_configs"
            self.model_to_agents[model_name].append(agent_name)
            self.model_to_agent_indices[model_name].append(i)

        self._use_data_parallel = False
        self._is_compiled = False

        self.model_class_names = {}
        for model_name, model in self.models.items():
            if isinstance(model, RNNModel):
                self.model_class_names[model_name] = 'RNNModel'
            elif isinstance(model, Conv1DModel):
                self.model_class_names[model_name] = 'Conv1DModel'
            elif isinstance(model, AttentionModel):
                self.model_class_names[model_name] = 'AttentionModel'
            else:
                self.model_class_names[model_name] = model.__class__.__name__

    def forward(self, observations: torch.Tensor, traj_padding_mask: torch.Tensor, alive_mask: torch.Tensor) -> Dict[str, Any]:
        q_val = [None for _ in range(len(self.agent_model_dict))]
        for (model_name, model), (_, fe) in zip(self.models.items(),
                                                self.feature_extractors.items()):
            selected_agents = self.model_to_agents[model_name]
            idx = self.model_to_agent_indices[model_name]
            # observation shape: (Batch Size, Agent Number, Time Step, Feature Dimensions) (B, N, T, F)
            obs = observations[:,idx]
            # (B, N, T, (obs_shape)) -> (B*N*T, (obs_shape))
            bs = obs.shape[0]
            n_agents = len(selected_agents)
            ts = obs.shape[2]
            obs_shape = list(obs.shape[3:])

            # Use class name checking instead of isinstance
            model_class_name = self.model_class_names[model_name]
            if model_class_name == 'Conv1DModel':
                # (B, N, T, *(obs_shape)) -> (B*N*T, *(obs_shape))
                obs = obs.reshape(bs*n_agents*ts, *obs_shape).to(self.device)
                obs_vectorized = fe(obs) # (B*N*T, (obs_shape)) -> (B*N*T, F)
                obs_vectorized = obs_vectorized.reshape(bs*n_agents, ts, -1) # (B*N*T, F) -> (B*N, T, F)
                obs_vectorized = obs_vectorized.permute(0, 2, 1) # (B*N, T, F) -> (B*N, F, T)
                q_selected = model(obs_vectorized)
            elif model_class_name == 'RNNModel':
                obs = obs.reshape(bs*n_agents*ts, *obs_shape).to(self.device)
                obs_vectorized = fe(obs) # (B*N*T, (obs_shape)) -> (B*N*T, F)
                obs_vectorized = obs_vectorized.reshape(bs*n_agents, ts, -1) # (B*N*T, F) -> (B*N, T, F)
                model.train() # cudnn RNN backward can only be called in training mode
                q_selected = model(obs_vectorized)
            elif model_class_name == 'AttentionModel':
                obs = obs.reshape(bs*n_agents*ts, *obs_shape).to(self.device)
                obs_vectorized = fe(obs) # (B*N*T, (obs_shape)) -> (B*N*T, F)
                obs_vectorized = obs_vectorized.reshape(bs*n_agents, ts, -1) # (B*N*T, F) -> (B*N, T, F)
                mask = traj_padding_mask[:,idx]
                mask = mask.reshape(bs*n_agents, ts)
                q_selected = model(obs_vectorized, mask)
            else:
                obs = obs[:,:,-1, :] # (B, N, T, *(obs_shape)) -> (B, N, *(obs_shape))
                obs = obs.reshape(bs*n_agents, *obs_shape).to(self.device) # (B, N, *(obs_shape)) -> (B*N, *(obs_shape))
                obs_vectorized = fe(obs) # (B*N, *(obs_shape)) -> (B*N, F) /  if use_data_parallel (D, B/D*N, F)
                obs_vectorized = obs_vectorized.reshape(bs*n_agents, -1) # if use_data_parallel (D, B/D*N, F) -> (B*N, F)
                q_selected = model(obs_vectorized)

            q_selected = q_selected.reshape(bs, n_agents, -1) # (B, N, Action Space)
            q_selected = q_selected.permute(1, 0, 2)  # (N, B, Action Space)

            for i, q in zip(idx, q_selected):
                q_val[i] = q

        q_val = torch.stack(q_val).to(self.device) # (N, B, Action Space)
        q_val = q_val.permute(1, 0, 2)  # (B, N, Action Space)
        q_val = q_val * alive_mask.unsqueeze(-1)

        return {'q_val': q_val}

    def act(
            self,
            observations: Dict[str, np.ndarray],
            state: np.ndarray,
            avail_actions: Dict[str, Any],
            traj_padding_mask: np.ndarray,
            alive_agents: List[str],
            epsilon: float = .0
        ) -> Dict[str, Any]:
        """
        Select actions based on Q-values and exploration with action masking.

        Args:
            observations (dict): Dictionary mapping agent IDs to observation arrays.
                Each observation array should have shape compatible with the agent's observation space.
            state (numpy array): Global state information for generating communication graph.
            avail_actions (dict): Dictionary mapping agent IDs to either action masks (numpy arrays)
                                or action spaces (gymnasium.spaces.Space). Each mask is a 1D array where 1
                                indicates available actions, and 0 indicates unavailable actions.
            traj_padding_mask (numpy array): Padding mask for trajectory processing.
                This is used to handle variable-length trajectories by indicating which positions
                contain valid data vs padding.
            alive_agents (list): List of agent IDs that are currently alive/active in the environment.
                Only these agents will have their actions returned in the output.
            epsilon (float): Exploration rate.
                - 0.0: Always choose optimal actions (greedy)
                - 1.0: Always choose random actions (pure exploration)
                - Values between 0.0 and 1.0: Mix of exploration and exploitation

        Returns:
            dict: Selected actions for each agent, with action mask applied, and edge indices.
                - 'actions': Dictionary mapping only alive agents to their selected actions
                - 'all_actions': Dictionary mapping all agents to their selected actions (including dead ones)
        """
        # Convert observations to tensor format
        obs = [observations[agent] for agent in self.agent_model_dict.keys()]
        obs = np.stack(obs)
        obs = torch.tensor(obs).unsqueeze(0).to(dtype=torch.float, device=self.device)

        padding_mask = torch.tensor(traj_padding_mask, dtype=torch.bool) # (T)
        padding_mask = torch.stack([padding_mask] * len(self.agent_model_dict), dim=0) # (N, T)
        padding_mask = padding_mask.unsqueeze(0).to(self.device) # (1, N, T)

        alive_mask = torch.tensor([agent in set(alive_agents) for agent in self.agent_model_dict.keys()])
        alive_mask = alive_mask.unsqueeze(0).to(self.device)

        # Get Q-values
        with torch.no_grad():
            ret = self.forward(obs, padding_mask, alive_mask)
            q_values = ret['q_val']
            q_values = q_values.detach().cpu().numpy().squeeze()

        # Handle different types of avail_actions
        if isinstance(next(iter(avail_actions.values())), np.ndarray):
            # Action masking case
            action_masks = np.array([avail_actions[agent_id] for agent_id in self.agent_model_dict.keys()])

            # Apply action masks to Q-values
            masked_q_values = np.where(action_masks == 1, q_values, -np.inf)

            # Get optimal actions
            optimal_actions = np.argmax(masked_q_values, axis=-1).astype(np.int64)

            # Generate random actions according to action masks
            mask_probs = action_masks / np.sum(action_masks, axis=1, keepdims=True)
            random_actions = np.array([
                np.random.choice(len(probs), p=probs)
                for probs in mask_probs
            ]).astype(np.int64)
        else:
            # Action space sampling case
            optimal_actions = np.argmax(q_values, axis=-1).astype(np.int64)
            random_actions = np.array([avail_actions[agent].sample() for agent in self.agent_model_dict.keys()]).astype(np.int64)

        # Epsilon-greedy action selection
        random_choices = np.random.binomial(1, epsilon, len(self.agent_model_dict)).astype(np.int64)
        actions = random_choices * random_actions + (1 - random_choices) * optimal_actions
        actions = actions.astype(np.int64).tolist()

        # Create action dictionary
        all_actions = {agent: action for agent, action in zip(self.agent_model_dict.keys(), actions)}
        actual_actions = {agent: all_actions[agent] for agent in alive_agents}

        return {'actions': actual_actions, 'all_actions': all_actions}

    def set_agent_group_params(self, params: Dict[str, dict]) -> 'AgentGroup':
        feature_extractor_params = params.get("feature_extractor", {})
        model_params = params.get("model", {})
        for (model_name, fe), (_, model) in zip(
                self.feature_extractors.items(),
                self.models.items()
        ):
            fe.load_state_dict(feature_extractor_params[model_name])
            model.load_state_dict(model_params[model_name])

        return self

    def get_agent_group_params(self) -> Dict[str, dict]:
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

    def zero_grad(self) -> 'AgentGroup':
        self.optimizer.zero_grad()
        return self

    def step(self) -> 'AgentGroup':
        for p in self.params_to_optimize:
            torch.nn.utils.clip_grad_norm_(
                p['params'],
                max_norm=5.0
            )
        self.optimizer.step()
        return self

    def lr_scheduler_step(self, reward) -> 'AgentGroup':
        if not self.lr_scheduler:
            return self
        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.lr_scheduler.step(reward)
        else:
            self.lr_scheduler.step()
        return self

    def to(self, device: str) -> 'AgentGroup':
        for (_, model), (_, fe) in zip(self.models.items(), self.feature_extractors.items()):
            model.to(device)
            fe.to(device)
        self.device = device
        return self

    def eval(self) -> 'AgentGroup':
        for (_, model), (_, fe) in zip(self.models.items(), self.feature_extractors.items()):
            model.eval()
            fe.eval()
        return self

    def train(self) -> 'AgentGroup':
        for (_, model), (_, fe) in zip(self.models.items(), self.feature_extractors.items()):
            model.train()
            fe.train()
        return self

    def share_memory(self) -> 'AgentGroup':
        for (_, model), (_, fe) in zip(self.models.items(), self.feature_extractors.items()):
            model.share_memory()
            fe.share_memory()
        return self

    def wrap_data_parallel(self) -> 'AgentGroup':
        for id in self.models.keys():
            self.models[id] = DataParallel(self.models[id])
            self.feature_extractors[id] = DataParallel(self.feature_extractors[id])
        self._use_data_parallel = True
        return self

    def unwrap_data_parallel(self) -> 'AgentGroup':
        for id in self.models.keys():
            self.models[id] = self.models[id].module
            self.feature_extractors[id] = self.feature_extractors[id].module
        self._use_data_parallel = False
        return self

    def save_params(self, path: str) -> 'AgentGroup':
        os.makedirs(path, exist_ok=True)
        for (model_name, model), (_, fe) in zip(
            self.models.items(),
            self.feature_extractors.items()):
            model_dir = os.path.join(path, model_name)
            os.makedirs(model_dir, exist_ok=True)
            torch.save(fe.state_dict(), os.path.join(model_dir, 'feature_extractor.pth'))
            torch.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))
        return self

    def load_params(self, path: str) -> 'AgentGroup':
        for (model_name, model), (_, fe) in zip(
            self.models.items(),
            self.feature_extractors.items()):
            model_dir = os.path.join(path, model_name)
            fe.load_state_dict(torch.load(os.path.join(model_dir, 'feature_extractor.pth')))
            model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pth')))
        return self

    def compile_models(self) -> 'AgentGroup':
        for id in self.models.keys():
            self.models[id] = torch.compile(self.models[id])
            self.feature_extractors[id] = torch.compile(self.feature_extractors[id])
        self._is_compiled = True
        return self

    def reset(self) -> 'AgentGroup':
        return self