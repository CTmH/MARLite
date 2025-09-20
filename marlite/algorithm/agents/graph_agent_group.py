import numpy as np
import torch
import os
from copy import deepcopy
from typing import Dict, List, Any
from torch.nn import DataParallel
from marlite.algorithm.model import TimeSeqModel, RNNModel, Conv1DModel, AttentionModel
from marlite.algorithm.model.model_config import ModelConfig
from marlite.algorithm.agents.agent_group import AgentGroup
from marlite.algorithm.graph_builder import GraphBuilderConfig
from marlite.util.optimizer_config import OptimizerConfig
from marlite.util.lr_scheduler_config import LRSchedulerConfig

class GraphAgentGroup(AgentGroup):
    def __init__(self,
                agent_model_dict: Dict[str, str],
                feature_extractor_configs: Dict[str, ModelConfig],
                encoder_configs: Dict[str, ModelConfig],
                decoder_configs: Dict[str, ModelConfig],
                graph_builder_config: GraphBuilderConfig,
                graph_model_config: ModelConfig,
                optimizer_config: OptimizerConfig,
                lr_scheduler_config: LRSchedulerConfig=None,
                device = 'cpu') -> None:
        super().__init__()
        self.device = device
        self.agent_model_dict = agent_model_dict
        self.feature_extractors = {model_name: config.get_model() for model_name, config in feature_extractor_configs.items()}
        self.encoders = {model_name: config.get_model() for model_name, config in encoder_configs.items()}
        self.models = self.encoders # For compatibility
        self.decoders = {model_name: config.get_model() for model_name, config in decoder_configs.items()}
        self.graph_model = graph_model_config.get_model()  # Graph model for message passing
        self.graph_builder = graph_builder_config.get_graph_builder()
        self.params_to_optimize = [{'params': extractor.parameters()} for extractor in self.feature_extractors.values()]
        self.params_to_optimize += [{'params': encoder.parameters()} for encoder in self.encoders.values()]
        self.params_to_optimize += [{'params': decoder.parameters()} for decoder in self.decoders.values()]
        self.params_to_optimize += [{'params': self.graph_builder.parameters()}]
        self.params_to_optimize += [{'params': self.graph_model.parameters()}]
        self.optimizer = optimizer_config.get_optimizer(self.params_to_optimize)
        self.lr_scheduler = None
        if lr_scheduler_config:
            self.lr_scheduler = lr_scheduler_config.get_lr_scheduler(self.optimizer)

        # Initialize model_to_agent dictionary and model_to_agent_indices dictionary
        self.model_to_agents = {model_name:[] for model_name in encoder_configs.keys()}
        self.model_to_agent_indices = {model_name:[] for model_name in encoder_configs.keys()}
        for i, (agent_name, model_name) in enumerate(self.agent_model_dict.items()):
            assert model_name in self.model_to_agents.keys(), f"Model {model_name} not found in model_configs"
            self.model_to_agents[model_name].append(agent_name)
            self.model_to_agent_indices[model_name].append(i)

        self._use_data_parallel = False
        self._is_compiled = False

        self.model_class_names = {}
        for model_name, model in self.encoders.items():
            if isinstance(model, RNNModel):
                self.model_class_names[model_name] = 'RNNModel'
            elif isinstance(model, Conv1DModel):
                self.model_class_names[model_name] = 'Conv1DModel'
            elif isinstance(model, AttentionModel):
                self.model_class_names[model_name] = 'AttentionModel'
            else:
                self.model_class_names[model_name] = model.__class__.__name__

    def forward(self,
                observations: Dict[str, np.ndarray],
                states: np.ndarray,
                traj_padding_mask: torch.Tensor,
                alive_mask: torch.Tensor,
                edge_indices: List[np.ndarray] | None = None
        ) -> Dict[str, Any]:
        raise NotImplementedError

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
            state (numpy array): Global state information for generating communication graph.
            avail_actions (dict): Dictionary mapping agent IDs to either action masks (numpy arrays)
                                or action spaces (gymnasium.spaces.Space). Each mask is a 1D array where 1
                                indicates available actions, and 0 indicates unavailable actions.
            epsilon (float): Exploration rate.

        Returns:
            dict: Selected actions for each agent, with action mask applied, and edge indices.
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

        with torch.no_grad():
            ret = self.forward(obs, np.expand_dims(state, axis=0), padding_mask, alive_mask)
            q_values = ret['q_val']  # (1, num_agents, num_actions)
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
            random_actions = np.array([avail_actions[key].sample() for key in avail_actions.keys()]).astype(np.int64)

        # Epsilon-greedy action selection
        random_choices = np.random.binomial(1, epsilon, len(self.agent_model_dict)).astype(np.int64)
        actions = random_choices * random_actions + (1 - random_choices) * optimal_actions
        actions = actions.astype(np.int64).tolist()

        # Create action dictionary
        all_actions = {agent: action for agent, action in zip(self.agent_model_dict.keys(), actions)}
        actual_actions = {agent: all_actions[agent] for agent in alive_agents}

        return {'actions': actual_actions, 'all_actions': all_actions, 'edge_indices': ret['edge_indices'][0]}

    def set_agent_group_params(self, params: Dict[str, dict]) -> 'AgentGroup':
        feature_extractor_params = params.get("feature_extractor", {})
        encoder_params = params.get("encoder", {})
        decoder_params = params.get("decoder", {})
        graph_builder_params = params.get("graph_builder", {})
        graph_model_params = params.get("graph_model", {})
        for (model_name, enc), (_, fe), (_, dec) in zip(
                self.encoders.items(),
                self.feature_extractors.items(),
                self.decoders.items()
        ):
            enc.load_state_dict(encoder_params[model_name])
            fe.load_state_dict(feature_extractor_params[model_name])
            dec.load_state_dict(decoder_params[model_name])

        self.graph_builder.load_state_dict(graph_builder_params)
        self.graph_model.load_state_dict(graph_model_params)

        return self

    def get_agent_group_params(self) -> Dict[str, dict]:
        feature_extractor_params = {
            model_name: deepcopy(fe.state_dict())
            for model_name, fe in self.feature_extractors.items()
        }
        encoder_params = {
            model_name: deepcopy(model.state_dict())
            for model_name, model in self.encoders.items()
        }
        decoder_params = {
            model_name: deepcopy(dec.state_dict())
            for model_name, dec in self.decoders.items()
        }
        graph_builder_params = deepcopy(self.graph_builder.state_dict())
        comm_model_params = deepcopy(self.graph_model.state_dict())
        params = {
            "encoder": encoder_params,
            "feature_extractor": feature_extractor_params,
            "decoder": decoder_params,
            "graph_builder": graph_builder_params,
            "graph_model": comm_model_params,
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
        for (_, enc), (_, fe), (_, dec) in zip(
            self.encoders.items(),
            self.feature_extractors.items(),
            self.decoders.items()):
            enc.to(device)
            fe.to(device)
            dec.to(device)
        self.graph_builder.to(device)
        self.graph_model.to(device)
        self.device = device
        return self

    def eval(self) -> 'AgentGroup':
        for (_, enc), (_, fe), (_, dec) in zip(
            self.encoders.items(),
            self.feature_extractors.items(),
            self.decoders.items()):
            enc.eval()
            fe.eval()
            dec.eval()
        self.graph_builder.eval()
        self.graph_model.eval()
        return self

    def train(self) -> 'AgentGroup':
        for (_, enc), (_, fe), (_, dec) in zip(
            self.encoders.items(),
            self.feature_extractors.items(),
            self.decoders.items()):
            enc.train()
            fe.train()
            dec.train()
        self.graph_builder.train()
        self.graph_model.train()
        return self

    def share_memory(self) -> 'AgentGroup':
        for (_, enc), (_, fe), (_, dec) in zip(
            self.encoders.items(),
            self.feature_extractors.items(),
            self.decoders.items()):
            enc.share_memory()
            fe.share_memory()
            dec.share_memory()
        self.graph_builder.share_memory()
        self.graph_model.share_memory()
        return self

    def wrap_data_parallel(self) -> 'AgentGroup':
        for id in self.encoders.keys():
            self.encoders[id] = DataParallel(self.encoders[id])
            self.feature_extractors[id] = DataParallel(self.feature_extractors[id])
            self.decoders[id] = DataParallel(self.decoders[id])
        self.graph_builder = DataParallel(self.graph_builder)
        #self.graph_model = DataParallel(self.graph_model)
        self._use_data_parallel = True
        return self

    def unwrap_data_parallel(self) -> 'AgentGroup':
        for id in self.encoders.keys():
            self.encoders[id] = self.encoders[id].module
            self.feature_extractors[id] = self.feature_extractors[id].module
            self.decoders[id] = self.decoders[id].module
        self.graph_builder = self.graph_builder.module
        #self.graph_model = self.graph_model.module
        self._use_data_parallel = False
        return self

    def save_params(self, path: str) -> 'AgentGroup':
        os.makedirs(path, exist_ok=True)
        for (model_name, enc), (_, fe), (_, dec) in zip(
            self.encoders.items(),
            self.feature_extractors.items(),
            self.decoders.items()):
            model_dir = os.path.join(path, model_name)
            os.makedirs(model_dir, exist_ok=True)
            torch.save(fe.state_dict(), os.path.join(model_dir, 'feature_extractor.pth'))
            torch.save(enc.state_dict(), os.path.join(model_dir, 'encoder.pth'))
            torch.save(dec.state_dict(), os.path.join(model_dir, 'decoder.pth'))
        torch.save(self.graph_builder.state_dict(), os.path.join(path, 'graph_builder.pth'))
        torch.save(self.graph_model.state_dict(), os.path.join(path, 'graph_model.pth'))
        return self

    def load_params(self, path: str) -> 'AgentGroup':
        for (model_name, enc), (_, fe), (_, dec) in zip(
            self.encoders.items(),
            self.feature_extractors.items(),
            self.decoders.items()):
            model_dir = os.path.join(path, model_name)
            fe.load_state_dict(torch.load(os.path.join(model_dir, 'feature_extractor.pth'),
                                          map_location=torch.device('cpu')))
            enc.load_state_dict(torch.load(os.path.join(model_dir, 'encoder.pth'),
                                           map_location=torch.device('cpu')))
            dec.load_state_dict(torch.load(os.path.join(model_dir, 'decoder.pth'),
                                           map_location=torch.device('cpu')))
        self.graph_builder.load_state_dict(torch.load(os.path.join(path, 'graph_builder.pth'),
                                                    map_location=torch.device('cpu')))
        self.graph_model.load_state_dict(torch.load(os.path.join(path, 'graph_model.pth'),
                                                    map_location=torch.device('cpu')))
        return self

    def compile_models(self) -> 'AgentGroup':
        for id in self.encoders.keys():
            self.encoders[id] = torch.compile(self.encoders[id])
            self.feature_extractors[id] = torch.compile(self.feature_extractors[id])
            self.decoders[id] = torch.compile(self.decoders[id])
        #self.graph_builder = torch.compile(self.graph_builder)
        self.graph_model = torch.compile(self.graph_model)
        self._is_compiled = True
        return self

    def reset(self) -> 'AgentGroup':
        if isinstance(self.graph_builder, DataParallel):
            self.graph_builder.module.reset()
        else:
            self.graph_builder.reset()
        return self