import numpy as np
import torch
import os
from torch.optim import Optimizer
from copy import deepcopy
from typing import Dict, Tuple, Type, List, Any
from torch.nn import DataParallel
from torch_geometric.data import Batch, Data
from torch_geometric.utils import unbatch
from ..model.model_config import ModelConfig
from ..model import TimeSeqModel, RNNModel
from .agent_group import AgentGroup
from ..graph_builder import GraphBuilderConfig
from src.util.optimizer_config import OptimizerConfig

class GNNAgentGroup(AgentGroup):
    def __init__(self,
                agent_model_dict: Dict[str, str],
                feature_extractor_configs: Dict[str, ModelConfig],
                encoder_configs: Dict[str, ModelConfig],
                decoder_configs: Dict[str, ModelConfig],
                graph_builder_config: GraphBuilderConfig,
                graph_model_config: ModelConfig,
                optimizer_config: OptimizerConfig,
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

        # Initialize model_to_agent dictionary and model_to_agent_indices dictionary
        self.model_to_agents = {model_name:[] for model_name in encoder_configs.keys()}
        self.model_to_agent_indices = {model_name:[] for model_name in encoder_configs.keys()}
        for i, (agent_name, model_name) in enumerate(self.agent_model_dict.items()):
            assert model_name in self.model_to_agents.keys(), f"Model {model_name} not found in model_configs"
            self.model_to_agents[model_name].append(agent_name)
            self.model_to_agent_indices[model_name].append(i)

    def forward(self,
                observations: Dict[str, np.ndarray],
                states: np.ndarray,
                edge_indices: Type[List[np.ndarray] | None] = None) -> Dict[str, Any]:
        msg = [None for _ in range(len(self.agent_model_dict))]
        for (model_name, fe), (_, enc) in zip(self.feature_extractors.items(),
                                                self.encoders.items()):
            selected_agents = self.model_to_agents[model_name]
            idx = self.model_to_agent_indices[model_name]
            # observation shape: (Batch Size, Agent Number, Time Step, Feature Dimensions) (B, N, T, F)
            obs = observations[:,idx]
            obs = torch.Tensor(obs) # (B, N, T, *(obs_shape))
            bs = obs.shape[0]
            n_agents = len(selected_agents)
            ts = obs.shape[2]
            obs_shape = list(obs.shape[3:])
            if isinstance(enc, TimeSeqModel):
                # (B, N, T, *(obs_shape)) -> (B*N*T, *(obs_shape))
                obs = obs.reshape(bs*n_agents*ts, *obs_shape).to(self.device)
                obs_vectorized = fe(obs) # (B*N*T, (obs_shape)) -> (B*N*T, F)
                obs_vectorized = obs_vectorized.reshape(bs*n_agents, ts, -1) # (B*N*T, F) -> (B*N, T, F)
                if isinstance(enc, RNNModel):
                    enc.train() # cudnn RNN backward can only be called in training mode
                msg_selected = enc(obs_vectorized) # (B*N, T, F) -> (B*N, F)
            else:
                obs = obs[:,:,-1, :] # (B, N, T, *(obs_shape)) -> (B, N, *(obs_shape))
                obs = obs.reshape(bs*n_agents, *obs_shape).to(self.device) # (B, N, *(obs_shape)) -> (B*N, *(obs_shape))
                obs_vectorized = fe(obs) # (B*N, *(obs_shape)) -> (B*N, F)
                msg_selected = enc(obs_vectorized) # (B*N, F) -> (B*N, F)

            msg_selected = msg_selected.reshape(bs, n_agents, -1) # (B, N, F)
            msg_selected = msg_selected.permute(1, 0, 2)  # (N, B, F)

            for i, m in zip(idx, msg_selected):
                msg[i] = m

        msg = torch.stack(msg).to(self.device) # (N, B, F)
        msg = msg.permute(1, 0, 2)  # (B, N, F)

        # Build Graph
        if edge_indices is None:  # If edge_indices are not provided
            adj_matrices, edge_indices = self.graph_builder(states)

        # Communication between agents using the graph model.
        batch_data = [None for i in range(bs)]
        for i in range(bs):
            batch_data[i] = Data(
                x = msg[i],
                edge_index = torch.Tensor(edge_indices[i]).to(device=self.device, dtype=torch.int)
            )
        batch_data = Batch.from_data_list(batch_data)
        x, e, batch = batch_data.x, batch_data.edge_index, batch_data.batch
        batch_h = self.graph_model(x, e)
        hidden_states = unbatch(batch_h, batch) # (B, N, Hidden Size)
        hidden_states = torch.stack(hidden_states)

        q_val = [None for _ in range(len(self.agent_model_dict))]
        for model_name, dec in self.decoders.items():
            selected_agents = self.model_to_agents[model_name]
            idx = self.model_to_agent_indices[model_name]
            h = hidden_states[:,idx]
            h = torch.Tensor(h) # (B, N, Hidden Size)
            bs = h.shape[0]
            n_agents = len(selected_agents)
            hidden_size = h.shape[-1]
            h = h.reshape(bs*n_agents, hidden_size) # (B*N, Hidden Size)
            q_selected = dec(h)
            q_selected = q_selected.reshape(bs, n_agents, -1) # (B, N, Action)
            q_selected = q_selected.permute(1, 0, 2)  # (N, B, Action)

            for i, m in zip(idx, q_selected):
                q_val[i] = m

        q_val = torch.stack(q_val).to(self.device) # (N, B, F)
        q_val = q_val.permute(1, 0, 2)  # (B, N, F)

        return {'q_val': q_val, 'edge_indices': edge_indices}

    def act(self, observations: Dict[str, np.ndarray], state: np.ndarray, avail_actions: Dict, epsilon: float = .0):
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
        obs = [observations[ag] for ag in self.agent_model_dict.keys()]
        obs = np.stack(obs)
        obs = np.expand_dims(obs, axis=0)
        with torch.no_grad():
            ret = self.forward(obs, np.expand_dims(state, axis=0))
            q_values = ret['q_val'] # (1, num_agents, num_actions)
            q_values = q_values.detach().cpu().numpy().squeeze()
        random_choices = np.random.binomial(1, epsilon, len(self.agent_model_dict)).astype(np.int64)
        random_actions = [avail_actions[key].sample() for key in avail_actions.keys()]
        random_actions = np.array(random_actions).astype(np.int64)

        actions = random_choices * random_actions \
        + (1 - random_choices) * np.argmax(q_values, axis=-1).astype(np.int64)
        actions = actions.astype(np.int64).tolist()

        actions = {agent_id: action for agent_id, action in zip(self.agent_model_dict.keys(), actions)}

        return {'actions': actions, 'edge_indices': ret['edge_indices'][0]}

    def set_agent_group_params(self, params: Dict[str, dict]) -> Type[AgentGroup]:
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

    def get_agent_group_params(self) -> Type[AgentGroup]:
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

    def zero_grad(self) -> Type[AgentGroup]:
        self.optimizer.zero_grad()
        return self

    def step(self) -> Type[AgentGroup]:
        for p in self.params_to_optimize:
            torch.nn.utils.clip_grad_norm_(
                p['params'],
                max_norm=5.0
            )
        self.optimizer.step()
        return self

    def to(self, device: str) -> Type[AgentGroup]:
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

    def eval(self) -> Type[AgentGroup]:
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

    def train(self) -> Type[AgentGroup]:
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

    def share_memory(self) -> Type[AgentGroup]:
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

    def wrap_data_parallel(self) -> Type[AgentGroup]:
        for id in self.models.keys():
            self.encoders[id] = DataParallel(self.encoders[id])
            self.feature_extractors[id] = DataParallel(self.feature_extractors[id])
            self.decoders[id] = DataParallel(self.decoders[id])
        self.graph_builder = DataParallel(self.graph_builder)
        self.graph_model = DataParallel(self.graph_model)
        return self

    def unwrap_data_parallel(self) -> Type[AgentGroup]:
        for id in self.models.keys():
            self.encoders[id] = self.encoders[id].module
            self.feature_extractors[id] = self.feature_extractors[id].module
            self.decoders[id] = self.decoders[id].module
        self.graph_builder = self.graph_builder.module
        self.graph_model = self.graph_model.module
        return self

    def save_params(self, path: str) -> Type[AgentGroup]:
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

    def load_params(self, path: str) -> Type[AgentGroup]:
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

    def reset(self) -> Type[AgentGroup]:
        self.graph_builder.reset()
        return self