import numpy as np
import torch
from typing import Dict, List, Any
from torch_geometric.data import Batch, Data
from torch_geometric.utils import unbatch
from marlite.algorithm.model.model_config import ModelConfig
from marlite.algorithm.model import TimeSeqModel, RNNModel
from marlite.algorithm.agents.graph_agent_group import GraphAgentGroup
from marlite.algorithm.graph_builder import GraphBuilderConfig
from marlite.util.optimizer_config import OptimizerConfig

class GNNAgentGroup(GraphAgentGroup):
    def __init__(self,
                agent_model_dict: Dict[str, str],
                feature_extractor_configs: Dict[str, ModelConfig],
                encoder_configs: Dict[str, ModelConfig],
                decoder_configs: Dict[str, ModelConfig],
                graph_builder_config: GraphBuilderConfig,
                graph_model_config: ModelConfig,
                optimizer_config: OptimizerConfig,
                device = 'cpu') -> None:
        super().__init__(
            agent_model_dict,
            feature_extractor_configs,
            encoder_configs,
            decoder_configs,
            graph_builder_config,
            graph_model_config,
            optimizer_config,
            device=device,
        )

    def forward(self,
                observations: Dict[str, np.ndarray],
                states: np.ndarray,
                traj_padding_mask: torch.Tensor,
                alive_mask: torch.Tensor,
                edge_indices: List[np.ndarray] | None = None
        ) -> Dict[str, Any]:
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

            model_class_name = self.model_class_names[model_name]
            if model_class_name == 'TimeSeqModel':
                # (B, N, T, *(obs_shape)) -> (B*N*T, *(obs_shape))
                obs = obs.reshape(bs*n_agents*ts, *obs_shape).to(self.device)
                obs_vectorized = fe(obs) # (B*N*T, (obs_shape)) -> (B*N*T, F)
                obs_vectorized = obs_vectorized.reshape(bs*n_agents, ts, -1) # (B*N*T, F) -> (B*N, T, F)
                obs_vectorized = obs_vectorized.permute(0, 2, 1) #  (B*N, T, F) -> (B*N, F, T)
                msg_selected = enc(obs_vectorized) # (B*N, F, T) -> (B*N, F)
            elif model_class_name == 'RNNModel':
                obs = obs.reshape(bs*n_agents*ts, *obs_shape).to(self.device)
                obs_vectorized = fe(obs) # (B*N*T, (obs_shape)) -> (B*N*T, F)
                obs_vectorized = obs_vectorized.reshape(bs*n_agents, ts, -1) # (B*N*T, F) -> (B*N, T, F)
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
            adj_matrix, edge_indices = self.graph_builder(states)

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