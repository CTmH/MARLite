import numpy as np
import torch
from typing import Dict, List, Any
from torch.nn import DataParallel
from marlite.algorithm.model.model_config import ModelConfig
from marlite.algorithm.agents.graph_agent_group import GraphAgentGroup
from marlite.algorithm.graph_builder import GraphBuilderConfig
from marlite.util.optimizer_config import OptimizerConfig
from marlite.util.lr_scheduler_config import LRSchedulerConfig

class G2ANetAgentGroup(GraphAgentGroup):
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
        super().__init__(
            agent_model_dict,
            feature_extractor_configs,
            encoder_configs,
            decoder_configs,
            graph_builder_config,
            graph_model_config,
            optimizer_config,
            lr_scheduler_config,
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
            if model_class_name == 'Conv1DModel':
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
            elif model_class_name == 'AttentionModel':
                obs = obs.reshape(bs*n_agents*ts, *obs_shape).to(self.device)
                obs_vectorized = fe(obs) # (B*N*T, (obs_shape)) -> (B*N*T, F)
                obs_vectorized = obs_vectorized.reshape(bs*n_agents, ts, -1) # (B*N*T, F) -> (B*N, T, F)
                mask = traj_padding_mask[:,idx]
                mask = mask.reshape(bs*n_agents, ts)
                msg_selected = enc(obs_vectorized, mask) # (B*N, T, F) -> (B*N, F)
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
        local_obs = msg

        # Build Graph
        adj_matrix, edge_indices = self.graph_builder(msg)

        # Communication between agents using the graph model.
        hidden_states = self.graph_model(msg, adj_matrix) # (B, N, Hidden Size)

        q_val = [None for _ in range(len(self.agent_model_dict))]
        emb_size = hidden_states.shape[-1] + local_obs.shape[-1]
        for model_name, dec in self.decoders.items():
            selected_agents = self.model_to_agents[model_name]
            idx = self.model_to_agent_indices[model_name]
            h = hidden_states[:,idx]
            lo = local_obs[:,idx] # (B, N, F)
            h = torch.Tensor(h) # (B, N, Hidden Size)
            bs = h.shape[0]
            n_agents = len(selected_agents)
            emb = torch.cat((h, lo), dim=-1)
            emb = emb.reshape(bs*n_agents, emb_size) # (B*N, Hidden Size)
            q_selected = dec(emb)
            q_selected = q_selected.reshape(bs, n_agents, -1) # (B, N, Action)
            q_selected = q_selected.permute(1, 0, 2)  # (N, B, Action)

            for i, m in zip(idx, q_selected):
                q_val[i] = m

        q_val = torch.stack(q_val).to(self.device) # (N, B, F)
        q_val = q_val.permute(1, 0, 2)  # (B, N, F)

        return {'q_val': q_val, 'edge_indices': edge_indices}

    def wrap_data_parallel(self) -> 'GraphAgentGroup':
        for id in self.encoders.keys():
            self.encoders[id] = DataParallel(self.encoders[id])
            self.feature_extractors[id] = DataParallel(self.feature_extractors[id])
            self.decoders[id] = DataParallel(self.decoders[id])
        #self.graph_builder = DataParallel(self.graph_builder)
        self.graph_model = DataParallel(self.graph_model)
        self._use_data_parallel = True
        return self

    def unwrap_data_parallel(self) -> 'GraphAgentGroup':
        for id in self.encoders.keys():
            self.encoders[id] = self.encoders[id].module
            self.feature_extractors[id] = self.feature_extractors[id].module
            self.decoders[id] = self.decoders[id].module
        #self.graph_builder = self.graph_builder.module
        self.graph_model = self.graph_model.module
        self._use_data_parallel = False
        return self

    def compile_models(self) -> 'GraphAgentGroup':
        for id in self.encoders.keys():
            self.encoders[id] = torch.compile(self.encoders[id])
            self.feature_extractors[id] = torch.compile(self.feature_extractors[id])
            self.decoders[id] = torch.compile(self.decoders[id])
        self.graph_builder = torch.compile(self.graph_builder)
        self.graph_model = torch.compile(self.graph_model)
        self._is_compiled = True
        return self