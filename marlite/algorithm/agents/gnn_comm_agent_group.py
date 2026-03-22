import os
import numpy as np
import torch
from typing import Dict, List, Any
from copy import deepcopy
from torch.nn.parallel import DistributedDataParallel as DDP
from marlite.algorithm.model.model_config import ModelConfig
from marlite.algorithm.agents.graph_agent_group import GraphAgentGroup
from marlite.algorithm.graph_builder import GraphBuilderConfig
from marlite.util.optimizer_config import OptimizerConfig
from marlite.util.lr_scheduler_config import LRSchedulerConfig
from marlite.util.prob_util import process_probabilistic_output


class ObsGNNCommAgentGroup(GraphAgentGroup):
    def __init__(
        self,
        agent_model_dict: Dict[str, str],
        feature_extractor_configs: Dict[str, ModelConfig],
        encoder_configs: Dict[str, ModelConfig],
        decoder_configs: Dict[str, ModelConfig],
        graph_builder_config: GraphBuilderConfig,
        graph_model_config: ModelConfig,
        optimizer_config: OptimizerConfig,
        lr_scheduler_config: LRSchedulerConfig = None,
        device="cpu",
    ) -> None:
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

    def _compute_local_state_estimates(self, msg, edge_indices):
        """Compute local state estimates from messages and local observations."""
        # Compute graph embeddings using parent method
        embedding = self.compute_graph_embeddings(msg, edge_indices)

        local_state_estimates = (
            embedding  # For non-probabilistic case, embedding is the estimate
        )

        return local_state_estimates, edge_indices

    def forward(
        self,
        observations: Dict[str, np.ndarray],
        states: np.ndarray,
        traj_padding_mask: torch.Tensor,
        alive_mask: torch.Tensor,
        edge_indices: List[np.ndarray] | None = None,
    ) -> Dict[str, Any]:
        msg, local_obs = self._process_observations(observations, traj_padding_mask)

        # Build Graph
        if edge_indices is None:  # If edge_indices are not provided
            adj_matrix, edge_indices = self.graph_builder(states)

        local_state_estimates, edge_indices = self._compute_local_state_estimates(
            msg, edge_indices
        )

        hidden_states = torch.cat(
            (local_state_estimates, local_obs), dim=-1
        )  # (B, N, Hidden Size + F_local_obs)

        q_val = self._process_decoders(hidden_states)

        return {
            "q_val": q_val,
            "edge_indices": edge_indices,
            "local_state_estimates": local_state_estimates,
        }


class SeqGNNCommAgentGroup(GraphAgentGroup):
    def __init__(
        self,
        agent_model_dict: Dict[str, str],
        feature_extractor_configs: Dict[str, ModelConfig],
        encoder_configs: Dict[str, ModelConfig],
        decoder_configs: Dict[str, ModelConfig],
        graph_builder_config: GraphBuilderConfig,
        graph_model_config: ModelConfig,
        optimizer_config: OptimizerConfig,
        lr_scheduler_config: LRSchedulerConfig = None,
        device="cpu",
    ) -> None:
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

    def _compute_local_state_estimates(self, msg, edge_indices):
        """Compute local state estimates from messages and local observations."""
        # Compute graph embeddings using parent method
        embedding = self.compute_graph_embeddings(msg, edge_indices)

        local_state_estimates = (
            embedding  # For non-probabilistic case, embedding is the estimate
        )

        return local_state_estimates, edge_indices

    def forward(
        self,
        observations: Dict[str, np.ndarray],
        states: np.ndarray,
        traj_padding_mask: torch.Tensor,
        alive_mask: torch.Tensor,
        edge_indices: List[np.ndarray] | None = None,
    ) -> Dict[str, Any]:
        msg, local_obs = self._process_sequences(observations, traj_padding_mask)

        # Build Graph
        if edge_indices is None:  # If edge_indices are not provided
            adj_matrix, edge_indices = self.graph_builder(states)

        local_state_estimates, edge_indices = self._compute_local_state_estimates(
            msg, edge_indices
        )

        hidden_states = torch.cat(
            (local_state_estimates, local_obs), dim=-1
        )  # (B, N, Hidden Size + F)

        q_val = self._process_decoders(hidden_states)

        return {
            "q_val": q_val,
            "edge_indices": edge_indices,
            "local_state_estimates": local_state_estimates,
        }


class ProbObsGNNCommAgentGroup(ObsGNNCommAgentGroup):
    """Probabilistic observation-based GNN communication agent group.

    This class extends ObsGNNCommAgentGroup with probabilistic message generation.
    The graph model outputs both mean and variance for message generation.
    """

    def __init__(
        self,
        agent_model_dict: Dict[str, str],
        feature_extractor_configs: Dict[str, ModelConfig],
        encoder_configs: Dict[str, ModelConfig],
        decoder_configs: Dict[str, ModelConfig],
        graph_builder_config: GraphBuilderConfig,
        graph_model_config: ModelConfig,
        optimizer_config: OptimizerConfig,
        lr_scheduler_config: LRSchedulerConfig = None,
        deterministic_eval: bool = True,
        device="cpu",
    ) -> None:
        super().__init__(
            agent_model_dict=agent_model_dict,
            feature_extractor_configs=feature_extractor_configs,
            encoder_configs=encoder_configs,
            decoder_configs=decoder_configs,
            graph_builder_config=graph_builder_config,
            graph_model_config=graph_model_config,
            optimizer_config=optimizer_config,
            lr_scheduler_config=lr_scheduler_config,
            device=device,
        )
        self.deterministic_eval = deterministic_eval

    def _compute_local_state_estimates(self, msg, edge_indices):
        """Compute local state estimates from messages and local observations with probabilistic output."""
        # Compute graph embeddings using parent method
        embedding = self.compute_graph_embeddings(msg, edge_indices)

        # Process probabilistic output
        deterministic = self.deterministic_eval and not self.graph_model.training
        estimates, log_var, mu, std = process_probabilistic_output(
            embedding, deterministic
        )  # All (B, N, F)

        return estimates, edge_indices, mu, std, log_var

    def forward(
        self,
        observations: Dict[str, np.ndarray],
        states: np.ndarray,
        traj_padding_mask: torch.Tensor,
        alive_mask: torch.Tensor,
        edge_indices: List[np.ndarray] | None = None,
    ) -> Dict[str, Any]:
        msg, local_obs = self._process_observations(observations, traj_padding_mask)

        # Build Graph
        if edge_indices is None:  # If edge_indices are not provided
            adj_matrix, edge_indices = self.graph_builder(states)

        estimates, edge_indices, mu, std, log_var = self._compute_local_state_estimates(
            msg, edge_indices
        )

        hidden_states = torch.cat(
            (estimates, local_obs), dim=-1
        )  # (B, N, Hidden Size + F_local_obs)

        q_val = self._process_decoders(hidden_states)

        return {
            "q_val": q_val,
            "edge_indices": edge_indices,
            "local_state_estimates": estimates,
            "mu": mu,
            "std": std,
            "log_var": log_var,
        }


class ProbSeqGNNCommAgentGroup(SeqGNNCommAgentGroup):
    """Probabilistic sequence-based GNN communication agent group.

    This class extends SeqGNNCommAgentGroup with probabilistic message generation.
    The graph model outputs both mean and variance for message generation.
    """

    def __init__(
        self,
        agent_model_dict: Dict[str, str],
        feature_extractor_configs: Dict[str, ModelConfig],
        encoder_configs: Dict[str, ModelConfig],
        decoder_configs: Dict[str, ModelConfig],
        graph_builder_config: GraphBuilderConfig,
        graph_model_config: ModelConfig,
        optimizer_config: OptimizerConfig,
        lr_scheduler_config: LRSchedulerConfig = None,
        deterministic_eval: bool = True,
        device="cpu",
    ) -> None:
        super().__init__(
            agent_model_dict=agent_model_dict,
            feature_extractor_configs=feature_extractor_configs,
            encoder_configs=encoder_configs,
            decoder_configs=decoder_configs,
            graph_builder_config=graph_builder_config,
            graph_model_config=graph_model_config,
            optimizer_config=optimizer_config,
            lr_scheduler_config=lr_scheduler_config,
            device=device,
        )
        self.deterministic_eval = deterministic_eval

    def _compute_local_state_estimates(self, msg, edge_indices):
        """Compute local state estimates from messages and local observations with probabilistic output."""
        # Compute graph embeddings using parent method
        embedding = self.compute_graph_embeddings(msg, edge_indices)

        # Process probabilistic output
        deterministic = self.deterministic_eval and not self.graph_model.training
        estimates, log_var, mu, std = process_probabilistic_output(
            embedding, deterministic
        )  # All (B, N, F)

        return estimates, edge_indices, mu, std, log_var

    def forward(
        self,
        observations: Dict[str, np.ndarray],
        states: np.ndarray,
        traj_padding_mask: torch.Tensor,
        alive_mask: torch.Tensor,
        edge_indices: List[np.ndarray] | None = None,
    ) -> Dict[str, Any]:
        msg, local_obs = self._process_sequences(observations, traj_padding_mask)

        # Build Graph
        if edge_indices is None:  # If edge_indices are not provided
            adj_matrix, edge_indices = self.graph_builder(states)

        estimates, edge_indices, mu, std, log_var = self._compute_local_state_estimates(
            msg, edge_indices
        )

        hidden_states = torch.cat(
            (estimates, local_obs), dim=-1
        )  # (B, N, Hidden Size + F_local_obs)

        q_val = self._process_decoders(hidden_states)

        return {
            "q_val": q_val,
            "edge_indices": edge_indices,
            "local_state_estimates": estimates,
            "mu": mu,
            "std": std,
            "log_var": log_var,
        }


class DualPathBasedGNNCommAgentGroup(GraphAgentGroup):
    """
    Base class for dual-path GNN communication agent groups.

    This dual-path architecture allows specialized representation learning:
    - One path for local observation encoding (used in Q-value computation)
    - One path for message generation (used in communication)
    """

    def __init__(
        self,
        agent_model_dict: Dict[str, str],
        feature_extractor_configs: Dict[str, ModelConfig],  # For observation processing
        encoder_configs: Dict[str, ModelConfig],
        decoder_configs: Dict[str, ModelConfig],
        graph_builder_config: GraphBuilderConfig,
        graph_model_config: ModelConfig,
        optimizer_config: OptimizerConfig,
        lr_scheduler_config: LRSchedulerConfig = None,
        enable_rl_grad_to_msg_aggr: bool = True,
        device="cpu",
    ) -> None:
        super().__init__(
            agent_model_dict=agent_model_dict,
            feature_extractor_configs=feature_extractor_configs,
            encoder_configs=encoder_configs,
            decoder_configs=decoder_configs,
            graph_builder_config=graph_builder_config,
            graph_model_config=graph_model_config,
            optimizer_config=optimizer_config,
            lr_scheduler_config=lr_scheduler_config,
            device=device,
        )

        # Separate feature extractors for message generation
        self.msg_feature_extractors = {}
        self.msg_encoders = {}

        self.enable_rl_grad_to_msg_aggr = enable_rl_grad_to_msg_aggr

    def _process_observations(self, observations, traj_padding_mask):
        """Process observations using dual-path architecture."""
        # Process through observation path (for Q-value computation)
        msg = [None for _ in range(len(self.agent_model_dict))]
        local_obs = [None for _ in range(len(self.agent_model_dict))]

        for model_name in self.feature_extractors.keys():
            enc = self.encoders[model_name]
            fe = self.feature_extractors[model_name]
            msg_fe = self.msg_feature_extractors[model_name]
            selected_agents = self.model_to_agents[model_name]
            idx = self.model_to_agent_indices[model_name]
            # observation shape: (Batch Size, Agent Number, Time Step, Feature Dimensions) (B, N, T, F)
            obs = observations[:, idx]  # (B, N, T, *(obs_shape))
            bs = obs.shape[0]
            n_agents = len(selected_agents)
            ts = obs.shape[2]
            obs_shape = list(obs.shape[3:])

            last_obs = obs[:, :, -1, :]
            last_obs = last_obs.reshape(bs * n_agents, *obs_shape).to(self.device)
            msg_selected = msg_fe(last_obs)  # (B*N, F)
            msg_selected = msg_selected.reshape(bs, n_agents, -1)
            msg_selected = msg_selected.permute(1, 0, 2)  # (N, B, F)

            # Use class name checking instead of isinstance
            model_class_name = self.model_class_names[model_name]
            if model_class_name == "Conv1DModel":
                obs = obs.reshape(bs * n_agents * ts, *obs_shape).to(
                    self.device
                )  # (B, N, T, *(obs_shape)) -> (B*N*T, *(obs_shape))
                obs_vectorized = fe(obs)  # (B*N*T, (obs_shape)) -> (B*N*T, F)
                obs_vectorized = obs_vectorized.reshape(
                    bs * n_agents, ts, -1
                )  # (B*N*T, F) -> (B*N, T, F)
                obs_vectorized = obs_vectorized.permute(
                    0, 2, 1
                )  # (B*N, T, F) -> (B*N, F, T)
                local_obs_selected = enc(obs_vectorized)  # (B*N, F, T) -> (B*N, F)
            elif model_class_name == "RNNModel":
                obs = obs.reshape(bs * n_agents * ts, *obs_shape).to(self.device)
                obs_vectorized = fe(obs)  # (B*N*T, (obs_shape)) -> (B*N*T, F)
                obs_vectorized = obs_vectorized.reshape(
                    bs * n_agents, ts, -1
                )  # (B*N*T, F) -> (B*N, T, F)
                enc.train()  # cudnn RNN backward can only be called in training mode
                local_obs_selected = enc(obs_vectorized)  # (B*N, T, F) -> (B*N, F)
            elif model_class_name == "AttentionModel":
                obs = obs.reshape(bs * n_agents * ts, *obs_shape).to(self.device)
                obs_vectorized = fe(obs)  # (B*N*T, (obs_shape)) -> (B*N*T, F)
                obs_vectorized = obs_vectorized.reshape(
                    bs * n_agents, ts, -1
                )  # (B*N*T, F) -> (B*N, T, F)
                mask = traj_padding_mask[:, idx]
                mask = mask.reshape(bs * n_agents, ts)
                local_obs_selected = enc(
                    obs_vectorized, mask
                )  # (B*N, T, F) -> (B*N, F)
            else:
                obs = obs[
                    :, :, -1, :
                ]  # (B, N, T, *(obs_shape)) -> (B, N, *(obs_shape))
                obs = obs.reshape(bs * n_agents, *obs_shape).to(
                    self.device
                )  # (B, N, *(obs_shape)) -> (B*N, *(obs_shape))
                obs_vectorized = fe(obs)  # (B*N, *(obs_shape)) -> (B*N, F)
                local_obs_selected = enc(obs_vectorized)  # (B*N, F) -> (B*N, F)

            local_obs_selected = local_obs_selected.reshape(
                bs, n_agents, -1
            )  # (B, N, F)
            local_obs_selected = local_obs_selected.permute(1, 0, 2)  # (N, B, F)

            for i, m, lo in zip(idx, msg_selected, local_obs_selected):
                msg[i] = m
                local_obs[i] = lo

        msg = torch.stack(msg).to(self.device)  # (N, B, F)
        msg = msg.permute(1, 0, 2)  # (B, N, F)
        local_obs = torch.stack(local_obs).to(self.device)  # (N, B, F)
        local_obs = local_obs.permute(1, 0, 2)  # (B, N, F)

        return msg, local_obs

    def set_agent_group_params(self, params: Dict[str, dict]) -> "GraphAgentGroup":
        """Override to handle message feature extractors"""
        super().set_agent_group_params(params)

        msg_feature_extractor_params = params.get("msg_feature_extractor", {})
        for model_name, fe in self.msg_feature_extractors.items():
            if model_name in msg_feature_extractor_params:
                fe.load_state_dict(msg_feature_extractor_params[model_name])
        msg_encoders_params = params.get("msg_encoders", {})
        for model_name, fe in self.msg_encoders.items():
            if model_name in msg_encoders_params:
                fe.load_state_dict(msg_encoders_params[model_name])

        return self

    def get_agent_group_params(self) -> Dict[str, dict]:
        """Override to include message feature extractors"""
        params = super().get_agent_group_params()
        msg_feature_extractor_params = {
            model_name: deepcopy(fe.state_dict())
            for model_name, fe in self.msg_feature_extractors.items()
        }
        msg_encoders_params = {
            model_name: deepcopy(fe.state_dict())
            for model_name, fe in self.msg_encoders.items()
        }
        if self.msg_feature_extractors:
            params["msg_feature_extractor"] = msg_feature_extractor_params
        if self.msg_encoders:
            params["msg_encoders"] = msg_encoders_params
        return params

    def to(self, device: str) -> "GraphAgentGroup":
        """Move all components to device"""
        super().to(device)
        for _, fe in self.msg_feature_extractors.items():
            fe.to(device)
        for _, enc in self.msg_encoders.items():
            enc.to(device)
        return self

    def eval(self) -> "GraphAgentGroup":
        """Set all components to evaluation mode"""
        super().eval()
        for _, fe in self.msg_feature_extractors.items():
            fe.eval()
        for _, enc in self.msg_encoders.items():
            enc.eval()
        return self

    def train(self) -> "GraphAgentGroup":
        """Set all components to training mode"""
        super().train()
        for _, fe in self.msg_feature_extractors.items():
            fe.train()
        for _, enc in self.msg_encoders.items():
            enc.train()
        return self

    def share_memory(self) -> "GraphAgentGroup":
        """Share memory for all components"""
        super().share_memory()
        for _, fe in self.msg_feature_extractors.items():
            fe.share_memory()
        for _, enc in self.msg_encoders.items():
            enc.share_memory()
        return self

    def wrap_data_parallel(self, device_id: int = 0) -> "GraphAgentGroup":
        """Wrap all components with DistributedDataParallel"""
        device = f"cuda:{device_id}"
        super().wrap_data_parallel(device_id)
        for id in self.msg_feature_extractors.keys():
            self.msg_feature_extractors[id] = self.msg_feature_extractors[id].to(device)
            self.msg_feature_extractors[id] = DDP(
                self.msg_feature_extractors[id], device_ids=[device_id]
            )
        for id in self.msg_encoders.keys():
            self.msg_encoders[id] = self.msg_encoders[id].to(device)
            self.msg_encoders[id] = DDP(self.msg_encoders[id], device_ids=[device_id])
        return self

    def unwrap_data_parallel(self) -> "GraphAgentGroup":
        """Unwrap DistributedDataParallel from all components"""
        super().unwrap_data_parallel()
        for id in self.msg_feature_extractors.keys():
            self.msg_feature_extractors[id] = self.msg_feature_extractors[
                id
            ].module.cpu()
        for id in self.msg_encoders.keys():
            self.msg_encoders[id] = self.msg_encoders[id].module.cpu()
        return self

    def save_params(self, path: str) -> "GraphAgentGroup":
        """Save all parameters including message feature extractors and encoders"""
        super().save_params(path)
        os.makedirs(path, exist_ok=True)
        for model_name, fe in self.msg_feature_extractors.items():
            model_dir = os.path.join(path, model_name)
            os.makedirs(model_dir, exist_ok=True)
            torch.save(
                fe.state_dict(), os.path.join(model_dir, "msg_feature_extractor.pth")
            )
        for model_name, enc in self.msg_encoders.items():
            model_dir = os.path.join(path, model_name)
            os.makedirs(model_dir, exist_ok=True)
            torch.save(enc.state_dict(), os.path.join(model_dir, "msg_encoder.pth"))
        return self

    def load_params(self, path: str) -> "GraphAgentGroup":
        """Load all parameters including message feature extractors and encoders"""
        super().load_params(path)
        for model_name, fe in self.msg_feature_extractors.items():
            model_dir = os.path.join(path, model_name)
            fe.load_state_dict(
                torch.load(
                    os.path.join(model_dir, "msg_feature_extractor.pth"),
                    map_location=torch.device("cpu"),
                )
            )
        for model_name, enc in self.msg_encoders.items():
            model_dir = os.path.join(path, model_name)
            enc.load_state_dict(
                torch.load(
                    os.path.join(model_dir, "msg_encoder.pth"),
                    map_location=torch.device("cpu"),
                )
            )
        return self

    def compile_models(self) -> "GraphAgentGroup":
        """Compile all models for performance"""
        super().compile_models()
        for id in self.msg_feature_extractors.keys():
            self.msg_feature_extractors[id] = torch.compile(
                self.msg_feature_extractors[id]
            )
        for id in self.msg_encoders.keys():
            self.msg_encoders[id] = torch.compile(self.msg_encoders[id])
        return self


class DualPathObsGNNCommAgentGroup(DualPathBasedGNNCommAgentGroup):
    """Dual-path observation-based GNN communication agent group.

    This class uses separate paths for message generation and local observation encoding.
    """

    def __init__(
        self,
        agent_model_dict: Dict[str, str],
        feature_extractor_configs: Dict[str, ModelConfig],  # For observation processing
        msg_feature_extractor_configs: Dict[str, ModelConfig],  # For message generation
        encoder_configs: Dict[str, ModelConfig],
        decoder_configs: Dict[str, ModelConfig],
        graph_builder_config: GraphBuilderConfig,
        graph_model_config: ModelConfig,
        optimizer_config: OptimizerConfig,
        lr_scheduler_config: LRSchedulerConfig = None,
        enable_rl_grad_to_msg_aggr: bool = True,
        device="cpu",
    ) -> None:
        super().__init__(
            agent_model_dict=agent_model_dict,
            feature_extractor_configs=feature_extractor_configs,
            encoder_configs=encoder_configs,
            decoder_configs=decoder_configs,
            graph_builder_config=graph_builder_config,
            graph_model_config=graph_model_config,
            optimizer_config=optimizer_config,
            lr_scheduler_config=lr_scheduler_config,
            enable_rl_grad_to_msg_aggr=enable_rl_grad_to_msg_aggr,
            device=device,
        )

        # Separate feature extractors for message generation
        self.msg_feature_extractors = {
            model_name: config.get_model()
            for model_name, config in msg_feature_extractor_configs.items()
        }

        # Add message feature extractors to parameters to optimize
        self.params_to_optimize += [
            {"params": extractor.parameters()}
            for extractor in self.msg_feature_extractors.values()
        ]

        # Recreate optimizer with all parameters
        self.optimizer = optimizer_config.get_optimizer(self.params_to_optimize)
        self.lr_scheduler = None
        if lr_scheduler_config:
            self.lr_scheduler = lr_scheduler_config.get_lr_scheduler(self.optimizer)

    def _compute_local_state_estimates(self, msg, edge_indices):
        """Compute local state estimates from messages and local observations."""
        # Compute graph embeddings using parent method
        embedding = self.compute_graph_embeddings(msg, edge_indices)

        if not self.enable_rl_grad_to_msg_aggr:
            embedding = embedding.detach()

        local_state_estimates = (
            embedding  # For non-probabilistic case, embedding is the estimate
        )

        return local_state_estimates, edge_indices

    def forward(
        self,
        observations: Dict[str, np.ndarray],
        states: np.ndarray,
        traj_padding_mask: torch.Tensor,
        alive_mask: torch.Tensor,
        edge_indices: List[np.ndarray] | None = None,
    ) -> Dict[str, Any]:
        msg, local_obs = self._process_observations(observations, traj_padding_mask)

        # Build Graph
        if edge_indices is None:  # If edge_indices are not provided
            adj_matrix, edge_indices = self.graph_builder(states)

        local_state_estimates, edge_indices = self._compute_local_state_estimates(
            msg, edge_indices
        )

        hidden_states = torch.cat(
            (local_state_estimates, local_obs), dim=-1
        )  # (B, N, Hidden Size + F_local_obs)

        q_val = self._process_decoders(hidden_states)

        return {
            "q_val": q_val,
            "edge_indices": edge_indices,
            "local_state_estimates": local_state_estimates,
        }


class DualPathProbObsGNNCommAgentGroup(DualPathObsGNNCommAgentGroup):
    """Dual-path probabilistic observation-based GNN communication agent group.

    This class combines dual-path architecture with probabilistic message generation.
    """

    def __init__(
        self,
        agent_model_dict: Dict[str, str],
        feature_extractor_configs: Dict[str, ModelConfig],  # For observation processing
        msg_feature_extractor_configs: Dict[str, ModelConfig],  # For message generation
        encoder_configs: Dict[str, ModelConfig],
        decoder_configs: Dict[str, ModelConfig],
        graph_builder_config: GraphBuilderConfig,
        graph_model_config: ModelConfig,
        optimizer_config: OptimizerConfig,
        lr_scheduler_config: LRSchedulerConfig = None,
        enable_rl_grad_to_msg_aggr: bool = True,
        deterministic_eval: bool = True,
        device="cpu",
    ) -> None:
        super().__init__(
            agent_model_dict=agent_model_dict,
            feature_extractor_configs=feature_extractor_configs,
            msg_feature_extractor_configs=msg_feature_extractor_configs,
            encoder_configs=encoder_configs,
            decoder_configs=decoder_configs,
            graph_builder_config=graph_builder_config,
            graph_model_config=graph_model_config,
            optimizer_config=optimizer_config,
            lr_scheduler_config=lr_scheduler_config,
            enable_rl_grad_to_msg_aggr=enable_rl_grad_to_msg_aggr,
            device=device,
        )
        self.deterministic_eval = deterministic_eval

    def _compute_local_state_estimates(self, msg, edge_indices):
        """Compute local state estimates from messages and local observations with probabilistic output."""
        # Compute graph embeddings using parent method
        embedding = self.compute_graph_embeddings(msg, edge_indices)

        # Process probabilistic output
        deterministic = self.deterministic_eval and not self.graph_model.training
        estimates, log_var, mu, std = process_probabilistic_output(
            embedding, deterministic
        )  # All (B, N, F)
        if not self.enable_rl_grad_to_msg_aggr:
            estimates = estimates.detach()

        return estimates, edge_indices, mu, std, log_var

    def forward(
        self,
        observations: Dict[str, np.ndarray],
        states: np.ndarray,
        traj_padding_mask: torch.Tensor,
        alive_mask: torch.Tensor,
        edge_indices: List[np.ndarray] | None = None,
    ) -> Dict[str, Any]:
        msg, local_obs = self._process_observations(observations, traj_padding_mask)

        # Build Graph
        if edge_indices is None:  # If edge_indices are not provided
            adj_matrix, edge_indices = self.graph_builder(states)

        estimates, edge_indices, mu, std, log_var = self._compute_local_state_estimates(
            msg, edge_indices
        )

        hidden_states = torch.cat(
            (estimates, local_obs), dim=-1
        )  # (B, N, Hidden Size + F_local_obs)

        q_val = self._process_decoders(hidden_states)

        return {
            "q_val": q_val,
            "edge_indices": edge_indices,
            "local_state_estimates": estimates,
            "mu": mu,
            "std": std,
            "log_var": log_var,
        }
