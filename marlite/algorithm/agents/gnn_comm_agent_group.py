import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any
from torch.nn.parallel import DistributedDataParallel as DDP
from marlite.algorithm.model.model_config import ModelConfig
from marlite.algorithm.agents.graph_agent_group import GraphAgentGroup
from marlite.algorithm.graph_builder import GraphBuilderConfig
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
    ) -> None:
        super().__init__(
            agent_model_dict,
            feature_extractor_configs,
            encoder_configs,
            decoder_configs,
            graph_builder_config,
            graph_model_config,
        )

    def _compute_local_state_estimates(self, msg, edge_indices):
        """Compute local state estimates from messages and local observations."""
        # Compute graph embeddings using parent method
        embedding = self.compute_graph_embeddings(msg, edge_indices)

        local_state_estimates = (
            embedding  # For non-probabilistic case, embedding is the estimate
        )

        return local_state_estimates

    def forward(
        self,
        observations: torch.Tensor,
        states: torch.Tensor,
        traj_padding_mask: torch.Tensor,
        alive_mask: torch.Tensor,
        edge_indices: List[np.ndarray] | None = None,
    ) -> Dict[str, Any]:
        msg, local_obs = self._process_observations(observations, traj_padding_mask)

        # Build Graph
        if edge_indices is None:  # If edge_indices are not provided
            adj_matrix, edge_indices = self.graph_builder(states)

        local_state_estimates = self._compute_local_state_estimates(msg, edge_indices)

        hidden_states = torch.cat(
            (local_state_estimates, local_obs), dim=-1
        )  # (B, N, Hidden Size + F_local_obs)

        q_val = self._process_decoders(hidden_states)
        q_val = q_val * alive_mask.unsqueeze(-1)

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
    ) -> None:
        super().__init__(
            agent_model_dict,
            feature_extractor_configs,
            encoder_configs,
            decoder_configs,
            graph_builder_config,
            graph_model_config,
        )

    def _compute_local_state_estimates(self, msg, edge_indices):
        """Compute local state estimates from messages and local observations."""
        # Compute graph embeddings using parent method
        embedding = self.compute_graph_embeddings(msg, edge_indices)

        local_state_estimates = (
            embedding  # For non-probabilistic case, embedding is the estimate
        )

        return local_state_estimates

    def forward(
        self,
        observations: Dict[str, np.ndarray],
        states: torch.Tensor,
        traj_padding_mask: torch.Tensor,
        alive_mask: torch.Tensor,
        edge_indices: List[np.ndarray] | None = None,
    ) -> Dict[str, Any]:
        msg, local_obs = self._process_observations(observations, traj_padding_mask)

        # Build Graph
        if edge_indices is None:  # If edge_indices are not provided
            adj_matrix, edge_indices = self.graph_builder(states)

        local_state_estimates = self._compute_local_state_estimates(msg, edge_indices)

        hidden_states = torch.cat(
            (local_state_estimates, local_obs), dim=-1
        )  # (B, N, Hidden Size + F)

        q_val = self._process_decoders(hidden_states)
        q_val = q_val * alive_mask.unsqueeze(-1)

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
        deterministic_eval: bool = True,
    ) -> None:
        super().__init__(
            agent_model_dict=agent_model_dict,
            feature_extractor_configs=feature_extractor_configs,
            encoder_configs=encoder_configs,
            decoder_configs=decoder_configs,
            graph_builder_config=graph_builder_config,
            graph_model_config=graph_model_config,
        )
        self.deterministic_eval = deterministic_eval

    def _compute_local_state_estimates(self, msg, edge_indices):
        """Compute local state estimates from messages and local observations with probabilistic output."""
        # Compute graph embeddings using parent method
        embedding = self.compute_graph_embeddings(msg, edge_indices)

        # Process probabilistic output
        deterministic = self.deterministic_eval and not self.training
        estimates, log_var, mu, std = process_probabilistic_output(
            embedding, deterministic
        )  # All (B, N, F)

        return estimates, mu, std, log_var

    def forward(
        self,
        observations: Dict[str, np.ndarray],
        states: torch.Tensor,
        traj_padding_mask: torch.Tensor,
        alive_mask: torch.Tensor,
        edge_indices: List[np.ndarray] | None = None,
    ) -> Dict[str, Any]:
        msg, local_obs = self._process_observations(observations, traj_padding_mask)

        # Build Graph
        if edge_indices is None:  # If edge_indices are not provided
            adj_matrix, edge_indices = self.graph_builder(states)

        estimates, mu, std, log_var = self._compute_local_state_estimates(
            msg, edge_indices
        )

        hidden_states = torch.cat(
            (estimates, local_obs), dim=-1
        )  # (B, N, Hidden Size + F_local_obs)

        q_val = self._process_decoders(hidden_states)
        q_val = q_val * alive_mask.unsqueeze(-1)

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
        deterministic_eval: bool = True,
    ) -> None:
        super().__init__(
            agent_model_dict=agent_model_dict,
            feature_extractor_configs=feature_extractor_configs,
            encoder_configs=encoder_configs,
            decoder_configs=decoder_configs,
            graph_builder_config=graph_builder_config,
            graph_model_config=graph_model_config,
        )
        self.deterministic_eval = deterministic_eval

    def _compute_local_state_estimates(self, msg, edge_indices):
        """Compute local state estimates from messages and local observations with probabilistic output."""
        # Compute graph embeddings using parent method
        embedding = self.compute_graph_embeddings(msg, edge_indices)

        # Process probabilistic output
        deterministic = self.deterministic_eval and not self.training
        estimates, log_var, mu, std = process_probabilistic_output(
            embedding, deterministic
        )  # All (B, N, F)

        return estimates, mu, std, log_var

    def forward(
        self,
        observations: Dict[str, np.ndarray],
        states: torch.Tensor,
        traj_padding_mask: torch.Tensor,
        alive_mask: torch.Tensor,
        edge_indices: List[np.ndarray] | None = None,
    ) -> Dict[str, Any]:
        msg, local_obs = self._process_observations(observations, traj_padding_mask)

        # Build Graph
        if edge_indices is None:  # If edge_indices are not provided
            adj_matrix, edge_indices = self.graph_builder(states)

        estimates, mu, std, log_var = self._compute_local_state_estimates(
            msg, edge_indices
        )

        hidden_states = torch.cat(
            (estimates, local_obs), dim=-1
        )  # (B, N, Hidden Size + F_local_obs)

        q_val = self._process_decoders(hidden_states)
        q_val = q_val * alive_mask.unsqueeze(-1)

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
        enable_rl_grad_to_msg_aggr: bool = False,
    ) -> None:
        super().__init__(
            agent_model_dict=agent_model_dict,
            feature_extractor_configs=feature_extractor_configs,
            encoder_configs=encoder_configs,
            decoder_configs=decoder_configs,
            graph_builder_config=graph_builder_config,
            graph_model_config=graph_model_config,
        )

        self.msg_feature_extractors = nn.ModuleDict()
        self.msg_encoders = nn.ModuleDict()

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
            msg_selected = msg_selected.reshape(bs, n_agents, -1)  # (B, N, F)

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

            for j, agent_idx in enumerate(idx):
                msg[agent_idx] = msg_selected[:, j, :]  # (B, F)
                local_obs[agent_idx] = local_obs_selected[:, j, :]  # (B, F)

        msg = torch.stack(msg, dim=1).to(self.device)  # (B, N, F)
        local_obs = torch.stack(local_obs, dim=1).to(self.device)  # (B, N, F)

        return msg, local_obs


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
        enable_rl_grad_to_msg_aggr: bool = False,
    ) -> None:
        super().__init__(
            agent_model_dict=agent_model_dict,
            feature_extractor_configs=feature_extractor_configs,
            encoder_configs=encoder_configs,
            decoder_configs=decoder_configs,
            graph_builder_config=graph_builder_config,
            graph_model_config=graph_model_config,
            enable_rl_grad_to_msg_aggr=enable_rl_grad_to_msg_aggr,
        )

        self.msg_feature_extractors = nn.ModuleDict()
        for model_name, config in msg_feature_extractor_configs.items():
            self.msg_feature_extractors[model_name] = config.get_model()

    def _compute_local_state_estimates(self, msg, edge_indices):
        """Compute local state estimates from messages and local observations."""
        # Compute graph embeddings using parent method
        embedding = self.compute_graph_embeddings(msg, edge_indices)

        if not self.enable_rl_grad_to_msg_aggr:
            embedding = embedding.detach()

        local_state_estimates = (
            embedding  # For non-probabilistic case, embedding is the estimate
        )

        return local_state_estimates

    def forward(
        self,
        observations: Dict[str, np.ndarray],
        states: torch.Tensor,
        traj_padding_mask: torch.Tensor,
        alive_mask: torch.Tensor,
        edge_indices: List[np.ndarray] | None = None,
    ) -> Dict[str, Any]:
        msg, local_obs = self._process_observations(observations, traj_padding_mask)

        # Build Graph
        if edge_indices is None:  # If edge_indices are not provided
            adj_matrix, edge_indices = self.graph_builder(states)

        local_state_estimates = self._compute_local_state_estimates(msg, edge_indices)

        hidden_states = torch.cat(
            (local_state_estimates, local_obs), dim=-1
        )  # (B, N, Hidden Size + F_local_obs)

        q_val = self._process_decoders(hidden_states)
        q_val = q_val * alive_mask.unsqueeze(-1)

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
        enable_rl_grad_to_msg_aggr: bool = False,
        deterministic_eval: bool = True,
    ) -> None:
        super().__init__(
            agent_model_dict=agent_model_dict,
            feature_extractor_configs=feature_extractor_configs,
            msg_feature_extractor_configs=msg_feature_extractor_configs,
            encoder_configs=encoder_configs,
            decoder_configs=decoder_configs,
            graph_builder_config=graph_builder_config,
            graph_model_config=graph_model_config,
            enable_rl_grad_to_msg_aggr=enable_rl_grad_to_msg_aggr,
        )
        self.deterministic_eval = deterministic_eval

    def _compute_local_state_estimates(self, msg, edge_indices):
        """Compute local state estimates from messages and local observations with probabilistic output."""
        # Compute graph embeddings using parent method
        embedding = self.compute_graph_embeddings(msg, edge_indices)

        # Process probabilistic output
        deterministic = self.deterministic_eval and not self.training
        estimates, log_var, mu, std = process_probabilistic_output(
            embedding, deterministic
        )  # All (B, N, F)

        return estimates, mu, std, log_var

    def forward(
        self,
        observations: Dict[str, np.ndarray],
        states: torch.Tensor,
        traj_padding_mask: torch.Tensor,
        alive_mask: torch.Tensor,
        edge_indices: List[np.ndarray] | None = None,
    ) -> Dict[str, Any]:
        msg, local_obs = self._process_observations(observations, traj_padding_mask)

        # Build Graph
        if edge_indices is None:  # If edge_indices are not provided
            adj_matrix, edge_indices = self.graph_builder(states)

        estimates, mu, std, log_var = self._compute_local_state_estimates(
            msg, edge_indices
        )

        if self.enable_rl_grad_to_msg_aggr:
            hidden_states = torch.cat(
                (estimates, local_obs), dim=-1
            )  # (B, N, Hidden Size + F_local_obs)
        else:
            hidden_states = torch.cat(
                (estimates.detach(), local_obs), dim=-1
            )  # (B, N, Hidden Size + F_local_obs), gradient truncated

        q_val = self._process_decoders(hidden_states)
        q_val = q_val * alive_mask.unsqueeze(-1)

        return {
            "q_val": q_val,
            "edge_indices": edge_indices,
            "local_state_estimates": estimates,
            "mu": mu,
            "std": std,
            "log_var": log_var,
        }
