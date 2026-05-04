import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any
from torch_geometric.data import Batch, Data
from torch_geometric.utils import unbatch
from marlite.algorithm.model import RNNModel, Conv1DModel, AttentionModel
from marlite.algorithm.model.model_config import ModelConfig
from marlite.algorithm.agents.agent_group import AgentGroup
from marlite.algorithm.graph_builder import GraphBuilderConfig


class GraphAgentGroup(AgentGroup):
    def __init__(
        self,
        agent_model_dict: Dict[str, str],
        feature_extractor_configs: Dict[str, ModelConfig],
        encoder_configs: Dict[str, ModelConfig],
        decoder_configs: Dict[str, ModelConfig],
        graph_builder_config: GraphBuilderConfig,
        graph_model_config: ModelConfig,
    ) -> None:
        super().__init__()
        self.agent_model_dict = agent_model_dict

        self.feature_extractors = nn.ModuleDict()
        for model_name, config in feature_extractor_configs.items():
            self.feature_extractors[model_name] = config.get_model()

        self.encoders = nn.ModuleDict()
        for model_name, config in encoder_configs.items():
            self.encoders[model_name] = config.get_model()

        self.models = self.encoders  # For compatibility

        self.decoders = nn.ModuleDict()
        for model_name, config in decoder_configs.items():
            self.decoders[model_name] = config.get_model()

        self.add_module("graph_model", graph_model_config.get_model())
        self.add_module("graph_builder", graph_builder_config.get_graph_builder())

        # Initialize model_to_agent dictionary and model_to_agent_indices dictionary
        self.model_to_agents = {model_name: [] for model_name in encoder_configs.keys()}
        self.model_to_agent_indices = {
            model_name: [] for model_name in encoder_configs.keys()
        }
        for i, (agent_name, model_name) in enumerate(self.agent_model_dict.items()):
            assert model_name in self.model_to_agents.keys(), (
                f"Model {model_name} not found in model_configs"
            )
            self.model_to_agents[model_name].append(agent_name)
            self.model_to_agent_indices[model_name].append(i)

        self.model_class_names = {}
        for model_name, model in self.encoders.items():
            if isinstance(model, RNNModel):
                self.model_class_names[model_name] = "RNNModel"
            elif isinstance(model, Conv1DModel):
                self.model_class_names[model_name] = "Conv1DModel"
            elif isinstance(model, AttentionModel):
                self.model_class_names[model_name] = "AttentionModel"
            else:
                self.model_class_names[model_name] = model.__class__.__name__

    def compute_graph_embeddings(
        self, msg: torch.Tensor, edge_indices: List[np.ndarray]
    ):
        """
        Compute graph embeddings by passing messages through the graph model.

        Args:
            msg: Message tensor of shape (batch_size, num_agents, feature_dim)
            edge_indices: Edge indices for the graph

        Returns:
            embedding: Embedding tensor of shape (batch_size, num_agents, hidden_dim)
        """
        bs = msg.shape[0]

        # Communication between agents using the graph model.
        batch_data = [None for i in range(bs)]
        for i in range(bs):
            batch_data[i] = Data(
                x=msg[i],
                edge_index=torch.Tensor(edge_indices[i]).to(
                    device=self.device, dtype=torch.int
                ),
            )
        batch_data = Batch.from_data_list(batch_data)
        x, e, batch = batch_data.x, batch_data.edge_index, batch_data.batch
        batch_h = self.graph_model(x, e)
        embedding = unbatch(batch_h, batch)  # (B, N, Hidden Size)
        embedding = torch.stack(embedding)

        return embedding

    def _process_observations(self, observations, traj_padding_mask):
        """Common observation processing logic for observation-based models."""
        msg = [None for _ in range(len(self.agent_model_dict))]
        local_obs = [None for _ in range(len(self.agent_model_dict))]

        for (model_name, fe), (_, enc) in zip(
            self.feature_extractors.items(), self.encoders.items()
        ):
            selected_agents = self.model_to_agents[model_name]
            idx = self.model_to_agent_indices[model_name]
            # observation shape: (Batch Size, Agent Number, Time Step, Feature Dimensions) (B, N, T, F)
            obs = observations[:, idx]  # (B, N, T, *(obs_shape))
            bs = obs.shape[0]
            n_agents = len(selected_agents)
            ts = obs.shape[2]
            obs_shape = list(obs.shape[3:])
            # Use class name checking instead of isinstance
            model_class_name = self.model_class_names[model_name]

            if model_class_name == "Conv1DModel":
                # (B, N, T, *(obs_shape)) -> (B*N*T, *(obs_shape))
                obs = obs.reshape(bs * n_agents * ts, *obs_shape).to(self.device)
                obs_vectorized = fe(obs)  # (B*N*T, (obs_shape)) -> (B*N*T, F)
                obs_vectorized = obs_vectorized.reshape(
                    bs, n_agents, ts, -1
                )  # (B*N*T, F) -> (B, N, T, F)
                msg_selected = obs_vectorized[:, :, -1, :]  # (B, N, T, F) -> (B, N, F)
                obs_vectorized = obs_vectorized.reshape(
                    bs * n_agents, ts, -1
                )  # (B, N, T, F) -> (B*N, T, F)
                obs_vectorized = obs_vectorized.permute(
                    0, 2, 1
                )  # (B*N, T, F) -> (B*N, F, T)
                local_obs_selected = enc(obs_vectorized)  # (B*N, F, T) -> (B*N, F)
            elif model_class_name == "RNNModel":
                obs = obs.reshape(bs * n_agents * ts, *obs_shape).to(self.device)
                obs_vectorized = fe(obs)  # (B*N*T, (obs_shape)) -> (B*N*T, F)
                obs_vectorized = obs_vectorized.reshape(
                    bs, n_agents, ts, -1
                )  # (B*N*T, F) -> (B, N, T, F)
                msg_selected = obs_vectorized[:, :, -1, :]  # (B, N, T, F) -> (B, N, F)
                obs_vectorized = obs_vectorized.reshape(
                    bs * n_agents, ts, -1
                )  # (B, N, T, F) -> (B*N, T, F)
                enc.train()  # cudnn RNN backward can only be called in training mode
                local_obs_selected = enc(obs_vectorized)  # (B*N, T, F) -> (B*N, F)
            elif model_class_name == "AttentionModel":
                obs = obs.reshape(bs * n_agents * ts, *obs_shape).to(self.device)
                obs_vectorized = fe(obs)  # (B*N*T, (obs_shape)) -> (B*N*T, F)
                obs_vectorized = obs_vectorized.reshape(
                    bs, n_agents, ts, -1
                )  # (B*N*T, F) -> (B, N, T, F)
                msg_selected = obs_vectorized[:, :, -1, :]  # (B, N, T, F) -> (B, N, F)
                obs_vectorized = obs_vectorized.reshape(
                    bs * n_agents, ts, -1
                )  # (B, N, T, F) -> (B*N, T, F)
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
                msg_selected = obs_vectorized.reshape(
                    bs, n_agents, -1
                )  # (B*N, F) -> (B, N, F)
                local_obs_selected = enc(obs_vectorized)  # (B*N, F) -> (B*N, F)

            local_obs_selected = local_obs_selected.reshape(
                bs, n_agents, -1
            )  # (B, N, F)
            local_obs_selected = local_obs_selected.permute(1, 0, 2)  # (N, B, F)
            msg_selected = msg_selected.permute(1, 0, 2)  # (N, B, F)

            for i, m, lo in zip(idx, msg_selected, local_obs_selected):
                msg[i] = m
                local_obs[i] = lo

        msg = torch.stack(msg).to(self.device)  # (N, B, F)
        msg = msg.permute(1, 0, 2)  # (B, N, F)
        local_obs = torch.stack(local_obs).to(self.device)  # (N, B, F)
        local_obs = local_obs.permute(1, 0, 2)  # (B, N, F)

        return msg, local_obs

    def _process_sequences(self, observations, traj_padding_mask):
        """Common sequence processing logic for sequence-based models."""
        local_obs = [None for _ in range(len(self.agent_model_dict))]

        for (model_name, fe), (_, enc) in zip(
            self.feature_extractors.items(), self.encoders.items()
        ):
            selected_agents = self.model_to_agents[model_name]
            idx = self.model_to_agent_indices[model_name]
            # observation shape: (Batch Size, Agent Number, Time Step, Feature Dimensions) (B, N, T, F)
            obs = observations[:, idx]  # (B, N, T, *(obs_shape))
            bs = obs.shape[0]
            n_agents = len(selected_agents)
            ts = obs.shape[2]
            obs_shape = list(obs.shape[3:])

            model_class_name = self.model_class_names[model_name]
            if model_class_name == "Conv1DModel":
                # (B, N, T, *(obs_shape)) -> (B*N*T, *(obs_shape))
                obs = obs.reshape(bs * n_agents * ts, *obs_shape).to(self.device)
                obs_vectorized = fe(obs)  # (B*N*T, (obs_shape)) -> (B*N*T, F)
                obs_vectorized = obs_vectorized.reshape(
                    bs * n_agents, ts, -1
                )  # (B*N*T, F) -> (B*N, T, F)
                obs_vectorized = obs_vectorized.permute(
                    0, 2, 1
                )  #  (B*N, T, F) -> (B*N, F, T)
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

            for i, m in zip(idx, local_obs_selected):
                local_obs[i] = m

        local_obs = torch.stack(local_obs).to(self.device)  # (N, B, F)
        local_obs = local_obs.permute(1, 0, 2)  # (B, N, F)
        msg = local_obs

        return msg, local_obs

    def _process_decoders(self, hidden_states):
        """Common decoder processing logic for all models."""
        q_val = [None for _ in range(len(self.agent_model_dict))]
        for model_name, dec in self.decoders.items():
            selected_agents = self.model_to_agents[model_name]
            idx = self.model_to_agent_indices[model_name]
            h = hidden_states[:, idx]
            bs = h.shape[0]
            n_agents = len(selected_agents)
            hidden_size = h.shape[-1]
            h = h.reshape(bs * n_agents, hidden_size)  # (B*N, Hidden Size)
            q_selected = dec(h)
            q_selected = q_selected.reshape(bs, n_agents, -1)  # (B, N, Action)
            q_selected = q_selected.permute(1, 0, 2)  # (N, B, Action)

            for i, m in zip(idx, q_selected):
                q_val[i] = m

        q_val = torch.stack(q_val).to(self.device)  # (N, B, F)
        q_val = q_val.permute(1, 0, 2)  # (B, N, F)

        return q_val

    def forward(
        self,
        observations: torch.Tensor,
        states: np.ndarray,
        traj_padding_mask: torch.Tensor,
        alive_mask: torch.Tensor,
        edge_indices: List[np.ndarray] | None = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def act(
        self,
        observations: Dict[str, np.ndarray],
        state: np.ndarray,
        avail_actions: Dict[str, Any],
        traj_padding_mask: np.ndarray,
        alive_agents: List[str],
        epsilon: float = 0.0,
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

        padding_mask = torch.tensor(traj_padding_mask, dtype=torch.bool)  # (T)
        padding_mask = torch.stack(
            [padding_mask] * len(self.agent_model_dict), dim=0
        )  # (N, T)
        padding_mask = padding_mask.unsqueeze(0).to(self.device)  # (1, N, T)

        alive_mask = torch.tensor(
            [agent in set(alive_agents) for agent in self.agent_model_dict.keys()]
        )
        alive_mask = alive_mask.unsqueeze(0).to(self.device)

        with torch.no_grad():
            ret = self(
                obs, np.expand_dims(state, axis=0), padding_mask, alive_mask
            )
            q_values = ret["q_val"]  # (1, num_agents, num_actions)
            q_values = q_values.detach().cpu().numpy().squeeze()

        # Handle different types of avail_actions
        if isinstance(next(iter(avail_actions.values())), np.ndarray):
            # Action masking case
            action_masks = np.array(
                [avail_actions[agent_id] for agent_id in self.agent_model_dict.keys()]
            )

            # Apply action masks to Q-values
            masked_q_values = np.where(action_masks == 1, q_values, -np.inf)

            # Get optimal actions
            optimal_actions = np.argmax(masked_q_values, axis=-1).astype(np.int64)

            # Generate random actions according to action masks
            mask_probs = action_masks / np.sum(action_masks, axis=1, keepdims=True)
            random_actions = np.array(
                [np.random.choice(len(probs), p=probs) for probs in mask_probs]
            ).astype(np.int64)
        else:
            # Action space sampling case
            optimal_actions = np.argmax(q_values, axis=-1).astype(np.int64)
            random_actions = np.array(
                [
                    avail_actions[agent].sample()
                    for agent in self.agent_model_dict.keys()
                ]
            ).astype(np.int64)

        # Epsilon-greedy action selection
        random_choices = np.random.binomial(
            1, epsilon, len(self.agent_model_dict)
        ).astype(np.int64)
        actions = (
            random_choices * random_actions + (1 - random_choices) * optimal_actions
        )
        actions = actions.astype(np.int64).tolist()

        # Create action dictionary
        all_actions = {
            agent: action
            for agent, action in zip(self.agent_model_dict.keys(), actions)
        }
        actual_actions = {agent: all_actions[agent] for agent in alive_agents}

        return {
            "actions": actual_actions,
            "all_actions": all_actions,
            "edge_indices": ret["edge_indices"][0],
        }

    def reset(self) -> "GraphAgentGroup":
        self.graph_builder.reset()
        return self
