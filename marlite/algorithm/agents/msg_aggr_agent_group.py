import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.init as init
from marlite.algorithm.model.model_config import ModelConfig
from marlite.algorithm.model import RNNModel, Conv1DModel, AttentionModel, MaskedModel
from marlite.algorithm.agents.agent_group import AgentGroup
from marlite.util.prob_util import process_probabilistic_output


class MsgAggrAgentGroup(AgentGroup):
    def __init__(
        self,
        agent_model_dict: Dict[str, str],
        feature_extractor_configs: Dict[str, ModelConfig],
        encoder_configs: Dict[str, ModelConfig],
        decoder_configs: Dict[str, ModelConfig],
        aggr_model_config: ModelConfig,
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

        self.add_module("aggr_model", aggr_model_config.get_model())

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

        self._use_data_parallel = False

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

        if isinstance(self.aggr_model, MaskedModel):
            self.aggr_model_class_name = "MaskedModel"
        else:
            self.aggr_model_class_name = self.aggr_model.__class__.__name__

    def _process_observations(self, observations, traj_padding_mask):
        """Common observation processing logic for observation-based models."""
        msg = [None for _ in range(len(self.agent_model_dict))]
        encoded = [None for _ in range(len(self.agent_model_dict))]

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
                encoded_selected = enc(obs_vectorized)  # (B*N, F, T) -> (B*N, F)
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
                encoded_selected = enc(obs_vectorized)  # (B*N, T, F) -> (B*N, F)
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
                encoded_selected = enc(obs_vectorized, mask)  # (B*N, T, F) -> (B*N, F)
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
                encoded_selected = enc(obs_vectorized)  # (B*N, F) -> (B*N, F)

            encoded_selected = encoded_selected.reshape(bs, n_agents, -1)  # (B, N, F)
            encoded_selected = encoded_selected.permute(1, 0, 2)  # (N, B, F)
            msg_selected = msg_selected.permute(1, 0, 2)  # (N, B, F)

            for i, m, c in zip(idx, msg_selected, encoded_selected):
                msg[i] = m
                encoded[i] = c

        msg = torch.stack(msg).to(self.device)  # (N, B, F)
        msg = msg.permute(1, 0, 2)  # (B, N, F)
        encoded = torch.stack(encoded).to(self.device)  # (N, B, F)
        encoded = encoded.permute(1, 0, 2)  # (B, N, F)

        return msg, encoded

    def _process_sequences(self, observations, traj_padding_mask):
        """Common sequence processing logic for sequence-based models."""
        encoded = [None for _ in range(len(self.agent_model_dict))]

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
                    bs * n_agents, ts, -1
                )  # (B*N*T, F) -> (B*N, T, F)
                obs_vectorized = obs_vectorized.permute(
                    0, 2, 1
                )  # (B*N, T, F) -> (B*N, F, T)
                encoded_selected = enc(obs_vectorized)  # (B*N, F, T) -> (B*N, F)
            elif model_class_name == "RNNModel":
                obs = obs.reshape(bs * n_agents * ts, *obs_shape).to(self.device)
                obs_vectorized = fe(obs)  # (B*N*T, (obs_shape)) -> (B*N*T, F)
                obs_vectorized = obs_vectorized.reshape(
                    bs * n_agents, ts, -1
                )  # (B*N*T, F) -> (B*N, T, F)
                enc.train()  # cudnn RNN backward can only be called in training mode
                encoded_selected = enc(obs_vectorized)  # (B*N, T, F) -> (B*N, F)
            elif model_class_name == "AttentionModel":
                obs = obs.reshape(bs * n_agents * ts, *obs_shape).to(self.device)
                obs_vectorized = fe(obs)  # (B*N*T, (obs_shape)) -> (B*N*T, F)
                obs_vectorized = obs_vectorized.reshape(
                    bs * n_agents, ts, -1
                )  # (B*N*T, F) -> (B*N, T, F)
                mask = traj_padding_mask[:, idx]
                mask = mask.reshape(bs * n_agents, ts)
                encoded_selected = enc(obs_vectorized, mask)  # (B*N, T, F) -> (B*N, F)
            else:
                obs = obs[
                    :, :, -1, :
                ]  # (B, N, T, *(obs_shape)) -> (B, N, *(obs_shape))
                obs = obs.reshape(bs * n_agents, *obs_shape).to(
                    self.device
                )  # (B, N, *(obs_shape)) -> (B*N, *(obs_shape))
                obs_vectorized = fe(obs)  # (B*N, *(obs_shape)) -> (B*N, F)
                encoded_selected = enc(obs_vectorized)  # (B*N, F) -> (B*N, F)

            encoded_selected = encoded_selected.reshape(bs, n_agents, -1)  # (B, N, F)
            encoded_selected = encoded_selected.permute(1, 0, 2)  # (N, B, F)

            for i, m in zip(idx, encoded_selected):
                encoded[i] = m

        encoded = torch.stack(encoded).to(self.device)  # (N, B, F)
        encoded = encoded.permute(1, 0, 2)  # (B, N, F)

        return encoded, encoded

    def forward(self, observations, traj_padding_mask, alive_mask):
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

        padding_mask = torch.tensor(traj_padding_mask, dtype=torch.bool)  # (T)
        padding_mask = torch.stack(
            [padding_mask] * len(self.agent_model_dict), dim=0
        )  # (N, T)
        padding_mask = padding_mask.unsqueeze(0).to(self.device)  # (1, N, T)

        alive_mask = torch.tensor(
            [agent in set(alive_agents) for agent in self.agent_model_dict.keys()]
        )
        alive_mask = alive_mask.unsqueeze(0).to(self.device)

        # Get Q-values
        with torch.no_grad():
            ret = self.forward(obs, padding_mask, alive_mask)
            q_values = ret["q_val"]
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

        return {"actions": actual_actions, "all_actions": all_actions}

    def reset(self) -> "MsgAggrAgentGroup":
        return self

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


class DualPathBasedMsgAggrAgentGroup(MsgAggrAgentGroup):
    """
    Base class for dual-path message aggregation agent groups.

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
        aggr_model_config: ModelConfig,
        enable_rl_grad_to_msg_aggr: bool = False,
    ) -> None:
        super().__init__(
            agent_model_dict=agent_model_dict,
            feature_extractor_configs=feature_extractor_configs,
            encoder_configs=encoder_configs,
            decoder_configs=decoder_configs,
            aggr_model_config=aggr_model_config,
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

    def reset(self) -> "DualPathBasedMsgAggrAgentGroup":
        return self


class ObsMsgAggrAgentGroup(MsgAggrAgentGroup):
    def forward(
        self,
        observations: torch.Tensor,
        traj_padding_mask: torch.Tensor,
        alive_mask: torch.Tensor,
    ) -> Dict[str, Any]:
        msg, encoded = self._process_observations(observations, traj_padding_mask)
        msg_detach = msg.detach()

        # Aggregate message
        if self.aggr_model_class_name == "MaskedModel":
            aggregated_msg = self.aggr_model(
                msg_detach, alive_mask
            )  # (B, N, F) -> (B, F)
        else:
            aggregated_msg = self.aggr_model(msg_detach)  # (B, N, F) -> (B, F)
        aggregated_msg_expand = aggregated_msg.unsqueeze(1).expand(
            -1, len(self.agent_model_dict), -1
        )  # (B, N, F)

        hidden_states = torch.cat(
            (encoded, aggregated_msg_expand), dim=-1
        )  # (B, N, Hidden Size(F_local_obs + F_aggregated_msg))

        # Process decoders
        q_val = self._process_decoders(hidden_states)

        return {"q_val": q_val, "aggregated_msg": aggregated_msg}


class SeqMsgAggrAgentGroup(MsgAggrAgentGroup):
    def forward(
        self,
        observations: torch.Tensor,
        traj_padding_mask: torch.Tensor,
        alive_mask: torch.Tensor,
    ) -> Dict[str, Any]:
        msg, encoded = self._process_sequences(observations, traj_padding_mask)
        msg_detach = msg.detach()

        # Aggregate message
        if self.aggr_model_class_name == "MaskedModel":
            aggregated_msg = self.aggr_model(
                msg_detach, alive_mask
            )  # (B, N, F) -> (B, F)
        else:
            aggregated_msg = self.aggr_model(msg_detach)  # (B, N, F) -> (B, F)
        aggregated_msg_expand = aggregated_msg.unsqueeze(1).expand(
            -1, len(self.agent_model_dict), -1
        )  # (B, N, F)

        hidden_states = torch.cat(
            (encoded, aggregated_msg_expand), dim=-1
        )  # (B, N, Hidden Size(F_local_obs + F_aggregated_msg))

        # Process decoders
        q_val = self._process_decoders(hidden_states)

        return {"q_val": q_val, "aggregated_msg": aggregated_msg}


class ProbObsMsgAggrAgentGroup(MsgAggrAgentGroup):
    def __init__(
        self,
        agent_model_dict: Dict[str, str],
        feature_extractor_configs: Dict[str, ModelConfig],
        encoder_configs: Dict[str, ModelConfig],
        decoder_configs: Dict[str, ModelConfig],
        aggr_model_config: ModelConfig,
        deterministic_eval: bool = True,
    ) -> None:
        super().__init__(
            agent_model_dict=agent_model_dict,
            feature_extractor_configs=feature_extractor_configs,
            encoder_configs=encoder_configs,
            decoder_configs=decoder_configs,
            aggr_model_config=aggr_model_config,
        )
        self.deterministic_eval = deterministic_eval

    def forward(
        self,
        observations: torch.Tensor,
        traj_padding_mask: torch.Tensor,
        alive_mask: torch.Tensor,
    ) -> Dict[str, Any]:
        msg, encoded = self._process_observations(observations, traj_padding_mask)

        # Aggregate message and split into mean and log variance
        if self.aggr_model_class_name == "MaskedModel":
            aggr_output = self.aggr_model(msg, alive_mask)  # (B, N, F) -> (B, 2*F)
        else:
            aggr_output = self.aggr_model(msg)  # (B, N, F) -> (B, 2*F)

        # Process probabilistic output
        deterministic = self.deterministic_eval and not self.aggr_model.training
        aggregated_msg, log_var, mu, std = process_probabilistic_output(
            aggr_output, deterministic
        )
        aggregated_msg_expand = aggregated_msg.unsqueeze(1).expand(
            -1, len(self.agent_model_dict), -1
        )  # (B, N, F)

        hidden_states = torch.cat(
            (encoded, aggregated_msg_expand), dim=-1
        )  # (B, N, Hidden Size(F_local_obs + F_aggregated_msg))

        # Process decoders
        q_val = self._process_decoders(hidden_states)

        return {
            "q_val": q_val,
            "aggregated_msg": aggregated_msg,
            "mu": mu,
            "std": std,
            "log_var": log_var,
        }


class ProbSeqMsgAggrAgentGroup(MsgAggrAgentGroup):
    def __init__(
        self,
        agent_model_dict: Dict[str, str],
        feature_extractor_configs: Dict[str, ModelConfig],
        encoder_configs: Dict[str, ModelConfig],
        decoder_configs: Dict[str, ModelConfig],
        aggr_model_config: ModelConfig,
        deterministic_eval: bool = True,
    ) -> None:
        super().__init__(
            agent_model_dict=agent_model_dict,
            feature_extractor_configs=feature_extractor_configs,
            encoder_configs=encoder_configs,
            decoder_configs=decoder_configs,
            aggr_model_config=aggr_model_config,
        )
        self.deterministic_eval = deterministic_eval

    def forward(
        self,
        observations: torch.Tensor,
        traj_padding_mask: torch.Tensor,
        alive_mask: torch.Tensor,
    ) -> Dict[str, Any]:
        msg, encoded = self._process_sequences(observations, traj_padding_mask)

        # Aggregate message
        if self.aggr_model_class_name == "MaskedModel":
            aggr_output = self.aggr_model(msg, alive_mask)  # (B, N, F) -> (B, F)
        else:
            aggr_output = self.aggr_model(msg)  # (B, N, F) -> (B, F)

        # Process probabilistic output
        deterministic = self.deterministic_eval and not self.aggr_model.training
        aggregated_msg, log_var, mu, std = process_probabilistic_output(
            aggr_output, deterministic
        )
        aggregated_msg_expand = aggregated_msg.unsqueeze(1).expand(
            -1, len(self.agent_model_dict), -1
        )  # (B, N, F)

        hidden_states = torch.cat(
            (encoded, aggregated_msg_expand), dim=-1
        )  # (B, N, Hidden Size(F_local_obs + F_aggregated_msg))

        # Process decoders
        q_val = self._process_decoders(hidden_states)

        return {
            "q_val": q_val,
            "aggregated_msg": aggregated_msg,
            "mu": mu,
            "std": std,
            "log_var": log_var,
        }


class DualPathObsMsgAggrAgentGroup(DualPathBasedMsgAggrAgentGroup):
    def __init__(
        self,
        agent_model_dict: Dict[str, str],
        feature_extractor_configs: Dict[str, ModelConfig],  # For observation processing
        msg_feature_extractor_configs: Dict[str, ModelConfig],  # For message generation
        encoder_configs: Dict[str, ModelConfig],
        decoder_configs: Dict[str, ModelConfig],
        aggr_model_config: ModelConfig,
        enable_rl_grad_to_msg_aggr: bool = False,
    ) -> None:
        super().__init__(
            agent_model_dict=agent_model_dict,
            feature_extractor_configs=feature_extractor_configs,
            encoder_configs=encoder_configs,
            decoder_configs=decoder_configs,
            aggr_model_config=aggr_model_config,
            enable_rl_grad_to_msg_aggr=enable_rl_grad_to_msg_aggr,
        )

        self.msg_feature_extractors = nn.ModuleDict()
        for model_name, config in msg_feature_extractor_configs.items():
            self.msg_feature_extractors[model_name] = config.get_model()

    def forward(
        self,
        observations: torch.Tensor,
        traj_padding_mask: torch.Tensor,
        alive_mask: torch.Tensor,
    ) -> Dict[str, Any]:
        msg, local_obs = self._process_observations(observations, traj_padding_mask)

        # Aggregate message
        if self.aggr_model_class_name == "MaskedModel":
            aggregated_msg = self.aggr_model(msg, alive_mask)  # (B, N, F) -> (B, F)
        else:
            aggregated_msg = self.aggr_model(msg)  # (B, N, F) -> (B, F)
        aggregated_msg_expand = aggregated_msg.unsqueeze(1).expand(
            -1, len(self.agent_model_dict), -1
        )  # (B, N, F)
        if not self.enable_rl_grad_to_msg_aggr:
            aggregated_msg_expand = aggregated_msg_expand.detach()

        hidden_states = torch.cat(
            (local_obs, aggregated_msg_expand), dim=-1
        )  # (B, N, Hidden Size(F_local_obs + F_aggregated_msg))

        # Process decoders
        q_val = self._process_decoders(hidden_states)

        return {"q_val": q_val, "aggregated_msg": aggregated_msg}


"""
class DualPathSeqMsgAggrAgentGroup(DualPathMsgAggrAgentGroup):

    def forward(self, observations: torch.Tensor, traj_padding_mask: torch.Tensor, alive_mask: torch.Tensor) -> Dict[str, Any]:
        msg, local_obs = self._process_dual_path_observations(observations, traj_padding_mask)

        # Aggregate message
        if self.aggr_model_class_name == 'MaskedModel':
            aggregated_msg = self.aggr_model(msg, alive_mask) # (B, N, F) -> (B, F)
        else:
            aggregated_msg = self.aggr_model(msg) # (B, N, F) -> (B, F)
        aggregated_msg_expand = aggregated_msg.unsqueeze(1).expand(-1, len(self.agent_model_dict), -1).detach()  # (B, N, F)

        hidden_states = torch.cat((local_obs, aggregated_msg_expand), dim=-1)  # (B, N, Hidden Size(F_local_obs + F_aggregated_msg))

        # Process decoders
        q_val = self._process_decoders(hidden_states)

        return {'q_val': q_val, 'aggregated_msg': aggregated_msg}
"""


class DualPathProbObsMsgAggrAgentGroup(DualPathObsMsgAggrAgentGroup):
    def __init__(
        self,
        agent_model_dict: Dict[str, str],
        feature_extractor_configs: Dict[str, ModelConfig],  # For observation processing
        msg_feature_extractor_configs: Dict[str, ModelConfig],  # For message generation
        encoder_configs: Dict[str, ModelConfig],
        decoder_configs: Dict[str, ModelConfig],
        aggr_model_config: ModelConfig,
        enable_rl_grad_to_msg_aggr: bool = False,
        deterministic_eval: bool = True,
    ) -> None:
        super().__init__(
            agent_model_dict=agent_model_dict,
            feature_extractor_configs=feature_extractor_configs,
            msg_feature_extractor_configs=msg_feature_extractor_configs,
            encoder_configs=encoder_configs,
            decoder_configs=decoder_configs,
            aggr_model_config=aggr_model_config,
            enable_rl_grad_to_msg_aggr=enable_rl_grad_to_msg_aggr,
        )
        self.deterministic_eval = deterministic_eval

    def forward(
        self,
        observations: torch.Tensor,
        traj_padding_mask: torch.Tensor,
        alive_mask: torch.Tensor,
    ) -> Dict[str, Any]:
        msg, local_obs = self._process_observations(observations, traj_padding_mask)

        # Aggregate message and split into mean and log variance
        if self.aggr_model_class_name == "MaskedModel":
            aggr_output = self.aggr_model(msg, alive_mask)  # (B, N, F) -> (B, 2*F)
        else:
            aggr_output = self.aggr_model(msg)  # (B, N, F) -> (B, 2*F)

        # Process probabilistic output
        deterministic = self.deterministic_eval and not self.aggr_model.training
        aggregated_msg, log_var, mu, std = process_probabilistic_output(
            aggr_output, deterministic
        )
        aggregated_msg_expand = aggregated_msg.unsqueeze(1).expand(
            -1, len(self.agent_model_dict), -1
        )  # (B, N, F)
        if not self.enable_rl_grad_to_msg_aggr:
            aggregated_msg_expand = aggregated_msg_expand.detach()

        hidden_states = torch.cat(
            (local_obs, aggregated_msg_expand), dim=-1
        )  # (B, N, Hidden Size(F_local_obs + F_aggregated_msg))

        # Process decoders
        q_val = self._process_decoders(hidden_states)

        return {
            "q_val": q_val,
            "aggregated_msg": aggregated_msg,
            "mu": mu,
            "std": std,
            "log_var": log_var,
        }


"""
class DualPathProbSeqMsgAggrAgentGroup(DualPathMsgAggrAgentGroup, ProbMsgAggrAgentGroup):

    def forward(self, observations: torch.Tensor, traj_padding_mask: torch.Tensor, alive_mask: torch.Tensor) -> Dict[str, Any]:
        msg, local_obs = self._process_dual_path_observations(observations, traj_padding_mask)

        # Aggregate message
        if self.aggr_model_class_name == 'MaskedModel':
            aggr_output = self.aggr_model(msg, alive_mask) # (B, N, F) -> (B, F)
        else:
            aggr_output = self.aggr_model(msg) # (B, N, F) -> (B, F)

        # Process probabilistic output
        aggregated_msg, mu, std = self._process_probabilistic_output(aggr_output)
        aggregated_msg_expand = aggregated_msg.unsqueeze(1).expand(-1, len(self.agent_model_dict), -1)  # (B, N, F)

        hidden_states = torch.cat((local_obs, aggregated_msg_expand), dim=-1)  # (B, N, Hidden Size(F_local_obs + F_aggregated_msg))

        # Process decoders
        q_val = self._process_decoders(hidden_states)

        return {'q_val': q_val, 'aggregated_msg': aggregated_msg, 'mu': mu, 'std': std, 'log_var': log_var}
"""


def _init_msg_extractor(m):
    if hasattr(m, "weight") and m.weight is not None:
        init.normal_(m.weight, mean=0.0, std=0.001)
    if hasattr(m, "bias") and m.bias is not None:
        init.zeros_(m.bias)
