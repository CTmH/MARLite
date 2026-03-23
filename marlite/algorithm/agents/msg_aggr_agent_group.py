import numpy as np
import torch
import os
from copy import deepcopy
from typing import Dict, Any, List
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.init as init
from marlite.algorithm.model.model_config import ModelConfig
from marlite.algorithm.model import RNNModel, Conv1DModel, AttentionModel, MaskedModel
from marlite.algorithm.agents.agent_group import AgentGroup
from marlite.util.optimizer_config import OptimizerConfig
from marlite.util.lr_scheduler_config import LRSchedulerConfig
from marlite.util.prob_util import process_probabilistic_output


class MsgAggrAgentGroup(AgentGroup):
    def __init__(
        self,
        agent_model_dict: Dict[str, str],
        feature_extractor_configs: Dict[str, ModelConfig],
        encoder_configs: Dict[str, ModelConfig],
        decoder_configs: Dict[str, ModelConfig],
        aggr_model_config: ModelConfig,
        optimizer_config: OptimizerConfig,
        lr_scheduler_config: LRSchedulerConfig = None,
        device="cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.agent_model_dict = agent_model_dict
        self.feature_extractors = {
            model_name: config.get_model()
            for model_name, config in feature_extractor_configs.items()
        }
        self.encoders = {
            model_name: config.get_model()
            for model_name, config in encoder_configs.items()
        }
        self.models = self.encoders  # For compatibility
        self.decoders = {
            model_name: config.get_model()
            for model_name, config in decoder_configs.items()
        }
        self.aggr_model = aggr_model_config.get_model()  # Message aggregator model
        self.params_to_optimize = [
            {"params": extractor.parameters()}
            for extractor in self.feature_extractors.values()
        ]
        self.params_to_optimize += [
            {"params": encoder.parameters()} for encoder in self.encoders.values()
        ]
        self.params_to_optimize += [
            {"params": decoder.parameters()} for decoder in self.decoders.values()
        ]
        self.params_to_optimize += [{"params": self.aggr_model.parameters()}]
        self.optimizer = optimizer_config.get_optimizer(self.params_to_optimize)
        self.lr_scheduler = None
        if lr_scheduler_config:
            self.lr_scheduler = lr_scheduler_config.get_lr_scheduler(self.optimizer)

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
        self._is_compiled = False

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

    def set_agent_group_params(self, params: Dict[str, dict]) -> "AgentGroup":
        feature_extractor_params = params.get("feature_extractor", {})
        encoder_params = params.get("encoder", {})
        decoder_params = params.get("decoder", {})
        aggr_model_params = params.get("aggr_model", {})
        for (model_name, enc), (_, fe), (_, dec) in zip(
            self.encoders.items(),
            self.feature_extractors.items(),
            self.decoders.items(),
        ):
            enc.load_state_dict(encoder_params[model_name])
            fe.load_state_dict(feature_extractor_params[model_name])
            dec.load_state_dict(decoder_params[model_name])

        self.aggr_model.load_state_dict(aggr_model_params)

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
        aggr_model_params = deepcopy(self.aggr_model.state_dict())
        params = {
            "encoder": encoder_params,
            "feature_extractor": feature_extractor_params,
            "decoder": decoder_params,
            "aggr_model": aggr_model_params,
        }
        return params

    def zero_grad(self) -> "AgentGroup":
        self.optimizer.zero_grad()
        return self

    def step(self) -> "AgentGroup":
        for p in self.params_to_optimize:
            torch.nn.utils.clip_grad_norm_(p["params"], max_norm=5.0)
        self.optimizer.step()
        return self

    def lr_scheduler_step(self, reward) -> "AgentGroup":
        if not self.lr_scheduler:
            return self
        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.lr_scheduler.step(reward)
        else:
            self.lr_scheduler.step()
        return self

    def to(self, device: str) -> "AgentGroup":
        for (_, enc), (_, fe), (_, dec) in zip(
            self.encoders.items(),
            self.feature_extractors.items(),
            self.decoders.items(),
        ):
            enc.to(device)
            fe.to(device)
            dec.to(device)
        self.aggr_model.to(device)
        self.device = device
        return self

    def eval(self) -> "AgentGroup":
        for (_, enc), (_, fe), (_, dec) in zip(
            self.encoders.items(),
            self.feature_extractors.items(),
            self.decoders.items(),
        ):
            enc.eval()
            fe.eval()
            dec.eval()
        self.aggr_model.eval()
        return self

    def train(self) -> "AgentGroup":
        for (_, enc), (_, fe), (_, dec) in zip(
            self.encoders.items(),
            self.feature_extractors.items(),
            self.decoders.items(),
        ):
            enc.train()
            fe.train()
            dec.train()
        self.aggr_model.train()
        return self

    def share_memory(self) -> "AgentGroup":
        for (_, enc), (_, fe), (_, dec) in zip(
            self.encoders.items(),
            self.feature_extractors.items(),
            self.decoders.items(),
        ):
            enc.share_memory()
            fe.share_memory()
            dec.share_memory()
        self.aggr_model.share_memory()
        return self

    def wrap_data_parallel(self, device_id: int = 0) -> "AgentGroup":
        """Wrap models with DistributedDataParallel.

        Args:
            device_id: The GPU device ID to use for this process
        """
        device = f"cuda:{device_id}"

        def has_trainable_params(module):
            """Check if module has any trainable parameters."""
            return any(p.requires_grad for p in module.parameters())

        for id in self.encoders.keys():
            self.encoders[id] = self.encoders[id].to(device)
            if has_trainable_params(self.encoders[id]):
                self.encoders[id] = DDP(self.encoders[id], device_ids=[device_id])
            self.feature_extractors[id] = self.feature_extractors[id].to(device)
            if has_trainable_params(self.feature_extractors[id]):
                self.feature_extractors[id] = DDP(
                    self.feature_extractors[id], device_ids=[device_id]
                )
            self.decoders[id] = self.decoders[id].to(device)
            if has_trainable_params(self.decoders[id]):
                self.decoders[id] = DDP(self.decoders[id], device_ids=[device_id])
        self.aggr_model = self.aggr_model.to(device)
        if has_trainable_params(self.aggr_model):
            self.aggr_model = DDP(self.aggr_model, device_ids=[device_id])
        self._use_data_parallel = True
        self.device = device
        return self

    def unwrap_data_parallel(self) -> "AgentGroup":
        """Unwrap DistributedDataParallel from models."""
        for id in self.encoders.keys():
            if isinstance(self.encoders[id], DDP):
                self.encoders[id] = self.encoders[id].module.cpu()
            else:
                self.encoders[id] = self.encoders[id].cpu()
            if isinstance(self.feature_extractors[id], DDP):
                self.feature_extractors[id] = self.feature_extractors[id].module.cpu()
            else:
                self.feature_extractors[id] = self.feature_extractors[id].cpu()
            if isinstance(self.decoders[id], DDP):
                self.decoders[id] = self.decoders[id].module.cpu()
            else:
                self.decoders[id] = self.decoders[id].cpu()
        if isinstance(self.aggr_model, DDP):
            self.aggr_model = self.aggr_model.module.cpu()
        else:
            self.aggr_model = self.aggr_model.cpu()
        self._use_data_parallel = False
        self.device = "cpu"
        return self

    def save_params(self, path: str) -> "AgentGroup":
        os.makedirs(path, exist_ok=True)
        for (model_name, enc), (_, fe), (_, dec) in zip(
            self.encoders.items(),
            self.feature_extractors.items(),
            self.decoders.items(),
        ):
            model_dir = os.path.join(path, model_name)
            os.makedirs(model_dir, exist_ok=True)
            torch.save(
                fe.state_dict(), os.path.join(model_dir, "feature_extractor.pth")
            )
            torch.save(enc.state_dict(), os.path.join(model_dir, "encoder.pth"))
            torch.save(dec.state_dict(), os.path.join(model_dir, "decoder.pth"))
        torch.save(self.aggr_model.state_dict(), os.path.join(path, "aggr_model.pth"))
        return self

    def load_params(self, path: str) -> "AgentGroup":
        for (model_name, enc), (_, fe), (_, dec) in zip(
            self.encoders.items(),
            self.feature_extractors.items(),
            self.decoders.items(),
        ):
            model_dir = os.path.join(path, model_name)
            fe.load_state_dict(
                torch.load(
                    os.path.join(model_dir, "feature_extractor.pth"),
                    map_location=torch.device("cpu"),
                )
            )
            enc.load_state_dict(
                torch.load(
                    os.path.join(model_dir, "encoder.pth"),
                    map_location=torch.device("cpu"),
                )
            )
            dec.load_state_dict(
                torch.load(
                    os.path.join(model_dir, "decoder.pth"),
                    map_location=torch.device("cpu"),
                )
            )
        self.aggr_model.load_state_dict(
            torch.load(
                os.path.join(path, "aggr_model.pth"), map_location=torch.device("cpu")
            )
        )
        return self

    def compile_models(self) -> "AgentGroup":
        for id in self.encoders.keys():
            self.encoders[id] = torch.compile(self.encoders[id])
            self.feature_extractors[id] = torch.compile(self.feature_extractors[id])
            self.decoders[id] = torch.compile(self.decoders[id])
        self.aggr_model = torch.compile(self.aggr_model)
        self._is_compiled = True
        return self

    def reset(self) -> "AgentGroup":
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
            aggr_model_config=aggr_model_config,
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

    def set_agent_group_params(self, params: Dict[str, dict]) -> "AgentGroup":
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

    def to(self, device: str) -> "AgentGroup":
        """Move all components to device"""
        super().to(device)
        for _, fe in self.msg_feature_extractors.items():
            fe.to(device)
        for _, enc in self.msg_encoders.items():
            enc.to(device)
        return self

    def eval(self) -> "AgentGroup":
        """Set all components to evaluation mode"""
        super().eval()
        for _, fe in self.msg_feature_extractors.items():
            fe.eval()
        for _, enc in self.msg_encoders.items():
            enc.eval()
        return self

    def train(self) -> "AgentGroup":
        """Set all components to training mode"""
        super().train()
        for _, fe in self.msg_feature_extractors.items():
            fe.train()
        for _, enc in self.msg_encoders.items():
            enc.train()
        return self

    def share_memory(self) -> "AgentGroup":
        """Share memory for all components"""
        super().share_memory()
        for _, fe in self.msg_feature_extractors.items():
            fe.share_memory()
        for _, enc in self.msg_encoders.items():
            enc.share_memory()
        return self

    def wrap_data_parallel(self, device_id: int = 0) -> "AgentGroup":
        """Wrap all components with DistributedDataParallel"""
        device = f"cuda:{device_id}"

        def has_trainable_params(module):
            """Check if module has any trainable parameters."""
            return any(p.requires_grad for p in module.parameters())

        super().wrap_data_parallel(device_id)
        for id in self.msg_feature_extractors.keys():
            self.msg_feature_extractors[id] = self.msg_feature_extractors[id].to(device)
            if has_trainable_params(self.msg_feature_extractors[id]):
                self.msg_feature_extractors[id] = DDP(
                    self.msg_feature_extractors[id], device_ids=[device_id]
                )
        for id in self.msg_encoders.keys():
            self.msg_encoders[id] = self.msg_encoders[id].to(device)
            if has_trainable_params(self.msg_encoders[id]):
                self.msg_encoders[id] = DDP(self.msg_encoders[id], device_ids=[device_id])
        return self

    def unwrap_data_parallel(self) -> "AgentGroup":
        """Unwrap DistributedDataParallel from all components"""
        super().unwrap_data_parallel()
        for id in self.msg_feature_extractors.keys():
            if isinstance(self.msg_feature_extractors[id], DDP):
                self.msg_feature_extractors[id] = self.msg_feature_extractors[
                    id
                ].module.cpu()
            else:
                self.msg_feature_extractors[id] = self.msg_feature_extractors[id].cpu()
        for id in self.msg_encoders.keys():
            if isinstance(self.msg_encoders[id], DDP):
                self.msg_encoders[id] = self.msg_encoders[id].module.cpu()
            else:
                self.msg_encoders[id] = self.msg_encoders[id].cpu()
        return self

    def save_params(self, path: str) -> "AgentGroup":
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

    def load_params(self, path: str) -> "AgentGroup":
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

    def compile_models(self) -> "AgentGroup":
        """Compile all models for performance"""
        super().compile_models()
        for id in self.msg_feature_extractors.keys():
            self.msg_feature_extractors[id] = torch.compile(
                self.msg_feature_extractors[id]
            )
        for id in self.msg_encoders.keys():
            self.msg_encoders[id] = torch.compile(self.msg_encoders[id])
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
            aggr_model_config=aggr_model_config,
            optimizer_config=optimizer_config,
            lr_scheduler_config=lr_scheduler_config,
            device=device,
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
            aggr_model_config=aggr_model_config,
            optimizer_config=optimizer_config,
            lr_scheduler_config=lr_scheduler_config,
            device=device,
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
            aggr_model_config=aggr_model_config,
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
            aggr_model_config=aggr_model_config,
            optimizer_config=optimizer_config,
            lr_scheduler_config=lr_scheduler_config,
            enable_rl_grad_to_msg_aggr=enable_rl_grad_to_msg_aggr,
            device=device,
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
