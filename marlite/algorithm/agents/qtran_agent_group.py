import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List
from marlite.algorithm.model.model_config import ModelConfig
from marlite.algorithm.model import RNNModel, Conv1DModel, AttentionModel
from marlite.algorithm.agents.agent_group import AgentGroup


class QTRANAgentGroup(AgentGroup):
    def __init__(
        self,
        agent_model_dict: Dict[str, str],
        encoder_configs: Dict[str, ModelConfig],
        decoder_configs: Dict[str, ModelConfig],
        feature_extractors_configs: Dict[str, ModelConfig],
    ) -> None:
        super().__init__()
        self.agent_model_dict = agent_model_dict

        self.encoders = nn.ModuleDict()
        for model_name, config in encoder_configs.items():
            self.encoders[model_name] = config.get_model()

        self.decoders = nn.ModuleDict()
        for model_name, config in decoder_configs.items():
            self.decoders[model_name] = config.get_model()

        self.feature_extractors = nn.ModuleDict()
        for model_name, config in feature_extractors_configs.items():
            self.feature_extractors[model_name] = config.get_model()

        self.model_to_agents = {model_name: [] for model_name in encoder_configs.keys()}
        self.model_to_agent_indices = {
            model_name: [] for model_name in encoder_configs.keys()
        }
        for i, (agent_name, model_name) in enumerate(self.agent_model_dict.items()):
            assert model_name in self.model_to_agents.keys(), (
                f"Model {model_name} not found in encoder_configs"
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

    def forward(
        self,
        observations: torch.Tensor,
        traj_padding_mask: torch.Tensor,
        alive_mask: torch.Tensor,
    ) -> Dict[str, Any]:
        q_val = [None for _ in range(len(self.agent_model_dict))]
        enc_outs = [None for _ in range(len(self.agent_model_dict))]
        for (model_name, enc), (_, dec), (_, fe) in zip(
            self.encoders.items(), self.decoders.items(), self.feature_extractors.items()
        ):
            idx = self.model_to_agent_indices[model_name]
            obs = observations[:, idx]
            bs = obs.shape[0]
            n_agents = len(idx)
            ts = obs.shape[2]
            obs_shape = list(obs.shape[3:])

            model_class_name = self.model_class_names[model_name]
            if model_class_name == "Conv1DModel":
                obs = obs.reshape(bs * n_agents * ts, *obs_shape).to(self.device)
                obs_vectorized = fe(obs)
                obs_vectorized = obs_vectorized.reshape(bs * n_agents, ts, -1)
                obs_vectorized = obs_vectorized.permute(0, 2, 1)
                enc_out = enc(obs_vectorized)
            elif model_class_name == "RNNModel":
                obs = obs.reshape(bs * n_agents * ts, *obs_shape).to(self.device)
                obs_vectorized = fe(obs)
                obs_vectorized = obs_vectorized.reshape(bs * n_agents, ts, -1)
                enc.train()
                enc_out = enc(obs_vectorized)
            elif model_class_name == "AttentionModel":
                obs = obs.reshape(bs * n_agents * ts, *obs_shape).to(self.device)
                obs_vectorized = fe(obs)
                obs_vectorized = obs_vectorized.reshape(bs * n_agents, ts, -1)
                mask = traj_padding_mask[:, idx]
                mask = mask.reshape(bs * n_agents, ts)
                enc_out = enc(obs_vectorized, mask)
            else:
                obs = obs[:, :, -1, :]
                obs = obs.reshape(bs * n_agents, *obs_shape).to(self.device)
                obs_vectorized = fe(obs)
                obs_vectorized = obs_vectorized.reshape(bs * n_agents, -1)
                enc_out = enc(obs_vectorized)

            q_flat = dec(enc_out)
            q_selected = q_flat.reshape(bs, n_agents, -1)
            enc_reshaped = enc_out.reshape(bs, n_agents, -1)

            for j, agent_idx in enumerate(idx):
                q_val[agent_idx] = q_selected[:, j, :]
                enc_outs[agent_idx] = enc_reshaped[:, j, :]

        q_val = torch.stack(q_val, dim=1).to(self.device)  # type: ignore[arg-type]
        enc_out = torch.stack(enc_outs, dim=1).to(self.device)  # type: ignore[arg-type]
        q_val = q_val * alive_mask.unsqueeze(-1)
        return {"q_val": q_val, "enc_out": enc_out}

    def act(
        self,
        observations: Dict[str, np.ndarray],
        state: np.ndarray,
        avail_actions: Dict[str, Any],
        traj_padding_mask: np.ndarray,
        alive_agents: List[str],
        epsilon: float = 0.0,
    ) -> Dict[str, Any]:
        obs = [observations[agent] for agent in self.agent_model_dict.keys()]
        obs = np.stack(obs)
        obs = torch.tensor(obs).unsqueeze(0).to(dtype=torch.float, device=self.device)

        padding_mask = torch.tensor(traj_padding_mask, dtype=torch.bool)
        padding_mask = torch.stack(
            [padding_mask] * len(self.agent_model_dict), dim=0
        )
        padding_mask = padding_mask.unsqueeze(0).to(self.device)

        alive_mask = torch.tensor(
            [agent in set(alive_agents) for agent in self.agent_model_dict.keys()]
        )
        alive_mask = alive_mask.unsqueeze(0).to(self.device)

        with torch.no_grad():
            ret = self(obs, padding_mask, alive_mask)
            q_values = ret["q_val"]
            q_values = q_values.detach().cpu().numpy().squeeze()

        if isinstance(next(iter(avail_actions.values())), np.ndarray):
            action_masks = np.array(
                [avail_actions[agent_id] for agent_id in self.agent_model_dict.keys()]
            )
            masked_q_values = np.where(action_masks == 1, q_values, -np.inf)
            optimal_actions = np.argmax(masked_q_values, axis=-1).astype(np.int64)
            mask_probs = action_masks / np.sum(action_masks, axis=1, keepdims=True)
            random_actions = np.array(
                [np.random.choice(len(probs), p=probs) for probs in mask_probs]
            ).astype(np.int64)
        else:
            optimal_actions = np.argmax(q_values, axis=-1).astype(np.int64)
            random_actions = np.array(
                [
                    avail_actions[agent].sample()
                    for agent in self.agent_model_dict.keys()
                ]
            ).astype(np.int64)

        random_choices = np.random.binomial(
            1, epsilon, len(self.agent_model_dict)
        ).astype(np.int64)
        actions = (
            random_choices * random_actions + (1 - random_choices) * optimal_actions
        )
        actions = actions.astype(np.int64).tolist()

        all_actions = {
            agent: action
            for agent, action in zip(self.agent_model_dict.keys(), actions)
        }
        actual_actions = {agent: all_actions[agent] for agent in alive_agents}

        return {"actions": actual_actions, "all_actions": all_actions}

    def reset(self) -> "QTRANAgentGroup":
        return self
