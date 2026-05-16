import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Dict, Any, List

from marlite.algorithm.model.model_config import ModelConfig
from marlite.algorithm.model import TimeSeqModel, RNNModel, Conv1DModel, AttentionModel
from marlite.algorithm.agents.agent_group import AgentGroup


class MAPPOAgentGroup(AgentGroup):
    def __init__(
        self,
        agent_model_dict: Dict[str, str],
        model_configs: Dict[str, ModelConfig],
        feature_extractors_configs: Dict[str, ModelConfig],
    ) -> None:
        super().__init__()
        self.agent_model_dict = agent_model_dict

        self.models = nn.ModuleDict()
        for model_name, config in model_configs.items():
            self.models[model_name] = config.get_model()

        self.feature_extractors = nn.ModuleDict()
        for model_name, config in feature_extractors_configs.items():
            self.feature_extractors[model_name] = config.get_model()

        self.model_to_agents = {model_name: [] for model_name in model_configs.keys()}
        self.model_to_agent_indices = {
            model_name: [] for model_name in model_configs.keys()
        }
        for i, (agent_name, model_name) in enumerate(self.agent_model_dict.items()):
            assert model_name in self.model_to_agents.keys(), (
                f"Model {model_name} not found in model_configs"
            )
            self.model_to_agents[model_name].append(agent_name)
            self.model_to_agent_indices[model_name].append(i)

        self.model_class_names = {}
        for model_name, model in self.models.items():
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
        action_logits_list = [None for _ in range(len(self.agent_model_dict))]
        for (model_name, model), (_, fe) in zip(
            self.models.items(), self.feature_extractors.items()
        ):
            selected_agents = self.model_to_agents[model_name]
            idx = self.model_to_agent_indices[model_name]
            obs = observations[:, idx]
            bs = obs.shape[0]
            n_agents = len(selected_agents)
            ts = obs.shape[2]
            obs_shape = list(obs.shape[3:])

            model_class_name = self.model_class_names[model_name]
            if model_class_name == "Conv1DModel":
                obs = obs.reshape(bs * n_agents * ts, *obs_shape).to(self.device)
                obs_vectorized = fe(obs)
                obs_vectorized = obs_vectorized.reshape(bs * n_agents, ts, -1)
                obs_vectorized = obs_vectorized.permute(0, 2, 1)
                output = model(obs_vectorized)
            elif model_class_name == "RNNModel":
                obs = obs.reshape(bs * n_agents * ts, *obs_shape).to(self.device)
                obs_vectorized = fe(obs)
                obs_vectorized = obs_vectorized.reshape(bs * n_agents, ts, -1)
                model.train()
                output = model(obs_vectorized)
            elif model_class_name == "AttentionModel":
                obs = obs.reshape(bs * n_agents * ts, *obs_shape).to(self.device)
                obs_vectorized = fe(obs)
                obs_vectorized = obs_vectorized.reshape(bs * n_agents, ts, -1)
                mask = traj_padding_mask[:, idx]
                mask = mask.reshape(bs * n_agents, ts)
                output = model(obs_vectorized, mask)
            else:
                obs = obs[:, :, -1, :]
                obs = obs.reshape(bs * n_agents, *obs_shape).to(self.device)
                obs_vectorized = fe(obs)
                obs_vectorized = obs_vectorized.reshape(bs * n_agents, -1)
                output = model(obs_vectorized)

            output = output.reshape(bs, n_agents, -1)
            output = output.permute(1, 0, 2)

            for i, q in zip(idx, output):
                action_logits_list[i] = q

        action_logits = torch.stack(action_logits_list).to(self.device)
        action_logits = action_logits.permute(1, 0, 2)

        return {"action_logits": action_logits}

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
            logits = ret["action_logits"].squeeze(0)
            logits = logits.detach()

        action_mask_array = isinstance(next(iter(avail_actions.values())), np.ndarray)

        if action_mask_array:
            action_masks = torch.tensor(
                np.array([avail_actions[agent] for agent in self.agent_model_dict.keys()]),
                dtype=torch.bool,
                device=self.device,
            )
        else:
            action_masks = None

        alive_flag = torch.tensor(
            [agent in set(alive_agents) for agent in self.agent_model_dict.keys()],
            dtype=torch.bool,
            device=self.device,
        )

        all_actions = {}
        all_log_probs = {}

        for i, agent in enumerate(self.agent_model_dict.keys()):
            if alive_flag[i]:
                if action_masks is not None:
                    masked_logits = logits[i].clone()
                    masked_logits[~action_masks[i]] = -float("inf")
                    dist = Categorical(logits=masked_logits)
                else:
                    dist = Categorical(logits=logits[i])

                if np.random.random() < epsilon:
                    action = dist.sample()
                else:
                    action = dist.probs.argmax()

                log_prob = dist.log_prob(action)
                all_actions[agent] = action.cpu().item()
                all_log_probs[agent] = log_prob.cpu().item()
            else:
                all_actions[agent] = 0
                all_log_probs[agent] = 0.0

        actual_actions = {agent: all_actions[agent] for agent in alive_agents}
        log_probs = {agent: all_log_probs[agent] for agent in alive_agents}

        return {
            "actions": actual_actions,
            "all_actions": all_actions,
            "log_probs": log_probs,
            "all_log_probs": all_log_probs,
        }

    def reset(self) -> "MAPPOAgentGroup":
        return self
