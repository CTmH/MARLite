"""G2ANetMAPPOAgentGroup — G2ANet communication graph for MAPPO.

Extends :class:`GraphAgentGroup` with G2ANet-style graph attention.
The agent observations are encoded via feature_extractor → encoder,
then a G2ANet graph builder produces a weighted adjacency matrix.
Message passing is done via ``MatrixGCNModel`` (batch matrix multiply),
and the decoder produces **action logits** (not Q-values) for PPO.
"""

import numpy as np
import torch
from torch.distributions import Categorical
from typing import Dict, List, Any

from marlite.algorithm.model.model_config import ModelConfig
from marlite.algorithm.agents.graph_agent_group import GraphAgentGroup
from marlite.algorithm.graph_builder import GraphBuilderConfig


class G2ANetMAPPOAgentGroup(GraphAgentGroup):
    """G2ANet agent group that outputs action logits for on-policy MAPPO.

    Inherits graph-builder and graph-model infrastructure from
    :class:`GraphAgentGroup`.  The ``forward()`` mirrors
    :class:`G2ANetAgentGroup` but returns ``action_logits`` instead of
    ``q_val``.  The ``act()`` samples from a categorical distribution
    and returns per-action log-probabilities required by PPO.
    """

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

    def forward(
        self,
        observations: torch.Tensor,
        states: np.ndarray,
        traj_padding_mask: torch.Tensor,
        alive_mask: torch.Tensor,
        edge_indices: List[np.ndarray] | None = None,
    ) -> Dict[str, Any]:
        """G2ANet forward pass producing action logits.

        Parameters
        ----------
        observations : (B, N, T, *obs_shape)
            Agent observations stacked by agent index.
        states : (B, H, W, C) or (B, C, H, W)
            Global state (unused by G2ANet — graph is built from encoded
            observations alone).
        traj_padding_mask : (B, N, T)
            Padding mask for padded trajectory steps.
        alive_mask : (B, N)
            Alive agent mask for the last timestep.
        edge_indices : list, optional
            Pre-computed edge indices (not used; the graph builder
            regenerates them from encoded observations).

        Returns
        -------
        dict with keys ``action_logits`` (B, N, A) and ``edge_indices``.
        """
        msg = [None for _ in range(len(self.agent_model_dict))]
        for (model_name, fe), (_, enc) in zip(
            self.feature_extractors.items(), self.encoders.items()
        ):
            selected_agents = self.model_to_agents[model_name]
            idx = self.model_to_agent_indices[model_name]
            obs = observations[:, idx]            # (B, n_sel, T, *obs_shape)
            obs = torch.Tensor(obs)
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
                msg_selected = enc(obs_vectorized)
            elif model_class_name == "RNNModel":
                obs = obs.reshape(bs * n_agents * ts, *obs_shape).to(self.device)
                obs_vectorized = fe(obs)
                obs_vectorized = obs_vectorized.reshape(bs * n_agents, ts, -1)
                enc.train()
                msg_selected = enc(obs_vectorized)
            elif model_class_name == "AttentionModel":
                obs = obs.reshape(bs * n_agents * ts, *obs_shape).to(self.device)
                obs_vectorized = fe(obs)
                obs_vectorized = obs_vectorized.reshape(bs * n_agents, ts, -1)
                mask = traj_padding_mask[:, idx]
                mask = mask.reshape(bs * n_agents, ts)
                msg_selected = enc(obs_vectorized, mask)
            else:
                obs = obs[:, :, -1, :]
                obs = obs.reshape(bs * n_agents, *obs_shape).to(self.device)
                obs_vectorized = fe(obs)
                msg_selected = enc(obs_vectorized)

            msg_selected = msg_selected.reshape(bs, n_agents, -1)
            msg_selected = msg_selected.permute(1, 0, 2)

            for i, m in zip(idx, msg_selected):
                msg[i] = m

        msg = torch.stack(msg).to(self.device)     # (N, B, F)
        msg = msg.permute(1, 0, 2)                  # (B, N, F)
        local_obs = msg

        # Build graph from encoded observations.
        adj_matrix, edge_indices = self.graph_builder(msg)

        # Propagate messages through the graph.
        hidden_states = self.graph_model(msg, adj_matrix)  # (B, N, H)

        # Decoder: concat graph output + local observation, then project.
        action_logits = [None for _ in range(len(self.agent_model_dict))]
        emb_size = hidden_states.shape[-1] + local_obs.shape[-1]
        for model_name, dec in self.decoders.items():
            selected_agents = self.model_to_agents[model_name]
            idx = self.model_to_agent_indices[model_name]
            h = hidden_states[:, idx]         # (B, n_sel, H)
            lo = local_obs[:, idx]            # (B, n_sel, F)
            bs = h.shape[0]
            n_agents = len(selected_agents)
            emb = torch.cat((h, lo), dim=-1)
            emb = emb.reshape(bs * n_agents, emb_size)
            logits_selected = dec(emb)
            logits_selected = logits_selected.reshape(bs, n_agents, -1)
            logits_selected = logits_selected.permute(1, 0, 2)

            for i, m in zip(idx, logits_selected):
                action_logits[i] = m

        action_logits = torch.stack(action_logits).to(self.device)  # (N, B, A)
        action_logits = action_logits.permute(1, 0, 2)             # (B, N, A)

        return {"action_logits": action_logits, "edge_indices": edge_indices}

    def act(
        self,
        observations: Dict[str, np.ndarray],
        state: np.ndarray,
        avail_actions: Dict[str, Any],
        traj_padding_mask: np.ndarray,
        alive_agents: List[str],
        epsilon: float = 0.0,
    ) -> Dict[str, Any]:
        """Select actions by sampling from the categorical distribution.

        Extends the base :meth:`GraphAgentGroup.act` with log-probability
        computation required by PPO.
        """
        obs = [observations[agent] for agent in self.agent_model_dict.keys()]
        obs = np.stack(obs)
        obs = torch.tensor(obs).unsqueeze(0).to(
            dtype=torch.float, device=self.device
        )

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
            ret = self(
                obs,
                np.expand_dims(state, axis=0),
                padding_mask,
                alive_mask,
            )
            logits = ret["action_logits"].squeeze(0).detach()
            edge_idx_result = ret["edge_indices"]

        action_mask_array = isinstance(
            next(iter(avail_actions.values())), np.ndarray
        )
        if action_mask_array:
            action_masks = torch.tensor(
                np.array([
                    avail_actions[agent]
                    for agent in self.agent_model_dict.keys()
                ]),
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

                action = dist.sample()
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
            "edge_indices": edge_idx_result[0],
        }

    def reset(self) -> "G2ANetMAPPOAgentGroup":
        return self