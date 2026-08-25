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

from marlite.algorithm.agents.g2anet_agent_group import G2ANetAgentGroup


class G2ANetMAPPOAgentGroup(G2ANetAgentGroup):
    """G2ANet agent group that outputs action logits for on-policy MAPPO.

    Inherits the full ``forward()`` from :class:`~G2ANetAgentGroup` and
    renames the ``q_val`` key to ``action_logits``.  The ``act()`` method
    samples from a categorical distribution and returns per-action
    log-probabilities required by PPO.
    """

    def forward(
        self,
        observations: torch.Tensor,
        states: torch.Tensor,
        traj_padding_mask: torch.Tensor,
        alive_mask: torch.Tensor,
        edge_indices: List[np.ndarray] | None = None,
    ) -> Dict[str, Any]:
        result = super().forward(
            observations, states, traj_padding_mask, alive_mask, edge_indices,
        )
        return {"action_logits": result["q_val"], "edge_indices": result["edge_indices"]}

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
            states_tensor = torch.from_numpy(state).float().unsqueeze(0).to(
                device=self.device
            )
            ret = self(
                obs,
                states_tensor,
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