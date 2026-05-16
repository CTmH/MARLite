"""VAE Group Consensus MAPPO agent group.

Extends GroupConsensusAgentGroup to output action logits (for PPO policy)
instead of Q-values.  The VAE consensus pipeline (dual encoders, group merging,
scattering, and reparameterisation) is inherited unchanged.
"""

import numpy as np
import torch
from torch.distributions import Categorical
from typing import Dict, Any, List, Optional

from marlite.algorithm.agents.group_consensus_agent_group import (
    GroupConsensusAgentGroup,
)


class VAEGroupConsensusMAPPOAgentGroup(GroupConsensusAgentGroup):
    """GroupConsensus agent group that outputs action logits for PPO.

    The forward pass is identical to the parent class except that the decoder
    output is returned under the key ``"action_logits"`` instead of ``"q_val"``.
    The ``act()`` method samples from a categorical distribution and returns
    per-action log-probabilities, as required by on-policy PPO.

    All VAE consensus functionality (latent estimation, group merging,
    scattering, reparameterisation) is inherited from
    :class:`GroupConsensusAgentGroup`.
    """

    def forward(
        self,
        observations: torch.Tensor,
        states: np.ndarray,
        traj_padding_mask: torch.Tensor,
        alive_mask: torch.Tensor,
        group_indices: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        result = super().forward(
            observations, states, traj_padding_mask, alive_mask, group_indices
        )
        result["action_logits"] = result.pop("q_val")
        return result

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
                obs, np.expand_dims(state, axis=0), padding_mask, alive_mask
            )
            logits = ret["action_logits"].squeeze(0).detach()
            group_indices_arr = ret["group_indices"].squeeze(0)

        action_mask_array = isinstance(
            next(iter(avail_actions.values())), np.ndarray
        )
        if action_mask_array:
            action_masks = torch.tensor(
                np.array(
                    [
                        avail_actions[agent]
                        for agent in self.agent_model_dict.keys()
                    ]
                ),
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

        actual_actions = {
            agent: all_actions[agent] for agent in alive_agents
        }
        log_probs = {
            agent: all_log_probs[agent] for agent in alive_agents
        }

        all_group_indices = {
            agent: int(gid)
            for agent, gid in zip(
                self.agent_model_dict.keys(), group_indices_arr
            )
        }
        actual_group_indices = {
            agent: all_group_indices[agent] for agent in alive_agents
        }

        return {
            "actions": actual_actions,
            "all_actions": all_actions,
            "log_probs": log_probs,
            "all_log_probs": all_log_probs,
            "group_indices": actual_group_indices,
            "all_group_indices": all_group_indices,
        }
