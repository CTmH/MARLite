import numpy as np
import torch
from marlite.util.action_distribution import masked_categorical
from typing import Dict, Any, List

from marlite.algorithm.agents.qmix_agent_group import QMIXAgentGroup


class MAPPOAgentGroup(QMIXAgentGroup):
    """Agent group that outputs action logits for PPO instead of Q-values.

    Inherits all model management and observation-processing logic from
    :class:`QMIXAgentGroup`.  The forward pass is identical except that
    the model output is returned under the key ``"action_logits"``.
    The ``act()`` method samples from a categorical distribution and
    returns per-action log-probabilities.
    """

    def forward(
        self,
        observations: torch.Tensor,
        traj_padding_mask: torch.Tensor,
        alive_mask: torch.Tensor,
    ) -> Dict[str, Any]:
        result = super().forward(observations, traj_padding_mask, alive_mask)
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
            ret = self(obs, padding_mask, alive_mask)
            logits = ret["action_logits"].squeeze(0).detach()

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
                dist = masked_categorical(
                    logits[i], None if action_masks is None else action_masks[i]
                )

                action = dist.sample()
                log_prob = dist.log_prob(action)
                all_actions[agent] = action.cpu().item()
                all_log_probs[agent] = log_prob.cpu().item()
            else:
                all_actions[agent] = 0
                all_log_probs[agent] = 0.0

        actual_actions = {agent: all_actions[agent] for agent in alive_agents}

        return {
            "actions": actual_actions,
            "all_actions": all_actions,
            "all_log_probs": all_log_probs,
        }
