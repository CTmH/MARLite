"""QPLEX-style positive transformation and duplex-dueling value factorization.

The configured attention transformation is applied before the optional positive
value head. Both chosen and feasible maximum utilities use the SAME weights and
biases. Joint-action-dependent positive advantage weights preserve feasible IGM.
Local advantages are detached, as in the reference QPLEX implementation.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from marlite.algorithm.model import ModelConfig, MaskedModel
from marlite.algorithm.critic.mixer import Mixer
from marlite.util.value_learning import masked_action_values


class QPLEXMixer(Mixer):
    """QPLEX Duplex Dueling Mixer — computes ``Q_tot(τ, a)`` via
    the duplex dueling factorisation (paper Eq. 8–11).

    The mixer owns five sub-modules:

    1. ``feature_extractor`` — optional state encoding.
    2. ``transformation`` — :class:`QplexTransformation` (Eq. 7).
    3. ``value_w_final`` / ``value_v`` — extra value-stream weights
       and biases (``hyper_w_final`` / ``V`` in PyMARL).
    4. ``joint_attention`` — :class:`QplexJointAttention` (Eq. 9, 10).

    Args:
        transformation: ``ModelConfig`` for :class:`QplexTransformation`.
        joint_attention: ``ModelConfig`` for :class:`QplexJointAttention`.
        value_stream_w_final: ``ModelConfig`` for a network
            ``state → n_agents``.  The weights are made positive via
            ``abs() + ε`` (PyMARL's ``hyper_w_final``).
        value_stream_v: ``ModelConfig`` for a network
            ``state → n_agents`` (PyMARL's ``V`` bias).
        feature_extractor: Optional ``ModelConfig`` for state encoding.
            Defaults to identity.
        action_dim: Number of discrete actions per agent.
        n_agents: Number of agents.
        weighted_head: If ``True`` (default), apply an additional positive
            affine head AFTER the attention transformation. If ``False``,
            sum the attention-transformed utilities without the extra head.
        is_minus_one: If ``True`` (default), the advantage uses
            ``λ_i − 1`` (Eq. 11b).  If ``False``, uses ``λ_i``
            directly (ablation, matches DMAQer without the offset).
    """

    def __init__(
        self,
        transformation: Dict,
        joint_attention: Dict,
        value_stream_w_final: Dict,
        value_stream_v: Dict,
        feature_extractor: Optional[Dict] = None,
        action_dim: int = 0,
        n_agents: int = 0,
        weighted_head: bool = True,
        is_minus_one: bool = True,
    ):
        super().__init__()
        if feature_extractor is None:
            feature_extractor = {"model_type": "Identity"}
        self.feature_extractor = ModelConfig(**feature_extractor).get_model()
        self._fe_is_masked = isinstance(self.feature_extractor, MaskedModel)

        # Reusable ModelConfig-registered modules.
        self.transformation = ModelConfig(**transformation).get_model()
        self.joint_attention = ModelConfig(**joint_attention).get_model()

        # Value stream sub-networks (PyMARL's hyper_w_final and V).
        self.value_w_final = ModelConfig(**value_stream_w_final).get_model()
        self.value_v = ModelConfig(**value_stream_v).get_model()

        # Scalars.
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.weighted_head = weighted_head
        self.is_minus_one = is_minus_one

    def forward(
        self,
        q_value_from_agents: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        alive_mask: torch.Tensor,
        padding_mask: torch.Tensor,
        avail_actions: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Mix a full (B, N, A) table; avail_actions has the same shape.

        Inactive agents contribute neither values nor advantages. An all-dead
        sample has a finite zero joint value. Current and target calls must both
        supply their corresponding available-action masks.
        """
        bs = q_value_from_agents.size(0)
        alive = alive_mask[:, -1].to(device=q_value_from_agents.device, dtype=torch.bool)
        states_last = states[:, -1]
        encoded_state = (
            self.feature_extractor(states_last, alive)
            if self._fe_is_masked else self.feature_extractor(states_last)
        )
        feasible_q = masked_action_values(q_value_from_agents, avail_actions, alive)
        max_q = feasible_q.max(-1).values
        effective_actions = actions.long().masked_fill(~alive, 0)
        chosen_q = q_value_from_agents.gather(-1, effective_actions.unsqueeze(-1)).squeeze(-1)
        chosen_q = chosen_q.masked_fill(~alive, 0.)
        actions_onehot = F.one_hot(effective_actions, num_classes=self.action_dim).to(encoded_state)
        joint_actions = actions_onehot.reshape(bs, self.n_agents * self.action_dim)

        # Even a nonlinear transformation must be independent of the selected
        # joint action. Its optional utility input is the detached feasible maximum,
        # not chosen_q; otherwise different actions could change the value baseline.
        weights, bias, att_reg = self.transformation(max_q.detach(), encoded_state)
        transformed_q = weights * chosen_q + bias
        transformed_max = weights * max_q + bias
        if self.weighted_head:
            head_weights = self.value_w_final(encoded_state).abs() + 1e-10
            head_bias = self.value_v(encoded_state)
            transformed_q = head_weights * transformed_q + head_bias
            transformed_max = head_weights * transformed_max + head_bias
        transformed_q = transformed_q.masked_fill(~alive, 0.)
        transformed_max = transformed_max.masked_fill(~alive, 0.)

        value = transformed_q.sum(-1)
        local_advantage = (transformed_q - transformed_max).detach()
        joint_weights = self.joint_attention(encoded_state, joint_actions)
        if self.is_minus_one:
            joint_weights = joint_weights - 1.
        advantage = (joint_weights * local_advantage).sum(-1)
        return {
            "q_tot": value + advantage,
            "v_tot": value,
            "a_tot": advantage,
            "att_reg": att_reg,
        }
