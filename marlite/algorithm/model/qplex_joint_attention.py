"""QPLEX Joint Advantage Attention module (paper Eq. 9, Eq. 10).

Corresponds to PyMARL's ``DMAQ_SI_Weight``.  Computes per-agent
importance weights ``λ_i(τ, a)`` via a multi-head attention structure
over the concatenation of global state and joint one-hot actions.

**Paper reference — Importance weights λ_i**
--------------------------------------------------
The joint advantage (Eq. 9) is:

    A_tot(τ, a) = Σ_{i=1}^{n} λ_i(τ, a) · A_i(τ, a_i)

with ``λ_i(τ, a) > 0``.  The implementation uses the multi-head
attention from Eq. 10:

    λ_i(τ, a) = Σ_{k=1}^{K}  |v_k(τ)| · σ(φ_{i,k}(τ)) · σ(λ_{i,k}(τ, a))

where:
- ``|v_k(τ)|``  = absolute value of the key network output (per-head).
- ``σ(φ_{i,k})`` = sigmoid of the agent-extractor output.
- ``σ(λ_{i,k})`` = sigmoid of the action-extractor output on
  ``[state; joint_onehot]``.
- Heads are summed, not concatenated.

**Variable correspondences (code → paper)**
----------------------------------------------
+-------------------------------+------------------------------+
| Code variable                 | Paper symbol                 |
+-------------------------------+------------------------------+
| ``λ`` (return value)          | ``λ_i(τ, a)`` (Eq. 9, 10)   |
| ``x_key = |keys[i](s)| + ε``  | ``|v_k(τ)|`` in Eq. 10      |
| ``x_agents = sigmoid(...)``   | ``σ(φ_{i,k}(τ))`` in Eq. 10 |
| ``x_action = sigmoid(...)``   | ``σ(λ_{i,k}(τ,a))`` in Eq. 10|
+-------------------------------+------------------------------+

The sigmoid activation introduces sparsity in the credit assignment
(paper Section 3.2).
"""

from typing import Dict, List, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from marlite.algorithm.model.model_config import ModelConfig


class QplexJointAttention(nn.Module):
    """QPLEX joint advantage attention — produces per-agent
    importance weights ``λ_i(τ, a)`` (paper Eq. 9, 10).

    For each head ``k`` (Eq. 10):

        weight_{i,k} = |key_k(s)| · σ(agent_k(s)) · σ(action_k([s, a]))

    The heads are **summed** (not concatenated) to give the final
    ``λ_i(τ, a)``.

    Args:
        state_dim: Dimensionality of the (encoded) global state.
        action_dim: Number of discrete actions per agent.
        n_agents: Number of agents.
        n_head: Number of attention heads.
        key_configs: ``n_head`` ModelConfig dicts.
            Each maps ``state → 1`` (the ``|v_k(τ)|`` gate in Eq. 10).
        agent_extractor_configs: ``n_head`` ModelConfig dicts.
            Each maps ``state → n_agents`` (the ``σ(φ_{i,k}(τ))``
            term in Eq. 10).
        action_extractor_configs: ``n_head`` ModelConfig dicts.
            Each maps ``[state; onehot(a)]`` → ``n_agents``
            (the ``σ(λ_{i,k}(τ,a))`` term in Eq. 10).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_agents: int,
        n_head: int,
        key_configs: List[Dict],
        agent_extractor_configs: List[Dict],
        action_extractor_configs: List[Dict],
    ):
        super().__init__()
        if not (
            len(key_configs)
            == len(agent_extractor_configs)
            == len(action_extractor_configs)
            == n_head
        ):
            raise ValueError(
                f"key_configs, agent_extractor_configs, and action_extractor_configs "
                f"must each have length n_head={n_head}"
            )

        # -- scalars ------------------------------------------------------------
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.n_head = n_head

        from marlite.algorithm.model.model_config import ModelConfig

        # -- sub-networks -------------------------------------------------------

        # Per-head key: state → 1.
        # After ``abs()`` this becomes the positive scalar ``v_k(τ) > 0``
        # in Eq. 10, acting as a per-head gate.
        self.keys = nn.ModuleList(
            [ModelConfig(**cfg).get_model() for cfg in key_configs]
        )

        # Per-head agent extractor: state → n_agents.
        # After sigmoid, this is ``σ(φ_{i,k}(τ))`` in Eq. 10 — a per-head
        # per-agent attention bias that depends only on the state.
        self.agent_extractors = nn.ModuleList(
            [ModelConfig(**cfg).get_model() for cfg in agent_extractor_configs]
        )

        # Per-head action extractor: (state + N*A) → n_agents.
        # After sigmoid, this is ``σ(λ_{i,k}(τ,a))`` in Eq. 10 — a
        # per-head per-agent action-dependent credit weight.
        self.action_extractors = nn.ModuleList()
        for cfg in action_extractor_configs:
            d = dict(cfg)
            d.setdefault("in_features", state_dim + n_agents * action_dim)
            self.action_extractors.append(ModelConfig(**d).get_model())

    def forward(
        self, states: torch.Tensor, joint_actions_onehot: torch.Tensor
    ) -> torch.Tensor:
        """Compute per-agent importance weights ``λ_i(τ, a)``
        (paper Eq. 9, 10).

        Args:
            states: Encoded global state, shape ``(B, state_dim)``.
            joint_actions_onehot: Joint one-hot action tensor, shape
                ``(B, N * A)`` — the flattening of per-agent one-hot
                action vectors.

        Returns:
            ``λ_i(τ, a)`` of shape ``(B, N)``, strictly positive.
        """
        bs = states.size(0)
        states_flat = states.reshape(-1, self.state_dim)

        # Concatenate state and joint actions → input for the action
        # extractors (the ``[s, a]`` concatenation in Eq. 10).
        actions_flat = joint_actions_onehot.reshape(-1, self.n_agents * self.action_dim)
        data = torch.cat([states_flat, actions_flat], dim=-1)  # (B, state_dim + N*A)

        head_attend_weights: List[torch.Tensor] = []

        # ------------------------------------------------------------------
        # Multi-head attention (Eq. 10).
        # ------------------------------------------------------------------
        for i in range(self.n_head):
            # Per-head key: |key(s)| + ε  →  v_k(τ) > 0.
            x_key = torch.abs(self.keys[i](states_flat)) + 1e-10         # (B, 1)

            # Per-head agent term: σ(φ_{i,k}(τ)).
            x_agents = torch.sigmoid(self.agent_extractors[i](states_flat))  # (B, N)

            # Per-head action term: σ(λ_{i,k}(τ, a)).
            x_action = torch.sigmoid(self.action_extractors[i](data))        # (B, N)

            # Elementwise product: the final per-head per-agent weight
            # (the product of the three sigmoid/abs-gated terms).
            weights = x_key * x_agents * x_action                        # (B, N)
            head_attend_weights.append(weights)

        # ------------------------------------------------------------------
        # Sum over heads (Eq. 10).
        # ------------------------------------------------------------------
        lambda_w = torch.stack(head_attend_weights, dim=1).sum(dim=1)    # (B, N)

        return lambda_w  # λ_i(τ, a) in Eq. 9
