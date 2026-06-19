"""QPLEX Transformation module (paper Section 3.2, Eq. 7, Eq. 10).

Corresponds to PyMARL's ``Qatten_Weight``.  Produces per-agent
transformation weights ``w_final`` and a per-agent bias ``v`` from the
global state via a multi-head cross-attention structure.

**Paper reference — Transformation network**
--------------------------------------------------
The Transformation module computes, for each agent ``i``:

    V_i(τ) = w_i(τ) · V_i(τ_i) + b_i(τ)             (Eq. 7, left)
    A_i(τ, a_i) = w_i(τ) · A_i(τ_i, a_i)             (Eq. 7, right)

where ``w_i(τ) > 0`` is a positive weight and ``b_i(τ)`` is a
state-dependent bias.

The module also implements the multi-head attention structure used
internally to produce those weights (Eq. 10):

    λ_i(τ, a) = Σ_{k=1}^{K} λ_{i,k}(τ, a) · φ_{i,k}(τ) · v_k(τ)

    with λ_{i,k}, φ_{i,k} = sigmoid(·),  v_k(τ) > 0

The selector / key / head-weight networks correspond to the three
components in the product above.

**Variable correspondences (code → paper)**
----------------------------------------------
+------------------+------------------------------+
| Code variable    | Paper symbol                 |
+------------------+------------------------------+
| ``w_final``      | ``w_i(τ)`` (Eq. 7)           |
| ``v``            | ``b_i(τ)`` (Eq. 7 bias)      |
| ``selectors``    | query network, part of Eq. 10 |
| ``keys``         | key network, part of Eq. 10   |
| ``head_weight``  | ``v_k(τ)`` in Eq. 10          |
+------------------+------------------------------+

The per-head ``λ_{i,k}`` and ``φ_{i,k}`` are implicitly handled by
the per-head selector/key attention weights passed through sigmoid
inside the :class:`QplexJointAttention` module (see that file for
the advantage-specific attention).
"""

from typing import Dict, List, Optional, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from marlite.algorithm.model.model_config import ModelConfig


class QplexTransformation(nn.Module):
    """QPLEX Transformation network — computes per-agent ``w_i(τ)`` and
    ``b_i(τ)`` from the global state (paper Eq. 7).

    The network uses a multi-head cross-attention between a **selector**
    (query vector, one per head) derived from the global state and
    per-agent **key** vectors (one per agent per head) also derived
    from the state (or from state + Q-value when ``nonlinear=True``).

    The attention softmax over the agent dimension produces per-head
    per-agent weights that are optionally re-weighted by a learned
    per-head importance (``v_k(τ)`` in Eq. 10) and summed across heads.

    Args:
        state_dim: Dimensionality of the (encoded) global state.
        n_agents: Number of agents.
        n_head: Number of attention heads.
        embed_dim: Embedding (hidden) dimension used by the attention.
        selector_configs: ``n_head`` ModelConfig dicts.  Each maps the
            state to an ``embed_dim``-sized query vector
            (``state → embed_dim``).
        key_configs: ``n_head`` ModelConfig dicts.

            * ``nonlinear=False``: each maps the state to
              ``n_agents * embed_dim``, reshaped internally to
              ``(B, n_agents, embed_dim)``.
            * ``nonlinear=True``: each maps
              ``[state; agent_q]`` → ``embed_dim``, applied per-agent.

        v_config: ModelConfig — maps state → 1 (the bias ``b_i`` in
            Eq. 7, broadcast to all agents).
        head_weight_config: Optional ModelConfig — maps state →
            ``n_head`` (the ``v_k`` importance in Eq. 10).
            ``None`` → uniform head weighting.
        nonlinear: If ``True``, the agent's own Q-value is concatenated
            to the key input (PyMARL's ``nonlinear`` flag).  This gives
            each agent a direct influence on its transformation weight.
        attend_reg_coef: Coefficient for the attention magnitude
            regulariser ``Σ_k (logit_k)²`` (optional, 0 = off).
    """

    def __init__(
        self,
        state_dim: int,
        n_agents: int,
        n_head: int,
        embed_dim: int,
        selector_configs: List[Dict],
        key_configs: List[Dict],
        v_config: Dict,
        head_weight_config: Optional[Dict] = None,
        nonlinear: bool = False,
        attend_reg_coef: float = 0.0,
    ):
        super().__init__()
        if len(selector_configs) != n_head:
            raise ValueError(
                f"selector_configs must have length n_head={n_head}, "
                f"got {len(selector_configs)}"
            )
        if len(key_configs) != n_head:
            raise ValueError(
                f"key_configs must have length n_head={n_head}, "
                f"got {len(key_configs)}"
            )

        # -- scalars ------------------------------------------------------------
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.n_head = n_head          # K in Eq. 10 (number of attention heads)
        self.embed_dim = embed_dim
        self.nonlinear = nonlinear
        self.attend_reg_coef = attend_reg_coef

        from marlite.algorithm.model.model_config import ModelConfig

        # -- sub-networks -------------------------------------------------------
        # Per-head queries (selector networks): state → embed_dim.
        # These correspond to the query in the multi-head attention of
        # the Transformation module (the ``φ_{i,k}`` or ``λ_{i,k}``
        # projection in Eq. 10, depending on the head).
        self.selectors = nn.ModuleList(
            [ModelConfig(**cfg).get_model() for cfg in selector_configs]
        )

        # Per-head key networks.
        # non-nonlinear: state → (n_agents * embed_dim), reshaped.
        # nonlinear:     (state + Q_i) → embed_dim, applied per-agent.
        if nonlinear:
            key_input_dim = state_dim + 1     # state + per-agent Q-value
            key_output_dim = embed_dim
        else:
            key_input_dim = state_dim
            key_output_dim = n_agents * embed_dim
        self.keys = nn.ModuleList()
        for cfg in key_configs:
            d = dict(cfg)
            d.setdefault("in_features", key_input_dim)
            d.setdefault("out_features", key_output_dim)
            self.keys.append(ModelConfig(**d).get_model())

        # V(s) network: state → 1.
        # This produces b_i(τ) in Eq. 7 (the state-dependent bias,
        # broadcast to all agents).
        self.V = ModelConfig(**v_config).get_model()

        # Optional per-head importance: state → n_head.
        # This corresponds to v_k(τ) > 0 in Eq. 10 (positive key per head).
        self.head_weight = (
            ModelConfig(**head_weight_config).get_model()
            if head_weight_config is not None
            else None
        )

    def forward(
        self, agent_qs: torch.Tensor, states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute per-agent transformation weights and bias
        (paper Eq. 7) along with an optional attention regulariser.

        Args:
            agent_qs: Per-agent chosen Q-values, shape ``(B, N)``.
                Only used when ``nonlinear=True``; ignored otherwise.
            states: Encoded global state, shape ``(B, state_dim)``.

        Returns:
            ``(w_final, v, att_reg)``:

            * **w_final** — ``w_i(τ)`` in Eq. 7, shape ``(B, N)``,
              positive.
            * **v** — ``b_i(τ)`` in Eq. 7, shape ``(B, N)``.
            * **att_reg** — scalar regulariser (0 if
              ``attend_reg_coef == 0``).
        """
        bs = states.size(0)
        states_flat = states.reshape(-1, self.state_dim)

        # ------------------------------------------------------------------
        # 1. Prepare per-agent key input when nonlinear (Eq. 7 with
        #    nonlinear key-query interaction).
        # ------------------------------------------------------------------
        if self.nonlinear:
            # Eq. 7: the transformation can optionally take the per-agent
            # Q-value Q_i(τ_i, a_i) as additional input so that the key
            # network sees ``[state; Q_i(τ_i, a_i)]`` and can produce
            # agent-specific keys even when the state is the same for all
            # agents.  This alleviates the partial observability problem.
            # Per-agent key input: the global state (same for all agents)
            # concatenated with the agent's own Q-value (different per agent).
            # This follows the paper's Eq. 7 where the transformation can
            # incorporate individual Q-values into the weight computation,
            # and matches PyMARL's ``nonlinear`` flag in Qatten_Weight.
            #
            # states_flat:      (B, state_dim)
            # agent_qs:         (B, N)
            # key_input:        (B, N, state_dim + 1) — per agent,
            #                    ``[state, Q_i(τ_i, a_i)]``
            states_expanded = states_flat.unsqueeze(1).expand(
                -1, self.n_agents, -1
            )                                                              # (B, N, state_dim)
            unit_q = agent_qs.reshape(bs, self.n_agents, 1)               # (B, N, 1)
            key_input = torch.cat([states_expanded, unit_q], dim=-1)      # (B, N, state_dim + 1)
        else:
            key_input = None

        head_attend_logits: List[torch.Tensor] = []
        head_attend_weights: List[torch.Tensor] = []

        # ------------------------------------------------------------------
        # 2. Per-head multi-head attention (Eq. 10).
        # ------------------------------------------------------------------
        for i in range(self.n_head):
            # Query: selectors[i](state) → embed_dim.
            sel = self.selectors[i](states_flat)

            # Key: for each agent, key from state (or state + agent_q).
            if self.nonlinear and key_input is not None:
                key_flat = key_input.reshape(bs * self.n_agents, -1)
                key_out_flat = self.keys[i](key_flat)                    # (B*N, embed_dim)
                key_out = key_out_flat.reshape(bs, self.n_agents, self.embed_dim)
            else:
                key_flat = self.keys[i](states_flat)
                key_out = key_flat.reshape(bs, self.n_agents, self.embed_dim)
            # key_out: (B, n_agents, embed_dim) — one key per agent.

            # Cross-attention: ``sel^T · key_i`` (scaled dot product).
            # (B, 1, embed_dim) x (B, embed_dim, N) → (B, 1, N)
            attend_logits = torch.matmul(
                sel.unsqueeze(1),
                key_out.permute(0, 2, 1),
            )

            # Scale and softmax over the agent dimension → per-head
            # per-agent attention weight.
            scaled_attend_logits = attend_logits / (self.embed_dim ** 0.5)
            attend_weights = F.softmax(scaled_attend_logits, dim=-1)     # (B, 1, N)

            head_attend_logits.append(attend_logits)
            head_attend_weights.append(attend_weights)

        # ------------------------------------------------------------------
        # 3. Aggregate heads → w_final.
        # ------------------------------------------------------------------
        # Stack all heads: (B, n_head, 1, N) → (B, n_head, N).
        head_attend = torch.stack(head_attend_weights, dim=1).squeeze(2)

        if self.head_weight is not None:
            # v_k(τ) in Eq. 10: per-head positive importance weight.
            w_head = torch.abs(self.head_weight(states_flat)) + 1e-10    # (B, n_head)
            head_attend = head_attend * w_head.unsqueeze(-1)

        # Sum over heads → (B, N).  This is w_i(τ) in Eq. 7.
        w_final = head_attend.sum(dim=1) + 1e-10                         # (B, N), > 0

        # ------------------------------------------------------------------
        # 4. V(s) bias (b_i(τ) in Eq. 7).
        # ------------------------------------------------------------------
        v_scalar = self.V(states_flat)                                   # (B, 1)
        v = v_scalar.view(-1, 1).expand(-1, self.n_agents)               # (B, N), broadcast

        # ------------------------------------------------------------------
        # 5. Attention magnitude regulariser (optional, see Appendix B).
        # ------------------------------------------------------------------
        att_reg: torch.Tensor
        if self.attend_reg_coef > 0 and len(head_attend_logits) > 0:
            reg_terms = torch.stack(
                [(logit ** 2).mean() for logit in head_attend_logits]
            )
            att_reg = self.attend_reg_coef * reg_terms.sum()
        else:
            att_reg = torch.zeros((), device=states.device, dtype=states.dtype)

        return w_final, v, att_reg
