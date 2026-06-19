"""QPLEX Mixer — Duplex Dueling value factorisation (paper Eq. 8–11).

Implements the joint Q-function ``Q_tot`` from

    "Duplex Dueling Multi-Agent Q-Learning" (Wang et al., ICLR 2021).

The mixer composes the :class:`QplexTransformation` and
:class:`QplexJointAttention` modules (plus two value-stream sub-nets)
to form the **duplex dueling** architecture in the paper's Figure 1a.

**Full QPLEX factorisation (paper Eq. 11)**
--------------------------------------------------
The joint action-value is decomposed as:

    Q_tot(τ, a) = V_tot(τ) + A_tot(τ, a)                            (Eq. 11a)

where

    V_tot(τ) = Σ_i V_i(τ)                                            (Eq. 8)
    A_tot(τ, a) = Σ_i λ_i(τ, a) · A_i(τ, a_i)                      (Eq. 9)

Expanding V_i and A_i via the duplex structure (Eq. 7) gives the
equivalent form:

    Q_tot(τ, a) = Σ_i Q_i(τ, a_i) + Σ_i (λ_i(τ, a) − 1) · Ã_i(τ, a_i)  (Eq. 11b)

where ``Ã_i`` is the local advantage with **stopped gradients**
(paper line 468, PyMARL ``.detach()`` on line 42 of ``dmaq_general.py``).

**Implementation outline (graph in paper Figure 1a)**
--------------------------------------------------------
1. **State encoding** — optional feature extractor (supports masked
   models for variable-length agent groups).
2. **Transformation** (Eq. 7) — ``QplexTransformation`` produces
   per-agent weights ``w_i(τ)`` and bias ``b_i(τ)`` from the state.
3. **Value stream** — produces the weighted sum ``V_tot(τ)`` from the
   per-agent Q-values (Eq. 8, optionally with the ``weighted_head``
   variant from PyMARL).
4. **Joint advantage** (Eq. 9) — ``QplexJointAttention`` produces
   ``λ_i(τ, a)`` from state + joint one-hot actions.  The local
   advantage ``A_i(τ, a_i)`` is constructed as
   ``(Q_i − max_i Q_i).detach()`` (paper line 468),
   then ``A_tot = Σ λ_i · Ã_i`` (or ``Σ (λ_i − 1) · Ã_i`` when
   ``is_minus_one=True``, matching Eq. 11b).
5. **Combine** — ``Q_tot = V_tot + A_tot`` (Eq. 11a).

**Variable correspondences (code → paper)**
----------------------------------------------
+-------------------------------+-----------------------------------------------+
| Code variable                 | Paper symbol                                  |
+-------------------------------+-----------------------------------------------+
| ``q_value_from_agents``       | ``Q_i(τ_i, a_i)`` (full per-agent table)      |
| ``chosen_q`` = gather(…, a)   | ``Q_i(τ_i, a_i)`` for the executed action     |
| ``max_q_i``                   | ``max_{a_i} Q_i(τ_i, a_i)`` = ``V_i(τ_i)``   |
| `local_adv = (chosen - max).detach()` | ``Ã_i(τ_i, a_i)`` in Eq. 11 (detached) |
| ``w_final`` (= w_i from trans.) | ``w_i(τ)`` in Eq. 7                        |
| ``v`` from transformation     | ``b_i(τ)`` in Eq. 7 (the V(s) bias)          |
| ``ws_w`` from value stream    | ``hyper_w_final`` in PyMARL (extra weighting) |
| ``ws_v`` from value stream    | ``V`` in PyMARL (extra bias)                  |
| ``V_tot``                     | ``V_tot(τ)`` (Eq. 8)                          |
| ``λ`` = ``lambda_w``          | ``λ_i(τ, a)`` (Eq. 9, 10)                     |
| ``A_tot``                     | ``A_tot(τ, a)`` (Eq. 9)                       |
| ``Q_tot = V_tot + A_tot``     | ``Q_tot(τ, a)`` (Eq. 11)                      |
| ``att_reg``                   | Attention magnitude regulariser (Appendix B)  |
+-------------------------------+-----------------------------------------------+
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from marlite.algorithm.model import ModelConfig, MaskedModel
from marlite.algorithm.critic.mixer import Mixer


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
        weighted_head: If ``True`` (default), the value stream applies
            ``ws_w · chosen_Q + ws_v`` before summing (PyMARL QPLEX
            default).  If ``False``, ``V_tot = Σ_i Q_i`` (Eq. 8).
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

    def forward(  # type: ignore[override]
        self,
        q_value_from_agents: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        alive_mask: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute QPLEX joint Q-function components (Eq. 8–11).

        Args:
            q_value_from_agents: Full per-agent Q-values, shape
                ``(B, N, A)`` — one value per agent per action.
            states: Global state sequence, shape ``(B, T, state_dim)``.
            actions: Per-agent action indices at the last timestep,
                shape ``(B, N)``.
            alive_mask: Per-agent alive flags, shape ``(B, T, N)``.
                Dead agents have their actions masked to 0 (no-op)
                when constructing the joint one-hot.
            padding_mask: Trajectory padding mask, shape ``(B, T)``.

        Returns:
            Dict with keys:

            * ``q_tot`` — ``Q_tot(τ, a)`` (Eq. 11), shape ``(B,)``.
            * ``v_tot`` — ``V_tot(τ)`` (Eq. 8), shape ``(B,)``.
            * ``a_tot`` — ``A_tot(τ, a)`` (Eq. 9), shape ``(B,)``.
            * ``att_reg`` — attention magnitude regulariser (scalar).
        """
        bs = q_value_from_agents.size(0)

        # ------------------------------------------------------------------
        # 1.  State encoding (paper figures: "State s" → encoded features).
        # ------------------------------------------------------------------
        states_last = states[:, -1, :]
        alive_last = alive_mask[:, -1, :]

        if self._fe_is_masked:
            mask_flat = (
                alive_last.reshape(alive_last.shape[0], -1)
                if alive_last.dim() > 1
                else alive_last
            )
            encoded_state = self.feature_extractor(states_last, mask_flat)
        else:
            encoded_state = self.feature_extractor(states_last)

        # ------------------------------------------------------------------
        # 2.  Joint action one-hot (for Eq. 9 — λ depends on joint action).
        # ------------------------------------------------------------------
        # Dead agents get action = 0 (no-op) so the joint action one-hot
        # does not receive out-of-range indices.
        effective_actions = actions * alive_last.long()
        actions_onehot = F.one_hot(
            effective_actions.long(), num_classes=self.action_dim
        ).to(dtype=encoded_state.dtype, device=encoded_state.device)       # (B, N, A)
        joint_actions = actions_onehot.view(bs, self.n_agents * self.action_dim)

        # ------------------------------------------------------------------
        # 3.  Extract chosen_Q and max_Q from the full Q-value table.
        #     (Eq. 7: V_i(τ_i) = max_a Q_i(τ_i, a);
        #      Eq. 11: Q_i(τ, a_i) is the chosen action.)
        # ------------------------------------------------------------------
        chosen_q = torch.gather(
            q_value_from_agents, dim=-1, index=actions.unsqueeze(-1)
        ).squeeze(-1)                                                       # (B, N)  — Q_i

        max_q_i = q_value_from_agents.max(dim=-1).values                   # (B, N)  — V_i(τ_i)

        # ------------------------------------------------------------------
        # 4.  Transformation → per-agent w_i(τ) and b_i(τ)  (Eq. 7).
        # ------------------------------------------------------------------
        w_final, v, att_reg = self.transformation(chosen_q, encoded_state)
        # w_final: (B, N) — w_i(τ)
        # v:       (B, N) — b_i(τ)

        # ------------------------------------------------------------------
        # 5.  Value stream → V_tot(τ)  (Eq. 8).
        # ------------------------------------------------------------------
        # PyMARL's ``hyper_w_final`` uses abs + ε for positivity and ``V``
        # as an additional per-agent bias on the per-agent Q-values.
        ws_w = torch.abs(self.value_w_final(encoded_state)) + 1e-10        # (B, N)
        ws_v = self.value_v(encoded_state)                                 # (B, N)

        if self.weighted_head:
            # V_tot = Σ_i (ws_w_i · Q_i + ws_v_i)
            # This matches PyMARL QPLEX's `weighted_head=True` behaviour
            # (the standard QPLEX formulation).
            transformed_q = ws_w * chosen_q + ws_v
            transformed_max = ws_w * max_q_i + ws_v
        else:
            # Ablation: V_tot = Σ_i Q_i  (Eq. 8).
            transformed_q = chosen_q
            transformed_max = max_q_i

        v_tot = transformed_q.sum(dim=-1)                                  # (B,) — V_tot(τ) in Eq. 8

        # ------------------------------------------------------------------
        # 6.  Joint advantage attention → λ_i(τ, a)  (Eq. 9, 10).
        # ------------------------------------------------------------------
        lambda_w = self.joint_attention(encoded_state, joint_actions)      # (B, N) — λ_i(τ, a)

        # ------------------------------------------------------------------
        # 7.  Joint advantage → A_tot(τ, a)  (Eq. 9, 11).
        # ------------------------------------------------------------------
        # The local advantage Ã_i(τ, a_i) is:
        #   Ã_i(τ, a_i) = (w_i(τ) · Q_i(τ_i, a_i) + b_i(τ))
        #               − (w_i(τ) · V_i(τ_i) + b_i(τ))
        #               = w_i(τ) · (Q_i(τ_i, a_i) − V_i(τ_i))
        #
        # **Crucially**, the local advantage is **detached** from the
        # computation graph (paper line 468, PyMARL ``dmaq_general.py``
        # line 42, ``dmaq_qatten.py`` line 35):
        #
        #     "We stop gradients of local advantage function A_i to
        #      increase the optimisation stability of the max operator
        #      of dueling structure." (paper Appendix B)
        #
        # This prevents the ``max`` in V_i = max_a Q_i from creating an
        # undesirable gradient path that would destabilise learning.
        local_adv = (transformed_q - transformed_max).detach()             # (B, N) — Ã_i(τ, a_i)

        if self.is_minus_one:
            # Q_tot(τ, a) = V_tot(τ) + Σ_i λ_i(τ, a) · Ã_i(τ, a_i)
            #              = Σ_i Q_i + Σ_i (λ_i(τ, a) − 1) · Ã_i(τ, a_i)   (Eq. 11b)
            a_tot = (local_adv * (lambda_w - 1.0)).sum(dim=-1)             # (B,) — A_tot(τ, a)
        else:
            # Ablation: A_tot = Σ_i λ_i · Ã_i  (without the −1 offset).
            a_tot = (local_adv * lambda_w).sum(dim=-1)                     # (B,)

        # ------------------------------------------------------------------
        # 8.  Combine → Q_tot(τ, a)  (Eq. 11a).
        # ------------------------------------------------------------------
        q_tot = v_tot + a_tot

        return {
            "q_tot": q_tot,       # Q_tot(τ, a)  (Eq. 11)
            "v_tot": v_tot,       # V_tot(τ)     (Eq. 8)
            "a_tot": a_tot,       # A_tot(τ, a)  (Eq. 9)
            "att_reg": att_reg,   # optional regulariser
        }
