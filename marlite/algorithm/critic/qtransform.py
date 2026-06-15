from typing import Dict
import torch
import torch.nn.functional as F
from marlite.algorithm.critic.mixer import Mixer
from marlite.algorithm.model import ModelConfig


class Qtransform(Mixer):
    """Qtransform (QTRAN-alt) joint action-value mixer.

    Implements the counterfactual joint network from Son et al. (2019),
    "QTRAN: Learning to Factorize with Transformation for Cooperative
    Multi-Agent Reinforcement Learning", Section 3.4.

    The mixer owns three sub-modules:
        * ``phi_net`` -- per-agent state-action encoder. Maps the
          concatenation ``[enc_out_i ; onehot(a_i)]`` to a per-agent
          feature ``phi_i``. Corresponds to ``h_Q,i`` in the paper.
        * ``psi_net`` -- per-agent individual (state-only) encoder. Maps
          ``enc_out_i`` (derived from agent ``i``'s partial observation)
          to a per-agent feature ``psi_i``. Corresponds to ``h_V,i`` in
          the paper. Operates on per-agent observation, NOT the global
          state ``s`` (the global state is consumed by ``StateValue``
          to produce ``V_jt``).
        * ``base_model`` -- the configurable joint Q projection head.
          Maps the aggregated counterfactual vector to per-agent Q-values
          ``Q_jt(τ, a, a_{-i})`` for all actions ``a ∈ U`` in one forward
          pass. Can be any ``ModelConfig`` (typically a stack of
          ``Linear`` layers wrapped via ``Custom``).

    Counterfactual aggregation (paper Eq. 7's input):

        vector_i = psi_i + mean_j(phi_j) - phi_i / N

    Algebraically this simplifies to ``vector_i = psi_i +
    (1/N) Σ_{j != i} phi_j``, which is independent of agent ``i``'s
    own action ``a_i``. This identity is what makes the single-forward
    implementation equivalent to explicitly enumerating all
    ``a ∈ U``: the ``MLP(vector)`` output's ``a``-th slot is
    interpretable as ``Q_jt(τ, (a, a_{-i}))``.

    Args:
        phi_net_config: ``ModelConfig`` for the per-agent state-action
            encoder producing ``phi_i``. Input dim must equal
            ``enc_out_dim + action_dim``.
        psi_net_config: ``ModelConfig`` for the per-agent state-only
            encoder producing ``psi_i``. Input dim must equal
            ``enc_out_dim``.
        base_model_config: ``ModelConfig`` for the joint Q projection
            head. Input dim must equal ``phi``'s output dim (which must
            match ``psi``'s output dim). Output dim must equal
            ``action_dim``.
        action_dim: number of discrete actions per agent.
    """

    def __init__(
        self,
        phi_net_config: ModelConfig,
        psi_net_config: ModelConfig,
        base_model_config: ModelConfig,
        action_dim: int,
    ):
        super().__init__()
        # Per-agent state-action feature: phi_i = phi_net([enc_out_i ; onehot(a_i)])
        self.phi_net = phi_net_config.get_model()
        # Per-agent individual (observation) feature: psi_i = psi_net(enc_out_i)
        self.psi_net = psi_net_config.get_model()
        # Configurable joint Q projection head (typically an MLP).
        self.base_model = base_model_config.get_model()
        self.action_dim = action_dim

    def forward(  # type: ignore[override]
        self,
        enc_out: torch.Tensor,
        actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute per-agent counterfactual joint Q-values.

        Args:
            enc_out: per-agent encoder output, shape ``(B, N, D_enc)``.
            actions: per-agent action indices, shape ``(B, N)``.

        Returns:
            dict with key ``"q_per_action"`` of shape
            ``(B, N, action_dim)``. ``q_per_action[b, i, a]`` is
            ``Q_jt(τ, (a, a_{-i}))`` -- the joint Q when agent ``i``
            takes action ``a`` and the other agents keep their actual
            actions ``a_{-i}``.
        """
        # One-hot encode actions and concatenate with enc_out, then pass
        # through phi_net to get per-agent state-action features.
        a_onehot = F.one_hot(actions.long(), num_classes=self.action_dim).to(enc_out)
        phi = self.phi_net(torch.cat([enc_out, a_onehot], dim=-1))
        # Per-agent state-only features (no action conditioning).
        psi = self.psi_net(enc_out)
        # Counterfactual aggregation: vector_i = psi_i + mean(phi) - phi_i / N.
        # Note: algebraically this is psi_i + (1/N) Σ_{j != i} phi_j, which
        # is independent of a_i. That is what enables the "1 forward pass
        # to enumerate all A counterfactuals" trick.
        n_agents = phi.size(1)
        enc_mean = phi.mean(dim=1, keepdim=True)
        vector = psi + enc_mean - phi / n_agents
        # Project aggregated vector to per-agent Q-values for all actions.
        q_per_action = self.base_model(vector)
        return {"q_per_action": q_per_action}
