"""MAPPOCritic — state-value critic that uses only the last timestep.

Follows the same pattern as QMixer (``critic/qmix_mixer.py``): the feature
extractor and base model operate only on the final timestep of the state
sequence, producing a single scalar value V(s_{T-1}).
"""

import torch
from typing import Dict

from marlite.algorithm.model import ModelConfig, MaskedModel
from marlite.algorithm.critic.critic import Critic


class MAPPOCritic(Critic):
    """Value network that estimates V(s) from the last timestep only.

    Parameters
    ----------
    base_model_config : ModelConfig
        Model that maps state features to a scalar value.
    feature_extractor_config : ModelConfig
        Feature extractor applied to the last timestep's state.
    """

    def __init__(
        self,
        base_model_config: ModelConfig,
        feature_extractor_config: ModelConfig,
    ):
        super().__init__()
        self.base_model = base_model_config.get_model()
        self.feature_extractor = feature_extractor_config.get_model()
        self._fe_is_masked = isinstance(self.feature_extractor, MaskedModel)

    def forward(
        self,
        states: torch.Tensor,
        alive_mask: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            states: ``(B, T, state_dim)`` or ``(B, T, N, state_dim)``
            alive_mask: ``(B, T)`` or ``(B, T, N)``
            padding_mask: ``(B, T)``

        Returns:
            ``{"v": (B, 1)}`` — value of the **last** timestep only.
        """
        # Use only the last timestep — ``...`` handles any trailing dims generically.
        states_last = states[:, -1, ...]
        alive_last = alive_mask[:, -1, ...]

        if self._fe_is_masked:
            # MaskedModel expects (B, state_dim) with an optional mask of shape (B, N)
            # or (B, H*W) etc.  Flatten the mask when it has trailing dims.
            mask_flat = alive_last.reshape(alive_last.shape[0], -1) if alive_last.dim() > 1 else alive_last
            state_features = self.feature_extractor(states_last, mask_flat)
        else:
            state_features = self.feature_extractor(states_last)

        # Flatten leading dims if the FE retained structure
        if state_features.dim() > 2:
            state_features = state_features.reshape(state_features.shape[0], -1)

        value = self.base_model(state_features)
        value = value.reshape(states.shape[0], -1)

        return {"v": value}
