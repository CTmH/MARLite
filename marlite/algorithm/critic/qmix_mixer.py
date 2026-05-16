"""Standard QMixer — uses only the last timestep of the state sequence."""

import torch
from typing import Dict

from marlite.algorithm.model import ModelConfig, MaskedModel
from marlite.algorithm.critic.mixer import Mixer


class QMixer(Mixer):
    def __init__(
        self, base_model_config: ModelConfig, feature_extractor_config: ModelConfig
    ):
        super().__init__()
        self.base_model = base_model_config.get_model()
        self.feature_extractor = feature_extractor_config.get_model()
        self._fe_is_masked = isinstance(self.feature_extractor, MaskedModel)

    def forward(
        self,
        q_value_from_agents: torch.Tensor,
        states: torch.Tensor,
        alive_mask: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        alive_mask_last = alive_mask[:, -1, :]
        states_last = states[:, -1, :]

        if self._fe_is_masked:
            encoded_states = self.feature_extractor(states_last, alive_mask_last)
        else:
            encoded_states = self.feature_extractor(states_last)

        masked_q_values = q_value_from_agents * alive_mask_last
        q_tot = self.base_model(masked_q_values, encoded_states)

        return {"q_tot": q_tot, "state_features": encoded_states}
