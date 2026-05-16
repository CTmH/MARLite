import torch
import torch.nn as nn
from typing import Dict

from marlite.algorithm.model import ModelConfig
from marlite.algorithm.model import MaskedModel


class MAPPOCritic(nn.Module):
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
        bs = states.shape[0]
        ts = states.shape[1]

        if states.dim() == 4:
            n_agents = states.shape[2]
            state_rest = list(states.shape[3:])
            states_flat = states.reshape(bs * ts, n_agents, *state_rest)
        else:
            states_flat = states.reshape(bs * ts, *states.shape[2:])

        if self._fe_is_masked and states.dim() == 4:
            alive_flat = alive_mask.reshape(bs * ts, n_agents)
            state_features = self.feature_extractor(states_flat, alive_flat)
        else:
            state_features = self.feature_extractor(states_flat)

        state_features = state_features.reshape(bs * ts, -1)
        value = self.base_model(state_features)
        value = value.reshape(bs, ts, -1)

        return {"v": value}
