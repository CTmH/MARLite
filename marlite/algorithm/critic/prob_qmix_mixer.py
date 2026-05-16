"""Probabilistic QMixer — models uncertainty through variational inference."""

import torch
from typing import Dict

from marlite.algorithm.model import ModelConfig, MaskedModel
from marlite.algorithm.critic.mixer import Mixer


class ProbQMixer(Mixer):
    def __init__(
        self,
        base_model_config: ModelConfig,
        feature_extractor_config: ModelConfig,
        deterministic_eval=True,
    ):
        super().__init__()
        self.base_model = base_model_config.get_model()
        self.feature_extractor = feature_extractor_config.get_model()
        self._fe_is_masked = isinstance(self.feature_extractor, MaskedModel)
        self.deterministic_eval = deterministic_eval

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
            encoded = self.feature_extractor(states_last, alive_mask_last)
        else:
            encoded = self.feature_extractor(states_last)

        dim = encoded.size(-1) // 2
        mu = encoded[:, :dim]
        log_var = encoded[:, dim:]
        std = torch.exp(0.5 * log_var)

        if self.deterministic_eval and not self.training:
            sample = mu
        else:
            eps = torch.randn_like(std)
            sample = mu + eps * std

        masked_q_values = q_value_from_agents * alive_mask_last
        q_tot = self.base_model(masked_q_values, sample)

        return {"q_tot": q_tot, "state_features": sample, "mu": mu, "std": std}
