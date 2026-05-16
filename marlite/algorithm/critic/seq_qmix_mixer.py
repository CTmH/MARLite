"""Sequential QMixer — processes full state sequence through a seq model."""

import torch
from typing import Dict

from marlite.algorithm.model import ModelConfig, RNNModel, Conv1DModel, AttentionModel, MaskedModel
from marlite.algorithm.critic.mixer import Mixer


class SeqQMixer(Mixer):
    def __init__(
        self,
        base_model_config: ModelConfig,
        feature_extractor_config: ModelConfig,
        seq_model_config: ModelConfig,
        state_feature_type: str = "Seq",
    ):
        super().__init__()
        self.base_model = base_model_config.get_model()
        self.feature_extractor = feature_extractor_config.get_model()
        self.seq_model = seq_model_config.get_model()
        self.state_feature_type = state_feature_type

        self._fe_is_masked = isinstance(self.feature_extractor, MaskedModel)

        if isinstance(self.seq_model, RNNModel):
            self._seq_model_class = "RNNModel"
        elif isinstance(self.seq_model, Conv1DModel):
            self._seq_model_class = "Conv1DModel"
        elif isinstance(self.seq_model, AttentionModel):
            self._seq_model_class = "AttentionModel"
        else:
            self._seq_model_class = self.seq_model.__class__.__name__

    def forward(
        self,
        q_value_from_agents: torch.Tensor,
        states: torch.Tensor,
        alive_mask: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        bs = q_value_from_agents.shape[0]
        ts = states.shape[1]
        state_shape = states.shape[2:]
        states_flat = states.reshape(bs * ts, *state_shape)

        if self._fe_is_masked:
            encoded = self.feature_extractor(
                states_flat, alive_mask.reshape(bs * ts, -1)
            )
        else:
            encoded = self.feature_extractor(states_flat)

        encoded = encoded.reshape(bs, ts, -1)
        last_encoded = encoded[:, -1, :]

        if self._seq_model_class == "Conv1DModel":
            encoded = encoded.permute(0, 2, 1)
            hidden = self.seq_model(encoded)
        elif self._seq_model_class == "RNNModel":
            hidden = self.seq_model(encoded)
        elif self._seq_model_class == "AttentionModel":
            hidden = self.seq_model(encoded, padding_mask)
        else:
            hidden = self.seq_model(encoded[:, -1, :])

        masked_q_values = q_value_from_agents * alive_mask[:, -1, :]
        q_tot = self.base_model(masked_q_values, hidden)

        if self.state_feature_type == "State":
            state_features = last_encoded
        else:
            state_features = hidden

        return {"q_tot": q_tot, "state_features": state_features}
