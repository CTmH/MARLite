import torch
from marlite.algorithm.model import (
    ModelConfig,
    RNNModel,
    Conv1DModel,
    AttentionModel,
    MaskedModel,
)


class SeqStateValue(torch.nn.Module):
    def __init__(
        self,
        base_model_config: ModelConfig,
        feature_extractor_config: ModelConfig,
        seq_model_config: ModelConfig,
    ):
        super().__init__()
        self.base_model = base_model_config.get_model()
        self.feature_extractor = feature_extractor_config.get_model()
        self.seq_model = seq_model_config.get_model()

        self._fe_is_masked = isinstance(self.feature_extractor, MaskedModel)

        if isinstance(self.seq_model, RNNModel):
            self._seq_model_class = "RNNModel"
        elif isinstance(self.seq_model, Conv1DModel):
            self._seq_model_class = "Conv1DModel"
        elif isinstance(self.seq_model, AttentionModel):
            self._seq_model_class = "AttentionModel"
        else:
            self._seq_model_class = self.seq_model.__class__.__name__

    def forward(self, states, alive_mask, padding_mask):
        bs = states.shape[0]
        ts = states.shape[1]

        states_flat = states.reshape(bs * ts, *states.shape[2:])

        if self._fe_is_masked and states.dim() > 3:
            alive_flat = alive_mask.reshape(bs * ts, -1)
            state_features = self.feature_extractor(states_flat, alive_flat)
        else:
            state_features = self.feature_extractor(states_flat)

        state_features = state_features.reshape(bs, ts, -1)

        if self._seq_model_class == "Conv1DModel":
            state_features = state_features.permute(0, 2, 1)
            hidden = self.seq_model(state_features)
        elif self._seq_model_class == "RNNModel":
            hidden = self.seq_model(state_features)
        elif self._seq_model_class == "AttentionModel":
            hidden = self.seq_model(state_features, padding_mask)
        else:
            hidden = self.seq_model(state_features[:, -1, :])

        value = self.base_model(hidden)
        value = value.reshape(bs, -1)
        return {"v": value}
