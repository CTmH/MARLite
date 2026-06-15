import torch
from marlite.algorithm.model import ModelConfig, MaskedModel


class StateValue(torch.nn.Module):
    def __init__(self, base_model_config: ModelConfig, feature_extractor_config: ModelConfig):
        super().__init__()
        self.base_model = base_model_config.get_model()
        self.feature_extractor = feature_extractor_config.get_model()
        self._fe_is_masked = isinstance(self.feature_extractor, MaskedModel)

    def forward(self, states, alive_mask, padding_mask):
        states_last = states[:, -1, ...]
        alive_last = alive_mask[:, -1, ...]

        if self._fe_is_masked:
            mask_flat = alive_last.reshape(alive_last.shape[0], -1) if alive_last.dim() > 1 else alive_last
            state_features = self.feature_extractor(states_last, mask_flat)
        else:
            state_features = self.feature_extractor(states_last)

        if state_features.dim() > 2:
            state_features = state_features.reshape(state_features.shape[0], -1)

        value = self.base_model(state_features)
        value = value.reshape(states.shape[0], -1)
        return {"v": value}
