"""Probabilistic Sequential QMixer — sequence model + variational inference."""

import torch
from typing import Dict

from marlite.algorithm.model import ModelConfig, RNNModel, Conv1DModel, AttentionModel, MaskedModel
from marlite.algorithm.critic.mixer import Mixer


class ProbSeqQMixer(Mixer):
    def __init__(
        self,
        base_model_config: ModelConfig,
        feature_extractor_config: ModelConfig,
        seq_model_config: ModelConfig,
        state_feature_type: str = "Seq",
        deterministic_eval=True,
    ):
        super().__init__()
        self.base_model = base_model_config.get_model()
        self.feature_extractor = feature_extractor_config.get_model()
        self.seq_model = seq_model_config.get_model()
        self.state_feature_type = state_feature_type
        self.deterministic_eval = deterministic_eval

        if state_feature_type == "State":
            self._forward_fn = self._forward_with_state_hidden
        else:
            self._forward_fn = self._forward_with_seq_hidden

        self._fe_is_masked = isinstance(self.feature_extractor, MaskedModel)

        if isinstance(self.seq_model, RNNModel):
            self._seq_model_class = "RNNModel"
        elif isinstance(self.seq_model, Conv1DModel):
            self._seq_model_class = "Conv1DModel"
        elif isinstance(self.seq_model, AttentionModel):
            self._seq_model_class = "AttentionModel"
        else:
            self._seq_model_class = self.seq_model.__class__.__name__

    def forward(self, q_value_from_agents, states, alive_mask, padding_mask):
        return self._forward_fn(q_value_from_agents, states, alive_mask, padding_mask)

    def _forward_with_state_hidden(self, q_value_from_agents, states, alive_mask, padding_mask):
        bs = q_value_from_agents.shape[0]
        ts = states.shape[1]
        state_shape = states.shape[2:]
        states_flat = states.reshape(bs * ts, *state_shape)

        if self._fe_is_masked:
            encoded = self.feature_extractor(states_flat, alive_mask.reshape(bs * ts, -1))
        else:
            encoded = self.feature_extractor(states_flat)

        dim = encoded.size(-1) // 2
        mu = encoded[:, :dim]
        log_var = encoded[:, dim:]
        std = torch.exp(0.5 * log_var)

        if self.deterministic_eval and not self.training:
            sample = mu
        else:
            eps = torch.randn_like(std)
            sample = mu + eps * std

        sample = sample.reshape(bs, ts, -1)
        mu = mu.reshape(bs, ts, -1)
        std = std.reshape(bs, ts, -1)
        last_sample = sample[:, -1, :]
        last_mu = mu[:, -1, :]
        last_std = std[:, -1, :]

        if self._seq_model_class == "Conv1DModel":
            sample = sample.permute(0, 2, 1)
            hidden = self.seq_model(sample)
        elif self._seq_model_class == "RNNModel":
            hidden = self.seq_model(sample)
        elif self._seq_model_class == "AttentionModel":
            hidden = self.seq_model(sample, padding_mask)
        else:
            hidden = self.seq_model(sample[:, -1, :])

        masked_q_values = q_value_from_agents * alive_mask[:, -1, :]
        q_tot = self.base_model(masked_q_values, hidden)

        return {"q_tot": q_tot, "state_features": last_sample, "mu": last_mu, "std": last_std}

    def _forward_with_seq_hidden(self, q_value_from_agents, states, alive_mask, padding_mask):
        bs = q_value_from_agents.shape[0]
        ts = states.shape[1]
        state_shape = states.shape[2:]
        states_flat = states.reshape(bs * ts, *state_shape)

        if self._fe_is_masked:
            encoded = self.feature_extractor(states_flat, alive_mask.reshape(bs * ts, -1))
        else:
            encoded = self.feature_extractor(states_flat)

        encoded = encoded.reshape(bs, ts, -1)

        if self._seq_model_class == "Conv1DModel":
            encoded = encoded.permute(0, 2, 1)
            hidden = self.seq_model(encoded)
        elif self._seq_model_class == "RNNModel":
            hidden = self.seq_model(encoded)
        elif self._seq_model_class == "AttentionModel":
            hidden = self.seq_model(encoded, padding_mask)
        else:
            hidden = self.seq_model(encoded[:, -1, :])

        dim = hidden.size(-1) // 2
        mu = hidden[:, :dim]
        log_var = hidden[:, dim:]
        std = torch.exp(0.5 * log_var)

        if self.deterministic_eval and not self.training:
            sample = mu
        else:
            eps = torch.randn_like(std)
            sample = mu + eps * std

        masked_q_values = q_value_from_agents * alive_mask[:, -1, :]
        q_tot = self.base_model(masked_q_values, sample)

        return {"q_tot": q_tot, "state_features": sample, "mu": mu, "std": std}
