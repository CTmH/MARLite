"""QPLEX with per-timestep state features and a temporal context encoder."""

from typing import Dict

from marlite.algorithm.model import ModelConfig, RNNModel, Conv1DModel, AttentionModel
from marlite.algorithm.critic.qplex_mixer import QPLEXMixer


class SeqQPLEXMixer(QPLEXMixer):
    """Use state history without changing QPLEX's value/advantage factorization.

    ``seq_model`` maps the encoded history to the state_dim expected by all
    QPLEX heads. Inputs and forward outputs match QPLEXMixer. As in SeqQMixer,
    attention receives the temporal padding mask, RNN/Conv1D process the window,
    and non-sequence models consume only the latest encoded state.
    """

    def __init__(self, seq_model: Dict, **kwargs):
        super().__init__(**kwargs)
        self.seq_model = ModelConfig(**seq_model).get_model()

    def _encode_states(self, states, alive_mask, padding_mask):
        batch_size, timesteps = states.shape[:2]
        states_flat = states.reshape(batch_size * timesteps, *states.shape[2:])
        if self._fe_is_masked:
            # Each historical state uses its OWN alive mask, not the last one.
            encoded = self.feature_extractor(
                states_flat, alive_mask.reshape(batch_size * timesteps, -1).bool()
            )
        else:
            encoded = self.feature_extractor(states_flat)
        encoded = encoded.reshape(batch_size, timesteps, -1)
        # Padding carries no state information, including for RNN/Conv1D models.
        encoded = encoded.masked_fill(padding_mask.unsqueeze(-1), 0.)

        if isinstance(self.seq_model, Conv1DModel):
            return self.seq_model(encoded.permute(0, 2, 1))
        if isinstance(self.seq_model, RNNModel):
            return self.seq_model(encoded)
        if isinstance(self.seq_model, AttentionModel):
            # An entirely padded history has zero context. Unmask one zero token
            # for that row so attention never evaluates an all-masked softmax.
            empty = padding_mask.all(dim=1)
            safe_mask = padding_mask.clone()
            safe_mask[:, -1] &= ~empty
            context = self.seq_model(encoded, safe_mask)
            return context.masked_fill(empty.unsqueeze(-1), 0.)
        return self.seq_model(encoded[:, -1])
