"""Centralized team-value model with its own recurrent state."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from marlite.algorithm.temporal import RecurrentState, TemporalEncoder


@dataclass
class ValueOutput:
    values: Tensor
    state: RecurrentState


class SequenceValue(nn.Module):
    """Frame extractor, temporal encoder and scalar head composed for V(history).

    State tensors use a singleton agent axis; actor state is never reused here.
    Inputs are B,T,..., with bool B,T validity/reset masks.
    """

    def __init__(self, frame_encoder: nn.Module, temporal: TemporalEncoder, head: nn.Module):
        super().__init__()
        self.frame_encoder, self.temporal, self.head = frame_encoder, temporal, head
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                raise ValueError("Sequence value models require batch-independent normalization")
            if isinstance(module, nn.modules.dropout._DropoutNd) and module.p:
                raise ValueError("Sequence value dropout must be zero")
            if isinstance(module, (nn.RNNBase, nn.MultiheadAttention)) and module.dropout:
                raise ValueError("Sequence value dropout must be zero")

    def initial_state(self, batch_size, device, dtype):
        return self.temporal.initial_state(batch_size, 1, device, dtype)

    def forward_sequence(self, states, initial_state, valid, reset):
        if valid.shape != states.shape[:2] or reset.shape != valid.shape:
            raise ValueError("Team-value masks must be B,T")
        if valid.dtype != torch.bool or reset.dtype != torch.bool:
            raise TypeError("Team-value masks must be bool")
        b, t = states.shape[:2]
        masked = states.masked_fill(~valid.reshape(b, t, *([1] * (states.ndim - 2))), 0)
        features = self.frame_encoder(masked.flatten(0, 1))
        if features.ndim != 2:
            raise ValueError("Value frame encoder must produce one vector per frame")
        result = self.temporal.forward_sequence(features.reshape(b, t, 1, -1), initial_state,
                                                valid[..., None], reset[..., None])
        values = self.head(result.features)
        if values.shape != (b, t, 1, 1):
            raise ValueError("Team-value head must produce one scalar per frame")
        return ValueOutput(values[..., 0, 0].masked_fill(~valid, 0), result.state)

    def step(self, states, recurrent_state, valid, reset):
        result = self.forward_sequence(states[:, None], recurrent_state, valid[:, None], reset[:, None])
        return ValueOutput(result.values[:, 0], result.state)

    def forward(self, states, initial_state, valid, reset):
        return self.forward_sequence(states, initial_state, valid, reset)
