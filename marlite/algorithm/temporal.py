"""Batch-first temporal encoders with explicit, caller-owned recurrent state.

Inputs are B,T,N,D. step() uses B,N,D. State leaves are batch-first; the
collector owns their lifetime. True masks mean valid/reset, never padding.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn


RecurrentState = dict[str, Tensor]


def detach_state(state: RecurrentState) -> RecurrentState:
    return {name: value.detach() for name, value in state.items()}


def select_state(state: RecurrentState, indices) -> RecurrentState:
    return {name: value[indices] for name, value in state.items()}


def reset_state(state: RecurrentState, reset: Tensor) -> RecurrentState:
    """Reset selected environments (B) or agent slots (B,N) without mutation."""
    result = {}
    for name, value in state.items():
        if value.shape[:reset.ndim] != reset.shape or reset.dtype != torch.bool:
            raise ValueError("Reset mask must match leading state axes")
        mask = reset.reshape(*reset.shape, *([1] * (value.ndim - reset.ndim)))
        result[name] = value.masked_fill(mask, 0)
    return result


@dataclass
class TemporalOutput:
    features: Tensor
    state: RecurrentState


class TemporalEncoder(nn.Module):
    """Small sequence protocol shared by policy and value encoders."""

    def initial_state(self, batch_size, n_agents, device, dtype):
        return {}

    def step(self, inputs, state, valid=None, reset=None):
        raise NotImplementedError

    def forward_sequence(self, inputs, initial_state, valid=None, reset=None):
        valid, reset = self._masks(inputs, valid, reset)
        state, outputs = initial_state, []
        for t in range(inputs.shape[1]):
            result = self.step(inputs[:, t], state, valid[:, t], reset[:, t])
            outputs.append(result.features)
            state = result.state
        return TemporalOutput(torch.stack(outputs, 1), state)

    @staticmethod
    def _masks(inputs, valid, reset):
        if inputs.ndim != 4 or inputs.shape[1] < 1:
            raise ValueError("Sequence inputs must be nonempty B,T,N,D")
        shape = inputs.shape[:3]
        valid = torch.ones(shape, dtype=torch.bool, device=inputs.device) if valid is None else valid
        reset = torch.zeros_like(valid) if reset is None else reset
        for mask in (valid, reset):
            if mask.shape != shape or mask.dtype != torch.bool or mask.device != inputs.device:
                raise ValueError("Sequence masks must be bool B,T,N on the input device")
        return valid, reset

    @staticmethod
    def _step_masks(inputs, valid, reset):
        if inputs.ndim != 3:
            raise ValueError("Step inputs must be B,N,D")
        valid, reset = TemporalEncoder._masks(
            inputs[:, None], None if valid is None else valid[:, None],
            None if reset is None else reset[:, None],
        )
        return valid[:, 0], reset[:, 0]


class FrameEncoder(TemporalEncoder):
    """Stateless feature projection with the same step/sequence contract."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.projection = nn.Linear(input_dim, hidden_dim)

    def step(self, inputs, state, valid=None, reset=None):
        valid, _ = self._step_masks(inputs, valid, reset)
        if state:
            raise ValueError("A stateless encoder cannot consume recurrent state")
        x = inputs.masked_fill(~valid[..., None], 0)
        return TemporalOutput(self.projection(x).masked_fill(~valid[..., None], 0), {})

    def forward_sequence(self, inputs, initial_state, valid=None, reset=None):
        valid, _ = self._masks(inputs, valid, reset)
        if initial_state:
            raise ValueError("A stateless encoder cannot consume recurrent state")
        x = inputs.masked_fill(~valid[..., None], 0)
        return TemporalOutput(self.projection(x).masked_fill(~valid[..., None], 0), {})


class RecurrentGRU(TemporalEncoder):
    """Fused GRU over uninterrupted chunks; reset-aware fallback at boundaries."""

    def __init__(self, input_dim, hidden_dim, layers=1, bias=True):
        super().__init__()
        self.hidden_dim, self.layers = hidden_dim, layers
        self.gru = nn.GRU(input_dim, hidden_dim, layers, bias=bias, batch_first=True)

    def initial_state(self, batch_size, n_agents, device, dtype):
        return {"h": torch.zeros(batch_size, n_agents, self.layers, self.hidden_dim,
                                 device=device, dtype=dtype)}

    def step(self, inputs, state, valid=None, reset=None):
        valid, reset = self._step_masks(inputs, valid, reset)
        b, n, _ = inputs.shape
        expected = (b, n, self.layers, self.hidden_dim)
        if set(state) != {"h"} or state["h"].shape != expected:
            raise ValueError(f"GRU state h must have shape {expected}")
        h = reset_state(state, reset | ~valid)["h"]
        h_in = h.reshape(b * n, self.layers, self.hidden_dim).transpose(0, 1).contiguous()
        x = inputs.masked_fill(~valid[..., None], 0).reshape(b * n, 1, -1)
        output, h_out = self.gru(x, h_in)
        next_h = h_out.transpose(0, 1).reshape(expected)
        next_h = next_h.masked_fill(~valid[..., None, None], 0)
        return TemporalOutput(output.reshape(b, n, self.hidden_dim).masked_fill(~valid[..., None], 0),
                              {"h": next_h})

    def forward_sequence(self, inputs, initial_state, valid=None, reset=None):
        valid, reset = self._masks(inputs, valid, reset)
        # Ordinary chunks use one fused recurrent call, not one Python call per frame.
        if not bool(valid.all()) or bool(reset[:, 1:].any()):
            return super().forward_sequence(inputs, initial_state, valid, reset)
        b, t, n, _ = inputs.shape
        expected = (b, n, self.layers, self.hidden_dim)
        if set(initial_state) != {"h"} or initial_state["h"].shape != expected:
            raise ValueError(f"GRU state h must have shape {expected}")
        h = reset_state(initial_state, reset[:, 0])["h"]
        h = h.reshape(b * n, self.layers, self.hidden_dim).transpose(0, 1).contiguous()
        x = inputs.transpose(1, 2).reshape(b * n, t, -1)
        output, h_out = self.gru(x, h)
        output = output.reshape(b, n, t, self.hidden_dim).transpose(1, 2)
        return TemporalOutput(output, {"h": h_out.transpose(0, 1).reshape(expected)})


class CausalAttention(TemporalEncoder):
    """Bounded causal attention with an explicit cache, no future-token pooling.

    Cached tensors are pre-attention embeddings. Each step reads at most
    context_length frames of the same agent; deaths and resets clear the cache.
    """

    def __init__(self, input_dim, hidden_dim, heads=4, context_length=128):
        super().__init__()
        if type(context_length) is not int or context_length < 1:
            raise ValueError("context_length must be a positive integer")
        self.hidden_dim, self.context_length = hidden_dim, context_length
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, heads, dropout=0., batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def initial_state(self, batch_size, n_agents, device, dtype):
        return {
            "cache": torch.zeros(batch_size, n_agents, self.context_length, self.hidden_dim,
                                 device=device, dtype=dtype),
            "valid": torch.zeros(batch_size, n_agents, self.context_length,
                                 device=device, dtype=torch.bool),
        }

    def step(self, inputs, state, valid=None, reset=None):
        valid, reset = self._step_masks(inputs, valid, reset)
        b, n, _ = inputs.shape
        expected = (b, n, self.context_length)
        if (set(state) != {"cache", "valid"} or state["valid"].shape != expected
                or state["cache"].shape != (*expected, self.hidden_dim)):
            raise ValueError("Attention cache shape does not match inputs")
        state = reset_state(state, reset | ~valid)
        current = self.projection(inputs.masked_fill(~valid[..., None], 0))
        cache = torch.cat((state["cache"][:, :, 1:], current[:, :, None]), 2)
        active = torch.cat((state["valid"][:, :, 1:], valid[..., None]), 2)
        # Give inactive rows one harmless key; mask their final output and state.
        safe = active.clone()
        safe[..., -1] = True
        query = current.reshape(b * n, 1, self.hidden_dim)
        keys = cache.reshape(b * n, self.context_length, self.hidden_dim)
        read, _ = self.attention(query, keys, keys,
                                 key_padding_mask=~safe.reshape(b * n, self.context_length),
                                 need_weights=False)
        output = self.norm(current + read.reshape(b, n, self.hidden_dim))
        return TemporalOutput(
            output.masked_fill(~valid[..., None], 0),
            {"cache": cache.masked_fill(~valid[..., None, None], 0), "valid": active},
        )


class WindowEncoder(TemporalEncoder):
    """Adapt a pooled history model without pretending it is a streaming RNN.

    The wrapped model accepts (B*N,W,D), padding_mask=True for padding, and
    returns (B*N,H). The explicit input cache preserves finite-window semantics.
    """

    def __init__(self, model, input_dim, hidden_dim, window_length):
        super().__init__()
        if type(window_length) is not int or window_length < 1:
            raise ValueError("window_length must be positive")
        self.model = model
        self.input_dim, self.hidden_dim = input_dim, hidden_dim
        self.window_length = window_length

    def initial_state(self, batch_size, n_agents, device, dtype):
        return {
            "window": torch.zeros(batch_size, n_agents, self.window_length, self.input_dim,
                                  device=device, dtype=dtype),
            "valid": torch.zeros(batch_size, n_agents, self.window_length,
                                 device=device, dtype=torch.bool),
        }

    def step(self, inputs, state, valid=None, reset=None):
        valid, reset = self._step_masks(inputs, valid, reset)
        b, n, _ = inputs.shape
        expected = (b, n, self.window_length)
        if (set(state) != {"window", "valid"} or state["valid"].shape != expected
                or state["window"].shape != (*expected, self.input_dim)):
            raise ValueError("Window state does not match input specification")
        state = reset_state(state, reset | ~valid)
        window = torch.cat((state["window"][:, :, 1:], inputs[:, :, None]), 2)
        active = torch.cat((state["valid"][:, :, 1:], valid[..., None]), 2)
        window = window.masked_fill(~active[..., None], 0)
        safe = active.clone()
        safe[..., -1] = True
        out = self.model(window.reshape(b * n, self.window_length, self.input_dim),
                         padding_mask=~safe.reshape(b * n, self.window_length))
        if out.shape != (b * n, self.hidden_dim):
            raise ValueError("Window model must output one hidden vector per agent")
        return TemporalOutput(out.reshape(b, n, self.hidden_dim).masked_fill(~valid[..., None], 0),
                              {"window": window, "valid": active})
