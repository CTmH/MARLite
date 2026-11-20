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
        if value.shape[: reset.ndim] != reset.shape or reset.dtype != torch.bool:
            raise ValueError("Reset mask must match leading state axes")
        mask = reset.reshape(*reset.shape, *([1] * (value.ndim - reset.ndim)))
        result[name] = value.masked_fill(mask, 0)
    return result


@dataclass
class TemporalOutput:
    features: Tensor
    state: RecurrentState


class TemporalEncoder(nn.Module):
    """Temporal feature encoder with caller-owned state.

    B is the batch size, T the time length, N the agent slots and D the input
    width. Step inputs have shape (B, N, D) and sequence inputs have shape
    (B, T, N, D). The boolean masks match the leading axes. True valid entries
    are live inputs and true reset entries start a new recurrent life before
    processing that frame.
    """

    def initial_state(self, batch_size, n_agents, device, dtype) -> RecurrentState:
        """Return initial state for a fresh batch, with a leading batch axis."""
        return {}

    def step(self, inputs, state, valid=None, reset=None) -> TemporalOutput:
        """Encode one frame and return its features and the next state."""
        raise NotImplementedError

    def forward_sequence(self, inputs, initial_state, valid=None, reset=None) -> TemporalOutput:
        """Encode every time step and return all features and the final state."""
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
        valid = (
            torch.ones(shape, dtype=torch.bool, device=inputs.device) if valid is None else valid
        )
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
            inputs[:, None],
            None if valid is None else valid[:, None],
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
        return {
            "h": torch.zeros(
                batch_size, n_agents, self.layers, self.hidden_dim, device=device, dtype=dtype
            )
        }

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
        return TemporalOutput(
            output.reshape(b, n, self.hidden_dim).masked_fill(~valid[..., None], 0), {"h": next_h}
        )

    def forward_sequence(self, inputs, initial_state, valid=None, reset=None):
        valid, reset = self._masks(inputs, valid, reset)
        b, t, n, _ = inputs.shape
        expected = (b, n, self.layers, self.hidden_dim)
        if set(initial_state) != {"h"} or initial_state["h"].shape != expected:
            raise ValueError(f"GRU state h must have shape {expected}")
        # Fuse intervals with unchanged live slots and no new lives. A dead slot
        # must not force every other agent through a Python loop for every frame.
        boundaries = (
            (reset[:, 1:].any(dim=(0, 2)) | (valid[:, 1:] != valid[:, :-1]).any(dim=(0, 2)))
            .nonzero()
            .flatten()
        )
        points = [0, *(boundaries + 1).tolist(), t]
        hidden, outputs = initial_state["h"], []
        for begin, end in zip(points[:-1], points[1:]):
            active = valid[:, begin]
            hidden = reset_state({"h": hidden}, reset[:, begin] | ~active)["h"]
            indices = active.flatten().nonzero().flatten()
            length = end - begin
            block = inputs.new_zeros(b * n, length, self.hidden_dim)
            next_hidden = inputs.new_zeros(b * n, self.layers, self.hidden_dim)
            if indices.numel():
                x = inputs[:, begin:end].transpose(1, 2).reshape(b * n, length, -1)[indices]
                h = hidden.reshape(b * n, self.layers, self.hidden_dim)[indices]
                y, h_out = self.gru(x, h.transpose(0, 1).contiguous())
                block[indices] = y
                next_hidden[indices] = h_out.transpose(0, 1)
            outputs.append(block.reshape(b, n, length, self.hidden_dim).transpose(1, 2))
            hidden = next_hidden.reshape(expected)
        return TemporalOutput(torch.cat(outputs, dim=1), {"h": hidden})


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
        self.attention = nn.MultiheadAttention(hidden_dim, heads, dropout=0.0, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def initial_state(self, batch_size, n_agents, device, dtype):
        return {
            "cache": torch.zeros(
                batch_size,
                n_agents,
                self.context_length,
                self.hidden_dim,
                device=device,
                dtype=dtype,
            ),
            "valid": torch.zeros(
                batch_size, n_agents, self.context_length, device=device, dtype=torch.bool
            ),
        }

    def step(self, inputs, state, valid=None, reset=None):
        valid, reset = self._step_masks(inputs, valid, reset)
        b, n, _ = inputs.shape
        expected = (b, n, self.context_length)
        if (
            set(state) != {"cache", "valid"}
            or state["valid"].shape != expected
            or state["cache"].shape != (*expected, self.hidden_dim)
        ):
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
        read, _ = self.attention(
            query,
            keys,
            keys,
            key_padding_mask=~safe.reshape(b * n, self.context_length),
            need_weights=False,
        )
        output = self.norm(current + read.reshape(b, n, self.hidden_dim))
        return TemporalOutput(
            output.masked_fill(~valid[..., None], 0),
            {"cache": cache.masked_fill(~valid[..., None, None], 0), "valid": active},
        )

    def forward_sequence(self, inputs, initial_state, valid=None, reset=None):
        """Evaluate all timesteps in one attention call with a causal band.

        Cache tokens belong to segment zero. Reset/death boundaries start new
        segments, so neither attention nor the returned cache crosses a life.
        The mask is O(B*N*heads*T*(context+T)); sequence length remains a user
        memory control, independently of the streaming cache length.
        """
        valid, reset = self._masks(inputs, valid, reset)
        b, t, n, _ = inputs.shape
        c, h = self.context_length, self.hidden_dim
        expected = (b, n, c)
        if (
            set(initial_state) != {"cache", "valid"}
            or initial_state["valid"].shape != expected
            or initial_state["valid"].dtype != torch.bool
            or initial_state["cache"].shape != (*expected, h)
        ):
            raise ValueError("Attention cache shape does not match inputs")
        active = valid.transpose(1, 2).reshape(b * n, t)
        boundaries = (reset | ~valid).transpose(1, 2).reshape(b * n, t)
        segments = boundaries.long().cumsum(-1)
        current = self.projection(inputs.masked_fill(~valid[..., None], 0))
        query = current.transpose(1, 2).reshape(b * n, t, h)
        keys = torch.cat((initial_state["cache"].reshape(b * n, c, h), query), dim=1)
        key_valid = torch.cat((initial_state["valid"].reshape(b * n, c), active), dim=1)
        key_segments = torch.cat((segments.new_zeros(b * n, c), segments), dim=1)
        query_index = torch.arange(c, c + t, device=inputs.device)
        key_index = torch.arange(c + t, device=inputs.device)
        band = (key_index[None] <= query_index[:, None]) & (
            key_index[None] > query_index[:, None] - c
        )
        allowed = band[None] & key_valid[:, None] & (segments[:, :, None] == key_segments[:, None])
        # Inactive queries still need one finite softmax entry. Their output and
        # state are masked away, matching step()'s harmless dummy key.
        dummy = (~active).nonzero()
        allowed[dummy[:, 0], dummy[:, 1], c + dummy[:, 1]] = True
        mask = (~allowed).repeat_interleave(self.attention.num_heads, dim=0)
        read, _ = self.attention(query, keys, keys, attn_mask=mask, need_weights=False)
        output = self.norm(query + read).reshape(b, n, t, h).transpose(1, 2)
        final_valid = key_valid[:, -c:] & (key_segments[:, -c:] == segments[:, -1:])
        final_valid &= active[:, -1:]
        cache = keys[:, -c:].masked_fill(~final_valid[..., None], 0)
        return TemporalOutput(
            output.masked_fill(~valid[..., None], 0),
            {"cache": cache.reshape(b, n, c, h), "valid": final_valid.reshape(b, n, c)},
        )


class WindowEncoder(TemporalEncoder):
    """Adapt a pooled history model without pretending it is a streaming RNN.

    The wrapped model accepts (B*N,W,D), padding_mask=True for padding, and
    returns (B*N,H). The explicit input cache preserves finite-window semantics.
    """

    def __init__(self, model, input_dim, hidden_dim, window_length, use_padding_mask=True):
        super().__init__()
        if type(window_length) is not int or window_length < 1:
            raise ValueError("window_length must be positive")
        self.model = model
        self.input_dim, self.hidden_dim = input_dim, hidden_dim
        self.window_length = window_length
        self.use_padding_mask = use_padding_mask

    def initial_state(self, batch_size, n_agents, device, dtype):
        return {
            "window": torch.zeros(
                batch_size, n_agents, self.window_length, self.input_dim, device=device, dtype=dtype
            ),
            "valid": torch.zeros(
                batch_size, n_agents, self.window_length, device=device, dtype=torch.bool
            ),
        }

    def step(self, inputs, state, valid=None, reset=None):
        valid, reset = self._step_masks(inputs, valid, reset)
        b, n, _ = inputs.shape
        expected = (b, n, self.window_length)
        if (
            set(state) != {"window", "valid"}
            or state["valid"].shape != expected
            or state["window"].shape != (*expected, self.input_dim)
        ):
            raise ValueError("Window state does not match input specification")
        state = reset_state(state, reset | ~valid)
        window = torch.cat((state["window"][:, :, 1:], inputs[:, :, None]), 2)
        active = torch.cat((state["valid"][:, :, 1:], valid[..., None]), 2)
        window = window.masked_fill(~active[..., None], 0)
        safe = active.clone()
        safe[..., -1] = True
        flat = window.reshape(b * n, self.window_length, self.input_dim)
        out = (
            self.model(flat, padding_mask=~safe.reshape(b * n, self.window_length))
            if self.use_padding_mask
            else self.model(flat)
        )
        if out.shape != (b * n, self.hidden_dim):
            raise ValueError("Window model must output one hidden vector per agent")
        return TemporalOutput(
            out.reshape(b, n, self.hidden_dim).masked_fill(~valid[..., None], 0),
            {"window": window, "valid": active},
        )
