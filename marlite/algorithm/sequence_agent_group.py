"""Composed AgentGroup contract for the replacement sequence learner pipeline.

This module intentionally does not inherit the legacy three/five-argument
AgentGroup forward API. The legacy hierarchy is removed after migration.
"""

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from marlite.algorithm.temporal import RecurrentState, TemporalEncoder


@dataclass
class DecisionInput:
    observations: Tensor
    alive: Tensor
    reset: Tensor
    context: dict[str, Tensor] = field(default_factory=dict)


@dataclass
class PolicyOutput:
    values: Tensor
    features: Tensor
    state: RecurrentState
    output_kind: str
    diagnostics: dict[str, Tensor] = field(default_factory=dict)


class SequenceAgentGroup(nn.Module):
    """Shared local policy built from frame, temporal and action components.

    Context is reserved for explicitly composed communication components. This
    local implementation rejects it rather than silently ignoring messages.
    step() returns B,N,A; forward_sequence() returns B,T,N,A. Neither samples.
    """

    def __init__(self, frame_encoder: nn.Module, temporal: TemporalEncoder,
                 action_head: nn.Module, output_kind="logits"):
        super().__init__()
        if output_kind not in ("logits", "q_values"):
            raise ValueError("output_kind must be logits or q_values")
        self.frame_encoder = frame_encoder
        self.temporal = temporal
        self.action_head = action_head
        self.output_kind = output_kind
        # Behavior replay must not depend on batch composition or fresh noise.
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                raise ValueError("Sequence policies require batch-independent normalization")
            if isinstance(module, nn.modules.dropout._DropoutNd) and module.p:
                raise ValueError("Sequence policy dropout must be zero")
            if isinstance(module, (nn.RNNBase, nn.MultiheadAttention)) and module.dropout:
                raise ValueError("Sequence policy dropout must be zero")

    def initial_state(self, batch_size, n_agents, device, dtype):
        return self.temporal.initial_state(batch_size, n_agents, device, dtype)

    def _encode(self, inputs, sequence):
        if inputs.context:
            raise ValueError("Local AgentGroup cannot consume communication context")
        axes = 3 if sequence else 2
        shape = inputs.observations.shape[:axes]
        if inputs.alive.shape != shape or inputs.reset.shape != shape:
            raise ValueError("Policy masks must match observation batch/time/agent axes")
        if inputs.alive.dtype != torch.bool or inputs.reset.dtype != torch.bool:
            raise TypeError("Policy validity and reset masks must be bool")
        mask = inputs.alive.reshape(*shape, *([1] * (inputs.observations.ndim - axes)))
        obs = inputs.observations.masked_fill(~mask, 0)
        encoded = self.frame_encoder(obs.flatten(0, axes - 1))
        if encoded.ndim != 2:
            raise ValueError("Frame encoder must output one vector per frame")
        return encoded.reshape(*shape, -1)

    def _output(self, encoded, state, alive):
        values = self.action_head(encoded)
        if values.ndim != encoded.ndim or values.shape[:-1] != encoded.shape[:-1]:
            raise ValueError("Action head must preserve batch/time/agent axes")
        return PolicyOutput(values.masked_fill(~alive[..., None], 0), encoded,
                            state, self.output_kind)

    def step(self, inputs: DecisionInput, recurrent_state):
        frames = self._encode(inputs, sequence=False)
        result = self.temporal.step(frames, recurrent_state, inputs.alive, inputs.reset)
        return self._output(result.features, result.state, inputs.alive)

    def forward_sequence(self, inputs: DecisionInput, initial_state):
        frames = self._encode(inputs, sequence=True)
        result = self.temporal.forward_sequence(frames, initial_state, inputs.alive, inputs.reset)
        return self._output(result.features, result.state, inputs.alive)

    def forward(self, inputs: DecisionInput, initial_state):
        return self.forward_sequence(inputs, initial_state)
