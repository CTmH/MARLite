"""Composed AgentGroup contract for the replacement sequence learner pipeline.

This module intentionally does not inherit the legacy three/five-argument
AgentGroup forward API. The legacy hierarchy is removed after migration.
"""

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from marlite.algorithm.temporal import RecurrentState, TemporalEncoder, TemporalOutput


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

    Communication is a component, not an algorithm-specific subclass. Local
    policies reject communication context instead of silently ignoring it.
    step() returns B,N,A; forward_sequence() returns B,T,N,A. Neither samples.
    """

    def __init__(self, frame_encoder: nn.Module, temporal: TemporalEncoder,
                 action_head: nn.Module, output_kind="logits", *, communication=None,
                 message_encoder=None, message_source="history", join="concat",
                 readout=None, detach_messages=False, detach_fusion=False,
                 message_temporal=None, message_projection=None):
        super().__init__()
        if output_kind not in ("logits", "q_values"):
            raise ValueError("output_kind must be logits or q_values")
        self.frame_encoder = frame_encoder
        self.temporal = temporal
        self.action_head = action_head
        self.output_kind = output_kind
        if message_source not in ("history", "frame", "observation"):
            raise ValueError("Message source must be history, frame or observation")
        if join not in ("concat", "replace", "add"):
            raise ValueError("Unknown local/communication join")
        if communication is None and (message_temporal is not None or message_projection is not None):
            raise ValueError("A message temporal branch requires communication")
        self.communication, self.message_encoder = communication, message_encoder
        self.message_temporal, self.message_projection = message_temporal, message_projection
        self.message_source, self.join = message_source, join
        self.readout = nn.Identity() if readout is None else readout
        self.detach_messages, self.detach_fusion = detach_messages, detach_fusion
        self.register_buffer("rl_gate", torch.tensor(1.))
        # Behavior replay must not depend on batch composition or fresh noise.
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                raise ValueError("Sequence policies require batch-independent normalization")
            if isinstance(module, nn.modules.dropout._DropoutNd) and module.p:
                raise ValueError("Sequence policy dropout must be zero")
            if isinstance(getattr(module, "dropout", None), (int, float)) and module.dropout:
                raise ValueError("Sequence policy dropout must be zero")

    def initial_state(self, batch_size, n_agents, device, dtype):
        state = self.temporal.initial_state(batch_size, n_agents, device, dtype)
        if any(key.startswith("message.") for key in state):
            raise ValueError("Local temporal state uses the reserved message prefix")
        if self.message_temporal is not None:
            state.update({"message." + key: value for key, value in
                          self.message_temporal.initial_state(batch_size, n_agents, device, dtype).items()})
        return state

    def _encode(self, inputs, sequence):
        if inputs.context and self.communication is None:
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

    def set_rl_gate(self, value):
        if not 0 <= value <= 1:
            raise ValueError("RL communication gate must be in [0,1]")
        self.rl_gate.fill_(value)

    def sample_context(self, alive, generator):
        if self.communication is not None and hasattr(self.communication, "sample_context"):
            return self.communication.sample_context(alive, generator)
        return {}

    def _features(self, inputs, state, *, sequence, for_action):
        frames = self._encode(inputs, sequence=sequence)
        local_state = {key: value for key, value in state.items() if not key.startswith("message.")}
        method = "forward_sequence" if sequence else "step"
        local = getattr(self.temporal, method)(frames, local_state, inputs.alive, inputs.reset)
        next_state = dict(local.state)
        messages = None
        # Warmup may skip the entire message/action path for a single temporal
        # branch. A dual-path policy must also reconstruct its own message state.
        if self.communication is not None and (for_action or self.message_temporal is not None):
            if self.message_source == "history":
                source = local.features
            elif self.message_source == "frame":
                source = frames
            else:
                axes = inputs.alive.ndim
                mask = inputs.alive.reshape(*inputs.alive.shape,
                                            *([1] * (inputs.observations.ndim - axes)))
                source = inputs.observations.masked_fill(~mask, 0)
            leading = inputs.alive.shape
            messages = (source if self.message_encoder is None else
                        self.message_encoder(source.reshape(-1, *source.shape[len(leading):])))
            messages = messages.reshape(*leading, -1)
            if self.message_temporal is not None:
                message_state = {key.removeprefix("message."): value for key, value in state.items()
                                 if key.startswith("message.")}
                result = getattr(self.message_temporal, method)(messages, message_state,
                                                                inputs.alive, inputs.reset)
                messages = result.features
                next_state.update({"message." + key: value for key, value in result.state.items()})
            if for_action and self.message_projection is not None:
                messages = self.message_projection(messages)
            messages = messages.masked_fill(~inputs.alive[..., None], 0)
        return TemporalOutput(local.features, next_state), messages

    def decode_actions(self, local, messages, alive, context=None, *, state=None):
        """Read actions from already encoded local features and sender messages.

        Offline codec training reuses frozen local features but reruns the frozen
        teacher's aggregation/readout WITH autograd. This explicit boundary lets
        message gradients flow without rebuilding or training the local encoder.
        """
        context = {} if context is None else context
        state = {} if state is None else state
        diagnostics = {"local_features": local}
        encoded = local
        if self.communication is not None:
            diagnostics["message_features"] = messages
            sent = messages.detach() if self.detach_messages else messages
            received = self.communication(sent, alive, context, queries=local)
            diagnostics.update(received.diagnostics)
            fused = received.features * self.rl_gate
            if self.detach_fusion:
                fused = fused.detach()
            if self.join == "concat":
                encoded = torch.cat((local, fused), -1)
            elif self.join == "replace":
                encoded = fused
            else:
                if local.shape != fused.shape:
                    raise ValueError("Additive communication must match local features")
                encoded = local + fused
        encoded = self.readout(encoded).masked_fill(~alive[..., None], 0)
        values = self.action_head(encoded)
        if values.ndim != encoded.ndim or values.shape[:-1] != encoded.shape[:-1]:
            raise ValueError("Action head must preserve batch/time/agent axes")
        return PolicyOutput(values.masked_fill(~alive[..., None], 0), encoded,
                            state, self.output_kind, diagnostics)

    def step(self, inputs: DecisionInput, recurrent_state):
        result, messages = self._features(inputs, recurrent_state, sequence=False, for_action=True)
        return self.decode_actions(result.features, messages, inputs.alive, inputs.context, state=result.state)

    def encode_sequence(self, inputs: DecisionInput, initial_state):
        """Reconstruct both temporal branches without communication/actions."""
        result, _ = self._features(inputs, initial_state, sequence=True, for_action=False)
        return result

    def forward_sequence(self, inputs: DecisionInput, initial_state):
        result, messages = self._features(inputs, initial_state, sequence=True, for_action=True)
        return self.decode_actions(result.features, messages, inputs.alive, inputs.context, state=result.state)

    def forward(self, inputs: DecisionInput, initial_state):
        return self.forward_sequence(inputs, initial_state)
