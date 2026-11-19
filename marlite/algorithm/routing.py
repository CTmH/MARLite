"""Per-model components composed into one functional sequence AgentGroup.

Roles follow stable agent-slot order. Their weights/state remain independent;
only feature/action outputs are padded to shared widths for tensorized learners.
No Trainer or optimizer special case is required for private models.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .temporal import TemporalEncoder, TemporalOutput


def _roles(roles, models):
    if not roles or any(
        not isinstance(role, str) or not role or "/" in role or "." in role for role in roles
    ):
        raise ValueError("Roles must be nonempty names without dots/slashes")
    if set(roles) != set(models):
        raise ValueError("Each used role must have exactly one model")
    return tuple(roles)


def _pad_width(values, width):
    if values.shape[-1] > width:
        raise ValueError("A routed output exceeds its common width")
    return F.pad(values, (0, width - values.shape[-1]))


class RoutedFrames(nn.Module):
    """Route flattened B*T*N frames to role encoders; preserve frame slot order."""

    def __init__(self, roles, models, output_dim, input_dims=None):
        super().__init__()
        self.roles, self.models = _roles(roles, models), nn.ModuleDict(models)
        if type(output_dim) is not int or output_dim < 1:
            raise ValueError("Routed output width must be positive")
        self.output_dim, self.input_dims = output_dim, input_dims or {}
        if self.input_dims.keys() - models.keys() or any(
            type(n) is not int or n < 1 for n in self.input_dims.values()
        ):
            raise ValueError("Input widths must name existing roles and be positive")

    def forward(self, inputs):
        return self.encode_frames(
            inputs, torch.ones(inputs.shape[0], dtype=torch.bool, device=inputs.device)
        )

    def encode_frames(self, inputs, alive):
        n = len(self.roles)
        if inputs.shape[0] % n or alive.shape != inputs.shape[:1] or alive.dtype != torch.bool:
            raise ValueError("Routed frames/masks must contain complete ordered agent sets")
        view = inputs.reshape(-1, n, *inputs.shape[1:])
        valid = alive.reshape(-1, n)
        output = inputs.new_zeros(view.shape[0], n, self.output_dim)
        for role, model in self.models.items():
            indices = [index for index, name in enumerate(self.roles) if name == role]
            active = valid[:, indices]
            if not active.any():
                continue  # Keep globally inactive private parameters at grad=None.
            selected = view[:, indices][active]
            if role in self.input_dims:
                if selected.ndim != 2 or not 0 < self.input_dims[role] <= selected.shape[-1]:
                    raise ValueError("Role input_dims select a positive flat feature prefix")
                selected = selected[:, : self.input_dims[role]]
            encoded = model(selected)
            if encoded.ndim != 2 or encoded.shape[0] != selected.shape[0]:
                raise ValueError("A role frame encoder must output one vector per frame")
            values = output.new_zeros(*active.shape, self.output_dim)
            values[active] = _pad_width(encoded, self.output_dim)
            output[:, indices] = values
        return output.flatten(0, 1)


class RoutedTemporal(TemporalEncoder):
    """Each role owns a separate temporal model and batch-first state subtree.

    State leaves use that role's subset of agent slots, not the full population.
    step/forward_sequence route global validity/reset masks into those subtrees.
    """

    def __init__(self, roles, models, hidden_dim):
        super().__init__()
        self.roles, self.models = _roles(roles, models), nn.ModuleDict(models)
        if type(hidden_dim) is not int or hidden_dim < 1:
            raise ValueError("Routed hidden width must be positive")
        self.hidden_dim = hidden_dim

    def initial_state(self, batch_size, n_agents, device, dtype):
        if n_agents != len(self.roles):
            raise ValueError("Routed role count differs from environment slots")
        return {
            role + "/" + key: value
            for role, model in self.models.items()
            for key, value in model.initial_state(
                batch_size, self.roles.count(role), device, dtype
            ).items()
        }

    def _forward(self, inputs, state, valid, reset, sequence):
        n_axis = 2 if sequence else 1
        if inputs.shape[n_axis] != len(self.roles):
            raise ValueError("Routed input agent count differs from configured roles")
        output = inputs.new_zeros(*inputs.shape[:-1], self.hidden_dim)
        next_state, consumed = {}, set()
        for role, model in self.models.items():
            indices = [index for index, name in enumerate(self.roles) if name == role]
            role_state = {
                key.removeprefix(role + "/"): value
                for key, value in state.items()
                if key.startswith(role + "/")
            }
            consumed.update(role + "/" + key for key in role_state)
            selected = inputs[:, :, indices] if sequence else inputs[:, indices]
            active = valid[:, :, indices] if sequence else valid[:, indices]
            starts = reset[:, :, indices] if sequence else reset[:, indices]
            if not active.any():
                new = model.initial_state(
                    inputs.shape[0], len(indices), inputs.device, inputs.dtype
                )
                if role_state.keys() != new.keys() or any(
                    role_state[key].shape != value.shape for key, value in new.items()
                ):
                    raise ValueError("Inactive routed state does not match its model contract")
                features = output[:, :, indices] if sequence else output[:, indices]
            else:
                result = (
                    model.forward_sequence(selected, role_state, active, starts)
                    if sequence
                    else model.step(selected, role_state, active, starts)
                )
                features, new = _pad_width(result.features, self.hidden_dim), result.state
            if sequence:
                output[:, :, indices] = features
            else:
                output[:, indices] = features
            next_state.update({role + "/" + key: value for key, value in new.items()})
        if consumed != set(state):
            raise ValueError("Unexpected routed recurrent-state keys")
        return TemporalOutput(output, next_state)

    def step(self, inputs, state, valid=None, reset=None):
        valid, reset = self._step_masks(inputs, valid, reset)
        return self._forward(inputs, state, valid, reset, False)

    def forward_sequence(self, inputs, initial_state, valid=None, reset=None):
        valid, reset = self._masks(inputs, valid, reset)
        return self._forward(inputs, initial_state, valid, reset, True)


class RoutedHead(nn.Module):
    """Private role readouts/heads; padded actions must be excluded by masks."""

    def __init__(self, roles, models, input_dim, output_dim, input_dims=None):
        super().__init__()
        self.roles, self.models = _roles(roles, models), nn.ModuleDict(models)
        self.in_features, self.out_features = input_dim, output_dim
        self.input_dims = input_dims or {}
        if any(type(n) is not int or n < 1 for n in (input_dim, output_dim)):
            raise ValueError("Routed head dimensions must be positive")
        if self.input_dims.keys() - models.keys() or any(
            type(n) is not int or not 0 < n <= input_dim for n in self.input_dims.values()
        ):
            raise ValueError("Head input widths must name existing roles and fit the common width")

    def forward(self, inputs):
        return self.forward_masked(
            inputs, torch.ones(inputs.shape[:-1], dtype=torch.bool, device=inputs.device)
        )

    def forward_masked(self, inputs, alive):
        if (
            inputs.shape[-2:] != (len(self.roles), self.in_features)
            or alive.shape != inputs.shape[:-1]
        ):
            raise ValueError("Routed head must receive its configured agent/feature axes")
        output = inputs.new_zeros(*inputs.shape[:-1], self.out_features)
        for role, model in self.models.items():
            indices = [index for index, name in enumerate(self.roles) if name == role]
            active = alive[..., indices]
            if not active.any():
                continue
            features = inputs[..., indices, :][active]
            if role in self.input_dims:
                features = features[..., : self.input_dims[role]]
            prediction = model(features)
            if prediction.ndim != 2 or prediction.shape[0] != features.shape[0]:
                raise ValueError("Role heads must preserve selected feature rows")
            values = output.new_zeros(*active.shape, self.out_features)
            values[active] = _pad_width(prediction, self.out_features)
            output[..., indices, :] = values
        return output
