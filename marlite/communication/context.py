"""Environment-side communication conditions, explicitly separate from policy features."""

import random

import numpy as np
import torch


class AllTogether:
    uses_global_state = False

    def reset(self, seed=None):
        pass

    def __call__(self, observations, state, alive, timestep):
        return {"groups": torch.zeros_like(alive, dtype=torch.long).masked_fill(~alive, -1)}


class FixedGroups:
    uses_global_state = False

    def __init__(self, groups):
        self.groups = torch.as_tensor(groups, dtype=torch.long)
        if self.groups.ndim != 1 or (self.groups < 0).any():
            raise ValueError("Fixed groups must be a nonnegative vector")

    def reset(self, seed=None):
        pass

    def __call__(self, observations, state, alive, timestep):
        if self.groups.shape != alive.shape:
            raise ValueError("Fixed groups do not match environment slots")
        return {"groups": self.groups.clone().masked_fill(~alive, -1)}


class FixedGraph:
    """Edges use sender,receiver order; adjacency uses receiver,sender."""

    uses_global_state = False

    def __init__(self, n_agents, edges, self_loops=False):
        self.adjacency = (
            torch.eye(n_agents, dtype=torch.float32)
            if self_loops
            else torch.zeros(n_agents, n_agents)
        )
        for sender, receiver in edges:
            if not 0 <= sender < n_agents or not 0 <= receiver < n_agents:
                raise ValueError("Fixed edge references an unknown slot")
            self.adjacency[receiver, sender] = 1.0

    def reset(self, seed=None):
        pass

    def __call__(self, observations, state, alive, timestep):
        if self.adjacency.shape != (*alive.shape, alive.shape[-1]):
            raise ValueError("Fixed graph does not match environment slots")
        return {"adjacency": self.adjacency.masked_fill(~(alive[:, None] & alive[None, :]), 0)}


class StateBuilderContext:
    """Use an existing environment-specific builder, with privilege recorded.

    Builders remain primitive topology constructors; they do not select actions.
    This adapter handles single-frame batch axes, not legacy training contracts.
    """

    uses_global_state = True

    def __init__(self, builder, field="groups", transpose_adjacency=False):
        if field not in ("groups", "adjacency"):
            raise ValueError("State builder must emit groups or adjacency")
        self.builder, self.field = builder, field
        self.transpose_adjacency = transpose_adjacency
        self.reset(seed=0)

    def reset(self, seed=None):
        if isinstance(self.builder, torch.nn.Module):
            self.builder.eval()  # Execution topology must honor its cache/update interval.
        if hasattr(self.builder, "reset"):
            self.builder.reset()
        self.numpy_rng = np.random.RandomState(seed)
        self.python_rng = random.Random(seed)
        self.torch_rng = torch.Generator()
        self.torch_rng.seed() if seed is None else self.torch_rng.manual_seed(seed)

    def __call__(self, observations, state, alive, timestep):
        # Constructors may use legacy global CPU RNGs. Isolate them per episode
        # without changing any GPU random stream or the caller's training RNG.
        numpy_state, python_state, torch_state = (
            np.random.get_state(),
            random.getstate(),
            torch.get_rng_state(),
        )
        np.random.set_state(self.numpy_rng.get_state())
        random.setstate(self.python_rng.getstate())
        torch.set_rng_state(self.torch_rng.get_state())
        try:
            result = self.builder(state[None])
        finally:
            self.numpy_rng.set_state(np.random.get_state())
            self.python_rng.setstate(random.getstate())
            self.torch_rng.set_state(torch.get_rng_state())
            np.random.set_state(numpy_state)
            random.setstate(python_state)
            torch.set_rng_state(torch_state)
        if isinstance(result, tuple):
            result = result[0]
        result = torch.as_tensor(result)[0].detach().cpu()
        if self.field == "groups":
            result = result.long().masked_fill(~alive, -1)
        else:
            if self.transpose_adjacency:
                result = result.transpose(-1, -2)
            result = result.float().masked_fill(~(alive[:, None] & alive[None, :]), 0)
        return {self.field: result}


def frame_context(frames, *, start=None, stop=None, batched=True, device=None):
    """Extract time-aligned execution fields, never the privileged state tensor."""
    time_slice = slice(start, stop)
    context = {}
    for name, value in frames.items():
        if name.startswith("context."):
            selected = value[:, time_slice] if batched else value[time_slice][None]
            context[name.removeprefix("context.")] = selected.to(device=device)
    return context
