"""Tensor-only episode contract for continuous-sequence training.

Frames include the real final observation. Transitions never contain a reset
observation from the following episode. All boolean masks use True for valid.
"""

from dataclasses import dataclass, field

import torch
from torch import Tensor


FRAME_FIELDS = {
    "observations": None,
    "alive": torch.bool,
    "available_actions": torch.bool,
    "agent_reset": torch.bool,
}
TRANSITION_FIELDS = {
    "actions": torch.int64,
    "rewards": None,
    "agent_terminated": torch.bool,
    "agent_truncated": torch.bool,
    "terminated": torch.bool,
    "truncated": torch.bool,
}


@dataclass
class Episode:
    """One environment episode with stable slots and T+1 decision frames.

    Additional tensor fields (states, old_value, old_log_prob, comm.*) use the
    same frame/transition time axes. Action-space objects belong in metadata.
    """

    agent_ids: tuple[str, ...]
    frames: dict[str, Tensor]
    transitions: dict[str, Tensor]
    policy_version: int = 0
    seed: int | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def length(self):
        return self.transitions["actions"].shape[0]

    def validate(self):
        if not self.agent_ids or len(set(self.agent_ids)) != len(self.agent_ids):
            raise ValueError("agent_ids must be nonempty and unique")
        for fields, required in ((self.frames, FRAME_FIELDS), (self.transitions, TRANSITION_FIELDS)):
            if missing := required.keys() - fields.keys():
                raise ValueError(f"Missing episode fields: {sorted(missing)}")
            for name, value in fields.items():
                if not isinstance(value, Tensor) or value.ndim == 0:
                    raise TypeError(f"{name} must be a time-indexed tensor")
                if value.device.type != "cpu" or value.requires_grad:
                    raise ValueError("Stored episodes must contain detached CPU tensors")
                if name in required and required[name] is not None and value.dtype != required[name]:
                    raise TypeError(f"Incorrect dtype for {name}: {value.dtype}")
                if value.is_floating_point() and not torch.isfinite(value).all():
                    raise ValueError(f"Nonfinite episode field: {name}")
        t, n = self.length, len(self.agent_ids)
        if t < 1 or type(self.policy_version) is not int or self.policy_version < 0:
            raise ValueError("Episodes need transitions and a nonnegative policy version")
        if any(v.shape[0] != t + 1 for v in self.frames.values()):
            raise ValueError("Frame fields must have T+1 entries")
        if any(v.shape[0] != t for v in self.transitions.values()):
            raise ValueError("Transition fields must have T entries")
        for name in FRAME_FIELDS:
            if self.frames[name].shape[:2] != (t + 1, n):
                raise ValueError(f"{name} must start with T+1,N")
        for name in ("alive", "agent_reset"):
            if self.frames[name].ndim != 2:
                raise ValueError(f"{name} must be T+1,N")
        available = self.frames["available_actions"]
        if available.ndim != 3 or available.shape[-1] < 1:
            raise ValueError("available_actions must be T+1,N,A")
        for name in ("actions", "rewards", "agent_terminated", "agent_truncated"):
            if self.transitions[name].shape != (t, n):
                raise ValueError(f"{name} must be T,N")
        if not self.transitions["rewards"].is_floating_point():
            raise TypeError("Rewards must be floating point")
        for name in ("terminated", "truncated"):
            if self.transitions[name].shape != (t,):
                raise ValueError(f"{name} must be T")
            if self.transitions[name][:-1].any():
                raise ValueError("An episode cannot contain transitions after a boundary")
        if not (self.transitions["terminated"][-1] | self.transitions["truncated"][-1]):
            raise ValueError("A complete episode must end at a boundary")
        alive = self.frames["alive"]
        if not self.frames["agent_reset"][0, alive[0]].all():
            raise ValueError("Initially alive agents must reset their recurrent state")
        born = alive[1:] & ~alive[:-1]
        if (born & ~self.frames["agent_reset"][1:]).any():
            raise ValueError("Newly alive agents must reset their recurrent state")
        if (alive & ~available.any(-1)).any():
            raise ValueError("Every live agent needs an available action")
        actions = self.transitions["actions"]
        if (actions < 0).any() or (actions >= available.shape[-1]).any():
            raise ValueError("Action index is outside the discrete action space")
        legal = available[:-1].gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        if (alive[:-1] & ~legal).any():
            raise ValueError("A live agent selected an unavailable action")
        for name in ("old_log_prob",):
            if name in self.transitions and self.transitions[name].shape != (t, n):
                raise ValueError(f"{name} must be T,N")
        if "old_value" in self.frames and self.frames["old_value"].shape != (t + 1,):
            raise ValueError("old_value must contain one team value per frame")
        return self

    def return_masks(self):
        """Bootstrap at a timeout, but never continue the trace across it."""
        bootstrap = ~self.transitions["terminated"] & self.frames["alive"][1:].any(-1)
        trace = ~(self.transitions["terminated"] | self.transitions["truncated"])
        trace = trace.clone()
        trace[-1] = False
        return bootstrap, trace

    def state_dict(self):
        return {
            "schema_version": 2, "agent_ids": self.agent_ids,
            "frames": self.frames, "transitions": self.transitions,
            "policy_version": self.policy_version, "seed": self.seed,
            "metadata": self.metadata,
        }

    @classmethod
    def from_state_dict(cls, state):
        state = dict(state)
        if state.pop("schema_version", None) != 2:
            raise ValueError("Unsupported episode schema")
        return cls(**state).validate()
