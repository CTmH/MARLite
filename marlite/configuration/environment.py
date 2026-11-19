"""Fresh ParallelEnv factories and homogeneous controlled-team specifications."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, asdict
import importlib
import math

import numpy as np

from .registry import ConfigurationError


@dataclass(frozen=True)
class EnvironmentSpec:
    agent_ids: tuple
    observation_shape: tuple
    state_shape: tuple
    action_dim: int

    @property
    def dimensions(self):
        return {
            "n_agents": len(self.agent_ids),
            "observation_shape": list(self.observation_shape),
            "state_shape": list(self.state_shape),
            "action_dim": self.action_dim,
            "observation_size": math.prod(self.observation_shape),
            "state_size": math.prod(self.state_shape),
            "joint_action_dim": len(self.agent_ids) * self.action_dim,
        }

    def state_dict(self):
        return asdict(self)


class EnvironmentFactory:
    """YAML chooses a module's parallel_env parameters and explicit wrappers.

    Environment modules remain normal Python plugins, not network constructors.
    Wrapper constructors come from the same component registry as other parts.
    Only wrappers requesting env_factory explicitly receive a fresh native factory.
    """

    def __init__(self, spec, registry):
        if not isinstance(spec, Mapping) or spec.keys() - {"module", "params", "wrappers"}:
            raise ConfigurationError("environment: expected module, params and optional wrappers")
        if not isinstance(spec.get("module"), str) or not isinstance(
            spec.get("params", {}), Mapping
        ):
            raise ConfigurationError("environment: module must be a string and params a mapping")
        if not isinstance(spec.get("wrappers", []), list):
            raise ConfigurationError("environment.wrappers must be a list")
        self.spec, self.registry = deepcopy(spec), registry

    def native(self):
        module = importlib.import_module(self.spec["module"])
        factory = getattr(module, "parallel_env", None)
        if not callable(factory):
            raise ConfigurationError("environment.module must expose parallel_env")
        return factory(**deepcopy(self.spec.get("params", {})))

    def __call__(self):
        env = self.native()
        try:
            for index, wrapper in enumerate(self.spec.get("wrappers", [])):
                # Wrapper ownership, not YAML, supplies simulator references.
                injected = {"env": env}
                if wrapper.get("type") == "SMACWrapper":
                    injected["env_factory"] = self.native
                env = self.registry.build(
                    wrapper, injected=injected, path=f"environment.wrappers[{index}]"
                )
            return env
        except Exception:
            env.close()
            raise

    def probe(self, seed=0):
        env = self()
        try:
            observations, _ = env.reset(seed=seed)
            agents = tuple(env.possible_agents)
            if not agents or len(set(agents)) != len(agents):
                raise ConfigurationError("Environment must have unique controlled-agent slots")
            shapes, actions = set(), set()
            for agent in agents:
                if agent not in observations:
                    raise ConfigurationError(
                        "Every possible slot must expose a real initial observation"
                    )
                observation = observations[agent]
                if isinstance(observation, Mapping):
                    observation = observation["observation"]
                array = np.asarray(observation)
                if not np.isfinite(array).all():
                    raise ConfigurationError("Initial observations must be finite")
                shapes.add(array.shape)
                space = env.action_space(agent)
                if not hasattr(space, "n") or getattr(space, "start", 0) != 0:
                    raise ConfigurationError("Current recipes require zero-based discrete actions")
                actions.add(int(space.n))
            if len(shapes) != 1 or len(actions) != 1:
                raise ConfigurationError(
                    "Use a controlled-team adapter for heterogeneous observations/actions"
                )
            state = np.asarray(env.state())
            if not state.size or not np.isfinite(state).all():
                raise ConfigurationError("Training state must be a nonempty finite tensor")
            return EnvironmentSpec(agents, shapes.pop(), state.shape, actions.pop())
        finally:
            env.close()
