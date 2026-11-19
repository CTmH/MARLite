"""One explicit component registry; configuration never specifies training code."""

from collections.abc import Mapping
from copy import deepcopy
import importlib
import inspect
import math
from pathlib import Path

import torch
from torch import nn


class ConfigurationError(ValueError):
    """A configuration error with a path to the offending component."""


class Registry:
    """Build nested constructor specs; extensions register ordinary Python objects.

    Specs contain type/params plus optional initialization/checkpoint boundaries.
    Plain mappings/lists remain data. Dimension references support only a named
    dimension, optional index and integer scale/offset; there is no Python eval,
    dotted import in YAML, graph-control DSL or implicit component sharing.
    """

    def __init__(self):
        self._factories = {}

    def register(self, name, factory, *, replace=False):
        if not isinstance(name, str) or not name:
            raise ValueError("A component name must be a nonempty string")
        if name in self._factories and not replace:
            raise ValueError(f"Component {name!r} is already registered")
        if not callable(factory) and not (isinstance(factory, tuple) and len(factory) == 2):
            raise TypeError("Factories must be callable or a lazy (module, attribute) pair")
        self._factories[name] = factory
        return factory

    @property
    def names(self):
        return tuple(sorted(self._factories))

    def _factory(self, name, path):
        if name not in self._factories:
            raise ConfigurationError(f"{path}: unknown component {name!r}")
        factory = self._factories[name]
        if isinstance(factory, tuple):
            module, attribute = factory
            try:
                factory = getattr(importlib.import_module(module), attribute)
            except (ImportError, AttributeError) as error:
                raise ConfigurationError(f"{path}: cannot load {name}: {error}") from error
            self._factories[name] = factory
        return factory

    def resolve(self, value, dimensions=None, *, path="config", construct=True):
        dimensions = dimensions or {}
        if isinstance(value, Mapping):
            if "$dim" in value:
                unknown = value.keys() - {"$dim", "index", "scale", "offset"}
                name = value["$dim"]
                if unknown or not isinstance(name, str) or name not in dimensions:
                    raise ConfigurationError(
                        f"{path}: invalid or unknown dimension reference {value!r}"
                    )
                result = deepcopy(dimensions[name])
                if "index" in value:
                    index = value["index"]
                    if (
                        not isinstance(result, (list, tuple))
                        or type(index) is not int
                        or not -len(result) <= index < len(result)
                    ):
                        raise ConfigurationError(f"{path}: invalid dimension index")
                    result = result[index]
                if "scale" in value or "offset" in value:
                    scale, offset = value.get("scale", 1), value.get("offset", 0)
                    if any(type(v) is not int for v in (result, scale, offset)):
                        raise ConfigurationError(f"{path}: dimension arithmetic requires integers")
                    result = result * scale + offset
                return result
            if construct and "type" in value:
                return self.build(value, dimensions, path=path)
            return {
                key: self.resolve(item, dimensions, path=f"{path}.{key}", construct=construct)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [
                self.resolve(item, dimensions, path=f"{path}[{index}]", construct=construct)
                for index, item in enumerate(value)
            ]
        if isinstance(value, float) and not math.isfinite(value):
            raise ConfigurationError(f"{path}: numeric configuration values must be finite")
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise ConfigurationError(f"{path}: unsupported configuration value {type(value).__name__}")

    def build(self, spec, dimensions=None, *, path="component", injected=None):
        if not isinstance(spec, Mapping) or not isinstance(spec.get("type"), str):
            raise ConfigurationError(f"{path}: expected a component type and params mapping")
        unknown = spec.keys() - {"type", "params", "initialization", "checkpoint"}
        if unknown:
            raise ConfigurationError(f"{path}: unknown spec fields {sorted(unknown)}")
        parameters = spec.get("params", {})
        if not isinstance(parameters, Mapping):
            raise ConfigurationError(f"{path}.params: expected a mapping")
        parameters = self.resolve(parameters, dimensions, path=path + ".params")
        injected = injected or {}
        if parameters.keys() & injected.keys():
            raise ConfigurationError(
                f"{path}: cannot override injected fields {sorted(parameters.keys() & injected.keys())}"
            )
        parameters.update(injected)
        factory = self._factory(spec["type"], path)
        try:
            inspect.signature(factory).bind(**parameters)
        except (TypeError, ValueError) as error:
            raise ConfigurationError(f"{path}: invalid constructor parameters: {error}") from error
        try:
            result = factory(**parameters)
        except (TypeError, ValueError) as error:
            raise ConfigurationError(f"{path}: {error}") from error
        if "initialization" in spec or "checkpoint" in spec:
            if not isinstance(result, nn.Module):
                raise ConfigurationError(f"{path}: initialization/checkpoint requires an nn.Module")
            from marlite.util.initialization import initialize_model

            initialize_model(result, spec.get("initialization"))
            if "checkpoint" in spec:
                checkpoint = spec["checkpoint"]
                if (
                    not isinstance(checkpoint, Mapping)
                    or checkpoint.keys() - {"path", "key"}
                    or "path" not in checkpoint
                ):
                    raise ConfigurationError(f"{path}: checkpoint needs path and optional key list")
                state = torch.load(
                    Path(checkpoint["path"]).expanduser(), weights_only=True, map_location="cpu"
                )
                keys = checkpoint.get("key", [])
                if not isinstance(keys, list) or any(
                    not isinstance(key, (str, int)) for key in keys
                ):
                    raise ConfigurationError(
                        f"{path}: checkpoint key must be a list of mapping keys"
                    )
                for key in keys:
                    state = state[key]
                result.load_state_dict(state, strict=True)
            result._initialization_boundary = True
        return result
