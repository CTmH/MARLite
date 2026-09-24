"""Explicit, opt-in initialization of model parameters.

Defaults visit supported projections only. Normalization, positional encodings,
and GAT attention vectors keep their native values unless explicitly selected.
Nested models with their own initialization/checkpoint are ownership boundaries:
an enclosing model may not overwrite them.
"""

import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn.parameter import UninitializedParameter
from torch_geometric.nn.dense.linear import Linear as GraphLinear
from torch_geometric.nn import GCNConv, GATConv


_METHODS = {
    "orthogonal": (nn.init.orthogonal_, {"gain"}),
    "xavier_uniform": (nn.init.xavier_uniform_, {"gain"}),
    "xavier_normal": (nn.init.xavier_normal_, {"gain"}),
    "kaiming_uniform": (nn.init.kaiming_uniform_, {"a", "mode", "nonlinearity"}),
    "kaiming_normal": (nn.init.kaiming_normal_, {"a", "mode", "nonlinearity"}),
    "normal": (nn.init.normal_, {"mean", "std"}),
    "uniform": (nn.init.uniform_, {"a", "b"}),
    "trunc_normal": (nn.init.trunc_normal_, {"mean", "std", "a", "b"}),
    "constant": (nn.init.constant_, {"value"}),
}
_PROJECTIONS = (
    nn.Linear, GraphLinear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
    nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
)
_CONTROLS = {"bias", "input", "recurrent", "forget_bias", "output_slices"}


def _validate_rule(rule):
    if not isinstance(rule, Mapping):
        raise ValueError("An initialization rule must be a mapping")
    scheme = rule.get("scheme")
    if scheme not in _METHODS:
        raise ValueError(f"Unknown initialization scheme: {scheme!r}")
    allowed = _METHODS[scheme][1] | _CONTROLS | {"scheme"}
    unknown = rule.keys() - allowed
    if unknown:
        raise ValueError(f"Unsupported {scheme} initialization options: {sorted(unknown)}")
    for name in _METHODS[scheme][1] | {"bias", "forget_bias"}:
        if name in rule and name not in {"mode", "nonlinearity"}:
            value = rule[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
    if scheme == "constant" and "value" not in rule:
        raise ValueError("constant initialization requires value")
    if scheme in {"normal", "trunc_normal"} and rule.get("std", 1) <= 0:
        raise ValueError("std must be positive")
    if scheme in {"uniform", "trunc_normal"}:
        default_a, default_b = (0, 1) if scheme == "uniform" else (-2, 2)
        if rule.get("a", default_a) >= rule.get("b", default_b):
            raise ValueError("Initialization bounds must satisfy a < b")
    for name in ("input", "recurrent"):
        if name in rule:
            _validate_rule(rule[name])
            if rule[name].keys() & _CONTROLS:
                raise ValueError(f"{name} accepts a weight rule only; set biases on the outer rule")


def _fill(tensor, rule):
    """Fill one logical weight, not an entire packed gate/QKV matrix."""
    if isinstance(tensor, UninitializedParameter):
        raise ValueError("Explicit initialization requires materialized parameters")
    scheme = rule["scheme"]
    if scheme in {"orthogonal", "xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal"}:
        if tensor.ndim < 2:
            raise ValueError(f"{scheme} requires a tensor with at least two dimensions")
    method, names = _METHODS[scheme]
    options = {key: rule[key] for key in names if key in rule}
    if scheme == "constant":
        options["val"] = options.pop("value")
    method(tensor, **options)


def _initialize_module(module, rule, seen):
    """Return whether this module has a supported parameter layout."""
    def weight(parameter, settings=rule, chunks=1):
        if parameter is None or id(parameter) in seen:
            return
        seen.add(id(parameter))
        if isinstance(parameter, UninitializedParameter):
            raise ValueError("Explicit initialization requires materialized parameters")
        for block in parameter.chunk(chunks, dim=0):
            _fill(block, settings)

    def bias(parameter):
        if parameter is not None and "bias" in rule and id(parameter) not in seen:
            seen.add(id(parameter))
            nn.init.constant_(parameter, rule["bias"])

    if isinstance(module, _PROJECTIONS):
        weight(module.weight)
        bias(module.bias)
    elif isinstance(module, (GCNConv, GATConv)):
        # PyG projections use GraphLinear, not nn.Linear. Handle them here so
        # exact graph-layer overrides and recursive defaults have equal coverage.
        for child in module.modules():
            if isinstance(child, GraphLinear):
                weight(child.weight)
                bias(child.bias)
        bias(module.bias)
        # GAT attention vectors retain native Glorot initialization unless an
        # explicit direct-parameter override selects att_src/att_dst/att_edge.
    elif isinstance(module, nn.MultiheadAttention):
        if module.in_proj_weight is not None:
            weight(module.in_proj_weight, chunks=3)
        else:
            for parameter in (module.q_proj_weight, module.k_proj_weight, module.v_proj_weight):
                weight(parameter)
        weight(module.out_proj.weight)
        for parameter in (module.in_proj_bias, module.out_proj.bias, module.bias_k, module.bias_v):
            bias(parameter)
    elif isinstance(module, nn.RNNBase):
        gates = 4 if isinstance(module, nn.LSTM) else 3 if isinstance(module, nn.GRU) else 1
        for name, parameter in module.named_parameters(recurse=False):
            if name.startswith("weight_ih"):
                weight(parameter, rule.get("input", rule), gates)
            elif name.startswith("weight_hh"):
                weight(parameter, rule.get("recurrent", rule), gates)
            elif name.startswith("weight_hr"):
                weight(parameter)  # Projected LSTM output, not a gate matrix.
            elif name.startswith("bias"):
                bias(parameter)
        if "forget_bias" in rule:
            if not isinstance(module, nn.LSTM) or not module.bias:
                raise ValueError("forget_bias requires an LSTM with bias")
            for name, parameter in module.named_parameters(recurse=False):
                if name.startswith("bias_"):
                    seen.add(id(parameter))
                    # PyTorch adds bias_ih and bias_hh: set the effective value once.
                    value = rule["forget_bias"] if name.startswith("bias_ih") else 0.0
                    nn.init.constant_(parameter.chunk(4)[1], value)
    else:
        return False
    if {"input", "recurrent", "forget_bias"} & rule.keys() and not isinstance(module, nn.RNNBase):
        raise ValueError("input/recurrent/forget_bias options require a recurrent module")
    if "output_slices" in rule:
        if not isinstance(module, (nn.Linear, GraphLinear)):
            raise ValueError("output_slices is supported only for Linear output rows")
        occupied = set()
        for entry in rule["output_slices"]:
            if not isinstance(entry, Mapping) or not {"start", "stop"} <= entry.keys():
                raise ValueError("Each output slice requires start and stop")
            settings = dict(entry)
            start, stop = settings.pop("start"), settings.pop("stop")
            if any(isinstance(v, bool) or not isinstance(v, int) for v in (start, stop)):
                raise ValueError("Output slice bounds must be integers")
            rows = set(range(start, stop))
            if not 0 <= start < stop <= module.weight.shape[0] or occupied & rows:
                raise ValueError("Output slices must be in range and non-overlapping")
            occupied.update(rows)
            _validate_rule(settings)
            if settings.keys() & (_CONTROLS - {"bias"}):
                raise ValueError("Output slices accept only a weight rule and bias")
            _fill(module.weight[start:stop], settings)
            if module.bias is not None and "bias" in settings:
                nn.init.constant_(module.bias[start:stop], settings["bias"])
    return True


@torch.no_grad()
def initialize_model(model: nn.Module, config: Mapping | None) -> None:
    """Apply defaults, then exact module overrides; never reset protected children.

    An override can select a direct ``parameter`` (e.g. GAT ``att_src`` or
    LayerNorm ``weight``). Such rules operate on that tensor exactly as stored.
    Unknown explicit targets fail rather than silently retaining defaults.
    """
    if config is None:
        return
    if not isinstance(config, Mapping):
        raise ValueError("initialization must be a mapping or null")
    defaults = dict(config)
    overrides = defaults.pop("overrides", [])
    if not isinstance(overrides, list):
        raise ValueError("initialization.overrides must be a list")
    if defaults:
        _validate_rule(defaults)
        if defaults.keys() & {"input", "recurrent", "forget_bias", "output_slices"}:
            raise ValueError("Layout-specific rules belong in module overrides")

    protected = set()
    protected_parameters = set()
    for name, module in model.named_modules():
        if name and getattr(module, "_initialization_boundary", False):
            protected.add(name)
            protected_parameters.update(id(p) for p in module.parameters())

    def is_protected(name):
        return any(name == root or name.startswith(root + ".") for root in protected)

    seen = set(protected_parameters)
    if defaults:
        for name, module in model.named_modules():
            if not is_protected(name):
                _initialize_module(module, defaults, seen)

    overridden = set()
    for entry in overrides:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("module"), str):
            raise ValueError("Each initialization override requires an exact module path")
        rule = dict(entry)
        name = rule.pop("module")
        parameter_name = rule.pop("parameter", None)
        try:
            module = model.get_submodule(name)
        except AttributeError as exc:
            raise ValueError(f"Unknown initialization module: {name!r}") from exc
        _validate_rule(rule)
        parameters = dict(module.named_parameters(recurse=False))
        if parameter_name is not None:
            if parameter_name not in parameters:
                raise ValueError(f"Unknown direct parameter: {name}.{parameter_name}")
            if rule.keys() & _CONTROLS:
                raise ValueError("Direct parameter rules accept only scheme-specific options")
            targets = {id(parameters[parameter_name])}
        else:
            targets = {id(p) for p in module.parameters()}
        if is_protected(name) or targets & protected_parameters:
            raise ValueError(f"Initialization cannot overwrite protected model: {name!r}")
        if parameter_name is not None:
            if targets & overridden:
                raise ValueError(f"Overlapping/shared initialization overrides: {name!r}")
            _fill(parameters[parameter_name], rule)
        else:
            touched = set()
            if not _initialize_module(module, rule, touched):
                raise ValueError(f"Unsupported initialization module: {name!r}; select a direct parameter")
            targets = touched
            if targets & overridden:
                raise ValueError(f"Overlapping/shared initialization overrides: {name!r}")
        overridden.update(targets)
