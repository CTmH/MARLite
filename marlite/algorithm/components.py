"""Small constructor-injected network components used by sequence recipes."""

import torch
from torch import nn


def sequential(layers):
    if not isinstance(layers, list) or any(not isinstance(layer, nn.Module) for layer in layers):
        raise ValueError("Sequential layers must be a list of modules")
    return nn.Sequential(*layers)


class Permute(nn.Module):
    """Explicit axes; distributed execution does not introduce extra tensor axes."""

    def __init__(self, dims):
        super().__init__()
        if sorted(dims) != list(range(len(dims))):
            raise ValueError("Permutation must contain every axis exactly once")
        self.dims = tuple(dims)

    def forward(self, inputs):
        if inputs.ndim != len(self.dims):
            raise ValueError("Permutation rank does not match its input")
        return inputs.permute(*self.dims)


class MaskChannels(nn.Module):
    """Remove forbidden observation channels before any learned local encoding."""

    def __init__(self, channels, dim=-1):
        super().__init__()
        self.channels, self.dim = tuple(channels), dim

    def forward(self, inputs):
        result = inputs.clone()
        indices = [slice(None)] * inputs.ndim
        indices[self.dim] = list(self.channels)
        result[tuple(indices)] = 0
        return result


class HyperNetwork(nn.Module):
    """A conditional MLP with explicitly injected weight/bias generators.

    Unlike QMIX's monotonic mixer this network has unconstrained weights by
    default. Positive weights are an explicit option for applications requiring
    monotonicity. Generators and activation are ordinary configurable modules.
    """

    def __init__(
        self,
        layer_dims,
        weight_generators,
        bias_generators,
        *,
        activation=None,
        positive_weights=False,
    ):
        super().__init__()
        if (
            len(layer_dims) < 2
            or any(type(n) is not int or n < 1 for n in layer_dims)
            or len(weight_generators) != len(layer_dims) - 1
            or len(bias_generators) != len(weight_generators)
        ):
            raise ValueError(
                "Hypernetwork needs positive layer widths and one generator pair per layer"
            )
        self.layer_dims = tuple(layer_dims)
        self.weight_generators = nn.ModuleList(weight_generators)
        self.bias_generators = nn.ModuleList(bias_generators)
        self.activation = nn.GELU() if activation is None else activation
        self.positive_weights = positive_weights

    def forward(self, inputs, condition):
        if inputs.shape[:-1] != condition.shape[:-1] or inputs.shape[-1] != self.layer_dims[0]:
            raise ValueError("Hypernetwork inputs and conditions must share leading axes")
        features = inputs
        for index, (weights, biases) in enumerate(
            zip(self.weight_generators, self.bias_generators)
        ):
            n_in, n_out = self.layer_dims[index : index + 2]
            generated, bias = weights(condition), biases(condition)
            if generated.shape != (*condition.shape[:-1], n_in * n_out) or bias.shape != (
                *condition.shape[:-1],
                n_out,
            ):
                raise ValueError("Generated hypernetwork parameters have incorrect widths")
            generated = generated.reshape(*condition.shape[:-1], n_in, n_out)
            if self.positive_weights:
                generated = generated.abs()
            features = torch.einsum("...i,...ij->...j", features, generated) + bias
            if index + 1 < len(self.weight_generators):
                features = self.activation(features)
        return features


def validate_replay_modules(module, *, scope="Sequence component"):
    """Keep replay and global-loss reduction independent of minibatch layout.

    Sampling affecting a loss/policy must use explicit recorded or frame-keyed
    randomness. Batch statistics and hidden per-forward noise are incompatible
    with that contract; LayerNorm/GroupNorm and controlled VAE draws are valid.
    """
    for child in module.modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            raise ValueError(f"{scope} requires batch-independent normalization")
        if isinstance(child, nn.RReLU):
            raise ValueError(f"{scope} cannot use implicit random activations")
        if isinstance(child, nn.modules.dropout._DropoutNd) and child.p:
            raise ValueError(f"{scope} dropout must be zero")
        if isinstance(getattr(child, "dropout", None), (int, float)) and child.dropout:
            raise ValueError(f"{scope} dropout must be zero")
