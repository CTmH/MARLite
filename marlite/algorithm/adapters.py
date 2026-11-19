"""Explicit adapters for pooled/masked neural component conventions."""

import torch
from torch import nn


class GaussianParameters(nn.Module):
    """Turn an existing (mean, log_variance) encoder into a concatenated tensor."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs):
        mean, log_variance = self.model(inputs)
        if mean.shape != log_variance.shape:
            raise ValueError("Gaussian encoder moments must have equal shapes")
        return torch.cat((mean, log_variance), -1)


class MaskedPool(nn.Module):
    """Read live sets with a model whose padding_mask uses True for padding.

    No order embeddings or extra token are inserted. Empty sets receive one
    harmless key to keep softmax finite and their result is then explicitly zero.
    Gaussian pooled encoders can concatenate their moment pair directly.
    """

    def __init__(
        self, model, gaussian_output=False, mask_argument="padding_mask", true_is_valid=False
    ):
        super().__init__()
        self.model, self.gaussian_output = model, gaussian_output
        self.mask_argument, self.true_is_valid = mask_argument, true_is_valid

    def forward(self, tokens, alive):
        if tokens.ndim != 3 or tokens.shape[:-1] != alive.shape or alive.dtype != torch.bool:
            raise ValueError("Masked pooling requires matching boolean live-token axes")
        empty = ~alive.any(-1)
        safe = alive.clone()
        safe[:, 0] |= empty
        result = self.model(
            tokens.masked_fill(~alive[..., None], 0),
            **{self.mask_argument: safe if self.true_is_valid else ~safe},
        )
        if self.gaussian_output:
            mean, log_variance = result
            if mean.shape != log_variance.shape:
                raise ValueError("Pooled Gaussian moments must have equal shapes")
            result = torch.cat((mean, log_variance), -1)
        if (
            not isinstance(result, torch.Tensor)
            or result.ndim != 2
            or result.shape[0] != tokens.shape[0]
        ):
            raise ValueError("Pooled models must return one vector per batch row")
        return result.masked_fill(empty[..., None], 0)
