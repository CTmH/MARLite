"""Joint objectives with one backward/optimizer step, including gradient surgery.

Gradient methods operate only on parameters used by both tasks. Pass the original
scalar losses as a sequence to retain autograd's distinction between unused and
zero gradients; a stacked tensor is also supported when parameters are shared.
"""

import math

import torch
import torch.distributed as dist
from torch import nn


def distributed_mean(value, distributed=False):
    """Average detached statistics without adding collectives to autograd."""
    result = value.detach().clone()
    if distributed:
        dist.all_reduce(result)
        result.div_(dist.get_world_size())
    return result


def reduce_mixed_gradients(parameters):
    """Average gradients safely when a task leaves parameters unused on a rank.

    Preserve None for globally unused parameters so optimizers do not apply weight
    decay or momentum to inactive heads. Every rank issues identical collectives.
    """
    parameters = tuple(dict.fromkeys(p for p in parameters if p.requires_grad))
    if not parameters:
        return
    used = torch.tensor(
        [p.grad is not None for p in parameters],
        device=parameters[0].device,
        dtype=torch.float32,
    )
    dist.all_reduce(used)
    for index in used.nonzero().flatten().tolist():
        parameter = parameters[index]
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        dist.all_reduce(parameter.grad)
        parameter.grad.div_(dist.get_world_size())


class LossMixer(nn.Module):
    """Base for task-vector losses; weights are applied after normalization."""

    def __init__(self, num_tasks=2, weights=None, reduction="sum"):
        super().__init__()
        if not isinstance(num_tasks, int) or num_tasks < 1:
            raise ValueError("num_tasks must be a positive integer")
        if reduction not in {"none", "sum", "mean"}:
            raise ValueError("reduction must be 'none', 'sum' or 'mean'")
        weights = torch.as_tensor(
            [1.0] * num_tasks if weights is None else weights, dtype=torch.float32
        )
        if (
            weights.shape != (num_tasks,)
            or not torch.isfinite(weights).all()
            or (weights < 0).any()
        ):
            raise ValueError("weights must contain one finite nonnegative value per task")
        self.num_tasks = num_tasks
        self.reduction = reduction
        self.register_buffer("weights", weights.detach().clone())

    def _loss_vector(self, losses):
        values = losses if isinstance(losses, torch.Tensor) else torch.stack(tuple(losses))
        if values.shape != (self.num_tasks,) or not values.is_floating_point():
            raise ValueError(f"losses must be a floating tensor of shape ({self.num_tasks},)")
        if not torch.isfinite(values).all():
            raise ValueError("losses must be finite")
        if self.weights.device != values.device:
            self.to(device=values.device)
        return values

    def _reduce(self, values):
        weighted = values * self.weights
        if self.reduction == "none":
            return weighted
        return weighted.mean() if self.reduction == "mean" else weighted.sum()


class WeightedSumLoss(LossMixer):
    """Positive weighted sum, valid for signed policy-gradient objectives."""

    def forward(self, losses, parameters=None, distributed=False):
        return self._reduce(self._loss_vector(losses))


class _GradientLoss(WeightedSumLoss):
    """Two-task gradient measurements on the shared part of supplied parameters."""

    def __init__(self, num_tasks=2, weights=None, reduction="sum", eps=1e-8):
        super().__init__(num_tasks, weights, reduction)
        if num_tasks != 2 or reduction == "none":
            raise ValueError("gradient mixers require two tasks and a scalar reduction")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        self.eps = eps

    def _shared_gradients(self, losses, parameters, distributed):
        if parameters is None:
            raise ValueError("gradient mixers require candidate shared parameters")
        parameters = tuple(dict.fromkeys(p for p in parameters if p.requires_grad))
        if not parameters:
            return (), (), ()
        local = []
        averaged = []
        presence = []
        for loss in losses:
            gradients = (
                torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
                if loss.requires_grad else (None,) * len(parameters)
            )
            flags = torch.tensor(
                [g is not None for g in gradients], device=parameters[0].device
            )
            flat = torch.cat(
                [
                    (g.detach() if g is not None else torch.zeros_like(p)).reshape(-1)
                    for p, g in zip(parameters, gradients)
                ]
            )
            local.append(flat)
            averaged.append(distributed_mean(flat, distributed))
            # All ranks must use the same shared subset, even for rank-local branches.
            presence.append(distributed_mean(flags.float(), distributed) > 0)
        shared = presence[0] & presence[1]
        sizes = [p.numel() for p in parameters]
        local = [g.split(sizes) for g in local]
        averaged = [g.split(sizes) for g in averaged]
        indices = shared.nonzero().flatten().tolist()
        return (
            tuple(parameters[i] for i in indices),
            tuple(torch.cat([g[i] for i in indices]) for g in local) if indices else (),
            tuple(torch.cat([g[i] for i in indices]) for g in averaged) if indices else (),
        )


class EMAGradNormLoss(_GradientLoss):
    """Bounded SSL weight from EMA gradient norms, not the GradNorm algorithm.

    Task 0 is RL. Its weight stays fixed; task 1's configured weight is the
    fallback when either shared gradient vanishes (e.g. a closed consensus gate).
    Otherwise target_ratio sets the weighted SSL/RL gradient-norm ratio.
    """

    def __init__(
        self, alpha=0.9, target_ratio=1.0, min_weight=0.0, max_weight=10.0, **kwargs
    ):
        super().__init__(**kwargs)
        if not math.isfinite(alpha) or not 0 <= alpha < 1:
            raise ValueError("alpha must be in [0, 1)")
        if not all(math.isfinite(v) for v in (target_ratio, min_weight, max_weight)):
            raise ValueError("gradient-weight parameters must be finite")
        if target_ratio < 0 or not 0 <= min_weight <= max_weight:
            raise ValueError("invalid target ratio or weight bounds")
        self.alpha = alpha
        self.target_ratio = target_ratio
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.register_buffer("norm_ema", torch.zeros(2))
        self.register_buffer("initialized", torch.tensor(False))
        self.register_buffer("effective_ssl_weight", self.weights[1].clone())

    def forward(self, losses, parameters=None, distributed=False):
        values = self._loss_vector(losses)
        weight = self.weights[1].detach().clone()
        if torch.is_grad_enabled() and self.training and self.weights.prod() > 0:
            _, _, gradients = self._shared_gradients(losses, parameters, distributed)
            if gradients:
                norms = torch.stack([g.norm() for g in gradients])
                if torch.isfinite(norms).all() and (norms > self.eps).all():
                    with torch.no_grad():
                        if self.initialized:
                            self.norm_ema.lerp_(norms.to(self.norm_ema), 1 - self.alpha)
                        else:
                            self.norm_ema.copy_(norms)
                            self.initialized.fill_(True)
                        weight = (
                            self.target_ratio * self.weights[0] * self.norm_ema[0]
                            / self.norm_ema[1].clamp_min(self.eps)
                        ).clamp(self.min_weight, self.max_weight).clone()
        elif self.initialized:
            weight = self.effective_ssl_weight.detach().clone()
        if self.training:
            self.effective_ssl_weight.copy_(weight)
        result = self.weights[0] * values[0] + weight * values[1]
        return result / 2 if self.reduction == "mean" else result


class PCGradLoss(_GradientLoss):
    """Symmetric two-task PCGrad on shared parameters; private gradients unchanged.

    The returned scalar displays the weighted loss but is a first-order surrogate:
    backward adds the projection correction. It is not a new scalar objective or
    a higher-order differentiable optimizer. All-reduce task gradients BEFORE
    projection; the worker's normal gradient reduction still handles private heads.
    """

    def forward(self, losses, parameters=None, distributed=False):
        values = self._loss_vector(losses)
        result = self._reduce(values)
        if not torch.is_grad_enabled() or not self.training or self.weights.prod() == 0:
            return result
        shared, local, averaged = self._shared_gradients(losses, parameters, distributed)
        if not shared:
            return result
        scale = self.weights / (2 if self.reduction == "mean" else 1)
        first, second = [g * w for g, w in zip(averaged, scale)]
        dot = torch.dot(first, second)
        # Each projection uses the ORIGINAL other-task gradient, not its projection.
        conflict = dot.clamp(max=0)
        projected = (
            first + second
            - conflict / second.square().sum().clamp_min(self.eps) * second
            - conflict / first.square().sum().clamp_min(self.eps) * first
        )
        correction = projected - sum(g * w for g, w in zip(local, scale))
        for parameter, delta in zip(shared, correction.split([p.numel() for p in shared])):
            # Zero forward value avoids cancellation with a large parameter dot product.
            result = result + ((parameter - parameter.detach()) * delta.view_as(parameter)).sum()
        return result
