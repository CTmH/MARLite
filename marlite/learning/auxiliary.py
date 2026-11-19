"""Optional representation learning, composed with either sequence learner.

Targets are training-only data. They never enter DecisionInput or collection.
Each loss population supplies a sum and a valid count; this preserves the same
objective for different rank partitions, padding lengths and group capacities.
"""

from dataclasses import dataclass
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

from marlite.algorithm.components import validate_replay_modules
from .sequence import validate_optimizer_ownership


@dataclass
class AuxiliaryTerm:
    total: torch.Tensor
    count: torch.Tensor
    weight: float = 1.


@dataclass
class Targets:
    values: torch.Tensor
    valid: torch.Tensor


def reparameterization_noise(batch, reference, update, seed):
    """Stable frame/slot noise independent of batch order and number of ranks.

    Only small latent tensors are generated on CPU. No process-global random
    state is consumed, and duplicate frames in overlapping chunks share noise.
    A new successful update receives a new draw. This is Monte Carlo VAE noise,
    separate from recorded randomness that may affect the behavior policy.
    """
    noise = torch.empty(reference.shape, dtype=reference.dtype, device="cpu")
    for b, source in enumerate(batch.sources):
        start = int(batch.frame_start[b])
        for t in range(reference.shape[1]):
            identity = f"{seed}:{update}:{source.episode_id}:{start + t}".encode()
            key = int.from_bytes(hashlib.blake2b(identity, digest_size=8).digest(), "little")
            generator = torch.Generator().manual_seed(key)
            noise[b, t] = torch.randn(reference.shape[2:], generator=generator, dtype=reference.dtype)
    return noise.to(reference.device)


class ObservationTargets:
    """Individual observations or group means, useful for generic baselines.

    Averaging is an explicit target definition, not entity deduplication. Tasks
    needing union/crops/semantic labels should inject a dedicated target builder.
    """

    def __init__(self, scope="agent"):
        if scope not in ("agent", "group"):
            raise ValueError("Observation target scope must be agent or group")
        self.scope = scope

    @torch.no_grad()
    def __call__(self, batch, diagnostics, length):
        obs = batch.frames["observations"][:, :length].float()
        alive = batch.frames["alive"][:, :length]
        if self.scope == "agent":
            return Targets(obs, alive)
        groups = diagnostics.get("groups")
        valid = diagnostics["group_valid"][:, :length]
        if groups is None:
            if valid.shape[-1] != 1:
                raise ValueError("Multi-group targets require group assignments")
            groups = torch.zeros_like(alive, dtype=torch.long)
        else:
            groups = groups[:, :length]
        b, t, n = alive.shape
        flat = obs.reshape(b * t, n, -1).masked_fill(~alive.reshape(b * t, n, 1), 0)
        gid = groups.reshape(b * t, n).masked_fill(~alive.reshape(b * t, n), 0)
        index = gid[..., None].expand_as(flat)
        target = flat.new_zeros(b * t, valid.shape[-1], flat.shape[-1]).scatter_add_(1, index, flat)
        count = flat.new_zeros(b * t, valid.shape[-1]).scatter_add_(1, gid, alive.reshape(b * t, n).float())
        target = target / count.clamp_min(1)[..., None]
        return Targets(target.reshape(b, t, valid.shape[-1], *obs.shape[3:]), valid)


class ConstructorTargets:
    """Adapt an environment label constructor to per-frame sequence targets.

    Existing constructors often consume only the last frame. Flattening B*T
    into singleton histories prevents silently discarding all earlier labels.
    `mask_is_padding` and `time_axis` are explicit because old constructors
    disagree about mask naming and whether they retain the singleton time axis.
    Adjacency convention here is receiver,sender; constructor edges are sender,
    receiver. Only detached topology is used to construct training labels.
    """

    def __init__(self, constructor, topology="groups", *, mask_is_padding=False,
                 time_axis=False, graph_source="behavior"):
        if topology not in ("groups", "graph") or graph_source not in ("behavior", "current"):
            raise ValueError("Invalid constructor topology/source")
        self.constructor, self.topology = constructor, topology
        self.mask_is_padding, self.time_axis, self.graph_source = mask_is_padding, time_axis, graph_source

    @torch.no_grad()
    def __call__(self, batch, diagnostics, length):
        def numpy_frames(name):
            value = batch.frames[name][:, :length]
            return value.flatten(0, 1).unsqueeze(1).detach().cpu().numpy()

        obs, states, alive = map(numpy_frames, ("observations", "states", "alive"))
        if self.topology == "groups":
            topology = diagnostics["groups"][:, :length].flatten(0, 1).detach().cpu().numpy()
        else:
            key = "behavior.adjacency"
            adjacency = (diagnostics["adjacency"][:, :length] if self.graph_source == "current" else
                         batch.transitions.get(key, batch.frames.get("context.adjacency")))
            if adjacency is None or adjacency.shape[1] < length:
                raise ValueError("Graph targets require recorded or explicitly selected current adjacency")
            adjacency = adjacency[:, :length].flatten(0, 1).detach().cpu()
            topology = [[(matrix != 0).nonzero().flip(-1).T.numpy()] for matrix in adjacency]
        values, valid = self.constructor.process(obs, states, topology, alive)
        values, valid = torch.as_tensor(values), torch.as_tensor(valid, dtype=torch.bool)
        if self.time_axis:
            if values.shape[1] != 1 or valid.shape[1] != 1:
                raise ValueError("Expected singleton constructor time axis")
            values, valid = values[:, 0], valid[:, 0]
        if self.mask_is_padding:
            valid = ~valid
        b = batch.filled.shape[0]
        if values.shape[0] != b * length or valid.shape[0] != b * length:
            raise ValueError("Constructor must label every requested frame")
        return Targets(values.reshape(b, length, *values.shape[1:]).to(batch.filled.device),
                       valid.reshape(b, length, *valid.shape[1:]).to(batch.filled.device))


class ReconstructionObjective(nn.Module):
    """AE/VAE reconstruction and agent/group KL, without a trainer subclass.

    `source` chooses group or agent latent diagnostics; deterministic graph
    features can use source='features'. The decoder accepts flattened unit
    vectors and returns the target feature shape, e.g. U,F or U,C,H,W. Optional
    entity masks have additional axes; average only the remaining feature axes.
    Group weighting can count each group once or weight by its live members.
    """

    def __init__(self, decoder, targets, *, source="group", variational=False,
                 reconstruction="mse", group_kl=0., agent_kl=0., group_weighting="groups",
                 noise_seed=0):
        super().__init__()
        if source not in ("agent", "group", "features"):
            raise ValueError("Unknown representation source")
        if variational and source == "features":
            raise ValueError("Variational reconstruction needs explicit Gaussian parameters")
        if not isinstance(reconstruction, nn.Module) and reconstruction not in ("mse", "bce", "bce_logits"):
            raise ValueError("Unknown reconstruction objective")
        if isinstance(reconstruction, nn.Module) and getattr(reconstruction, "reduction", None) != "none":
            raise ValueError("Custom reconstruction losses must return one loss per unit (reduction=none)")
        if group_weighting not in ("groups", "agents"):
            raise ValueError("Unknown group population weighting")
        if not all(math.isfinite(v) and v >= 0 for v in (group_kl, agent_kl)):
            raise ValueError("KL weights must be finite and nonnegative")
        self.decoder, self.targets, self.source = decoder, targets, source
        self.variational, self.reconstruction = variational, reconstruction
        self.group_kl, self.agent_kl, self.group_weighting = group_kl, agent_kl, group_weighting
        self.noise_seed = noise_seed

    def forward(self, output, batch, update=0, **context):
        length = batch.loss_mask.shape[1]
        diagnostics = output.diagnostics
        source = (diagnostics["local_state_estimates"] if self.source == "features" else
                  diagnostics[self.source + "_mu"])[:, :length]
        if self.variational:
            log_var = diagnostics[self.source + "_log_var"][:, :length]
            noise = reparameterization_noise(batch, source, update, self.noise_seed)
            source = source + (.5 * log_var).exp() * noise
        valid_units = (diagnostics["group_valid"][:, :length] if self.source == "group" else
                       batch.frames["alive"][:, :length])
        target = self.targets(batch, diagnostics, length)
        target_values = target.values.to(source)
        prediction = self.decoder(source.flatten(0, 2)).reshape(*source.shape[:3], *target_values.shape[3:])
        if prediction.shape != target_values.shape or target.valid.dtype != torch.bool:
            raise ValueError("Reconstruction targets must match predictions and use boolean validity")
        if target.valid.shape[:3] != source.shape[:3] or tuple(target_values.shape[:target.valid.ndim]) != tuple(target.valid.shape):
            raise ValueError("Target validity must describe leading target axes")
        valid = target.valid & (valid_units & batch.loss_mask[..., None]).reshape(
            *valid_units.shape, *([1] * (target.valid.ndim - 3)))
        # Invalid padding must not introduce NaNs or invalid BCE labels.
        mask = valid.reshape(*valid.shape, *([1] * (prediction.ndim - valid.ndim)))
        prediction = prediction.masked_fill(~mask, 0)
        target_values = target_values.masked_fill(~mask, 0)
        if isinstance(self.reconstruction, nn.Module):
            # Preserve set losses/foreground weighting without hiding their label
            # populations inside a scalar mean. Additional validity axes are
            # entity axes; the remaining target axes are feature channels.
            units = math.prod(source.shape[:3])
            entities = math.prod(valid.shape[3:])
            error = self.reconstruction(prediction.reshape(units, entities, -1),
                                        target_values.reshape(units, entities, -1),
                                        valid.reshape(units, entities))
            if error.shape != (units,):
                raise ValueError("Custom reconstruction must return a vector over B*T*U units")
            error = error.reshape(source.shape[:3])
            weights = valid.reshape(*source.shape[:3], entities).any(-1).to(source)
        else:
            if self.reconstruction == "mse":
                error = (prediction - target_values).square()
            elif self.reconstruction == "bce":
                error = F.binary_cross_entropy(prediction, target_values, reduction="none")
            else:
                error = F.binary_cross_entropy_with_logits(prediction, target_values, reduction="none")
            if error.ndim > valid.ndim:
                error = error.flatten(valid.ndim).mean(-1)
            weights = valid.to(source)
        if self.source == "group" and self.group_weighting == "agents":
            groups = diagnostics.get("groups")
            if groups is None:
                count = batch.frames["alive"][:, :length].sum(-1, keepdim=True).to(source)
            else:
                live = batch.frames["alive"][:, :length]
                gid = groups[:, :length].masked_fill(~live, 0)
                count = source.new_zeros(*source.shape[:3]).scatter_add_(2, gid, live.to(source))
            weights = weights * count.reshape(*count.shape, *([1] * (weights.ndim - 3)))
        terms = {"reconstruction_loss": AuxiliaryTerm((error * weights).sum(), weights.sum())}
        for level, coefficient in (("agent", self.agent_kl), ("group", self.group_kl)):
            if not coefficient:
                continue
            mean = diagnostics[level + "_mu"][:, :length]
            log_var = diagnostics[level + "_log_var"][:, :length]
            unit_valid = (diagnostics["group_valid"][:, :length] if level == "group" else
                          batch.frames["alive"][:, :length]) & batch.loss_mask[..., None]
            kl = .5 * (mean.square() + log_var.exp() - 1 - log_var).mean(-1)
            terms[level + "_kl"] = AuxiliaryTerm(kl[unit_valid].sum(), unit_valid.sum(), coefficient)
        return terms


class AuxiliaryLearning:
    """Own an optional objective, optimizer, task mixer and warmup counter.

    Warmup counts successful learner updates; SSL starts at `start_update`.
    The policy's detach flags and RL gate independently control whether RL may
    train messages. Disabled SSL heads retain grad=None (no weight decay drift).
    """

    def __init__(self, objective, optimizer=None, *, mixer=None, start_update=0):
        if type(start_update) is not int or start_update < 0:
            raise ValueError("Auxiliary start_update must be a nonnegative integer")
        validate_optimizer_ownership(((objective, optimizer),))
        self.objective, self.optimizer, self.mixer = objective, optimizer, mixer
        self.start_update, self.updates = start_update, 0

    def initialize(self, backend, modules):
        owned = {id(p) for module in modules for p in module.parameters()}
        if owned.intersection(map(id, self.objective.parameters())):
            raise ValueError("Auxiliary heads must not own actor/critic parameters")
        validate_replay_modules(self.objective, scope="Auxiliary objective")
        self.objective.to(backend.device)
        backend.synchronize(self.objective)
        if self.mixer is not None:
            if self.mixer.num_tasks != 2 or self.mixer.reduction == "none":
                raise ValueError("RL/SSL mixing needs two tasks and a scalar reduction")
            if any(p.requires_grad for p in self.mixer.parameters()):
                raise ValueError("Task mixers must not contain unowned trainable parameters")
            if not self.mixer.weights.any():
                raise ValueError("At least one learning task must have positive weight")
            self.mixer.to(backend.device)
            backend.synchronize(self.mixer)

    def zero_grad(self):
        self.objective.train()
        if self.mixer is not None:
            self.mixer.train()
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)

    def losses(self, output, batch, backend, **context):
        metrics = {}
        if self.updates < self.start_update:
            return None, metrics
        terms = self.objective(output, batch, self.updates, **context)
        total = None
        for name, term in terms.items():
            if not math.isfinite(term.weight) or term.weight < 0:
                raise ValueError("Auxiliary weights must be finite and nonnegative")
            count = backend.sum(term.count.to(device=backend.device, dtype=torch.float32))
            if term.weight and count:
                value = term.weight * term.total / count
                total = value if total is None else total + value
            metrics["ssl/" + name] = float(backend.sum(term.total) / count.clamp_min(1))
            metrics["ssl/" + name + "_count"] = float(count)
        return total, metrics

    def combine(self, rl_loss, ssl_loss, parameters, backend):
        if ssl_loss is None:
            return rl_loss
        if not torch.isfinite(ssl_loss):
            raise RuntimeError("Nonfinite auxiliary loss")
        if self.mixer is None:
            return rl_loss + ssl_loss
        # A zero weight disables a task's graph, not merely its gradient value.
        # Inactive private heads must keep grad=None even with optimizer momentum.
        if not self.mixer.weights[0]:
            rl_loss = rl_loss.detach()
        if not self.mixer.weights[1]:
            ssl_loss = ssl_loss.detach()
        return self.mixer((rl_loss, ssl_loss), parameters=parameters,
                          distributed=backend.distributed, distributed_reduction="sum")

    def step(self, backend, max_grad_norm, active):
        backend.reduce_gradients(self.objective.parameters())
        if self.optimizer is not None and active and any(p.grad is not None for p in self.objective.parameters()):
            torch.nn.utils.clip_grad_norm_(self.objective.parameters(), max_grad_norm)
            self.optimizer.step()
        if active:
            self.updates += 1

    def state_dict(self):
        return {"objective": self.objective.state_dict(), "updates": self.updates,
                "start_update": self.start_update,
                "optimizer": None if self.optimizer is None else self.optimizer.state_dict(),
                "mixer": None if self.mixer is None else self.mixer.state_dict()}

    def load_state_dict(self, state):
        if state["start_update"] != self.start_update:
            raise ValueError("Checkpoint auxiliary warmup differs from this learner")
        self.objective.load_state_dict(state["objective"])
        for name in ("optimizer", "mixer"):
            component = getattr(self, name)
            if (component is None) != (state[name] is None):
                raise ValueError("Checkpoint auxiliary composition differs from this learner")
            if component is not None:
                component.load_state_dict(state[name])
        self.updates = state["updates"]
