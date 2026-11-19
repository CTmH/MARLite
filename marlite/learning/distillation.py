"""C04 teacher references and task losses, independent of PPO's update formula."""

from collections import OrderedDict
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

from marlite.algorithm.sequence_agent_group import DecisionInput
from marlite.communication.context import frame_context
from marlite.util.action_distribution import masked_categorical
from marlite.util.communication_curriculum import sender_replacement_mask
from .auxiliary import AuxiliaryTerm


def _frame_seed(batch, row, time, update, seed):
    source = batch.sources[row]
    frame = int(batch.frame_start[row]) + time
    key = f"{seed}:{update}:{source.episode_id}:{frame}".encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")


class FrozenPolicyReference:
    """Bounded derived cache for one immutable, frozen teacher AgentGroup.

    Only local features and logits are cached; replacing a message MUST rerun
    aggregation/readout with autograd. Cache entries retain episode identity to
    prevent object-ID reuse, and never reuse a student's recurrent state.
    This is deliberately not a child nn.Module of an auxiliary objective: the
    Trainer owns and checkpoints this single teacher, not every loss using it.
    """

    def __init__(self, actor, capacity=16):
        if type(capacity) is not int or capacity < 1 or actor.output_kind != "logits":
            raise ValueError(
                "A frozen reference needs a categorical actor and positive cache capacity"
            )
        self.actor = actor.requires_grad_(False).eval()
        self.capacity, self.cache = capacity, OrderedDict()

    def clear(self):
        self.cache.clear()

    @torch.no_grad()
    def episode(self, episode):
        if self.actor.training or any(p.requires_grad for p in self.actor.parameters()):
            raise ValueError("Reference teacher must remain frozen and in evaluation mode")
        key = id(episode)
        if key not in self.cache:
            device = next(self.actor.parameters()).device
            inputs = DecisionInput(
                episode.frames["observations"][None].to(device),
                episode.frames["alive"][None].to(device),
                episode.frames["agent_reset"][None].to(device),
                frame_context(episode.frames, batched=False, device=device),
            )
            state = self.actor.initial_state(1, len(episode.agent_ids), device, torch.float32)
            result = self.actor.forward_sequence(inputs, state)
            values = {
                "local": result.diagnostics["local_features"][0].detach().cpu().clone(),
                "logits": result.values[0].detach().cpu().clone(),
            }
            self.cache[key] = (episode, values)
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key][1]

    def batch(self, batch, episodes, length):
        if episodes is None:
            raise ValueError("Teacher state reconstruction requires original episodes")
        result = {}
        for row, source in enumerate(batch.sources):
            values = self.episode(episodes[source.episode_id])
            start = int(batch.frame_start[row])
            for name, tensor in values.items():
                if name not in result:
                    result[name] = tensor.new_zeros(len(batch.sources), length, *tensor.shape[1:])
                selected = tensor[start : start + length]
                result[name][row, : selected.shape[0]] = selected
        device = batch.filled.device
        return {name: tensor.to(device) for name, tensor in result.items()}


def action_kl_term(prediction, target, batch, weight=1.0):
    """KL(teacher || student) over live learning frames, with legal-action masks."""
    length = batch.loss_mask.shape[1]
    alive = batch.frames["alive"][:, :length] & batch.loss_mask[..., None]
    available = batch.frames["available_actions"][:, :length]
    reference = masked_categorical(target, available, active_mask=alive)
    student = masked_categorical(prediction, available, active_mask=alive)
    values = torch.distributions.kl_divergence(reference, student)
    return AuxiliaryTerm(values[alive].sum(), alive.sum(), weight)


class ActionDistillationObjective(nn.Module):
    """Optional online teacher loss with update-count-based linear decay."""

    def __init__(self, reference, weight=0.1, decay_updates=1000):
        super().__init__()
        if (
            not math.isfinite(weight)
            or weight < 0
            or type(decay_updates) is not int
            or decay_updates < 1
        ):
            raise ValueError("Invalid online distillation schedule")
        self.reference, self.weight, self.decay_updates = reference, weight, decay_updates

    def forward(self, output, batch, update=0, *, episodes=None, sources=None):
        weight = self.weight * max(0.0, 1 - update / self.decay_updates)
        if not weight:
            return {}
        target = self.reference.batch(batch, episodes, batch.loss_mask.shape[1])["logits"]
        return {"online_distillation_loss": action_kl_term(output.values, target, batch, weight)}


class CompositeObjective(nn.Module):
    """Compose optional independent heads/losses without another Trainer hierarchy."""

    def __init__(self, objectives):
        super().__init__()
        self.objectives = nn.ModuleDict(objectives)

    def forward(self, output, batch, update=0, **context):
        terms = {}
        for objective in self.objectives.values():
            extra = objective(output, batch, update, **context)
            if terms.keys() & extra.keys():
                raise ValueError("Composed objectives must use distinct metric names")
            terms.update(extra)
        return terms


class FrameQueryTargets:
    """Adapt a training-only query provider, with stable per-frame CPU draws.

    Providers consume one frame of states/observations/group IDs. All random
    candidate generation is CPU-side and isolated from training RNG streams;
    changing rank partitions cannot change the sampled questions or labels.
    """

    def __init__(self, provider, seed=0, resample_each_update=True):
        self.provider, self.seed = provider, seed
        self.resample_each_update = resample_each_update

    @torch.no_grad()
    def __call__(self, batch, diagnostics, length, update):
        valid = diagnostics["group_valid"][:, :length].detach().cpu()
        result = {}
        for row in range(valid.shape[0]):
            for time in range(length):
                frame = {
                    key: batch.frames[name][row : row + 1, time : time + 1].detach().cpu()
                    for key, name in (
                        ("states", "states"),
                        ("observations", "observations"),
                        ("alive_mask", "alive"),
                        ("group_indices", "context.groups"),
                    )
                }
                generator = torch.Generator().manual_seed(
                    _frame_seed(
                        batch, row, time, update if self.resample_each_update else 0, self.seed
                    )
                )
                with torch.random.fork_rng(devices=[]):
                    # set_rng_state affects CPU only; manual_seed would also seed GPUs.
                    torch.set_rng_state(generator.get_state())
                    targets = self.provider(frame, valid[row : row + 1, time], "cpu")
                for name, tensor in targets.items():
                    if name not in result:
                        result[name] = tensor.new_empty(*valid.shape[:2], *tensor.shape[1:])
                    result[name][row, time] = tensor[0]
        return {name: tensor.to(batch.filled.device) for name, tensor in result.items()}


class RelationQueryObjective(nn.Module):
    """Optional task prediction trains only its head and the upstream group context."""

    def __init__(self, head, targets, weight=1.0):
        super().__init__()
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("Query loss weight must be finite and nonnegative")
        self.head, self.targets, self.weight = head, targets, weight

    def forward(self, output, batch, update=0, *, weight_override=None, **context):
        weight = self.weight if weight_override is None else weight_override
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("Query loss weight must be finite and nonnegative")
        if not weight:
            return {}
        length = batch.loss_mask.shape[1]
        diagnostics = output.diagnostics
        representation = diagnostics["relation_context"][:, :length]
        targets = self.targets(batch, diagnostics, length, update)
        queries = targets["relation_queries"].to(representation)
        labels = targets["relation_labels"].to(representation)
        valid = targets["relation_mask"]
        group_valid = diagnostics["group_valid"][:, :length] & batch.loss_mask[..., None]
        if (
            valid.dtype != torch.bool
            or valid.shape != labels.shape
            or queries.shape[:-1] != labels.shape
        ):
            raise ValueError("Query targets need matching axes and boolean validity")
        if labels.shape[:3] != group_valid.shape:
            raise ValueError("Query targets must match time and group axes")
        mask = valid & group_valid[..., None]
        if not ((labels[mask] >= 0) & (labels[mask] <= 1)).all():
            raise ValueError("Valid query labels must be finite and between zero and one")
        logits = self.head(representation, queries)
        total = F.binary_cross_entropy_with_logits(logits[mask], labels[mask], reduction="sum")
        return {"relation_prediction_loss": AuxiliaryTerm(total, mask.sum(), weight)}


def replacement_senders(batch, groups, alive, *, mode, stage, update, seed=0):
    """Select senders per frame, independently of labels and rank partition."""
    selected = torch.zeros_like(alive)
    for row in range(alive.shape[0]):
        for time in range(alive.shape[1]):
            generator = torch.Generator().manual_seed(_frame_seed(batch, row, time, update, seed))
            chosen = sender_replacement_mask(
                groups[row, time][None].cpu(),
                alive[row, time][None].cpu(),
                mode,
                stage,
                update,
                generator=generator,
            )
            selected[row, time] = chosen[0].to(selected.device)
    return selected


def codec_distillation(reference, codec, batch, episodes, selected):
    """Frozen teacher readout is differentiable ONLY through replaced messages."""
    length = batch.loss_mask.shape[1]
    target = reference.batch(batch, episodes, length)
    local = target["local"]
    decoded, _ = codec(local)
    alive = batch.frames["alive"][:, :length]
    if selected.shape != alive.shape or selected.dtype != torch.bool:
        raise ValueError("Sender replacements must be boolean B,T,N")
    messages = torch.where((selected & alive)[..., None], decoded, local)
    predicted = reference.actor.decode_actions(
        local, messages, alive, frame_context(batch.frames, stop=length)
    )
    return predicted, action_kl_term(predicted.values, target["logits"], batch)
