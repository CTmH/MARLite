"""Contiguous sequence slicing without materializing overlapping histories."""

from dataclasses import dataclass

import torch
from torch import Tensor

from .episode import Episode


@dataclass(frozen=True)
class SequenceSlice:
    episode_id: int
    start: int
    length: int
    burn_in: int = 0
    future: int = 0

    def validate(self, episode):
        if any(type(v) is not int for v in (self.start, self.length, self.burn_in, self.future)):
            raise TypeError("Sequence positions and lengths must be integers")
        if self.start < 0 or self.start >= episode.length or self.length < 1:
            raise ValueError("Invalid learning interval")
        if self.burn_in < 0 or self.future < 0:
            raise ValueError("Burn-in and future horizon cannot be negative")


@dataclass
class SequenceBatch:
    frames: dict[str, Tensor]
    transitions: dict[str, Tensor]
    filled: Tensor
    loss_mask: Tensor
    bootstrap_mask: Tensor
    trace_mask: Tensor
    sources: tuple[SequenceSlice, ...]
    frame_start: Tensor
    learning_start: Tensor

    @property
    def actor_mask(self):
        return self.loss_mask[..., None] & self.frames["alive"][:, :-1]

    def to(self, device):
        return SequenceBatch(
            {k: v.to(device) for k, v in self.frames.items()},
            {k: v.to(device) for k, v in self.transitions.items()},
            *(getattr(self, key).to(device) for key in
              ("filled", "loss_mask", "bootstrap_mask", "trace_mask")),
            self.sources, self.frame_start.to(device), self.learning_start.to(device),
        )


def partition_episode(episode_id, episode, length=32, burn_in=8, future=0):
    """Each transition is a learning target exactly once; short tails are kept."""
    if type(length) is not int or length < 1:
        raise ValueError("Sequence length must be positive")
    return [SequenceSlice(episode_id, start, length, burn_in, future)
            for start in range(0, episode.length, length)]


def collate_sequences(episodes: dict[int, Episode], sources):
    """Right-pad segments; distinguish warmup, learning, and bootstrap tail.

    Episodes must be validated at collection/store boundaries, not rescanned
    in full for every minibatch. Recurrent-state snapshots are read separately
    for initialization and never copied into each GPU training window.
    Agent slots/specifications must match. Ragged communication data must be
    tensorized by its producer.
    """
    sources = tuple(sources)
    if not sources:
        raise ValueError("Cannot collate an empty sequence batch")
    segments = []
    for source in sources:
        episode = episodes[source.episode_id]
        source.validate(episode)
        begin = max(0, source.start - source.burn_in)
        learn_end = min(episode.length, source.start + source.length)
        end = min(episode.length, learn_end + source.future)
        segments.append((episode, begin, learn_end, end))
    first = segments[0][0]
    for episode, *_ in segments:
        if episode.agent_ids != first.agent_ids:
            raise ValueError("A batch requires the same ordered agent slots")
        for fields in ("frames", "transitions"):
            actual, expected = getattr(episode, fields), getattr(first, fields)
            if actual.keys() != expected.keys() or any(
                actual[k].shape[1:] != expected[k].shape[1:] or actual[k].dtype != expected[k].dtype
                for k in expected
            ):
                raise ValueError("A batch requires matching field specifications")
    size = max(end - begin for _, begin, _, end in segments)
    frames = {k: v.new_zeros(len(sources), size + 1, *v.shape[1:])
              for k, v in first.frames.items()
              if not k.startswith(("actor_state.", "critic_state."))}
    transitions = {k: v.new_zeros(len(sources), size, *v.shape[1:]) for k, v in first.transitions.items()}
    filled = torch.zeros(len(sources), size, dtype=torch.bool)
    learning, bootstrap, trace = (torch.zeros_like(filled) for _ in range(3))
    starts, offsets = [], []
    for row, (source, segment) in enumerate(zip(sources, segments)):
        episode, begin, learn_end, end = segment
        count = end - begin
        for key, value in frames.items():
            value[row, :count + 1] = episode.frames[key][begin:end + 1]
        for key, value in transitions.items():
            value[row, :count] = episode.transitions[key][begin:end]
        filled[row, :count] = True
        learning[row, source.start - begin:learn_end - begin] = True
        boot, cont = episode.return_masks()
        bootstrap[row, :count] = boot[begin:end]
        trace[row, :count] = cont[begin:end]
        # A slice cannot recurse into an advantage that is not in this batch.
        trace[row, count - 1] = False
        starts.append(begin)
        offsets.append(source.start - begin)
    return SequenceBatch(frames, transitions, filled, learning, bootstrap, trace,
                         sources, torch.tensor(starts), torch.tensor(offsets))
