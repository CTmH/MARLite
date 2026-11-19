"""Episode replay with uniform or explicitly named reward-weighted chunks."""

from collections import OrderedDict
import math

import torch

from .episode import Episode
from .sequence import partition_episode


class EpisodeReplay:
    """Capacity is measured in complete episodes; slot IDs never get reused.

    Reward weighting is an experimental sampling distribution, NOT TD-error PER
    or an importance-corrected estimator. Sampling uses the caller's generator.
    """

    def __init__(self, capacity=128, sampling="uniform", alpha=.8):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("Replay capacity must be positive")
        if sampling not in ("uniform", "reward_weighted"):
            raise ValueError("Unsupported replay sampling distribution")
        if not math.isfinite(alpha) or alpha < 0:
            raise ValueError("Replay weighting exponent must be finite/nonnegative")
        self.capacity, self.sampling, self.alpha = capacity, sampling, alpha
        self.episodes = OrderedDict()
        self.next_id = 0

    def add(self, episode):
        episode.validate()
        key = self.next_id
        self.next_id += 1
        self.episodes[key] = episode
        while len(self.episodes) > self.capacity:
            self.episodes.popitem(last=False)
        return key

    def sample(self, count, length, generator):
        if type(count) is not int or count < 1:
            raise ValueError("Sample count must be positive")
        sources = [s for index, episode in self.episodes.items()
                   for s in partition_episode(index, episode, length, burn_in=0)]
        if not sources:
            raise ValueError("Cannot sample an empty replay")
        weights = torch.ones(len(sources), dtype=torch.float64)
        if self.sampling == "reward_weighted":
            rewards = {key: float(ep.transitions["rewards"].sum()) for key, ep in self.episodes.items()}
            low, high = min(rewards.values()), max(rewards.values())
            if high > low:
                weights = torch.tensor([((rewards[s.episode_id] - low) / (high - low) + 1e-6) ** self.alpha
                                        for s in sources], dtype=torch.float64)
        chosen = torch.multinomial(weights, count, replacement=True, generator=generator).tolist()
        return [sources[i] for i in chosen]

    def state_dict(self):
        return {"capacity": self.capacity, "sampling": self.sampling, "alpha": self.alpha,
                "next_id": self.next_id,
                "episodes": {key: ep.state_dict() for key, ep in self.episodes.items()}}

    def load_state_dict(self, state):
        if any(state[k] != getattr(self, k) for k in ("capacity", "sampling", "alpha")):
            raise ValueError("Replay specification mismatch")
        self.episodes = OrderedDict((key, Episode.from_state_dict(ep)) for key, ep in state["episodes"].items())
        self.next_id = state["next_id"]
        if len(self.episodes) > self.capacity or any(k >= self.next_id for k in self.episodes):
            raise ValueError("Invalid replay checkpoint IDs/capacity")
