"""Replay training policy composed with the same lifecycle and device backend."""

from copy import deepcopy
from pathlib import Path
import math

import torch

from .lifecycle import TrainingLifecycle


def _weights(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


class SequenceValueTrainer(TrainingLifecycle):
    """Train complete episodes from replay; evaluation stays held out by default.

    Optional rollback deliberately restores weights only, retains optimizer
    momentum/counters, and hard-reanchors targets. Optional validation replay is
    explicit and never applied to purpose='test' evaluation.
    """

    def __init__(self, learner, collector, replay, *, evaluation_collector=None,
                 seed=42, workdir=None, sequence_length=32, batch_sequences=32,
                 updates_per_iteration=4, epsilon_start=1., epsilon_end=.05,
                 epsilon_decay_steps=50000, rollback_interval=None,
                 cache_tolerance=.03, validation_replay_ratio=0.):
        super().__init__(learner, collector, evaluation_collector=evaluation_collector,
                         seed=seed, workdir=workdir)
        for name, value in (("sequence_length", sequence_length), ("batch_sequences", batch_sequences),
                            ("updates_per_iteration", updates_per_iteration),
                            ("epsilon_decay_steps", epsilon_decay_steps)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be positive")
        if rollback_interval is not None and (type(rollback_interval) is not int or rollback_interval < 1):
            raise ValueError("rollback_interval must be positive or None")
        if not 0 <= epsilon_end <= epsilon_start <= 1:
            raise ValueError("Invalid epsilon schedule")
        if not 0 <= validation_replay_ratio <= 1 or not math.isfinite(cache_tolerance) or cache_tolerance < 0:
            raise ValueError("Invalid optional training policy")
        self.replay = replay
        self.sequence_length, self.batch_sequences = sequence_length, batch_sequences
        self.updates_per_iteration = updates_per_iteration
        self.epsilon_start, self.epsilon_end, self.epsilon_decay_steps = epsilon_start, epsilon_end, epsilon_decay_steps
        self.rollback_interval, self.cache_tolerance = rollback_interval, cache_tolerance
        self.validation_replay_ratio = validation_replay_ratio
        self.rollback_weights = self._snapshot_weights() if self.rollback_interval is not None else None
        self.replay_files = {}
        self.algorithm = "sequence_" + type(learner.objective).__name__.removesuffix("Objective").lower()

    def _snapshot_weights(self):
        return {name: _weights(getattr(self.learner, name)) for name in ("actor", "critic", "objective")}

    def _add_to_replay(self, episodes):
        for index, episode in episodes.items():
            key = self.replay.add(episode)
            if index in self._last_collection_paths:
                self.replay_files[key] = self._last_collection_paths[index]

    def _release_evicted(self):
        backend = self.learner.backend
        backend.barrier()
        dead = set(self.replay_files) - self.replay.episodes.keys()
        for key in dead:
            path = Path(self.replay_files.pop(key))
            if backend.rank == 0:
                path.unlink(missing_ok=True)
                if path.parent.exists() and not any(path.parent.iterdir()):
                    path.parent.rmdir()
        backend.barrier()

    def train_iteration(self, n_episodes=8):
        if not self._at_boundary:
            raise RuntimeError("Cannot continue a partially failed training iteration")
        self._at_boundary = False
        rollback = self.rollback_interval is not None and self.iterations > 0 and self.iterations % self.rollback_interval == 0
        if rollback:
            for name, weights in self.rollback_weights.items():
                getattr(self.learner, name).load_state_dict(weights)
            self.learner.refresh_targets(hard=True)
        fraction = min(1., self.env_steps / self.epsilon_decay_steps)
        epsilon = self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)
        self._add_to_replay(self.collect(n_episodes, epsilon=epsilon))
        metrics = []
        for _ in range(self.updates_per_iteration):
            sources = self.replay.sample(self.batch_sequences, self.sequence_length, self.rng)
            backend = self.learner.backend
            metrics.append(self.learner.update(self.replay.episodes, sources[backend.rank::backend.world_size]))
        self._release_evicted()
        self.iterations += 1
        self.policy_version += 1
        self._at_boundary = True
        result = {}
        for key in metrics[0]:
            if key.endswith("_count") or key == "team_steps":
                result[key] = sum(m[key] for m in metrics)
            else:
                count_key = key + "_count"
                count = sum(m[count_key] for m in metrics)
                result[key] = sum(m[key] * m[count_key] for m in metrics) / max(count, 1)
        result.update(epsilon=epsilon, rollback_applied=rollback,
                      replay_episodes=len(self.replay.episodes))
        return result

    def evaluate(self, n_episodes=8, *, stochastic=False, purpose="validation"):
        result = super().evaluate(n_episodes, stochastic=stochastic, purpose=purpose)
        if purpose == "validation":
            previous = self.best_reward
            if self.rollback_interval is not None and (
                not math.isfinite(previous)
                or result["reward_mean"] >= previous - self.cache_tolerance * max(abs(previous), 1.)
            ):
                self.rollback_weights = self._snapshot_weights()
            count = int(n_episodes * self.validation_replay_ratio)
            if count:
                selected = torch.randperm(n_episodes, generator=self.rng)[:count].tolist()
                episodes = {i: self._last_evaluation[i] for i in selected} if self.learner.backend.rank == 0 else {}
                published = self._publish_episodes(episodes, f"validation_{self.evaluation_round}")
                self._add_to_replay(published)
                self._release_evicted()
            result["validation_replay_episodes"] = count
        self._last_evaluation = []
        return result

    def _spec(self):
        result = {key: getattr(self, key) for key in (
            "seed", "sequence_length", "batch_sequences", "updates_per_iteration",
            "epsilon_start", "epsilon_end", "epsilon_decay_steps", "rollback_interval",
            "cache_tolerance", "validation_replay_ratio")}
        result.update({key: getattr(self.learner, key) for key in (
            "gamma", "n_steps", "reward_aggregation", "burn_in", "state_strategy",
            "target_mode", "target_interval", "target_tau", "max_grad_norm")})
        result["replay"] = {key: getattr(self.replay, key) for key in ("capacity", "sampling", "alpha")}
        if self.learner.objective.value_network is not None:
            result["constraints"] = {key: getattr(self.learner.objective, key) for key in
                                     ("lambda_opt", "lambda_nopt", "mask_optimal_samples")}
        return result

    def _checkpoint_extra(self):
        # Embed replay tensors so a checkpoint remains valid after rolling files expire.
        return {"replay": self.replay.state_dict(), "rollback_weights": self.rollback_weights}

    def _load_checkpoint_extra(self, state):
        self.replay.load_state_dict(state["replay"])
        self.rollback_weights = deepcopy(state["rollback_weights"])
        self.replay_files = {}  # Restored tensors are owned by the checkpoint, not old files.
