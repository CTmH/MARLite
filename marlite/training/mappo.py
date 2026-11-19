"""On-policy lifecycle using the replacement sequence contracts."""

import json
import os
from pathlib import Path
import random
import shutil

import numpy as np
import torch

from marlite.data import partition_episode
from marlite.learning.mappo import prepare_ppo_targets
from marlite.util.randomness import derive_seed, preserve_rng_state, ROLLOUT_STREAM, EVALUATION_STREAM


def _rng_state():
    numpy = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (numpy[0], numpy[1].tolist(), *numpy[2:]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None,
        "mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
    }


def _restore_rng(state):
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"])
    if state["mps"] is not None:
        torch.mps.set_rng_state(state["mps"])


class SequenceMAPPOTrainer:
    """Training collection and held-out evaluation are separate operations.

    Checkpoints are legal only at complete iteration boundaries. PPO data is
    discarded there, so on-policy continuation needs no historical replay.
    Off-policy replay and C04 offline stages add their own state later.
    """

    def __init__(self, learner, collector, *, evaluation_collector=None, seed=42,
                 sequence_length=32, batch_sequences=32, ppo_epochs=4,
                 gamma=.99, gae_lambda=.95, normalize_advantages=True,
                 reward_aggregation="sum", workdir=None):
        for name, value in (("sequence_length", sequence_length), ("batch_sequences", batch_sequences),
                            ("ppo_epochs", ppo_epochs)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1:
            raise ValueError("Invalid GAE parameters")
        self.learner, self.collector = learner, collector
        self.evaluation_collector = evaluation_collector or collector
        self.seed = seed
        self.sequence_length, self.batch_sequences, self.ppo_epochs = sequence_length, batch_sequences, ppo_epochs
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.normalize_advantages, self.reward_aggregation = normalize_advantages, reward_aggregation
        self.workdir = Path(workdir) if workdir is not None else None
        if learner.backend.distributed and self.workdir is None:
            raise ValueError("Distributed training requires a shared workdir")
        self.rng = torch.Generator().manual_seed(derive_seed(seed, 30) if seed is not None else torch.seed())
        self.iterations = self.collection_round = self.evaluation_round = self.policy_version = 0
        self.env_steps = self.agent_steps = 0
        self.best_reward = -float("inf")
        self._at_boundary = True

    def _rng_context(self):
        device = self.learner.backend.device
        return preserve_rng_state(cuda_devices=[device.index] if device.type == "cuda" else [])

    def collect(self, n_episodes):
        if type(n_episodes) is not int or n_episodes < 1:
            raise ValueError("n_episodes must be positive")
        backend = self.learner.backend
        episodes, manifest = {}, None
        if backend.rank == 0:
            try:
                with self._rng_context():
                    episodes = {
                        index: self.collector.collect_episode(
                            self.learner.actor, self.learner.critic,
                            seed=derive_seed(self.seed, ROLLOUT_STREAM, self.collection_round, index),
                            policy_version=self.policy_version,
                        ) for index in range(n_episodes)
                    }
                if backend.distributed:
                    directory = self.workdir / "rollouts" / f"round_{self.collection_round}"
                    directory.mkdir(parents=True, exist_ok=True)
                    paths = {}
                    for index, episode in episodes.items():
                        path = directory / f"episode_{index}.pt"
                        torch.save(episode.state_dict(), path)
                        paths[index] = str(path.resolve())
                    manifest = {"paths": paths}
            except Exception as exc:
                if not backend.distributed:
                    raise
                manifest = {"error": repr(exc)}
        if backend.distributed:
            from marlite.data import Episode
            manifest = backend.broadcast_object(manifest)
            if "error" in manifest:
                raise RuntimeError("Collection failed on rank 0: " + manifest["error"])
            episodes = {index: Episode.from_state_dict(torch.load(path, weights_only=True, mmap=True))
                        for index, path in manifest["paths"].items()}
        self.collection_round += 1
        self.env_steps += sum(e.length for e in episodes.values())
        self.agent_steps += sum(int(e.frames["alive"][:-1].sum()) for e in episodes.values())
        return episodes

    def evaluate(self, n_episodes=8, *, stochastic=False):
        if type(n_episodes) is not int or n_episodes < 1:
            raise ValueError("n_episodes must be positive")
        # Evaluation neither adds training data nor consumes training RNG draws.
        backend = self.learner.backend
        result = None
        if backend.rank == 0:
            try:
                with self._rng_context():
                    episodes = [self.evaluation_collector.collect_episode(
                        self.learner.actor, self.learner.critic,
                        seed=derive_seed(self.seed, EVALUATION_STREAM, self.evaluation_round, i),
                        policy_version=self.policy_version, stochastic=stochastic,
                    ) for i in range(n_episodes)]
                scores = [float(e.transitions["rewards"].sum()) for e in episodes]
                result = {"reward_mean": float(np.mean(scores)),
                          "reward_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.,
                          "episode_rewards": scores, "episode_seeds": [e.seed for e in episodes]}
            except Exception as exc:
                if not backend.distributed:
                    raise
                result = {"error": repr(exc)}
        result = backend.broadcast_object(result)
        if "error" in result:
            raise RuntimeError("Evaluation failed on rank 0: " + result["error"])
        self.evaluation_round += 1
        return result

    def train_iteration(self, n_episodes=8):
        if not self._at_boundary:
            raise RuntimeError("Cannot continue a partially failed training iteration")
        self._at_boundary = False
        episodes = self.collect(n_episodes)
        prepared = prepare_ppo_targets(episodes, self.gamma, self.gae_lambda,
                                       self.normalize_advantages, self.reward_aggregation)
        sources = [source for index, episode in prepared.items()
                   for source in partition_episode(index, episode, self.sequence_length)]
        updates = []
        for _ in range(self.ppo_epochs):
            order = torch.randperm(len(sources), generator=self.rng).tolist()
            for start in range(0, len(order), self.batch_sequences):
                batch = [sources[i] for i in order[start:start + self.batch_sequences]]
                backend = self.learner.backend
                updates.append(self.learner.update(prepared, batch[backend.rank::backend.world_size]))
        if self.learner.backend.distributed:
            # All ranks finished reading this immutable on-policy dataset.
            self.learner.backend.barrier()
            if self.learner.backend.rank == 0:
                directory = self.workdir / "rollouts" / f"round_{self.collection_round - 1}"
                shutil.rmtree(directory)
            self.learner.backend.barrier()
        self.iterations += 1
        self.policy_version += 1
        self._at_boundary = True
        # Loss means are weighted by the population each objective normalizes.
        result = {}
        for key in updates[0]:
            denominator = "team_steps" if key == "critic_loss" else "agent_steps"
            if key in ("agent_steps", "team_steps"):
                result[key] = sum(m[key] for m in updates)
            else:
                total = sum(m[denominator] for m in updates)
                result[key] = sum(m[key] * m[denominator] for m in updates) / max(total, 1)
        return result

    def run(self, iterations, *, train_episodes=8, eval_episodes=8,
            eval_interval=1, checkpoint_interval=None):
        if type(iterations) is not int or iterations < 0:
            raise ValueError("iterations must be nonnegative")
        for name, value in (("eval_interval", eval_interval), ("checkpoint_interval", checkpoint_interval)):
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be positive or None")
        for _ in range(iterations):
            event = {"iteration": self.iterations + 1, "training": self.train_iteration(train_episodes)}
            if eval_interval is not None and self.iterations % eval_interval == 0:
                event["evaluation"] = self.evaluate(eval_episodes)
                if event["evaluation"]["reward_mean"] > self.best_reward:
                    self.best_reward = event["evaluation"]["reward_mean"]
                    if self.workdir is not None:
                        self.save_checkpoint(self.workdir / "checkpoints" / "best.pt")
            event.update(env_steps=self.env_steps, agent_steps=self.agent_steps,
                         actor_updates=self.learner.actor_updates, critic_updates=self.learner.critic_updates)
            if self.workdir is not None:
                if self.learner.backend.rank == 0:
                    self.workdir.mkdir(parents=True, exist_ok=True)
                    with (self.workdir / "events.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(event, allow_nan=False) + "\n")
                if checkpoint_interval is not None and self.iterations % checkpoint_interval == 0:
                    self.save_checkpoint(self.workdir / "checkpoints" / f"iteration_{self.iterations}.pt")
        return self

    def _spec(self):
        return {
            key: getattr(self, key) for key in
            ("seed", "sequence_length", "batch_sequences", "ppo_epochs", "gamma", "gae_lambda",
             "normalize_advantages", "reward_aggregation")
        } | {key: getattr(self.learner, key) for key in
             ("burn_in", "state_strategy", "clip_epsilon", "entropy_coef", "vf_coef", "max_grad_norm")}

    def save_checkpoint(self, path):
        if not self._at_boundary:
            raise RuntimeError("Save only at a completed training boundary")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "schema_version": 2, "algorithm": "sequence_mappo", "spec": self._spec(),
            "learner": self.learner.state_dict(),
            "rng_by_rank": self.learner.backend.gather_objects(_rng_state()),
            "world_size": self.learner.backend.world_size,
            "sampler_rng": self.rng.get_state(),
            "counters": {key: getattr(self, key) for key in
                         ("iterations", "collection_round", "evaluation_round", "policy_version",
                          "env_steps", "agent_steps", "best_reward")},
        }
        temporary = path.with_name(path.name + ".tmp")
        error = None
        if self.learner.backend.rank == 0:
            try:
                torch.save(state, temporary)
                os.replace(temporary, path)
            except Exception as exc:
                error = repr(exc)
            finally:
                temporary.unlink(missing_ok=True)
        error = self.learner.backend.broadcast_object(error)
        if error is not None:
            raise RuntimeError("Checkpoint save failed: " + error)

    def load_checkpoint(self, path):
        if not self._at_boundary:
            raise RuntimeError("Cannot restore during an update")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("schema_version") != 2 or state.get("algorithm") != "sequence_mappo":
            raise ValueError("Not a sequence MAPPO training checkpoint")
        if state["spec"] != self._spec():
            raise ValueError("Checkpoint training specification differs from this trainer")
        if state["world_size"] != self.learner.backend.world_size:
            raise ValueError("Exact training resume requires the same device topology")
        self.learner.load_state_dict(state["learner"])
        self.rng.set_state(state["sampler_rng"])
        for key, value in state["counters"].items():
            setattr(self, key, value)
        _restore_rng(state["rng_by_rank"][self.learner.backend.rank])
        return self
