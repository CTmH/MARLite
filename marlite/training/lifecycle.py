"""Shared collection, evaluation, logging and checkpoint lifecycle."""

import json
import os
from pathlib import Path
import random

import numpy as np
import torch

from marlite.util.randomness import (
    derive_seed,
    preserve_rng_state,
    ROLLOUT_STREAM,
    EVALUATION_STREAM,
    FINAL_TEST_STREAM,
)


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


class TrainingLifecycle:
    """Shallow lifecycle shared by PPO and replay trainers; no update formulas."""

    algorithm = None

    def __init__(self, learner, collector, *, evaluation_collector=None, seed=42, workdir=None):
        self.learner, self.collector = learner, collector
        self.evaluation_collector = evaluation_collector or collector
        self.seed = seed
        self.workdir = Path(workdir) if workdir is not None else None
        if learner.backend.distributed and self.workdir is None:
            raise ValueError("Distributed training requires a shared workdir")
        sampler_seed = derive_seed(seed, 30)
        if sampler_seed is None:
            backend = learner.backend
            sampler_seed = backend.broadcast_object(torch.seed() if backend.rank == 0 else None)
        self.rng = torch.Generator().manual_seed(sampler_seed)
        self.iterations = self.collection_round = self.evaluation_round = self.test_round = (
            self.policy_version
        ) = 0
        self.env_steps = self.agent_steps = 0
        self.best_reward = -float("inf")
        self._at_boundary = True
        self._last_evaluation = []
        self._last_collection_paths = {}

    def _publish_episodes(self, episodes, directory_name):
        backend = self.learner.backend
        self._last_collection_paths = {}
        if not backend.distributed:
            return episodes
        manifest = None
        if backend.rank == 0:
            try:
                directory = self.workdir / "rollouts" / directory_name
                directory.mkdir(parents=True, exist_ok=True)
                paths = {}
                for index, episode in episodes.items():
                    path = directory / f"episode_{index}.pt"
                    torch.save(episode.state_dict(), path)
                    paths[index] = str(path.resolve())
                manifest = {"paths": paths}
            except Exception as exc:
                manifest = {"error": repr(exc)}
        manifest = backend.broadcast_object(manifest)
        if "error" in manifest:
            raise RuntimeError("Publishing trajectories failed: " + manifest["error"])
        from marlite.data import Episode

        self._last_collection_paths = manifest["paths"]
        return {
            index: Episode.from_state_dict(torch.load(path, weights_only=True, mmap=True))
            for index, path in manifest["paths"].items()
        }

    def _checkpoint_extra(self):
        return {}

    def _load_checkpoint_extra(self, state):
        if state:
            raise ValueError("Unexpected algorithm-specific checkpoint data")

    def _rng_context(self):
        device = self.learner.backend.device
        return preserve_rng_state(cuda_devices=[device.index] if device.type == "cuda" else [])

    def collect(self, n_episodes, **collection_kwargs):
        if type(n_episodes) is not int or n_episodes < 1:
            raise ValueError("n_episodes must be positive")
        backend = self.learner.backend
        episodes, manifest = {}, None
        if backend.rank == 0:
            try:
                with self._rng_context():
                    episodes = {
                        index: self.collector.collect_episode(
                            self.learner.actor,
                            getattr(self.learner, "collection_critic", self.learner.critic),
                            seed=derive_seed(
                                self.seed, ROLLOUT_STREAM, self.collection_round, index
                            ),
                            policy_version=self.policy_version,
                            **collection_kwargs,
                        )
                        for index in range(n_episodes)
                    }
            except Exception as exc:
                if not backend.distributed:
                    raise
                manifest = {"error": repr(exc)}
        error = backend.broadcast_object(manifest)
        if error is not None:
            raise RuntimeError("Collection failed on rank 0: " + error["error"])
        episodes = self._publish_episodes(episodes, f"round_{self.collection_round}")
        self.collection_round += 1
        self.env_steps += sum(e.length for e in episodes.values())
        self.agent_steps += sum(int(e.frames["alive"][:-1].sum()) for e in episodes.values())
        return episodes

    def evaluate(self, n_episodes=8, *, stochastic=False, purpose="validation"):
        if type(n_episodes) is not int or n_episodes < 1:
            raise ValueError("n_episodes must be positive")
        if purpose not in ("validation", "test"):
            raise ValueError("Evaluation purpose must be validation or test")
        round_index = self.evaluation_round if purpose == "validation" else self.test_round
        stream = EVALUATION_STREAM if purpose == "validation" else FINAL_TEST_STREAM
        # Evaluation neither adds training data nor consumes training RNG draws.
        backend = self.learner.backend
        result = None
        if backend.rank == 0:
            try:
                with self._rng_context():
                    episodes = [
                        self.evaluation_collector.collect_episode(
                            self.learner.actor,
                            getattr(self.learner, "collection_critic", self.learner.critic),
                            seed=derive_seed(self.seed, stream, round_index, i),
                            policy_version=self.policy_version,
                            stochastic=stochastic,
                        )
                        for i in range(n_episodes)
                    ]
                self._last_evaluation = episodes if purpose == "validation" else []
                scores = [float(e.transitions["rewards"].sum()) for e in episodes]
                result = {
                    "reward_mean": float(np.mean(scores)),
                    "reward_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                    "episode_rewards": scores,
                    "episode_seeds": [e.seed for e in episodes],
                }
            except Exception as exc:
                if not backend.distributed:
                    raise
                result = {"error": repr(exc)}
        result = backend.broadcast_object(result)
        if "error" in result:
            raise RuntimeError("Evaluation failed on rank 0: " + result["error"])
        if purpose == "validation":
            self.evaluation_round += 1
        else:
            self.test_round += 1
        result["purpose"] = purpose
        return result

    def run(
        self,
        iterations,
        *,
        train_episodes=8,
        eval_episodes=8,
        eval_interval=1,
        checkpoint_interval=None,
    ):
        if type(iterations) is not int or iterations < 0:
            raise ValueError("iterations must be nonnegative")
        for name, value in (
            ("eval_interval", eval_interval),
            ("checkpoint_interval", checkpoint_interval),
        ):
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be positive or None")
        for _ in range(iterations):
            event = {
                "iteration": self.iterations + 1,
                "training": self.train_iteration(train_episodes),
            }
            if eval_interval is not None and self.iterations % eval_interval == 0:
                event["evaluation"] = self.evaluate(eval_episodes)
                self.learner.schedules.after_validation(event["evaluation"])
                if event["evaluation"]["reward_mean"] > self.best_reward:
                    self.best_reward = event["evaluation"]["reward_mean"]
                    if self.workdir is not None:
                        self.save_checkpoint(self.workdir / "checkpoints" / "best.pt")
            event.update(
                env_steps=self.env_steps,
                agent_steps=self.agent_steps,
                actor_updates=self.learner.actor_updates,
                critic_updates=self.learner.critic_updates,
            )
            if self.workdir is not None:
                if self.learner.backend.rank == 0:
                    self.workdir.mkdir(parents=True, exist_ok=True)
                    with (self.workdir / "events.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(event, allow_nan=False) + "\n")
                if checkpoint_interval is not None and self.iterations % checkpoint_interval == 0:
                    self.save_checkpoint(
                        self.workdir / "checkpoints" / f"iteration_{self.iterations}.pt"
                    )
        return self

    def save_checkpoint(self, path):
        if not self._at_boundary:
            raise RuntimeError("Save only at a completed training boundary")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "schema_version": 2,
            "algorithm": self.algorithm,
            "spec": self._spec(),
            "configuration": getattr(self, "configuration", None),
            "learner": self.learner.state_dict(),
            "extra": self._checkpoint_extra(),
            "rng_by_rank": self.learner.backend.gather_objects(_rng_state()),
            "world_size": self.learner.backend.world_size,
            "sampler_rng": self.rng.get_state(),
            "counters": {
                key: getattr(self, key)
                for key in (
                    "iterations",
                    "collection_round",
                    "evaluation_round",
                    "test_round",
                    "policy_version",
                    "env_steps",
                    "agent_steps",
                    "best_reward",
                )
            },
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
        if state.get("schema_version") != 2 or state.get("algorithm") != self.algorithm:
            raise ValueError("Checkpoint algorithm does not match this trainer")
        if state.get("configuration") != getattr(self, "configuration", None):
            raise ValueError(
                "Checkpoint experiment configuration differs; only run limits/workdir may change"
            )
        if state["spec"] != self._spec():
            raise ValueError("Checkpoint training specification differs from this trainer")
        if state["world_size"] != self.learner.backend.world_size:
            raise ValueError("Exact training resume requires the same device topology")
        self.learner.load_state_dict(state["learner"])
        self._load_checkpoint_extra(state["extra"])
        self.rng.set_state(state["sampler_rng"])
        for key, value in state["counters"].items():
            setattr(self, key, value)
        _restore_rng(state["rng_by_rank"][self.learner.backend.rank])
        return self
