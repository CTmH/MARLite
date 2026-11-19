"""Tensor trajectory summaries and value-free held-out policy evaluation."""

from pathlib import Path

import numpy as np
import torch

from marlite.util.randomness import derive_seed, preserve_rng_state, FINAL_TEST_STREAM


def statistics(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
    }


def summarize_episodes(episodes):
    episodes = tuple(episodes)
    if not episodes:
        raise ValueError("Analysis requires at least one complete episode")
    actions, rewards, lengths, sent_bytes, edges, metrics = {}, [], [], [], [], {}
    per_agent_rewards = {}
    for episode in episodes:
        episode.validate()
        rewards.append(float(episode.transitions["rewards"].sum()))
        lengths.append(episode.length)
        alive = episode.frames["alive"][:-1]
        chosen = episode.transitions["actions"]
        for index, agent in enumerate(episode.agent_ids):
            distribution = actions.setdefault(agent, {})
            for action in chosen[:, index][alive[:, index]].tolist():
                distribution[action] = distribution.get(action, 0) + 1
            per_agent_rewards.setdefault(agent, []).append(
                float(episode.transitions["rewards"][:, index].sum())
            )
        if "behavior.communication_bytes" in episode.transitions:
            sent_bytes.append(float(episode.transitions["behavior.communication_bytes"].sum()))
        adjacency = episode.transitions.get("behavior.adjacency")
        if adjacency is not None:
            edges.extend((adjacency != 0).sum((-1, -2)).tolist())
        for name, value in episode.metadata.get("metrics", {}).items():
            metrics.setdefault(name, []).append(float(value))
    return {
        "episode_reward": statistics(rewards),
        "episode_length": statistics(lengths),
        "agent_reward": {name: statistics(values) for name, values in per_agent_rewards.items()},
        "action_counts": actions,
        "communication_bytes_per_episode": statistics(sent_bytes),
        "active_edges_per_step": statistics(edges),
        "environment_metrics": {name: statistics(values) for name, values in metrics.items()},
    }


class PolicyEvaluator:
    """Evaluate only the actor; no teacher, optimizer, replay or central critic.

    This also permits new map sizes whose centralized training state has a
    different shape, provided the actor's legal observation/action schema fits.
    Evaluation uses a separate stream, exports actual tensor episodes if asked,
    and never invokes a training lifecycle or adds held-out data to replay.
    """

    def __init__(self, config, actor, collector, environment_spec, backend):
        self.config, self.actor, self.collector = config, actor, collector
        self.environment_spec, self.backend = environment_spec, backend
        self.checkpoint = None

    def load_checkpoint(self, path):
        path = Path(path).expanduser()
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        expected = {
            "mappo": "sequence_mappo",
            "qmix": "sequence_qmix",
            "qplex": "sequence_qplex",
            "qtran": "sequence_qtran",
            "c04": "sequence_distilled_group_mappo",
        }[self.config["recipe"]]
        if state.get("schema_version") != 2 or state.get("algorithm") != expected:
            raise ValueError("Checkpoint algorithm differs from the evaluation recipe")
        previous = state.get("configuration")
        if previous is not None:

            def architecture(config):
                def clean(value):
                    if isinstance(value, dict):
                        return {
                            key: clean(item)
                            for key, item in value.items()
                            if not ("type" in value and key in ("checkpoint", "initialization"))
                        }
                    if isinstance(value, list):
                        return [clean(item) for item in value]
                    return value

                defaults = config.get("defaults", {})
                return {
                    "recipe": config["recipe"],
                    "actor": clean(config.get("actor")),
                    "hidden_dim": defaults.get("hidden_dim", 64),
                    "heads": defaults.get("heads", 4),
                }

            if architecture(previous) != architecture(self.config):
                raise ValueError(
                    "Evaluation actor configuration differs from the trained architecture"
                )
        self.actor.load_state_dict(state["learner"]["actor"], strict=True)
        self.checkpoint = str(path.resolve())

    def evaluate(
        self, n_episodes=32, *, seed=None, round_index=0, stochastic=False, trajectories=None
    ):
        if (
            type(n_episodes) is not int
            or n_episodes < 1
            or type(round_index) is not int
            or round_index < 0
        ):
            raise ValueError(
                "Evaluation counts/round must be nonnegative integers, with at least one episode"
            )
        seed = self.config.get("seed", 42) if seed is None else seed
        report = None
        if self.backend.rank == 0:
            try:
                devices = [self.backend.device.index] if self.backend.device.type == "cuda" else []
                with preserve_rng_state(cuda_devices=devices):
                    episodes = [
                        self.collector.collect_episode(
                            self.actor,
                            seed=derive_seed(seed, FINAL_TEST_STREAM, round_index, index),
                            stochastic=stochastic,
                            record_values=False,
                        )
                        for index in range(n_episodes)
                    ]
                report = summarize_episodes(episodes)
                report.update(
                    checkpoint=self.checkpoint,
                    evaluation_seed=seed,
                    evaluation_round=round_index,
                    episode_seeds=[episode.seed for episode in episodes],
                    stochastic=stochastic,
                )
                if trajectories is not None:
                    destination = Path(trajectories).expanduser()
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "schema_version": 2,
                            "episodes": [episode.state_dict() for episode in episodes],
                        },
                        destination,
                    )
            except Exception as exc:
                report = {"error": repr(exc)}
        report = self.backend.broadcast_object(report)
        if "error" in report:
            raise RuntimeError("Policy evaluation failed: " + report["error"])
        return report
