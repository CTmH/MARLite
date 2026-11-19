"""On-policy iteration policy composed with shared lifecycle and PPO learner."""

import shutil

import torch

from marlite.data import partition_episode
from marlite.learning.mappo import prepare_ppo_targets
from .lifecycle import TrainingLifecycle
from .metrics import summarize_updates


class SequenceMAPPOTrainer(TrainingLifecycle):
    """Freeze complete rollout labels, then train continuous chunks for PPO epochs."""

    algorithm = "sequence_mappo"

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
        super().__init__(learner, collector, evaluation_collector=evaluation_collector,
                         seed=seed, workdir=workdir)
        self.sequence_length, self.batch_sequences, self.ppo_epochs = sequence_length, batch_sequences, ppo_epochs
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.normalize_advantages, self.reward_aggregation = normalize_advantages, reward_aggregation

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
        return summarize_updates(updates, default_count="agent_steps",
                                 count_overrides={"critic_loss": "team_steps"})

    def _spec(self):
        return {
            key: getattr(self, key) for key in
            ("seed", "sequence_length", "batch_sequences", "ppo_epochs", "gamma", "gae_lambda",
             "normalize_advantages", "reward_aggregation")
        } | {key: getattr(self.learner, key) for key in
             ("burn_in", "state_strategy", "clip_epsilon", "entropy_coef", "vf_coef", "max_grad_norm")}
