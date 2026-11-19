"""Interrupted simulators never publish a partial trajectory or leak model state."""

import pytest
import torch

from marlite.collection.parallel import ParallelCollector
from marlite.environment.errors import EnvironmentInterrupted, RetryPolicy
from test.test_sequence_mappo import TinyParallel, make_trainer


@pytest.mark.parametrize("where", ["constructor", "reset", "step"])
def test_recovery_recollects_same_seed_and_discards_partial_episode(where):
    trainer = make_trainer()
    made, seeds = [], []
    attempts = 0

    class Interrupted(TinyParallel):
        def reset(self, seed=None):
            seeds.append(seed)
            if where == "reset":
                raise EnvironmentInterrupted("simulator lost")
            return super().reset(seed)

        def step(self, actions):
            if self.time == 1:
                raise EnvironmentInterrupted("simulator lost")
            return super().step(actions)

    def factory():
        nonlocal attempts
        attempts += 1
        if attempts == 1 and where == "constructor":
            raise ConnectionError("backend unavailable")
        env = Interrupted() if attempts == 1 else TinyParallel()
        made.append(env)
        return env

    collector = ParallelCollector(factory, max_steps=4, retry={"initial_delay": 0.0})
    recovered = collector.collect_episode(trainer.learner.actor, trainer.learner.critic, seed=77)
    reference = ParallelCollector(TinyParallel, max_steps=4).collect_episode(
        trainer.learner.actor, trainer.learner.critic, seed=77
    )
    assert attempts == 2 and all(env.closed for env in made)
    assert seeds == ([] if where == "constructor" else [77])
    for fields in ("frames", "transitions", "state_snapshots"):
        for name, tensor in getattr(reference, fields).items():
            torch.testing.assert_close(getattr(recovered, fields)[name], tensor, rtol=0, atol=0)
    assert trainer.learner.actor.training


def test_recovery_budget_is_bounded_and_programming_errors_are_not_retried():
    trainer = make_trainer()
    count = 0

    def factory():
        nonlocal count
        count += 1
        raise TimeoutError("lost")

    collector = ParallelCollector(factory, retry={"max_retries": 2, "initial_delay": 0.0})
    with pytest.raises(TimeoutError):
        collector.collect_episode(trainer.learner.actor, trainer.learner.critic, seed=4)
    assert count == 3
    collector.env_factory = lambda: (_ for _ in ()).throw(ValueError("bad config"))
    with pytest.raises(ValueError, match="bad config"):
        collector.collect_episode(trainer.learner.actor, trainer.learner.critic, seed=4)
    assert trainer.learner.actor.training
    with pytest.raises(ValueError):
        RetryPolicy(max_retries=-1)
