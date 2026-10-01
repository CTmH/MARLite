import random

import pytest
import yaml

from marlite.rollout.rolloutmanager_config import RolloutManagerConfig
from marlite.rollout.env_retry import EnvRetryPolicy


@pytest.mark.parametrize("options", [
    {"strategy": "unknown"},
    {"max_retries": -1}, {"max_retries": 1.5}, {"max_retries": True},
    {"initial_delay": -1}, {"initial_delay": 121},
    {"backoff_factor": 0.5}, {"max_delay": float("inf")},
    {"jitter": -0.1}, {"jitter": 1.1}, {"jitter": float("nan")},
    {"initial_delay": "30"},
])
def test_invalid_policy(options):
    with pytest.raises(ValueError):
        EnvRetryPolicy(**options)


def test_backoff_bounds_and_independent_randomness(monkeypatch):
    ranges = []
    waits = []

    def uniform(self, low, high):
        ranges.append((low, high))
        return high

    monkeypatch.setattr(random.SystemRandom, "uniform", uniform)
    monkeypatch.setattr("marlite.rollout.env_retry.time.sleep", waits.append)
    state = random.getstate()
    policy = EnvRetryPolicy()
    for attempt in range(3):
        policy.wait(attempt, RuntimeError("lost"))
    for actual, expected in zip(ranges, [(20, 40), (40, 80), (80, 120)]):
        assert actual == pytest.approx(expected)
    assert waits == [40, 80, 120]
    assert random.getstate() == state
    with pytest.raises(RuntimeError, match="after recovery attempts"):
        policy.wait(3, RuntimeError("lost"))
    assert len(waits) == 3
    EnvRetryPolicy(initial_delay=0).wait(0, RuntimeError("lost"))
    assert waits[-1] == 0


@pytest.mark.parametrize("strategy,expected", [
    ("fixed", [30, 30, 30]),
    ("exponential", [30, 60, 120]),
    ("fixed_jitter", [40, 40, 40]),
    ("exponential_jitter", [40, 80, 120]),
])
def test_wait_strategies(strategy, expected, monkeypatch):
    waits = []
    monkeypatch.setattr(random.SystemRandom, "uniform", lambda self, low, high: high)
    monkeypatch.setattr("marlite.rollout.env_retry.time.sleep", waits.append)
    policy = EnvRetryPolicy(strategy=strategy)
    for attempt in range(3):
        policy.wait(attempt, RuntimeError("lost"))
    assert waits == expected


@pytest.mark.parametrize("manager_type", ["persistent-env", "multi-process"])
def test_yaml_policy_reaches_training_and_evaluation(manager_type):
    options = yaml.safe_load("""
    n_episodes: 2
    n_workers: 1
    traj_len: 3
    episode_limit: 3
    device: cpu
    env_retry:
      max_retries: 5
      strategy: fixed_jitter
      initial_delay: 10
      backoff_factor: 1.5
      max_delay: 60
      jitter: 0.2
    """)
    config = RolloutManagerConfig(
        manager_type=manager_type, worker_type=manager_type, **options,
    )
    for create in (config.create_manager, config.create_eval_manager):
        manager = create(None, b"", None, 0)
        assert manager.env_retry == EnvRetryPolicy(**options["env_retry"])
