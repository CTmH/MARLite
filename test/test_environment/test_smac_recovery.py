import numpy as np
import pytest
from gymnasium.spaces import Box
from types import SimpleNamespace
from marlite.environment.smac_wrapper import SC2_RECOVERABLE_ERRORS

SC2EpisodeInterrupted = SC2_RECOVERABLE_ERRORS[0]

import sys
from marlite.configuration.environment import EnvironmentFactory
from marlite.configuration.catalog import default_registry
from marlite.environment.smac_wrapper import SMACWrapper


class FakeSC2Env:
    possible_agents = ["marine_0"]

    def __init__(self, fail_reset=False):
        self.fail_reset = fail_reset
        self.seeds = []
        self.closed = False

    def state_space(self, agent):
        return Box(-1, 1, shape=(2,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.seeds.append(seed)
        if self.fail_reset:
            raise SC2EpisodeInterrupted("lost")
        return {"marine_0": np.zeros(2)}, {"marine_0": {}}

    def close(self):
        self.closed = True


def test_smac_wrapper_recreates_failed_env_with_same_seed():
    failed = FakeSC2Env(fail_reset=True)
    recovered = FakeSC2Env()
    wrapper = SMACWrapper(failed, env_factory=lambda: recovered, reset_retries=2)

    observations, _ = wrapper.reset(seed=42)

    assert list(observations) == ["marine_0"]
    assert failed.closed
    assert failed.seeds == recovered.seeds == [42]
    assert wrapper.env is recovered


def test_smac_wrapper_stops_after_bounded_retries():
    environments = [FakeSC2Env(fail_reset=True) for _ in range(3)]
    factory = iter(environments[1:]).__next__
    wrapper = SMACWrapper(environments[0], env_factory=factory, reset_retries=2)

    with pytest.raises(SC2EpisodeInterrupted):
        wrapper.reset(seed=7)

    assert all(env.closed and env.seeds == [7] for env in environments)


def test_env_config_supplies_smac_recreation_factory(monkeypatch):
    environments = [FakeSC2Env(fail_reset=True), FakeSC2Env()]
    factory = iter(environments).__next__
    monkeypatch.setitem(sys.modules, "smac_recovery_fake", SimpleNamespace(parallel_env=factory))
    wrapper = EnvironmentFactory(
        {
            "module": "smac_recovery_fake",
            "wrappers": [{"type": "SMACWrapper", "params": {"reset_retries": 2}}],
        },
        default_registry(),
    )()
    wrapper.reset(seed=13)

    assert environments[0].closed
    assert environments[1].seeds == [13]


def test_smac_wrapper_defers_recovery_to_rollout_by_default():
    failed = FakeSC2Env(fail_reset=True)

    def unexpected_restart():
        pytest.fail("Wrapper must not restart before rollout backoff")

    wrapper = SMACWrapper(failed, env_factory=unexpected_restart)
    with pytest.raises(SC2EpisodeInterrupted):
        wrapper.reset(seed=42)
    assert failed.closed
    assert failed.seeds == [42]
