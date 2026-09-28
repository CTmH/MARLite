import numpy as np
import pytest
from gymnasium.spaces import Box
from types import SimpleNamespace
from smac_pettingzoo.env.smacv2 import SC2EpisodeInterrupted

from marlite.environment import EnvConfig
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
    wrapper = SMACWrapper(failed, env_factory=lambda: recovered)

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
    module = SimpleNamespace(smacv2=SimpleNamespace(parallel_env=factory))
    monkeypatch.setattr(
        "marlite.environment.env_config.importlib.import_module",
        lambda name: module,
    )

    wrapper = EnvConfig("fake", "smacv2", wrapper={"type": "smac"}).create_env()
    wrapper.reset(seed=13)

    assert environments[0].closed
    assert environments[1].seeds == [13]
