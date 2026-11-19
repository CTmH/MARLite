import random
import numpy as np
import pytest
import torch
from torch import nn
from marlite.util.randomness import (
    configure_randomness,
    derive_seed,
    seed_everything,
    preserve_rng_state,
)
from marlite.util.initialization_diagnostics import inspect_initial_outputs
from marlite.algorithm.sequence_agent_group import PolicyOutput


def draws():
    return random.random(), np.random.rand(), torch.rand(3)


def assert_draws_equal(a, b):
    assert a[:2] == b[:2]
    torch.testing.assert_close(a[2], b[2])


def test_seeds_are_stable_and_streams_distinct():
    seeds = [derive_seed(42, stream, rank) for stream in range(7) for rank in range(4)]
    assert len(set(seeds)) == len(seeds)
    assert seeds == [derive_seed(42, stream, rank) for stream in range(7) for rank in range(4)]
    configure_randomness(42)
    first = draws()
    configure_randomness(42)
    assert_draws_equal(first, draws())
    assert derive_seed(None, 1) is None


@pytest.mark.parametrize("seed", [-1, True, 1.2, "42"])
def test_invalid_seed(seed):
    with pytest.raises(ValueError):
        configure_randomness(seed)


def test_default_configuration_is_a_noop_and_strict_mode_is_explicit():
    seed_everything(22)
    expected = draws()
    seed_everything(22)
    configure_randomness()
    assert_draws_equal(expected, draws())
    try:
        configure_randomness(22, True)
        assert torch.are_deterministic_algorithms_enabled()
    finally:
        configure_randomness(22, False)


def test_rng_preservation_even_on_failure():
    seed_everything(4)
    expected = draws()
    seed_everything(4)
    with pytest.raises(RuntimeError):
        with preserve_rng_state():
            draws()
            raise RuntimeError()
    assert_draws_equal(expected, draws())


def test_diagnostic_preserves_rng_modes_and_caches():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = nn.Dropout(0.9)
            self.cache = []

        def forward(self, x):
            self.cache.append(1)
            draws()
            return PolicyOutput(
                self.dropout(x),
                x,
                {},
                "logits",
                {
                    "agent_mu": torch.ones(1, 2, 2),
                    "group_mu": torch.tensor([[[1.0, 1.0], [99.0, 99.0]]]),
                    "group_valid": torch.tensor([[True, False]]),
                },
            )

    model = Model().train()
    seed_everything(9)
    expected = draws()
    seed_everything(9)
    metrics = inspect_initial_outputs(
        model,
        torch.zeros(1, 2, 3),
        action_mask=torch.tensor([[[True, True, False], [False, False, False]]]),
        active_mask=torch.tensor([[True, False]]),
    )
    assert metrics["policy_normalized_entropy"] == pytest.approx(1.0)
    assert metrics["group_mu_mean"] == 1.0
    assert model.training and model.dropout.training and not model.cache
    assert_draws_equal(expected, draws())
