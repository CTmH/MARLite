import random
import importlib
from copy import deepcopy
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch import nn

from marlite.util.randomness import (
    configure_randomness, derive_seed, seed_everything, preserve_rng_state,
    WORKER_STREAM, ROLLOUT_STREAM, EVALUATION_STREAM,
)
from marlite.util.initialization_diagnostics import inspect_initial_outputs
from marlite.rollout.rolloutmanager_config import RolloutManagerConfig
from marlite.rollout.rolloutmanager import RolloutManager
from marlite.trainer.trainer_worker_group.base_worker_group import worker_loop


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


def test_manager_round_and_episode_seed_assignment():
    config = RolloutManagerConfig(manager_type="persistent-env", worker_type="persistent-env", n_episodes=7)
    config.set_randomness(42)
    first = config._randomness_kwargs()
    second = config._randomness_kwargs()
    evaluation = config._randomness_kwargs(evaluation=True)
    assert first["seed"] == derive_seed(42, ROLLOUT_STREAM, 0)
    assert second["seed"] == derive_seed(42, ROLLOUT_STREAM, 1)
    assert evaluation["seed"] == derive_seed(42, EVALUATION_STREAM, 0)
    manager = RolloutManager(None, None, None, b"", 7, 2, 10, 0., "cpu", **first)
    first_seeds = manager._episode_seeds()
    assert first_seeds == [derive_seed(first["seed"], 0, i) for i in range(7)]
    assert first_seeds != manager._episode_seeds()


def test_worker_rank_stream_does_not_depend_on_model_construction():
    results = []

    class Worker:
        def __init__(self, count):
            torch.rand(count)

        def handle_command(self, *args):
            results.append(draws())
            return False

    for rank, count in [(0, 1), (0, 200), (1, 1)]:
        worker_loop(0, 0, rank, 2, "unused", Worker, {"count": count},
                    None, None, None, Mock(), None, Mock(), seed=42)
    assert_draws_equal(results[0], results[1])
    assert not torch.equal(results[0][2], results[2][2])
    seed_everything(derive_seed(42, WORKER_STREAM, 0))
    assert_draws_equal(results[0], draws())


def test_diagnostic_preserves_rng_modes_and_caches():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = nn.Dropout(.9)
            self.cache = []

        def forward(self, x):
            self.cache.append(1)
            draws()
            return {"action_logits": self.dropout(x),
                    "agent_mu": torch.ones(1, 2, 2),
                    "group_mu": torch.tensor([[[1., 1.], [99., 99.]]]),
                    "group_indices": torch.tensor([[0, -1]])}

    model = Model().train()
    seed_everything(9)
    expected = draws()
    seed_everything(9)
    metrics = inspect_initial_outputs(model, torch.zeros(1, 2, 3),
        action_mask=torch.tensor([[[True, True, False], [False, False, False]]]),
        active_mask=torch.tensor([[True, False]]))
    assert metrics["policy_normalized_entropy"] == pytest.approx(1.)
    assert metrics["group_mu_mean"] == 1.
    assert model.training and model.dropout.training and not model.cache
    assert_draws_equal(expected, draws())


@pytest.mark.parametrize("onpolicy", [True, False])
def test_actual_trainer_seed_and_configured_policy_head(tmp_path, onpolicy):
    from marlite.trainer.trainer_config import TrainerConfig
    from test.test_trainer.test_mappo_trainer import TestMAPPOTrainer
    from test.test_trainer.test_qmix_trainer import TestQMixTrainer

    fixture = TestMAPPOTrainer() if onpolicy else TestQMixTrainer()
    fixture.setUp()
    config = deepcopy(fixture.config)
    config["trainer"].update(seed=41, n_workers=0, train_device="cpu")
    decoder = config["agent_group"]["models"]["model1"]["decoder"]
    if onpolicy:
        # The fixture decoder is a single Linear inside CustomModel.
        decoder["initialization"] = {"scheme": "orthogonal", "gain": .01, "bias": 0}
    parameters = []
    samples = []
    losses = []
    for i, seed in enumerate([41, 41, 42]):
        config["trainer"].update(seed=seed, workdir=str(tmp_path / str(i)))
        trainer = TrainerConfig(config).create_trainer()
        parameters.append(torch.cat([p.detach().flatten() for p in trainer.eval_agent_group.parameters()]))
        samples.append(draws())
        assert trainer.rolloutmanager_config.seed == seed
        from test.test_trainer.test_multistep_training import small_episode
        trainer.replaybuffer.add_episode(small_episode(terminal=False))
        losses.append(trainer.learn(4, 2, times=1))
    torch.testing.assert_close(parameters[0], parameters[1])
    assert not torch.equal(parameters[0], parameters[2])
    assert_draws_equal(samples[0], samples[1])
    assert losses[0] == pytest.approx(losses[1], abs=1e-7)


@pytest.mark.parametrize("module_name,class_name,extra_model", [
    ("test_qtran_trainer", "TestQTRANTrainer", "eval_v_net"),
    ("test_vaegc_mappo_trainer", "TestVAEGCMAPPOTrainer", "ssl_model"),
    ("test_vae_group_consensus_trainer", "TestGroupConsensusTrainer", "ssl_model"),
])
def test_networks_built_before_base_trainer_are_seeded(tmp_path, module_name, class_name, extra_model):
    from marlite.trainer.trainer_config import TrainerConfig

    fixture = getattr(importlib.import_module("test.test_trainer." + module_name), class_name)()
    fixture.setUp()
    config = deepcopy(fixture.config)
    config["trainer"].update(seed=51, n_workers=0, train_device="cpu")
    parameters = []
    for i in range(2):
        config["trainer"]["workdir"] = str(tmp_path / str(i))
        trainer = TrainerConfig(config).create_trainer()
        parameters.append(torch.cat([
            p.detach().flatten() for p in getattr(trainer, extra_model).parameters()
        ]))
        torch.rand(113)  # Construction must not depend on the caller's RNG state.
    torch.testing.assert_close(parameters[0], parameters[1])


def test_consensus_head_configuration_and_real_diagnostic(tmp_path):
    from marlite.trainer.trainer_config import TrainerConfig
    from test.test_trainer.test_vaegc_mappo_trainer import TestVAEGCMAPPOTrainer

    fixture = TestVAEGCMAPPOTrainer()
    fixture.setUp()
    config = deepcopy(fixture.config)
    config["trainer"].update(seed=51, n_workers=0, train_device="cpu", workdir=str(tmp_path))
    config["agent_group"]["models"]["model1"]["group_estimate_feature_extractor"] = {
        "model_type": "Custom", "layers": [{"type": "Linear", "in_features": 18, "out_features": 18}],
        "initialization": {"overrides": [{
            "module": "model.0", "scheme": "normal", "std": .01, "bias": 0,
            "output_slices": [{"start": 9, "stop": 18, "scheme": "normal", "std": .001, "bias": 0}],
        }]},
    }
    trainer = TrainerConfig(config).create_trainer()
    active = torch.ones(1, 3, dtype=torch.bool)
    metrics = inspect_initial_outputs(
        trainer.eval_agent_group, torch.ones(1, 3, 2, 18), torch.ones(1, 54),
        torch.zeros(1, 3, 2, dtype=torch.bool), active, torch.zeros(1, 3, dtype=torch.long),
        action_mask=torch.ones(1, 3, 5, dtype=torch.bool), active_mask=active,
    )
    assert abs(metrics["agent_log_var_mean"]) < .02
    assert metrics["agent_log_var_std"] < .02
    assert all(np.isfinite(value) for value in metrics.values())
