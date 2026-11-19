"""Schedulers follow successful optimizer updates and survive exact resume."""

from copy import deepcopy
import sys
from types import SimpleNamespace

import pytest
import torch

from marlite.configuration.recipe import build_experiment
from marlite.configuration.registry import ConfigurationError
from marlite.learning.schedules import OptimizerSchedules, ScheduleBinding
from test.test_configuration import minimal_config
from test.test_sequence_mappo import TinyParallel, assert_tree_equal


def scheduler_config():
    return {"scheduler": {"type": "StepLR", "params": {"step_size": 1, "gamma": 0.5}}}


@pytest.mark.parametrize("algorithm", ["mappo", "qmix", "qplex", "qtran", "c04"])
def test_all_recipes_schedule_actual_updates_and_resume(monkeypatch, tmp_path, algorithm):
    monkeypatch.setitem(
        sys.modules, "schedule_test_env", SimpleNamespace(parallel_env=TinyParallel)
    )
    config = minimal_config("schedule_test_env", algorithm)
    config["schedulers"] = {"actor": scheduler_config(), "critic": scheduler_config()}
    if algorithm == "qtran":
        config["schedulers"]["objective"] = scheduler_config()
    if algorithm == "c04":
        config["teacher"]["schedulers"] = {"actor": scheduler_config()}
        config["offline"]["schedulers"] = {
            "codec": scheduler_config(),
            "policy": scheduler_config(),
        }
    experiment = build_experiment(config, workdir=tmp_path / "initial")
    trainer = experiment.trainer
    initial_lr = trainer.learner.actor_optimizer.param_groups[0]["lr"]
    if algorithm == "c04":
        experiment.teacher_trainer.train_iteration(2)
        assert experiment.teacher_trainer.learner.schedules.updates["actor"] > 0
    trainer.train_iteration(2)
    count = trainer.learner.actor_updates
    assert count > 0
    assert trainer.learner.actor_optimizer.param_groups[0]["lr"] == initial_lr * 0.5**count
    if algorithm == "c04":
        assert trainer.stage_schedules.updates == {"codec": 1, "policy": 1}
    checkpoint = tmp_path / "resume.pt"
    trainer.save_checkpoint(checkpoint)
    expected_metrics = trainer.train_iteration(2)
    expected = deepcopy(trainer.learner.state_dict())
    restored = build_experiment(config, workdir=tmp_path / "restored").trainer
    restored.load_checkpoint(checkpoint)
    assert expected_metrics == restored.train_iteration(2)
    assert_tree_equal(expected, restored.learner.state_dict())
    if algorithm == "c04":
        assert_tree_equal(
            trainer.stage_schedules.state_dict(), restored.stage_schedules.state_dict()
        )


def test_plateau_advances_on_validation_after_updates_only(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "schedule_test_env", SimpleNamespace(parallel_env=TinyParallel)
    )
    config = minimal_config("schedule_test_env")
    config["schedulers"] = {
        "actor": {
            "scheduler": {"type": "ReduceLROnPlateau", "params": {"mode": "max", "patience": 0}},
            "event": "validation",
            "metric": "reward_mean",
        }
    }
    trainer = build_experiment(config).trainer
    schedules = trainer.learner.schedules
    before = deepcopy(schedules.state_dict())
    trainer.evaluate(1, purpose="test")
    trainer.evaluate(1, purpose="validation")
    assert_tree_equal(before, schedules.state_dict())
    trainer.train_iteration(2)
    schedules.after_validation({"reward_mean": 1.0})
    initial_lr = trainer.learner.actor_optimizer.param_groups[0]["lr"]
    schedules.after_validation({"reward_mean": -1.0})  # No intervening update.
    assert trainer.learner.actor_optimizer.param_groups[0]["lr"] == initial_lr
    trainer.train_iteration(2)
    schedules.after_validation({"reward_mean": -1.0})
    assert trainer.learner.actor_optimizer.param_groups[0]["lr"] == initial_lr * 0.1
    # The ordinary lifecycle calls the schedule AFTER its validation event.
    trainer.run(1, train_episodes=1, eval_episodes=1)
    assert schedules.validated["actor"] == schedules.updates["actor"]


def test_invalid_scheduler_metric_and_absent_optimizer_are_rejected(monkeypatch):
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    with pytest.raises(ValueError, match="metric"):
        ScheduleBinding(torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer))
    with pytest.raises(ValueError, match="metric"):
        ScheduleBinding(torch.optim.lr_scheduler.StepLR(optimizer, 1), metric="loss")
    schedules = OptimizerSchedules(
        {
            "actor": ScheduleBinding(
                torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer), metric="loss"
            )
        }
    )
    with pytest.raises(ValueError, match="Missing"):
        schedules.after_update("actor", {})
    monkeypatch.setitem(
        sys.modules, "schedule_test_env", SimpleNamespace(parallel_env=TinyParallel)
    )
    config = minimal_config("schedule_test_env")
    config["schedulers"] = {"auxiliary": scheduler_config()}
    with pytest.raises(ConfigurationError, match="absent"):
        build_experiment(config)
