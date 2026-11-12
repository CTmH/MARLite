from types import SimpleNamespace

import pytest
import torch

from marlite.trainer.g2anet_mappo_trainer import GraphMAPPOTrainer
from marlite.trainer.mappo_trainer import MAPPOTrainer
from marlite.trainer.self_supervised_mappo_trainer import SelfSupervisedMAPPOTrainer


@pytest.fixture
def trainer_factory(tmp_path):
    def make(cls, *, empty_at=None, budget=False):
        trainer = cls.__new__(cls)
        trainer.checkpointdir = str(tmp_path)
        trainer.eval_metric_list = ["reward"]
        trainer.best_metrics = {}
        trainer.eval_agent_group = torch.nn.Linear(1, 1)
        trainer.eval_critic = torch.nn.Linear(1, 1)
        trainer.ssl_model = torch.nn.Linear(1, 1)
        trainer.loss_mixer = torch.nn.Linear(1, 1)
        trainer.agent_optimizer = trainer.critic_optimizer = trainer.ssl_optimizer = (
            SimpleNamespace(param_groups=[{"lr": 0.01}])
        )
        trainer.lr_scheduler = trainer.agent_lr_scheduler = trainer.ssl_lr_scheduler = None
        trainer.sample_mode = "ratio"
        trainer.sample_ratio = SimpleNamespace(get_value=lambda _: 1)
        trainer.replaybuffer_config = SimpleNamespace(
            create_replaybuffer=lambda: SimpleNamespace(buffer=[])
        )
        trainer.replaybuffer = trainer.replaybuffer_config.create_replaybuffer()
        trainer.rolloutmanager_config = SimpleNamespace(
            step_budget=10 if budget else None, collection_exhausted=False
        )
        events = []

        def evaluate():
            events.append("evaluate")
            next_iteration = events.count("evaluate")
            if next_iteration != empty_at:
                trainer.replaybuffer.buffer.append(object())
            return {"reward": {"mean": 0.0}}

        def learn(**kwargs):
            events.append("learn")
            if budget:
                trainer.rolloutmanager_config.collection_exhausted = True
            return {"loss": 0.0}

        @torch.no_grad()
        def sync():
            events.append("sync")
            for model in models(trainer):
                model.weight.fill_(trainer.current_epoch + 1)

        trainer.evaluate = evaluate
        trainer.learn = learn
        trainer._sync_params_to_workers = lambda: None
        trainer._sync_eval_params_from_workers = sync
        trainer.worker_group = SimpleNamespace(
            average_eval_params=lambda: events.append("average"), shutdown=lambda: None
        )
        trainer._prepare_rollout = lambda _: None
        trainer.save_intermediate_results = lambda *_: None
        trainer.save_best_model = lambda: None
        return trainer, events

    return make


def models(trainer):
    result = [trainer.eval_agent_group, trainer.eval_critic]
    if isinstance(trainer, SelfSupervisedMAPPOTrainer):
        result.extend((trainer.ssl_model, trainer.loss_mixer))
    return result


@pytest.mark.parametrize("cls", [MAPPOTrainer, GraphMAPPOTrainer, SelfSupervisedMAPPOTrainer])
def test_periodic_checkpoints_save_synchronized_weights(cls, trainer_factory, tmp_path):
    trainer, events = trainer_factory(cls)
    save = trainer.save_current_model

    def checked_save(name):
        assert events[-3:] == ["learn", "average", "sync"]
        return save(name)

    trainer.save_current_model = checked_save
    trainer.train(5, 1.0, checkpoint_interval=2)
    paths = sorted(tmp_path.glob("checkpoint_*"), key=lambda p: int(p.name.rsplit("_", 1)[1]))
    assert [int(p.name.rsplit("_", 1)[1]) for p in paths] == [2, 4]
    assert events.count("learn") == 5
    for path, iteration in zip(paths, [2, 4], strict=True):
        loaded, _ = trainer_factory(cls)
        loaded.load_checkpoint(path.name)
        for model in models(loaded):
            torch.testing.assert_close(model.weight, torch.full_like(model.weight, iteration))


@pytest.mark.parametrize("interval", [0, -1, True, 1.5, "2"])
def test_invalid_interval_fails_before_collection(interval, trainer_factory):
    trainer, events = trainer_factory(MAPPOTrainer)
    with pytest.raises(ValueError, match="checkpoint_interval"):
        trainer.train(1, 1.0, checkpoint_interval=interval)
    assert not events


def test_default_and_empty_iteration_do_not_save(trainer_factory, tmp_path):
    trainer, _ = trainer_factory(MAPPOTrainer)
    trainer.train(2, 1.0)
    trainer, events = trainer_factory(MAPPOTrainer, empty_at=2)
    trainer.train(2, 1.0, checkpoint_interval=2)
    assert events.count("learn") == 1
    assert not list(tmp_path.iterdir())


def test_budget_exit_keeps_checkpoint(trainer_factory, tmp_path):
    trainer, events = trainer_factory(MAPPOTrainer, budget=True)
    trainer.train(3, 1.0, checkpoint_interval=1)
    assert events == ["evaluate", "learn", "average", "sync"]
    assert len(list(tmp_path.glob("checkpoint_*_1/agent/agent.pth"))) == 1
