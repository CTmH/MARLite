"""Small CPU/GPU regressions, including QTRAN's actual outer training loop."""

from copy import deepcopy
import importlib
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from marlite.trainer import TrainerConfig
from marlite.util.serialization import get_state_dict, serialize_to_buffer


@pytest.mark.parametrize("device", ["cpu", "cuda:0", "dual"])
@pytest.mark.parametrize("family", ["qtran", "qplex", "qplex_ema", "g2anet_mappo", "g2anet_qmix", "mappo", "mappo_no_value"])
def test_priority_trainer_updates(tmp_path, device, family):
    if device != "cpu" and torch.cuda.device_count() < (2 if device == "dual" else 1):
        pytest.skip("Required GPUs unavailable")
    qplex_ema = family == "qplex_ema"
    if qplex_ema:
        family = "qplex"
    fixtures = {
        "qtran": ("qtran_trainer", "TestQTRANTrainer"),
        "qplex": ("qplex_trainer", "TestQPLEXTrainer"),
        "g2anet_mappo": ("g2anet_mappo_trainer", "TestGraphMAPPOTrainer"),
        "g2anet_qmix": ("graph_qmix_trainer_g2anet", "TestG2ANetQMIXTrainer"),
        "mappo": ("mappo_trainer", "TestMAPPOTrainer"),
        "mappo_no_value": ("mappo_trainer", "TestMAPPOTrainer"),
    }
    module, name = fixtures[family]
    fixture = getattr(importlib.import_module(f"test.test_trainer.test_{module}"), name)()
    fixture.setUp()
    config = deepcopy(fixture.config)
    config["trainer"].update(train_device=["cuda:0", "cuda:1"] if device == "dual" else device,
                             n_workers=0, workdir=str(tmp_path), seed=15)
    if family == "mappo_no_value":
        config["trainer"]["vf_coef"] = 0.
    if qplex_ema:
        config["trainer"].update(n_steps=3, target_update_mode="ema",
                                 target_update_tau=.2, target_update_interval=2)
        config["critic"]["transformation"]["attend_reg_coef"] = .01
    config["rollout"]["device"] = "cpu"
    trainer = TrainerConfig(config).create_trainer()
    try:
        params = serialize_to_buffer(get_state_dict(trainer.eval_agent_group))
        rollout = importlib.import_module("marlite.rollout.multiprocess_rollout")
        with patch.object(rollout, "SharedMemory", return_value=SimpleNamespace(buf=params, close=lambda: None)):
            episode = rollout.multiprocess_rollout(trainer.env_config, trainer.agent_group_config,
                ("in-process", len(params)), rnn_traj_len=config["replay_buffer"]["traj_len"],
                episode_limit=4, epsilon=1. if "mappo" in family else .5)
        trainer.replaybuffer.add_episode(episode)
        trainer.replaybuffer.add_episode(deepcopy(episode))
        before = deepcopy(get_state_dict(trainer.eval_agent_group))
        critic_before = deepcopy(get_state_dict(trainer.eval_critic))
        if family in ("qtran", "qplex"):
            evaluated = []
            def evaluate():
                evaluated.append(deepcopy(get_state_dict(trainer.eval_agent_group)))
                return {metric: {"mean": 1.} for metric in trainer.eval_metric_list}
            if family == "qtran":
                trainer.v_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(trainer.v_optimizer, .5)
            with patch.object(trainer, "collect_experience"), patch.object(trainer, "evaluate", side_effect=evaluate):
                metrics = trainer.train(epochs=2, target_first_metric=2., batch_size=2)
            assert len(evaluated) == 2
            for snapshot in evaluated:
                assert any(not torch.equal(before[key].cpu(), snapshot[key].cpu())
                           for key in dict(trainer.eval_agent_group.named_parameters()))
            # Best checkpoints must use the same layout as load_checkpoint.
            trainer.load_checkpoint("best")
        else:
            metrics = trainer.learn(4, 2, times=1)
            trainer._sync_eval_params_from_workers()
        assert all(np.isfinite(value) for value in metrics.values())
        after = get_state_dict(trainer.eval_agent_group)
        assert any(not torch.equal(before[key].cpu(), after[key].cpu())
                   for key in dict(trainer.eval_agent_group.named_parameters()))
        if family == "mappo_no_value":
            for key, value in trainer.eval_critic.state_dict().items():
                torch.testing.assert_close(value.cpu(), critic_before[key].cpu())
        if trainer.worker_group is not None:
            for command in trainer.worker_group.cmd_queues:
                command.put("SYNC_TO_MAIN")
            states = [queue.get() for queue in trainer.worker_group.param_queues]
            names = ["eval_agent_group", "eval_critic"]
            if family == "qtran":
                names += ["eval_v_net"]
            if family in ("qtran", "qplex"):
                names += ["target_agent_group", "target_critic"]
            for name in names:
                for key in dict(getattr(trainer, name).named_parameters()):
                    torch.testing.assert_close(states[0][name][key], states[1][name][key])
                    torch.testing.assert_close(states[0][name][key], get_state_dict(getattr(trainer, name))[key].cpu())
    finally:
        if trainer.worker_group is not None:
            trainer.worker_group.shutdown()


@pytest.mark.parametrize("size", [0, 1, 3, 5])
def test_qplex_rejects_uneven_batches_before_dispatch(size):
    from marlite.trainer.trainer_worker_group.qplex_worker_group import QPLEXWorkerGroup

    # No processes or queues: validation must happen before issuing commands.
    group = object.__new__(QPLEXWorkerGroup)
    group.world_size = 2
    with pytest.raises(ValueError, match="positive multiple"):
        group.train_step({"states": torch.empty(size, 3, 54)})
