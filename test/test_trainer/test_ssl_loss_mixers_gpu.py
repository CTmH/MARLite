"""Optional real two-GPU worker tests; only rollout IPC is replaced in-process."""

from copy import deepcopy
import importlib
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from marlite.trainer import TrainerConfig
from marlite.util.serialization import get_state_dict, serialize_to_buffer


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two GPUs required")
@pytest.mark.parametrize("mixer", ["weighted_sum", "pit_loss", "ema_grad_norm", "pcgrad"])
@pytest.mark.parametrize("family,mode", [
    ("mappo", "ae"), ("mappo", "vae"),
    ("qmix", "ae"), ("qmix", "vae"), ("graph", "vae"),
])
def test_real_ssl_workers(tmp_path, family, mode, mixer):
    fixtures = {
        "mappo": ("test_ae_gc_mappo_trainer", "TestAEGCMAPPOTrainer"),
        "qmix": ("test_ae_group_consensus_trainer", "TestAEGroupConsensusTrainer"),
        "graph": ("test_self_supervised_gnn", "TestVAEGraphQMIXBattle"),
    }
    module, name = fixtures[family]
    fixture = getattr(importlib.import_module(f"test.test_trainer.{module}"), name)()
    fixture.setUp()
    config = deepcopy(fixture.config)
    if family == "qmix" and mode == "vae":
        fixture = importlib.import_module(
            "test.test_trainer.test_vae_group_consensus_trainer"
        ).TestGroupConsensusTrainer()
        fixture.setUp()
        config = deepcopy(fixture.config)
    if family == "mappo" and mode == "vae":
        config["agent_group"]["consensus_mode"] = "vae"
        config["agent_group"]["models"]["model_0"]["group_estimate_feature_extractor"]["layers"][-1]["out_features"] = 128
        config["trainer"].update(consensus_mode="vae", kl_on_agent=True,
                                  kl_on_group=True, kl_divergence_weight=.001)
    if family != "graph":
        config["agent_group"]["enable_rl_grad_to_group_estimate"] = True
    config["trainer"].update(
        train_device=["cuda:0", "cuda:1"], n_workers=0, workdir=str(tmp_path),
        loss_mixer={"type": mixer, "weights": [1., .2]}, seed=42,
    )
    warmup = "warmup_iterations" if family == "mappo" else "warmup_epochs"
    config["trainer"][warmup] = 1
    config["rollout"]["device"] = "cpu"
    config["self_supervised_learning"]["data_constructor"]["n_workers"] = 0
    trainer = TrainerConfig(config).create_trainer()
    try:
        rollout = importlib.import_module("marlite.rollout.multiprocess_rollout")
        params = serialize_to_buffer(get_state_dict(trainer.eval_agent_group))
        with patch.object(rollout, "SharedMemory", return_value=SimpleNamespace(buf=params, close=lambda: None)):
            episode = rollout.multiprocess_rollout(
                trainer.env_config, trainer.agent_group_config, ("in-process", len(params)),
                rnn_traj_len=config["replay_buffer"]["traj_len"], episode_limit=4,
                epsilon=1. if family == "mappo" else .5,
            )
        trainer.replaybuffer.add_episode(episode)
        initial_mixer = deepcopy(trainer.loss_mixer.state_dict())
        before = deepcopy(get_state_dict(trainer.eval_agent_group))
        # Warmup must not initialize/update mixer statistics on either GPU.
        metrics = trainer.learn(4, 3, times=1)
        trainer._sync_eval_params_from_workers()
        for key, value in initial_mixer.items():
            torch.testing.assert_close(trainer.loss_mixer.state_dict()[key], value)
        trainer.current_epoch = 1
        trainer._sync_params_to_workers()
        metrics = trainer.learn(7, 3, times=1)
        assert all(np.isfinite(value) for value in metrics.values())
        trainer._sync_eval_params_from_workers()
        after = get_state_dict(trainer.eval_agent_group)
        parameter_names = dict(trainer.eval_agent_group.named_parameters())
        assert any(not torch.equal(before[key].cpu(), after[key].cpu()) for key in parameter_names)
        if mixer == "pit_loss":
            assert trainer.loss_mixer.step > 0
        if mixer == "ema_grad_norm":
            assert trainer.loss_mixer.initialized
        # Read both ranks directly: averaging their parameters first could hide drift.
        group = trainer.worker_group
        for queue in group.cmd_queues:
            queue.put("SYNC_TO_MAIN")
        states = [queue.get() for queue in group.param_queues]
        for name in ("eval_agent_group", "eval_critic", "ssl_model", "loss_mixer"):
            # BatchNorm running moments are local data statistics, not parameters
            # covered by the current worker gradient-reduction protocol.
            parameter_names = dict(getattr(trainer, name).named_parameters())
            for key, value in states[0][name].items():
                if name == "loss_mixer" or key in parameter_names:
                    torch.testing.assert_close(value, states[1][name][key], msg=f"{name}.{key}")
        trainer.save_current_model("gpu-mixer")
        trainer.load_checkpoint("gpu-mixer")
        for key, value in states[0]["loss_mixer"].items():
            torch.testing.assert_close(trainer.loss_mixer.state_dict()[key].cpu(), value.cpu())
        print(f"{family}/{mode}/{mixer}: {metrics}")
    finally:
        trainer.worker_group.shutdown()
