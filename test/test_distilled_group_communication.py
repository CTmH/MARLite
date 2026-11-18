from pathlib import Path

import pytest
import torch
import yaml

from marlite.algorithm.agents import AgentGroupConfig
from marlite.trainer.trainer_config import TrainerConfig
from marlite.util.communication_curriculum import sender_replacement_mask


@pytest.fixture
def config(tmp_path):
    example = Path(__file__).parents[1] / "examples" / "distilled_group_communication_mappo.yaml"
    config = yaml.safe_load(example.read_text())
    config["trainer"].update(train_device="cpu", workdir=str(tmp_path))
    config["trainer"]["offline"].update(
        train_episodes=1, validation_episodes=1, codec_updates=1, policy_updates=1
    )
    config["auxiliary_head"].update(
        probe_updates=1,
        label_provider={
            "type": "StateThresholdQueries",
            "indices": [0, 1], "thresholds": [0, 0],
            "queries": [[1] * 14, [0] * 14],
        },
    )
    return config


def test_hard_codec_and_receiver_isolation(config):
    actor = AgentGroupConfig(**config["agent_group"]).get_agent_group()
    observations = torch.randn(2, 8, 3, 9, 9, 29)
    padding = torch.zeros(2, 8, 3, dtype=torch.bool)
    alive = torch.ones(2, 8, dtype=torch.bool)
    groups = torch.tensor([[0, 0, 0, 0, 5, 5, 5, 5]]).repeat(2, 1)
    result = actor(observations, padding, alive, groups)
    codes = result["message_codes"]
    payload = actor.codec.pack(codes)
    assert payload.shape == (2, 8, 16)
    assert torch.equal(actor.codec.unpack(payload), codes)
    torch.testing.assert_close(actor.codec.decode(codes), result["decoded_tokens"])
    modified = observations.clone()
    modified[:, 4:] *= -2
    other = actor(modified, padding, alive, groups)
    torch.testing.assert_close(other["action_logits"][:, :4], result["action_logits"][:, :4])
    assert not any("teacher" in name or "auxiliary" in name for name in actor.state_dict())


def test_offline_stage_parameters_and_masked_online_loss(config, monkeypatch):
    trainer = TrainerConfig(config).create_trainer()
    student = trainer.eval_agent_group
    teacher = trainer.teacher_config.get_agent_group().requires_grad_(False).eval()
    trainer.teacher = teacher
    teacher_before = {key: value.clone() for key, value in teacher.state_dict().items()}

    batch = {
        "observations": torch.randn(2, 3, 8, 9, 9, 29),
        "timestep_padding_mask": torch.zeros(2, 3, dtype=torch.bool),
        "alive_mask": torch.ones(2, 3, 8, dtype=torch.bool),
        "group_indices": torch.zeros(2, 3, 8, dtype=torch.long),
        "states": torch.tensor([[[1.0, -1.0]] * 3, [[-1.0, 1.0]] * 3]),
        "avail_actions": torch.ones(2, 3, 8, 29, dtype=torch.bool),
    }
    batch["avail_actions"][..., -1] = False
    monkeypatch.setattr(trainer, "_collect_offline", lambda *_: None)
    monkeypatch.setattr(trainer, "_sample_offline", lambda *_: batch)
    run_stage = trainer._run_offline_stage
    changes = {}

    def check_stage(name, *args):
        before = {key: value.clone() for key, value in student.state_dict().items()}
        run_stage(name, *args)
        changes[name] = {
            key.split(".")[0] for key, value in student.state_dict().items()
            if not torch.equal(value, before[key])
        }

    monkeypatch.setattr(trainer, "_run_offline_stage", check_stage)
    trainer._train_offline()
    assert changes == {
        "probe": set(),
        "codec": {"codec"},
        "policy": {"fusion", "reader", "local_head", "action_head"},
    }
    for name, value in teacher.state_dict().items():
        assert torch.equal(value, teacher_before[name])
    assert all(parameter.grad is None for parameter in teacher.parameters())

    inputs, available = trainer._offline_inputs(batch)
    result = student(*inputs)
    objectives = trainer._minibatch_actor_objectives(
        result, batch, *inputs[:3], available
    )
    assert {"online_distillation_loss", "online_auxiliary_loss"} == objectives.keys()
    sum(objectives.values()).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in student.codec.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainer.student_auxiliary.parameters())
    trainer.actor_updates = trainer.online_distillation_updates
    objectives = trainer._minibatch_actor_objectives(result, batch, *inputs[:3], available)
    assert {"online_auxiliary_loss"} == objectives.keys()


@pytest.mark.parametrize("stage, count", [(0, 2), (1, 3), (2, 5)])
def test_progressive_sender_replacement(stage, count):
    groups = torch.tensor([[0, 0, 0, 1, 1, -1]])
    alive = groups >= 0
    selected = sender_replacement_mask(groups, alive, "progressive", stage, 0)
    assert selected.sum().item() == count
    assert not (selected & ~alive).any()
    assert torch.equal(sender_replacement_mask(groups, alive, "all", stage, 0), alive)
