import torch
import pytest
from marlite.algorithm.model.group_communication import (
    ScalarMessageCodec,
    GroupSetFusion,
    RelationalNeuralProcess,
)
from marlite.algorithm.agents.group_communication_agent_group import (
    GroupCommunicationMAPPOAgentGroup,
    GroupCommunicationQMIXAgentGroup,
)


@pytest.mark.parametrize("bits", [1, 2, 4, 8])
def test_codec_payload(bits):
    codec = ScalarMessageCodec(8, 16, bits)
    x = torch.randn(2, 3, 8, requires_grad=True)
    decoded, codes = codec(x)
    packed = codec.pack(codes)
    assert packed.shape[-1] == 16 * bits // 8
    assert torch.equal(codec.unpack(packed), codes)
    decoded.square().sum().backward()
    assert x.grad.abs().sum() > 0
    with pytest.raises(ValueError):
        codec.pack(codes.long())


def test_set_fusion_invariance_and_dead_members():
    torch.manual_seed(1)
    f = GroupSetFusion(8, 2, 3)
    x = torch.randn(2, 4, 8, requires_grad=True)
    groups = torch.tensor([[0, 0, 2, -1], [-1] * 4])
    alive = groups >= 0
    y, valid = f(x, groups, alive)
    perm = torch.tensor([2, 0, 3, 1])
    yp, vp = f(x[:, perm], groups[:, perm], alive[:, perm])
    torch.testing.assert_close(y, yp)
    assert torch.equal(valid, vp)
    assert y[1].count_nonzero() == 0
    y.square().sum().backward()
    assert x.grad[0, 3].count_nonzero() == 0


@pytest.mark.parametrize(
    "agent_class", [GroupCommunicationMAPPOAgentGroup, GroupCommunicationQMIXAgentGroup]
)
def test_np_training_does_not_expose_labels_to_actions(agent_class):
    torch.manual_seed(3)
    agent = agent_class(
        dict.fromkeys(["a", "b", "c"], "shared"),
        6,
        5,
        {"type": "Fixed", "group_ids": [0, 0, 0]},
        hidden_dim=8,
        heads=2,
        slots=2,
        message_dim=4,
        use_np=True,
        hyper_rank=2,
    )
    obs = torch.randn(2, 3, 2, 6)
    pad = torch.zeros(2, 3, 2, dtype=torch.bool)
    alive = torch.ones(2, 3, dtype=torch.bool)
    groups = torch.zeros(2, 2, 3, dtype=torch.long)
    batch = {
        "group_indices": groups,
        "relation_queries": torch.randn(2, 3, 4, 4),
        "relation_labels": torch.zeros(2, 3, 4),
        "relation_mask": torch.ones(2, 3, 4, dtype=torch.bool),
    }
    first = agent.training_forward(obs, pad, alive, batch)
    batch["relation_labels"].fill_(1)
    second = agent.training_forward(obs, pad, alive, batch)
    torch.testing.assert_close(first[agent.output_key], second[agent.output_key])
    (second[agent.output_key].square().mean() + second["auxiliary_loss"]).backward()
    assert agent.np.posterior.weight.grad.abs().sum() > 0
    assert agent.codec.encoder.weight.grad.abs().sum() > 0
    assert agent.np.prior.weight.grad.abs().sum() > 0
    alive.zero_()
    dead = agent(obs, pad, alive, groups[:, -1] + 99)
    assert dead[agent.output_key].count_nonzero() == 0


def test_np_empty_targets():
    np = RelationalNeuralProcess(8, 4, 3)
    loss = np.loss(
        torch.randn(1, 2, 8),
        torch.randn(1, 2, 3, 4),
        torch.full((1, 2, 3), float("nan")),
        torch.zeros(1, 2, 3, dtype=torch.bool),
    )
    assert loss.item() == 0
    loss.backward()


@pytest.mark.parametrize("algorithm", ["MAPPO", "QMIX", "QPLEX"])
@pytest.mark.parametrize("manager", ["multi-process", "persistent-env"])
def test_real_rollout_and_training(algorithm, manager, tmp_path):
    import importlib

    fixture = importlib.import_module(
        "test.test_trainer.test_" + algorithm.lower() + "_trainer"
    )
    cls = getattr(
        fixture,
        {
            "MAPPO": "TestMAPPOTrainer",
            "QMIX": "TestQMixTrainer",
            "QPLEX": "TestQPLEXTrainer",
        }[algorithm],
    )
    template = cls()
    template.setUp()
    config = template.config
    old = config["agent_group"]
    config["agent_group"] = dict(
        type="GroupCommunication" + ("QMIX" if algorithm == "QPLEX" else algorithm),
        agent_list=old["agent_list"],
        optimizer=old["optimizer"],
        observation_dim=18,
        action_dim=5,
        hidden_dim=8,
        heads=2,
        slots=2,
        message_dim=4,
        use_np=True,
        group_builder={"type": "Fixed", "group_ids": [0, 0, 0]},
        relation_targets={
            "indices": [0, 1],
            "queries": [[1, 0, 0, 0], [0, 1, 0, 0]],
            "thresholds": [0, 0],
        },
    )
    profile = "group_communication_" + (
        "qmix" if algorithm == "QPLEX" else algorithm.lower()
    )
    config["rollout"].update(
        manager_type=manager,
        worker_type=manager,
        episode_limit=3,
        traj_len=2,
        required_attrs=profile,
    )
    config["replay_buffer"].update(traj_len=2, required_attrs=profile)
    config["trainer"].update(workdir=str(tmp_path), train_device="cpu")
    from marlite.trainer import TrainerConfig

    trainer = TrainerConfig(config).create_trainer()
    try:
        before = trainer.eval_agent_group.codec.encoder.weight.detach().clone()
        trainer.collect_experience(0.2)
        episode = next(
            ep for ep in trainer.replaybuffer.episode_buffer.values() if ep is not None
        )
        assert len(episode["message_codes"]) == len(episode["observations"])
        from marlite.util.trajectory_dataset import trajectory_collate_fn
        from marlite.util.communication_training import agent_training_forward

        data = trainer.replaybuffer.sample(3)
        batch = trajectory_collate_fn(list(data))
        padding = batch["timestep_padding_mask"].unsqueeze(1).expand(-1, 3, -1)
        alive = batch["alive_mask"][:, -1].bool()
        observations = batch["observations"].transpose(1, 2).float()
        with torch.no_grad():
            ret = agent_training_forward(
                trainer.eval_agent_group, observations, padding, alive, batch
            )
            torch.testing.assert_close(
                ret["message_codes"], batch["message_codes"][:, -1]
            )
            if algorithm == "MAPPO":
                from marlite.util.action_distribution import masked_categorical

                logp = masked_categorical(
                    ret["action_logits"], active_mask=alive
                ).log_prob(batch["actions"][:, -1])
                torch.testing.assert_close(logp, batch["all_log_probs"][:, -1].float())
        posterior_before = trainer.eval_agent_group.np.posterior.weight.detach().clone()
        result = trainer.learn(sample_size=3, batch_size=3, times=1)
        assert not torch.equal(
            posterior_before, trainer.eval_agent_group.np.posterior.weight
        )
        # Exercise the actual worker loss on CPU, with only distributed collectives bypassed.
        module = importlib.import_module(
            "marlite.trainer.trainer_worker." + algorithm.lower() + "_worker"
        )
        worker_cls = getattr(module, algorithm + "Worker")
        worker = worker_cls.__new__(worker_cls)
        for name in (
            "eval_agent_group",
            "eval_critic",
            "agent_optimizer",
            "critic_optimizer",
            "gamma",
            "max_grad_norm",
            "reward_aggr_mode",
            "clip_epsilon",
            "entropy_coef",
            "vf_coef",
            "target_agent_group",
            "target_critic",
        ):
            if hasattr(trainer, name):
                setattr(worker, name, getattr(trainer, name))
        worker.device = torch.device("cpu")
        worker._reduce_agent_gradients = lambda: None
        worker._reduce_critic_gradients = lambda: None
        worker.reduce_gradients = lambda: None
        worker._update_target_after_batch = lambda: None
        worker_posterior = worker.eval_agent_group.np.posterior.weight.detach().clone()
        previous_updates = worker.eval_agent_group.distillation_updates.item()
        assert torch.isfinite(torch.tensor(worker.train_step(batch)["loss"]))
        assert not torch.equal(
            worker_posterior, worker.eval_agent_group.np.posterior.weight
        )
        assert (
            worker.eval_agent_group.distillation_updates.item() == previous_updates + 1
        )
        if algorithm in ("QMIX", "QPLEX"):
            trainer._update_target_after_batch()
            for key, value in trainer.eval_agent_group.state_dict().items():
                torch.testing.assert_close(
                    value, trainer.target_agent_group.state_dict()[key]
                )
        from marlite.replaybuffer.prioritized_replaybuffer import (
            PrioritizedReplayBuffer,
        )

        replay = PrioritizedReplayBuffer(2, 2, "episode_reward", required_attrs=profile)
        replay.add_episode(episode)
        assert "message_codes" in trajectory_collate_fn(list(replay.sample(2)))
        assert torch.isfinite(torch.tensor(result["loss"]))
        assert not torch.equal(before, trainer.eval_agent_group.codec.encoder.weight)
        trainer.save_current_model("roundtrip")
        trainer.load_checkpoint("roundtrip")
    finally:
        pass


@pytest.mark.parametrize(
    "agent_class", [GroupCommunicationMAPPOAgentGroup, GroupCommunicationQMIXAgentGroup]
)
def test_frozen_teacher_input_gradient_and_curriculum_restore(agent_class):
    from copy import deepcopy
    from marlite.util.communication_curriculum import CommunicationCurriculum

    student = agent_class(
        dict.fromkeys(["a", "b", "c"], "shared"),
        6,
        5,
        {"type": "Fixed", "group_ids": [0, 0, 0]},
        hidden_dim=8,
        heads=2,
        slots=2,
        message_dim=4,
    )
    teacher = deepcopy(student)
    teacher.message_mode = "raw"
    c = CommunicationCurriculum(student, teacher, (1, 1, 1))
    obs = torch.randn(2, 3, 2, 6)
    padding = torch.zeros(2, 3, 2, dtype=torch.bool)
    alive = torch.ones(2, 3, dtype=torch.bool)
    groups = torch.zeros(2, 3, dtype=torch.long)
    optimizer = torch.optim.Adam(student.parameters(), lr=0.001)
    for expected in (1, 2, 3):
        assert c.replacement_mask(groups, alive).sum(-1).tolist() == [
            expected,
            expected,
        ]
        c.step(optimizer, obs, padding, alive, groups)
        assert student.codec.encoder.weight.grad.abs().sum() > 0
        assert all(p.grad is None for p in c.teacher.parameters())
    restored = CommunicationCurriculum(deepcopy(student), teacher, (10, 10, 10))
    restored.load_state_dict(c.state_dict())
    assert restored.stage == 3
    with pytest.raises(RuntimeError):
        restored.loss(obs, padding, alive, groups)


def test_protocol_retransmissions_and_budget():
    from marlite.util.communication_protocol import MessagePacket, snapshot_packets

    a = MessagePacket(0, 3, 1, 5, 6, bytes(16))
    b = MessagePacket(1, 3, 1, 7, 8, bytes(16))
    assert len(a.encode()) == 25
    assert MessagePacket.decode(a.encode(), 16) == a
    assert snapshot_packets([b, a, a], 3, 1, [0, 1]) == [a, b]
    for messages in (
        [a],
        [a, MessagePacket(0, 3, 1, 5, 6, bytes(8))],
        [a, MessagePacket(1, 4, 1, 7, 8, bytes(16))],
    ):
        with pytest.raises(ValueError):
            snapshot_packets(messages, 3, 1, [0, 1])


def test_online_teacher_schedule_and_restore(tmp_path):
    from marlite.util.communication_training import advance_communication_step

    kwargs = dict(
        agent_model_dict={"a": "shared", "b": "shared"},
        observation_dim=6,
        action_dim=5,
        group_builder={"type": "Fixed", "group_ids": [0, 0]},
        hidden_dim=8,
        heads=2,
        slots=2,
        message_dim=4,
    )
    teacher = GroupCommunicationMAPPOAgentGroup(**kwargs, message_mode="raw")
    path = tmp_path / "teacher.pth"
    torch.save(teacher.state_dict(), path)
    student = GroupCommunicationMAPPOAgentGroup(
        **kwargs, teacher_checkpoint=path, distillation_steps=1
    )
    obs = torch.randn(1, 2, 2, 6)
    pad = torch.zeros(1, 2, 2, dtype=torch.bool)
    alive = torch.ones(1, 2, dtype=torch.bool)
    batch = {"group_indices": torch.zeros(1, 2, 2, dtype=torch.long)}
    ret = student.training_forward(obs, pad, alive, batch)
    ret["auxiliary_loss"].backward()
    assert all(p.grad is None for p in student.teacher.parameters())
    assert student.codec.encoder.weight.grad.abs().sum() > 0
    advance_communication_step(student)
    assert "auxiliary_loss" not in student.training_forward(obs, pad, alive, batch)
    restored = GroupCommunicationMAPPOAgentGroup(
        **kwargs, teacher_checkpoint=path, distillation_steps=1
    )
    restored.load_state_dict(student.state_dict())
    assert restored.distillation_updates.item() == 1
    restored.train()
    assert not restored.teacher.training


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS not available"
)
def test_mps_forward_backward_np():
    agent = GroupCommunicationMAPPOAgentGroup(
        {"a": "shared", "b": "shared"},
        6,
        5,
        {"type": "Fixed", "group_ids": [0, 0]},
        hidden_dim=8,
        heads=2,
        slots=2,
        message_dim=4,
        use_np=True,
    ).to("mps")
    obs = torch.randn(1, 2, 2, 6, device="mps")
    pad = torch.zeros(1, 2, 2, dtype=torch.bool, device="mps")
    alive = torch.ones(1, 2, dtype=torch.bool, device="mps")
    ret = agent(obs, pad, alive, torch.zeros(1, 2, dtype=torch.long, device="mps"))
    context = ret["relation_context"]
    queries = torch.randn(1, 2, 3, 4, device="mps")
    loss = agent.np.loss(
        context,
        queries,
        torch.zeros(1, 2, 3, device="mps"),
        ret["group_valid"].unsqueeze(-1).expand(-1, -1, 3),
    )
    (loss + ret["action_logits"].square().mean()).backward()
    assert torch.isfinite(agent.codec.encoder.weight.grad).all()
    samples = agent.np.predict(context, queries, samples=4)
    assert samples.shape == (4, 1, 2, 3)
    assert torch.isfinite(samples).all()
