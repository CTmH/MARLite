import numpy as np
import pytest
from magent2.environments.complementary_pursuit_v1 import parallel_env as NativeEnv
from marlite.environment.complementary_pursuit import parallel_env
from marlite.environment.complementary_events import (
    attack_outcomes,
    ATTACK_OFFSETS,
    local_escape_action,
)


class Scene(NativeEnv):
    def __init__(self, modes, damage=10):
        self.modes = modes
        self.positions = [(7, 5), (5, 7), (9, 7)][: len(modes)]
        super().__init__(map_size=15, effective_damage=damage)

    def generate_map(self):
        evader, pursuer = self.env.get_handles()
        self.env.add_agents(evader, "custom", pos=[(7, 7, 3), (12, 12, 3)])
        self.env.add_agents(pursuer, "custom", pos=[(*p, 3) for p in self.positions])


@pytest.mark.parametrize("modes", ["A", "AA", "AB", "AAB"])
@pytest.mark.parametrize("damage", [1.0, 5.0, 10.0])
def test_attack_queries_match_native(modes, damage):
    env = Scene(modes, damage)
    env.reset(seed=42)
    positions = {f"pursuer_{i}": p for i, p in enumerate(env.positions)}
    actions = {
        a: 5 + 12 * (modes[i] == "B") + ATTACK_OFFSETS.index((7 - p[0], 7 - p[1]))
        for i, (a, p) in enumerate(positions.items())
    }
    events, missed = attack_outcomes(
        positions, {0: (7, 7, 5), 1: (12, 12, 5)}, actions, effective_damage=damage
    )
    _, rewards, terms, _, _ = env.step(actions)
    assert events[0]["captured"] == terms["evader_0"]
    assert events[0]["effective"] == ("A" in modes and "B" in modes)
    assert not missed
    pool = 0.2 if events[0]["effective"] else 0.02
    pool += 1.0 if events[0]["captured"] else 0.0
    for agent in actions:
        critical = events[0]["critical"]
        expected = pool / len(actions) + (
            0.2 / len(critical) if agent in critical else 0.0
        )
        assert rewards[agent] == pytest.approx(expected)
    env.close()


@pytest.mark.parametrize("policy", ["random", "local_escape"])
def test_population_permissions_reset_and_truncation(policy):
    env = parallel_env(
        map_size=15,
        n_pursuers=3,
        n_evaders=5,
        minimap_mode=True,
        extra_features=True,
        max_cycles=3,
        prey_policy=policy,
    )
    first, _ = env.reset(seed=7)
    second, _ = env.reset(seed=7)
    assert len(second) == 3
    for a in first:
        np.testing.assert_array_equal(first[a], second[a])
        assert not second[a][..., [3, 6]].any()
        assert env.action_space(a).n == 29
    for _ in range(3):
        _, _, _, _, infos = env.step({a: 2 for a in env.agents})
    assert not env.agents and env.captured == 0
    assert all(i["capture_fraction"] == 0 for i in infos.values())
    env.reset(seed=19)
    state = env.state()
    walls = np.asarray(env.env.env._get_walls_info())
    assert state[..., 0].sum() == len(walls)
    assert np.all(state[walls[:, 0], walls[:, 1], 0] == 1)
    env.close()


def test_different_targets_and_solo_capture():
    positions = {"a": (5, 5), "b": (10, 10)}
    targets = {0: (5, 7, 5), 1: (10, 12, 5)}
    actions = {
        "a": 5 + ATTACK_OFFSETS.index((0, 2)),
        "b": 17 + ATTACK_OFFSETS.index((0, 2)),
    }
    events, _ = attack_outcomes(positions, targets, actions, ineffective_damage=6)
    assert all(not e["effective"] and e["captured"] for e in events.values())


def test_escape_reads_only_local_view():
    obs = np.zeros((3, 3, 7))
    obs[0, 1, 4] = 1
    assert local_escape_action(obs, np.random.default_rng(1), True) in (1, 3, 4)


def test_query_constructor_and_singleton_groups():
    import torch
    from marlite.util.complementary_targets import ComplementaryAttackQueries
    from marlite.algorithm.graph_builder.graph_util import build_partial_groups

    state = torch.zeros(1, 15, 15, 25)
    state[0, 7, 7, 1] = 1
    state[0, 7, 7, 2] = 1
    state[0, 7, 5, 3] = state[0, 5, 7, 3] = 1
    constructor = ComplementaryAttackQueries(n_queries=3)
    target = torch.tensor([[[[7, 7]] * 3]])
    positions = torch.tensor([[[[[7, 5], [5, 7], [9, 7]]] * 3]])
    modes = torch.tensor([[[[0, 1, 0]] * 3]])
    active = torch.tensor([[[[True, True, False]] * 3]])
    result = constructor.evaluate(
        state,
        torch.tensor([[True]]),
        target,
        positions,
        modes,
        active,
        torch.tensor([[[0, 1, 2]]]),
    )
    assert result["relation_labels"].tolist() == [[[1.0, 1.0, 1.0]]]
    assert result["relation_queries"].shape == (1, 1, 3, 14)
    assert result["relation_mask"].all()
    groups = build_partial_groups(np.array([[5, 2, 3]]), 8, "cityblock", 2, [5], [])
    assert groups.tolist() == [0]
    groups = build_partial_groups(
        np.array([[5, 2, 3], [6, 20, 20]]), 8, "cityblock", 2, [5, 6], []
    )
    assert groups[0] != groups[1] and (groups >= 0).all()


@pytest.mark.parametrize(
    "use_np,hyper_rank", [(False, 0), (True, 0), (False, 2), (True, 2)]
)
def test_configurable_cnn_and_shared_teacher(tmp_path, use_np, hyper_rank):
    import torch
    from marlite.algorithm.agents import AgentGroupConfig
    from marlite.util.communication_curriculum import CommunicationCurriculum

    config = dict(
        type="GroupCommunicationMAPPO",
        agent_list={"a": "shared", "b": "shared"},
        observation_dim=18,
        action_dim=29,
        hidden_dim=8,
        heads=2,
        slots=2,
        message_dim=4,
        group_builder={"type": "Fixed", "group_ids": [0, 0]},
        feature_extractor={
            "model_type": "Custom",
            "layers": [
                {"type": "Permute", "dims": [0, 3, 1, 2]},
                {
                    "type": "Conv2d",
                    "in_channels": 2,
                    "out_channels": 8,
                    "kernel_size": 1,
                },
                {"type": "AdaptiveAvgPool2d", "output_size": 1},
                {"type": "Flatten", "start_dim": 1},
            ],
        },
    )
    teacher_config = dict(config, message_mode="raw")
    teacher = AgentGroupConfig(**teacher_config).get_agent_group()
    path = tmp_path / "teacher.pth"
    torch.save(teacher.state_dict(), path)
    student = AgentGroupConfig(
        **dict(
            config,
            use_np=use_np,
            hyper_rank=hyper_rank,
            message_dim=8,
            teacher_config=teacher_config,
            teacher_checkpoint=path,
            distillation_steps=3,
        )
    ).get_agent_group()
    assert student.teacher.np is None
    assert (student.np is not None) == use_np
    obs = torch.randn(1, 2, 2, 3, 3, 2)
    pad = torch.zeros(1, 2, 2, dtype=torch.bool)
    alive = torch.ones(1, 2, dtype=torch.bool)
    groups = torch.zeros(1, 2, dtype=torch.long)
    curriculum = CommunicationCurriculum(student, teacher, (1, 1, 1))
    optimizer = torch.optim.Adam(student.parameters(), lr=0.001)
    before = student.codec.encoder.weight.detach().clone()
    head = student.action_head.weight.detach().clone()
    curriculum.readout_step(optimizer, obs, pad, alive, groups)
    torch.testing.assert_close(before, student.codec.encoder.weight)
    assert not torch.equal(head, student.action_head.weight)


@pytest.mark.parametrize("algorithm", ["mappo", "qmix"])
@pytest.mark.parametrize("mode", ["local", "raw", "quantized"])
@pytest.mark.parametrize("manager", ["multi-process", "persistent-env"])
def test_native_training_modes(algorithm, mode, manager, tmp_path):
    import yaml
    import torch
    from pathlib import Path
    from marlite.trainer import TrainerConfig

    config = yaml.safe_load(
        Path("examples/complementary_" + algorithm + ".yaml").read_text()
    )
    config["agent_group"]["message_mode"] = mode
    config["rollout"].update(
        episode_limit=3, traj_len=2, manager_type=manager, worker_type=manager
    )
    config["replay_buffer"]["traj_len"] = 2
    config["trainer"]["workdir"] = str(tmp_path)
    trainer = TrainerConfig(config).create_trainer()
    trainer.collect_experience(0.2)
    episode = next(
        ep for ep in trainer.replaybuffer.episode_buffer.values() if ep is not None
    )
    assert 0 <= episode["capture_fraction"] <= 1
    if mode == "local":
        assert len(episode["message_codes"][0]["pursuer_0"]) == 0
    expected_bytes = {"local": 0, "raw": 137, "quantized": 25}[mode]
    assert all(
        size == expected_bytes
        for frame in episode["communication_bytes"]
        for size in frame.values()
    )
    before = trainer.eval_agent_group.action_head.weight.detach().clone()
    result = trainer.learn(sample_size=3, batch_size=3, times=1)
    assert np.isfinite(result["loss"])
    assert not torch.equal(before, trainer.eval_agent_group.action_head.weight)


def test_wrapper_reports_capture_on_the_final_transition():
    from marlite.environment.complementary_pursuit import ComplementaryPursuit

    env = ComplementaryPursuit(Scene("AB"))
    env.reset(seed=1)
    actions = {
        "pursuer_0": 5 + ATTACK_OFFSETS.index((0, 2)),
        "pursuer_1": 17 + ATTACK_OFFSETS.index((2, 0)),
    }
    _, _, _, _, infos = env.step(actions)
    assert env.captured == 1
    assert all(i["capture_fraction"] == 0.5 for i in infos.values())
    env.close()


def test_curriculum_checkpoint_restores_optimizer_and_rng(tmp_path):
    import torch
    from copy import deepcopy
    from marlite.algorithm.agents import GroupCommunicationMAPPOAgentGroup
    from marlite.util.communication_curriculum import CommunicationCurriculum
    from marlite.util.optimizer_config import OptimizerConfig

    student = GroupCommunicationMAPPOAgentGroup(
        {"a": "shared", "b": "shared"},
        6,
        5,
        {"type": "Fixed", "group_ids": [0, 0]},
        hidden_dim=8,
        heads=2,
        slots=2,
        message_dim=4,
        encoder_lr_scale=0.25,
    )
    teacher = deepcopy(student)
    teacher.message_mode = "raw"
    curriculum = CommunicationCurriculum(student, teacher, (2, 2, 2))
    optimizer = OptimizerConfig(type="Adam", lr=0.001).get_optimizer(student)
    assert [g["lr"] for g in optimizer.param_groups] == [0.001, 0.00025]
    obs = torch.randn(1, 2, 2, 6)
    pad = torch.zeros(1, 2, 2, dtype=torch.bool)
    alive = torch.ones(1, 2, dtype=torch.bool)
    groups = torch.zeros(1, 2, dtype=torch.long)
    curriculum.step(optimizer, obs, pad, alive, groups)
    path = tmp_path / "curriculum.pth"
    curriculum.save_checkpoint(path, optimizer)
    expected = torch.rand(5)
    curriculum.step(optimizer, obs, pad, alive, groups)
    curriculum.load_checkpoint(path, optimizer)
    assert curriculum.updates.item() == 1
    torch.testing.assert_close(torch.rand(5), expected)
    assert optimizer.state


@pytest.mark.parametrize(
    "layer",
    [
        {"type": "BatchNorm1d", "num_features": 8},
        {"type": "Dropout", "p": 0.1},
        {"type": "MultiheadAttention", "embed_dim": 8, "num_heads": 2, "dropout": 0.1},
    ],
)
def test_encoder_rejects_batch_or_random_message_dependencies(layer):
    from marlite.algorithm.agents import GroupCommunicationMAPPOAgentGroup

    with pytest.raises(ValueError):
        GroupCommunicationMAPPOAgentGroup(
            {"a": "shared"},
            8,
            5,
            {"type": "Fixed", "group_ids": [0]},
            hidden_dim=8,
            heads=2,
            message_dim=4,
            feature_extractor={"model_type": "Custom", "layers": [layer]},
        )
