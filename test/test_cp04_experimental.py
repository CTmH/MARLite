from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from marlite.algorithm.agents import AgentGroupConfig
from marlite.experimental.cp04 import initialize_online, load_bundle, prepare
from marlite.trainer.trainer_config import TrainerConfig


@pytest.mark.parametrize("algorithm", ["mappo", "qplex"])
def test_cp04_preparation_and_fresh_handoff(tmp_path, algorithm):
    example = Path(__file__).resolve().parents[1] / "examples" / f"complementary_{algorithm}_8v25.yaml"
    config = yaml.safe_load(example.read_text())
    source = AgentGroupConfig(**config["agent_group"]).get_agent_group()
    checkpoint = tmp_path / "source.pth"
    torch.save(source.state_dict(), checkpoint)
    config["agent_group"]["type"] = "CP03" + algorithm.upper()
    config["agent_group"]["guidance_weight"] = 0.1
    source_config = tmp_path / "source.yaml"
    source_config.write_text(yaml.safe_dump(config, sort_keys=False))
    output = tmp_path / "prepared"
    before = checkpoint.read_bytes()

    manifest = prepare(source_config, checkpoint, output, seed=42)
    assert manifest == load_bundle(output)
    assert manifest["experimental"] and not manifest["teacher_behavior_verified"]
    assert checkpoint.read_bytes() == before
    assert manifest["teacher"]["message_mode"] == "raw"
    assert manifest["student_config"]["agent_group"]["message_mode"] == "quantized"
    assert "guidance_weight" not in manifest["teacher"]

    teacher = torch.load(output / "teacher.pth", weights_only=True)
    initial = torch.load(output / "actor_init.pth", weights_only=True)
    critic_weights = torch.load(output / "critic_init.pth", weights_only=True)
    for name, tensor in source.state_dict().items():
        torch.testing.assert_close(teacher[name], tensor, rtol=0, atol=0)
        if not name.startswith("codec.") and name != "distillation_updates":
            torch.testing.assert_close(initial[name], tensor, rtol=0, atol=0)
    assert not torch.equal(initial["codec.encoder.weight"], teacher["codec.encoder.weight"])

    online = deepcopy(manifest["student_config"])
    online["agent_group"].update(
        teacher_checkpoint=str(output / "teacher.pth"),
        teacher_config=manifest["teacher"],
        distillation_steps=5,
    )
    spec = TrainerConfig(online).trainer_kwargs
    actor = spec["agent_group_config"].get_agent_group()
    critic = spec["critic_config"].get_critic()
    trainer = SimpleNamespace(
        eval_agent_group=actor, eval_critic=critic, replaybuffer=SimpleNamespace(buffer=[])
    )
    if algorithm == "qplex":
        trainer.target_agent_group = deepcopy(actor)
        trainer.target_critic = deepcopy(critic)
    initialize_online(trainer, initial, critic_weights)
    for name, tensor in initial.items():
        torch.testing.assert_close(actor.state_dict()[name], tensor, rtol=0, atol=0)
    for name, tensor in teacher.items():
        torch.testing.assert_close(actor.teacher.state_dict()[name], tensor, rtol=0, atol=0)
    assert not any(p.requires_grad for p in actor.teacher.parameters())
    if algorithm == "qplex":
        for name, tensor in actor.state_dict().items():
            torch.testing.assert_close(trainer.target_agent_group.state_dict()[name], tensor)
    with pytest.raises(ValueError, match="pure student"):
        initialize_online(trainer, actor.state_dict(), critic_weights)
    trainer.replaybuffer.buffer.append("offline data")
    with pytest.raises(ValueError, match="fresh quantized"):
        initialize_online(trainer, initial, critic_weights)
    (output / "critic_init.pth").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_bundle(output)


def test_cp04_collection_keeps_heldout_maps_out_of_teacher_data(tmp_path, monkeypatch):
    import numpy as np

    from marlite.experimental import cp04

    source = {
        "artifacts": {"teacher.pth": "teacher-hash"},
        "teacher": {"agent_list": {"a": "shared", "b": "shared"}},
        "student_config": {"rollout": {"traj_len": 2}},
    }
    monkeypatch.setattr(cp04, "load_bundle", lambda _: source)
    visited = []

    def collect_teacher(config, teacher, checkpoint, seeds):
        visited.extend(seeds)
        return [{
            "episode_length": 1,
            "observations": [{"b": np.ones(3), "a": np.zeros(3)}],
            "states": [np.array([seed])],
            "alive_mask": [{"b": True, "a": True}],
            "group_indices": [{"b": 0, "a": 0}],
            "avail_actions": [{"b": np.array([True, False]), "a": np.ones(2, dtype=bool)}],
        } for seed in seeds]

    monkeypatch.setattr(cp04, "_collect_teacher", collect_teacher)
    split_path = tmp_path / "splits.yaml"
    splits = {
        "offline_train": [870001, 870002],
        "offline_validation": [880001],
        "online_validation": [890001],
        "test": [900001],
    }
    output = tmp_path / "data"
    split_path.write_text(yaml.safe_dump({**splits, "test": [840000]}))
    with pytest.raises(ValueError, match="overlap"):
        cp04.collect(tmp_path, split_path, output)
    assert not output.exists()
    split_path.write_text(yaml.safe_dump(splits))
    cp04.collect(tmp_path, split_path, output)
    assert visited == [870001, 870002, 880001]

    dataset = cp04.load_offline_dataset(output, "offline_train", "teacher-hash")
    assert dataset.sample_id_list == [(0, 0), (1, 0)]
    assert dataset[1]["timestep_padding_mask"] == [True, False]
    assert dataset[1]["states"][-1].item() == 870002
    assert dataset[1]["observations"][-1][:, 0].tolist() == [0, 1]
    assert dataset[1]["avail_actions"][-1].tolist() == [[True, True], [True, False]]
    with pytest.raises(ValueError, match="different CP04 teacher"):
        cp04.load_offline_dataset(output, "offline_train", "wrong-hash")


@pytest.mark.parametrize("replacement", ["progressive", "all"])
def test_cp04_warmup_exports_only_student(tmp_path, monkeypatch, replacement):
    from marlite.experimental import cp04

    example = Path(__file__).resolve().parents[1] / "examples/complementary_mappo_8v25.yaml"
    config = yaml.safe_load(example.read_text())
    teacher = AgentGroupConfig(**config["agent_group"]).get_agent_group()
    source = tmp_path / "teacher.pth"
    torch.save(teacher.state_dict(), source)
    bundle, data, output = (tmp_path / name for name in ("bundle", "data", "offline"))
    cp04.prepare(example, source, bundle, seed=42)
    data.mkdir()
    (data / "manifest.yaml").write_text("experimental: true\n")
    generator = torch.Generator().manual_seed(43)
    observations = torch.randn(2, 8, 9, 9, 29, generator=generator)
    sample = {
        "observations": observations.numpy(),
        "timestep_padding_mask": [True, False],
        "alive_mask": [[False] * 8, [True] * 8],
        "group_indices": [[-1] * 8, [0] * 8],
    }
    monkeypatch.setattr(cp04, "load_offline_dataset", lambda *_: [sample])
    args = (bundle, data, output, replacement, (1, 1, 1), 1, 1)
    cp04.warmup(*args, kl_only=True)
    state = torch.load(output / "offline_state.pth", weights_only=True)
    initial = torch.load(bundle / "actor_init.pth", weights_only=True)
    actor = torch.load(output / "actor.pth", weights_only=True)
    assert len(state["losses"]) == 4
    assert set(state["validation"]) == {"initial", "after_codec", "after_readout"}
    assert state["validation"]["after_codec"]["student"]["action_kl"] >= -1e-7
    assert int(state["curriculum"]["updates"]) == 3
    assert int(state["curriculum"]["readout_updates"]) == 1
    assert set(initial) == set(actor)
    assert actor["distillation_updates"].item() == 0
    assert not torch.equal(actor["codec.encoder.weight"], initial["codec.encoder.weight"])
    assert not torch.equal(actor["action_head.weight"], initial["action_head.weight"])
    for name in actor:
        if name.startswith(("features.", "history.")):
            assert torch.equal(actor[name], initial[name])
    cp04.warmup(*args, kl_only=True, resume=True)
    resumed = torch.load(output / "actor.pth", weights_only=True)
    assert all(torch.equal(resumed[key], value) for key, value in actor.items())


def test_cp04_fixed_relation_queries_preserve_replacement_rng():
    from marlite.experimental.cp04 import _fixed_relations

    batch = {"states": torch.zeros(1, 2, 9, 9, 25)}
    alive = torch.tensor([[True, True, False]])
    groups = torch.tensor([[0, 0, -1]])
    before = torch.get_rng_state().clone()
    first = _fixed_relations(batch, alive, groups, {}, seed=101)
    second = _fixed_relations(batch, alive, groups, {}, seed=101)
    assert torch.equal(before, torch.get_rng_state())
    assert all(torch.equal(first[k], second[k]) for k in first)
    assert first["relation_mask"][:, 0].any()
    assert not first["relation_mask"][:, 1:].any()


@pytest.mark.parametrize("algorithm", ["MAPPO", "QPLEX"])
def test_cp04_online_schedule_matches_rollout_managers(algorithm):
    from marlite.experimental.cp04 import _online_seed_schedule
    from marlite.rollout.rolloutmanager_config import RolloutManagerConfig

    example = Path(__file__).resolve().parents[1] / "examples" / f"complementary_{algorithm.lower()}_8v25.yaml"
    config = yaml.safe_load(example.read_text())
    config["trainer"]["seed"] = 42
    rounds = 2
    result = _online_seed_schedule(config, rounds, {"test": [900000]})
    managers = RolloutManagerConfig(**config["rollout"])
    managers.set_randomness(42)
    for expected in result["collection"]:
        manager = managers.create_manager(None, b"", None, epsilon=0)
        assert manager._episode_seeds() == expected
    for expected in result["trainer_diagnostics"]:
        manager = managers.create_eval_manager(None, b"", None, epsilon=0)
        assert manager._episode_seeds() == expected
    with pytest.raises(ValueError, match="collide"):
        _online_seed_schedule(config, rounds, {"test": [result["collection"][0][0]]})


def test_cp04_group_queries_use_positions_without_reading_prey():
    from marlite.experimental.cp04 import _fixed_relations

    states = torch.zeros(1, 1, 9, 9, 25)
    obs = torch.zeros(1, 1, 3, 9, 9, 29)
    for index, (x, y) in enumerate(((2, 3), (4, 3), (8, 8))):
        obs[0, 0, index, 4, 4, -2:] = torch.tensor([x, y]) / 9
        states[0, 0, x, y, 3] = 1
    batch = {
        "states": states,
        "observations": obs,
        "alive_mask": torch.ones(1, 1, 3, dtype=torch.bool),
        "group_indices": torch.tensor([[[0, 0, 1]]]),
    }
    alive, groups = batch["alive_mask"][:, -1], batch["group_indices"][:, -1]
    spec = {"proposal": "group_geometry", "n_queries": 120}
    empty = _fixed_relations(batch, alive, groups, spec, seed=7)
    assert not empty["relation_labels"].any()
    states[..., 1] = 1
    states[..., 2] = 1
    occupied = _fixed_relations(batch, alive, groups, spec, seed=7)
    assert torch.equal(occupied["relation_queries"], empty["relation_queries"])
    kind = occupied["relation_queries"][0, 0, :, :3].argmax(-1)
    labels = occupied["relation_labels"][0, 0]
    assert labels[kind == 1].any() and labels[kind == 2].any()
    singleton_kind = occupied["relation_queries"][0, 1, :, :3].argmax(-1)
    assert not occupied["relation_labels"][0, 1, singleton_kind != 0].any()
    assert not occupied["relation_mask"][0, 2].any()


def test_cp04_heldout_eval_requires_selected_checkpoint(tmp_path, monkeypatch):
    from marlite.experimental import cp04

    run, data = tmp_path / "run", tmp_path / "data"
    data.mkdir()
    checkpoint = run / "checkpoints/final/agent/agent.pth"
    checkpoint.parent.mkdir(parents=True)
    torch.save({"student": torch.ones(1), "teacher.weight": torch.ones(1)}, checkpoint)
    digest = cp04._digest(checkpoint)
    cp04._write_config(data / "manifest.yaml", {"splits": {"test": [901234]}})
    cp04._write_config(run / "manifest.yaml", {
        "complete": True,
        "kl_only_smoke": False,
        "data_manifest_sha256": cp04._digest(data / "manifest.yaml"),
        "final_actor_sha256": digest,
    })
    cp04._write_config(run / "config.yaml", {
        "trainer": {"type": "MAPPO"},
        "agent_group": {"type": "GroupCommunicationMAPPO", "teacher_checkpoint": "/missing"},
    })
    output = tmp_path / "test.yaml"
    with pytest.raises(ValueError, match="locked"):
        cp04.evaluate_checkpoint(run, data, checkpoint, "test", output)
    assert not output.exists()

    def frozen_rollout(config, spec, state, seeds, *, greedy):
        assert "teacher_checkpoint" not in spec
        assert not greedy
        assert list(state) == ["student"]
        assert seeds == [901234]
        return [{
            "capture_fraction": 0.4,
            "effective_hit_rate": 0.2,
            "missed_attack_rate": 0.3,
            "episode_length": 1,
            "communication_bytes": [{"a": 25, "b": 25}],
        }]

    monkeypatch.setattr(cp04, "_rollout_actor", frozen_rollout)
    result = cp04.evaluate_checkpoint(run, data, checkpoint, "test", output, digest)
    assert result["episodes"][0]["communication_bytes"] == 50
    assert result["mean_capture_fraction"] == 0.4


def test_structured_pursuit_control_has_exact_local_payload():
    from marlite.algorithm.model.group_communication import StructuredPursuitCodec

    codec = StructuredPursuitCodec(8, 25)
    frames = torch.zeros(1, 2, 9, 9, 29)
    frames[:, :, 4, 4, -2:] = torch.tensor([10, 11]) / 25
    frames[0, 0, 3, 4, 4:6] = torch.tensor([1.0, 0.5])
    frames[0, 0, 4, 5, 0] = 1.0
    frames[0, 1, 4, 6, 4:6] = torch.tensor([1.0, 1.0])
    decoded, codes = codec(frames)
    assert codec.pack(codes).shape == (1, 2, 16)
    first = codec.pack(codes)[0, 0].tolist()
    assert first == [10, 10, 128, 1, 11, 11, 0, 2] + [0] * 8
    frames[0, 1].zero_()
    _, modified = codec(frames)
    assert torch.equal(codes[0, 0], modified[0, 0])
    torch.testing.assert_close(codec.decode(codec.unpack(codec.pack(codes))), decoded)
    decoded.sum().backward()
    assert codec.decoder[0].weight.grad.abs().sum() > 0
    assert not hasattr(codec, "encoder")


def test_ss_gigc_precision_fusion_matches_scalar_calculation():
    from marlite.algorithm.model.cp04_baselines import gaussian_product

    parameters = torch.tensor([[[2.0, 0.0], [6.0, torch.log(torch.tensor(4.0))]]])
    members = torch.tensor([[[True, True], [False, False]]])
    group, valid = gaussian_product(parameters, members)
    torch.testing.assert_close(group[0, 0], torch.tensor([2.8, torch.log(torch.tensor(0.8))]))
    assert valid.tolist() == [[True, False]] and not group[0, 1].any()


@pytest.mark.parametrize("algorithm", ["MAPPO", "QMIX"])
@pytest.mark.parametrize("method", ["masia", "ss_gigc", "coalign"])
def test_adapted_paper_baselines_are_finite_rate_and_training_only(algorithm, method):
    torch.manual_seed(42)
    actor = AgentGroupConfig(
        type="GroupCommunication" + algorithm, agent_list={"a": "shared", "b": "shared"},
        observation_dim=9 * 9 * 29, action_dim=29,
        group_builder={"type": "Fixed", "group_ids": [0, 0]},
        hidden_dim=8, slots=2, heads=2, message_map_size=25,
        experimental_cp04_baseline=method,
    ).get_agent_group()
    obs = torch.rand(1, 2, 2, 9, 9, 29)
    obs[..., -2:] = 0.4
    alive = torch.tensor([[True, True]])
    groups = torch.zeros(1, 2, 2, dtype=torch.long)
    padding = torch.zeros(1, 2, 2, dtype=torch.bool)
    batch = {
        "observations": obs.transpose(1, 2),
        "states": torch.rand(1, 2, 25, 25, 25),
        "next_states": torch.rand(1, 2, 25, 25, 25),
        "actions": torch.zeros(1, 2, 2, dtype=torch.long),
        "group_indices": groups,
        "next_group_indices": groups,
    }
    first = actor.training_forward(obs, padding, alive, batch)
    second = actor.training_forward(obs, padding, alive, {
        **batch, "states": 1 - batch["states"], "next_states": 1 - batch["next_states"],
    })
    torch.testing.assert_close(first[actor.output_key], second[actor.output_key])
    codes = first["message_codes"]
    assert actor.codec.pack(codes).shape[-1] == 16
    torch.testing.assert_close(actor.codec.decode(codes), first["decoded_tokens"])
    loss = first["auxiliary_loss"]
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert actor.codec.encoder.weight.grad.abs().sum() > 0
    assert "auxiliary_loss" not in actor.training_forward(
        obs, padding, alive, {"next_group_indices": groups}, next_state=True
    )


def test_cp04_reference_requires_matching_local_config(tmp_path, monkeypatch):
    from marlite.experimental import cp04

    example = Path(__file__).resolve().parents[1] / "examples/complementary_mappo_8v25.yaml"
    config = yaml.safe_load(example.read_text())
    raw = AgentGroupConfig(**config["agent_group"]).get_agent_group()
    source = tmp_path / "source.pth"
    torch.save(raw.state_dict(), source)
    bundle, data = tmp_path / "bundle", tmp_path / "data"
    manifest = cp04.prepare(example, source, bundle, 42)
    data.mkdir()
    cp04._write_config(data / "manifest.yaml", {
        "teacher_sha256": manifest["artifacts"]["teacher.pth"],
        "splits": {"online_validation": [891111]},
    })

    def evaluate(config, spec, state, seeds, split, digest, output, *, greedy):
        assert seeds == [891111]
        assert "teacher." not in state
        return spec["message_mode"]

    monkeypatch.setattr(cp04, "_evaluate_actor", evaluate)
    assert cp04.evaluate_reference(bundle, data, "online_validation", tmp_path / "raw") == "raw"
    config["agent_group"]["message_mode"] = "local"
    config["agent_group"]["type"] = "CP03MAPPO"
    local = tmp_path / "local.yaml"
    cp04._write_config(local, config)
    assert cp04.evaluate_reference(
        bundle, data, "online_validation", tmp_path / "local",
        local_config=local, local_actor=source,
    ) == "local"
    config["environment"]["env_params"]["map_size"] = 24
    cp04._write_config(local, config)
    with pytest.raises(ValueError, match="paired raw"):
        cp04.evaluate_reference(
            bundle, data, "online_validation", tmp_path / "invalid",
            local_config=local, local_actor=source,
        )


def test_cp04_budget_audits_actual_truncated_batches():
    from marlite.experimental.cp04_budget import AuditedRollouts

    class Manager:
        def __init__(self, seed):
            self.seed = seed
            self.n_episodes = 2
            self.episode_limit = 5

        def generate_episodes(self):
            return [{"episode_length": self.episode_limit} for _ in range(self.n_episodes)]

    class Config:
        def create_manager(self, seed):
            return Manager(seed)

        create_eval_manager = create_manager

    audit = AuditedRollouts(Config(), 13)
    audit.create_manager(10).generate_episodes()
    audit.create_eval_manager(11).generate_episodes()
    audit.create_manager(12).generate_episodes()
    assert not audit.collection_exhausted
    audit.create_manager(13).generate_episodes()
    assert audit.collection_exhausted and audit.collection_steps == 13
    assert audit.diagnostic_steps == 10
    assert [batch["steps"] for batch in audit.batches] == [[5, 5], [5, 5], [1, 1], [1]]
    with pytest.raises(RuntimeError, match="exhausted"):
        audit.create_manager(14).generate_episodes()


def test_cp04_gradient_cosine_does_not_write_gradients():
    from marlite.util.communication_training import message_gradient_cosine

    codec = torch.nn.Linear(2, 1, bias=False)
    a, b = codec.weight[0]
    assert message_gradient_cosine(a, -a, codec) == -1.0
    assert message_gradient_cosine(a, b, codec) == 0.0
    assert message_gradient_cosine(a, 0 * b, codec) is None
    assert codec.weight.grad is None


def test_cp04_greedy_mappo_keeps_action_mask():
    import numpy as np

    actor = AgentGroupConfig(
        type="GroupCommunicationMAPPO", agent_list={"a": "shared"},
        observation_dim=6, action_dim=3, message_mode="local",
        group_builder={"type": "Fixed", "group_ids": [0]}, hidden_dim=8, heads=2,
    ).get_agent_group()
    with torch.no_grad():
        actor.action_head.weight.zero_()
        actor.action_head.bias.copy_(torch.tensor([2., 1., 3.]))
    result = actor.act(
        {"a": np.zeros((1, 6), dtype=np.float32)}, None,
        {"a": np.array([True, True, False])}, np.array([False]), ["a"],
        greedy_evaluation=True,
    )
    assert result["actions"] == {"a": 0}
