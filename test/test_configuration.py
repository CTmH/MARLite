"""The new configuration constructs components, never algorithm control flow."""

from copy import deepcopy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from marlite.algorithm.components import HyperNetwork
from marlite.configuration.catalog import default_registry
from marlite.configuration.environment import EnvironmentFactory
from marlite.configuration.recipe import build_experiment
from marlite.configuration.registry import ConfigurationError, Registry
from test.test_sequence_mappo import TinyParallel, assert_tree_equal


@pytest.fixture
def environment_module(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "configuration_test_env", SimpleNamespace(parallel_env=TinyParallel)
    )
    return "configuration_test_env"


def minimal_config(module, recipe="mappo"):
    config = {
        "version": 3,
        "recipe": recipe,
        "seed": 42,
        "environment": {"module": module, "params": {"death": True}},
        "defaults": {"hidden_dim": 8, "mixing_dim": 4, "heads": 2},
        "collector": {"max_steps": 4},
        "trainer": {"sequence_length": 2, "batch_sequences": 3},
        "run": {"iterations": 1, "train_episodes": 2, "eval_episodes": 2, "checkpoint_interval": 1},
    }
    if recipe in ("mappo", "c04"):
        config["trainer"]["ppo_epochs"] = 1
    else:
        config["trainer"]["updates_per_iteration"] = 2
        config["replay"] = {"capacity": 4}
    if recipe == "c04":
        config["teacher"] = {
            "source": "train",
            "trainer": {"sequence_length": 2, "batch_sequences": 3, "ppo_epochs": 1},
            "run": {"iterations": 1, "train_episodes": 2, "eval_interval": None},
            "online_distillation": {"enabled": True, "weight": 0.1, "decay_updates": 3},
        }
        config["offline"] = {
            "train_episodes": 2,
            "validation_episodes": 1,
            "codec_updates": 1,
            "policy_updates": 1,
            "offline_batch_sequences": 2,
            "optimizers": {"codec": {"type": "Adam", "params": {"lr": 0.002}}},
        }
    return config


def test_registry_resolves_dimensions_and_preserves_initialization_boundaries(tmp_path):
    registry = default_registry()
    child = {
        "type": "Linear",
        "params": {
            "in_features": {"$dim": "shape", "index": -1},
            "out_features": {"$dim": "agents", "scale": 2, "offset": 1},
        },
        "initialization": {"scheme": "constant", "value": 0.25, "bias": 0.5},
    }
    spec = {
        "type": "Sequential",
        "params": {"layers": [child, {"type": "GELU"}]},
        "initialization": {"scheme": "constant", "value": 10.0},
    }
    model = registry.build(spec, {"shape": [2, 3], "agents": 2})
    assert model[0].weight.shape == (5, 3) and model[0].weight.eq(0.25).all()
    path = tmp_path / "weights.pt"
    torch.save({"actor": model.state_dict()}, path)
    spec["checkpoint"] = {"path": str(path), "key": ["actor"]}
    restored = registry.build(spec, {"shape": [2, 3], "agents": 2})
    assert_tree_equal(model.state_dict(), restored.state_dict())
    assert isinstance(registry.build({"type": "DType", "params": {"name": "float32"}}), torch.dtype)


@pytest.mark.parametrize(
    "spec,pattern",
    [
        ({"type": "DoesNotExist"}, "unknown component"),
        ({"type": "Linear", "params": {"in_feature": 3, "out_features": 4}}, "constructor"),
        ({"type": "Linear", "in_features": 3}, "unknown spec"),
        (
            {"type": "Linear", "params": {"in_features": {"$dim": "missing"}, "out_features": 4}},
            "dimension",
        ),
    ],
)
def test_registry_reports_configuration_paths(spec, pattern):
    with pytest.raises(ConfigurationError, match=pattern) as error:
        default_registry().build(spec, path="actor.frame_encoder")
    assert "actor.frame_encoder" in str(error.value)


def test_registry_extensions_are_explicit_and_injected_fields_cannot_be_overridden():
    registry = Registry()
    registry.register("Identity", nn.Identity)
    with pytest.raises(ValueError, match="already registered"):
        registry.register("Identity", nn.Identity)
    registry.register("List", lambda items: tuple(items))
    with pytest.raises(ConfigurationError, match="override injected"):
        registry.build({"type": "List", "params": {"items": []}}, injected={"items": [1]})
    assert registry.build({"type": "List"}, injected={"items": [1]}) == (1,)


def test_unconstrained_hypernetwork_can_express_negative_slope():
    weight, bias = nn.Linear(1, 1), nn.Linear(1, 1)
    with torch.no_grad():
        weight.weight.fill_(0)
        weight.bias.fill_(-2)
        bias.weight.fill_(0)
        bias.bias.fill_(1)
    network = HyperNetwork([1, 1], [weight], [bias])
    x, condition = torch.tensor([[[1.0], [3.0]]]), torch.zeros(1, 2, 1)
    torch.testing.assert_close(network(x, condition), torch.tensor([[[-1.0], [-5.0]]]))
    network.positive_weights = True
    torch.testing.assert_close(network(x, condition), torch.tensor([[[3.0], [7.0]]]))


@pytest.mark.parametrize("recipe", ["mappo", "qmix", "qplex", "qtran", "c04"])
def test_new_recipes_train_and_resume(environment_module, recipe, tmp_path):
    config = minimal_config(environment_module, recipe)
    experiment = build_experiment(config, workdir=tmp_path / "run")
    experiment.run()
    checkpoint = tmp_path / "run" / "checkpoints" / "iteration_1.pt"
    assert checkpoint.exists()
    if recipe == "c04":
        assert experiment.teacher_trainer.iterations == 1
        assert experiment.trainer.phase == "online"
        assert all(p.grad is None for p in experiment.trainer.reference.actor.parameters())
    restored = build_experiment(config, workdir=tmp_path / "restored")
    restored.trainer.load_checkpoint(checkpoint)
    assert experiment.trainer.train_iteration(2) == restored.trainer.train_iteration(2)
    assert_tree_equal(
        experiment.trainer.learner.state_dict(), restored.trainer.learner.state_dict()
    )


def test_custom_network_and_optimizer_parameter_groups(environment_module):
    config = minimal_config(environment_module)
    config["actor"] = {
        "type": "SequenceAgentGroup",
        "params": {
            "frame_encoder": {
                "type": "Linear",
                "params": {"in_features": {"$dim": "observation_size"}, "out_features": 8},
            },
            "temporal": {"type": "RecurrentGRU", "params": {"input_dim": 8, "hidden_dim": 8}},
            "action_head": {
                "type": "Linear",
                "params": {"in_features": 8, "out_features": {"$dim": "action_dim"}},
            },
        },
    }
    config["optimizers"] = {
        "actor": {
            "type": "AdamW",
            "params": {"lr": 0.001},
            "groups": [{"prefixes": ["frame_encoder", "temporal"], "params": {"lr": 0.0001}}],
        }
    }
    experiment = build_experiment(config)
    assert [group["lr"] for group in experiment.trainer.learner.actor_optimizer.param_groups] == [
        0.0001,
        0.001,
    ]
    assert experiment.trainer.train_iteration(2)["agent_steps"] > 0


@pytest.mark.parametrize(
    "patch,pattern",
    [
        ({"version": 1}, "incompatible"),
        ({"recipe": "GraphMAPPO"}, "supported recipe"),
        ({"typo": 1}, "unknown fields"),
        ({"learner": {"actor": None}}, "wiring"),
        ({"trainer": {"workdir": "bad"}}, "wiring"),
        ({"dimensions": {"action_dim": 99}}, "override"),
        ({"learner": {"clip_epslion": 0.2}}, "learner"),
        ({"offline": {"learner": {}}}, "offline"),
    ],
)
def test_recipe_rejects_unknown_or_python_owned_controls(environment_module, patch, pattern):
    config = minimal_config(environment_module, "c04" if "offline" in patch else "mappo")
    config.update(patch)
    with pytest.raises(ConfigurationError, match=pattern):
        build_experiment(config)


def test_environment_factory_always_closes_probe(monkeypatch):
    env = TinyParallel()
    monkeypatch.setitem(
        sys.modules, "configuration_test_probe", SimpleNamespace(parallel_env=lambda: env)
    )
    factory = EnvironmentFactory({"module": "configuration_test_probe"}, default_registry())
    spec = factory.probe(1)
    assert spec.dimensions["observation_size"] == 3 and env.closed


def test_yaml_rejects_duplicate_explicit_keys_and_allows_anchor_overrides(tmp_path):
    from marlite.configuration.io import load_config

    path = tmp_path / "config.yaml"
    path.write_text("version: 3\nversion: 2\n")
    with pytest.raises(ConfigurationError, match="Duplicate YAML key"):
        load_config(path)
    path.write_text("base: &b {width: 8, depth: 2}\nmodel:\n  <<: *b\n  width: 16\n")
    assert load_config(path)["model"] == {"width": 16, "depth": 2}


def test_value_free_analysis_exports_real_episodes_and_does_not_train(environment_module, tmp_path):
    from marlite.data import Episode
    from marlite.learning.mappo import prepare_ppo_targets

    config = minimal_config(environment_module, "c04")
    trained = build_experiment(config, workdir=tmp_path / "train")
    trained.run()
    path = tmp_path / "train" / "checkpoints" / "final.pt"
    evaluator = build_experiment(config, policy_only=True)
    assert not hasattr(evaluator, "trainer") and not hasattr(evaluator, "teacher_trainer")
    evaluator.load_checkpoint(path)
    before = deepcopy(evaluator.actor.state_dict())
    report = evaluator.evaluate(2, seed=100, trajectories=tmp_path / "evaluation.pt")
    assert report["episode_length"]["count"] == 2
    assert report["communication_bytes_per_episode"]["mean"] > 0
    assert_tree_equal(before, evaluator.actor.state_dict())
    episodes = torch.load(tmp_path / "evaluation.pt", weights_only=True)["episodes"]
    restored = Episode.from_state_dict(episodes[0])
    assert "old_value" not in restored.frames
    assert not any(key.startswith("actor_state.") for key in restored.frames)
    with pytest.raises(ValueError, match="old_value"):
        prepare_ppo_targets({0: restored})
    again = evaluator.evaluate(2, seed=100)
    assert report == again
    assert (tmp_path / "train" / "resolved.yaml").exists()
    assert (tmp_path / "train" / "provenance.json").exists()
    assert list((tmp_path / "train" / "source").glob("*.zip"))


def test_checkpoint_rejects_changed_policy_semantics(environment_module, tmp_path):
    config = minimal_config(environment_module)
    experiment = build_experiment(config, workdir=tmp_path / "run")
    experiment.run()
    modified = deepcopy(config)
    modified["learner"] = {"entropy_coef": 0.2}
    restored = build_experiment(modified)
    with pytest.raises(ValueError, match="configuration"):
        restored.trainer.load_checkpoint(tmp_path / "run" / "checkpoints" / "final.pt")


def test_context_rng_is_episode_local_and_does_not_touch_training_rng():
    import numpy as np
    from marlite.communication.context import StateBuilderContext

    class RandomBuilder(nn.Module):
        def forward(self, state):
            assert not self.training
            return torch.from_numpy(np.random.randint(0, 3, size=(1, 2)))

    provider = StateBuilderContext(RandomBuilder())
    provider.reset(seed=7)
    before = np.random.get_state()
    first = provider(torch.zeros(2, 3), torch.zeros(3), torch.ones(2, dtype=torch.bool), 0)
    after = np.random.get_state()
    assert np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    provider.reset(seed=7)
    repeated = provider(torch.zeros(2, 3), torch.zeros(3), torch.ones(2, dtype=torch.bool), 0)
    assert torch.equal(first["groups"], repeated["groups"])


EXAMPLES = sorted(
    path
    for path in (Path(__file__).parents[1] / "examples" / "sequence").glob("*.yaml")
    if "magent" not in path.name
)


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda path: path.stem)
def test_all_mpe2_examples_execute_short_workflows(path, tmp_path):
    from marlite.configuration.io import load_config

    config = load_config(path)
    config["environment"]["params"]["max_cycles"] = 3
    config["collector"]["max_steps"] = 3
    config["trainer"].update(sequence_length=2, batch_sequences=2)
    if config["recipe"] in ("mappo", "c04"):
        config["trainer"]["ppo_epochs"] = 1
    else:
        config["trainer"]["updates_per_iteration"] = 1
    config["run"] = {"iterations": 1, "train_episodes": 2, "eval_episodes": 2}
    if config["recipe"] == "c04":
        config["teacher"]["run"] = {"iterations": 1, "train_episodes": 2, "eval_interval": None}
        config["teacher"]["trainer"].update(ppo_epochs=1, sequence_length=2, batch_sequences=2)
        config["offline"].update(
            codec_updates=1,
            policy_updates=1,
            train_episodes=2,
            validation_episodes=1,
            offline_batch_sequences=2,
        )
    experiment = build_experiment(config, workdir=tmp_path)
    experiment.run()
    evaluator = build_experiment(config, policy_only=True)
    evaluator.load_checkpoint(tmp_path / "checkpoints" / "final.pt")
    assert evaluator.evaluate(1)["episode_length"]["mean"] == 3


def test_batch_dependent_and_unrecorded_random_components_fail_before_training():
    from marlite.algorithm.components import validate_replay_modules

    with pytest.raises(ValueError, match="batch-independent"):
        validate_replay_modules(nn.Sequential(nn.Linear(3, 8), nn.BatchNorm1d(8)))
    with pytest.raises(ValueError, match="dropout"):
        validate_replay_modules(nn.Dropout(0.1))
    with pytest.raises(ValueError, match="random activations"):
        validate_replay_modules(nn.RReLU())


def test_spatial_decoder_does_not_depend_on_rank_batch_partition():
    from marlite.algorithm.model.decoder import NearestDecoder
    from marlite.algorithm.components import validate_replay_modules

    decoder = NearestDecoder(latent_dim=4, out_channels=2, window_size=5, hid_channels=4)
    validate_replay_modules(decoder)
    inputs = torch.randn(3, 4)
    together = decoder(inputs)
    apart = torch.cat([decoder(row[None]) for row in inputs])
    torch.testing.assert_close(together, apart, rtol=1e-5, atol=1e-6)


def test_invalid_online_run_is_rejected_before_teacher_training(environment_module, tmp_path):
    experiment = build_experiment(minimal_config(environment_module, "c04"), workdir=tmp_path)
    with pytest.raises(ConfigurationError, match="run.iterations"):
        experiment.run(run_options={"iterations": -1})
    assert experiment.teacher_trainer.iterations == 0
    assert not (tmp_path / "teacher" / "events.jsonl").exists()


def test_communication_rl_gate_is_a_configurable_persistent_parameter():
    registry = default_registry()
    actor = registry.build(
        {
            "type": "SequenceAgentGroup",
            "params": {
                "frame_encoder": {
                    "type": "Linear",
                    "params": {"in_features": 3, "out_features": 8},
                },
                "temporal": {"type": "RecurrentGRU", "params": {"input_dim": 8, "hidden_dim": 8}},
                "action_head": {"type": "Linear", "params": {"in_features": 8, "out_features": 3}},
                "rl_gate": 0.25,
            },
        }
    )
    assert actor.state_dict()["rl_gate"] == 0.25
    with pytest.raises(ConfigurationError, match="finite"):
        registry.build(
            {"type": "Linear", "params": {"in_features": float("nan"), "out_features": 3}}
        )


@pytest.mark.parametrize("initialize_from_teacher", [False, True])
def test_teacher_transfer_respects_student_initialization(
    environment_module, tmp_path, initialize_from_teacher
):
    config = minimal_config(environment_module, "c04")
    config["offline"].update(offline_enabled=False, initialize_from_teacher=initialize_from_teacher)
    experiment = build_experiment(config, workdir=tmp_path / "run")
    student = experiment.trainer.learner.actor
    before = deepcopy(student.action_head.state_dict())
    experiment.run(run_options={"iterations": 0})
    teacher = experiment.teacher_trainer.learner.actor
    reference = experiment.trainer.reference.actor
    assert experiment.teacher_trainer.learner.actor_updates > 0
    assert_tree_equal(teacher.state_dict(), reference.state_dict())
    expected = teacher.action_head.state_dict() if initialize_from_teacher else before
    assert_tree_equal(expected, student.action_head.state_dict())


def test_c04_can_load_a_frozen_teacher_from_yaml_checkpoint(environment_module, tmp_path):
    from marlite.configuration import defaults

    config = minimal_config(environment_module, "c04")
    teacher_run = build_experiment(config, workdir=tmp_path / "teacher")
    teacher_run.teacher_trainer.run(**teacher_run.teacher_run)
    state = teacher_run.teacher_trainer.learner.actor.state_dict()
    checkpoint = tmp_path / "trained_actor.pt"
    torch.save({"learner": {"actor": state}}, checkpoint)

    def raw_teacher():
        model = defaults.actor(teacher_run.environment_spec, 8, quantized=True, heads=2)
        model.communication = model.communication.aggregation
        return model

    registry = default_registry()
    registry.register("RawTeacher", raw_teacher)
    config["teacher"]["source"] = "checkpoint"
    config["teacher"]["actor"] = {
        "type": "RawTeacher",
        "checkpoint": {"path": str(checkpoint), "key": ["learner", "actor"]},
    }
    experiment = build_experiment(config, registry=registry, workdir=tmp_path / "student")
    assert experiment.teacher_trainer is None
    experiment.run()
    assert experiment.trainer.phase == "online"
    assert experiment.trainer.learner.actor_updates > 0
    assert_tree_equal(state, experiment.trainer.reference.actor.state_dict())
