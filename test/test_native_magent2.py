"""Opt-in native MAgent2 tests; Linux acceptance runs these explicitly."""

import os
import pytest

if os.environ.get("MARLITE_NATIVE_MAGENT2") != "1":
    pytest.skip(
        "Enable MARLITE_NATIVE_MAGENT2=1 for native-library integration", allow_module_level=True
    )

import numpy as np
from magent2.environments.complementary_pursuit_v1 import parallel_env as NativeEnv
from marlite.environment.complementary_pursuit import parallel_env
from marlite.environment.complementary_events import attack_outcomes, ATTACK_OFFSETS


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
        expected = pool / len(actions) + (0.2 / len(critical) if agent in critical else 0.0)
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


@pytest.mark.parametrize(
    "family,wrapper_type",
    [
        ("adversarial_pursuit_v4", "AdversarialPursuitPredator"),
        ("adversarial_pursuit_v4", "AdversarialPursuitPrey"),
        ("battlefield_v5", "BattleWrapper"),
    ],
)
@pytest.mark.parametrize("format", ["image", "channel_first", "vector"])
def test_native_opponent_wrapper_spaces_and_seeded_controller(family, wrapper_type, format):
    import importlib
    from marlite.configuration.catalog import default_registry
    from marlite.environment.controllers import RandomController

    native = importlib.import_module("magent2.environments." + family).parallel_env(
        map_size=46 if family == "battlefield_v5" else 20, max_cycles=3
    )
    wrapper = default_registry().build(
        {
            "type": wrapper_type,
            "params": {
                "channel_first": format == "channel_first",
                "vector_observation": format == "vector",
            },
        },
        injected={"env": native},
    )
    try:
        observation, _ = wrapper.reset(seed=42)
        assert wrapper.agents and wrapper.opponent_agents
        for agent in wrapper.agents:
            assert wrapper.observation_space(agent).contains(observation[agent])
        controller = RandomController()
        controller.reset(42, wrapper.possible_agents)
        for _ in range(3):
            actions = controller.actions(observation, wrapper.action_spaces, wrapper.agents)
            observation, _, _, _, _ = wrapper.step(actions)
        assert not wrapper.agents
    finally:
        wrapper.close()


@pytest.mark.parametrize("history", ["gru", "pooled_attention"])
def test_c04_native_short_teacher_offline_online_and_checkpoint(tmp_path, history):
    import torch
    import yaml
    from copy import deepcopy
    from pathlib import Path
    from marlite.configuration.recipe import build_experiment
    from test.test_sequence_mappo import assert_tree_equal

    config = yaml.safe_load(
        (Path(__file__).parents[1] / "examples/sequence/c04_magent2.yaml").read_text()
    )
    config["environment"]["params"]["max_cycles"] = 4
    config["collector"]["max_steps"] = 4
    config["communication_context"]["params"]["builder"]["params"]["n_workers"] = 0
    run = {"iterations": 1, "train_episodes": 2, "eval_episodes": 1, "eval_interval": 1}
    config["run"] = run
    config["teacher"]["run"] = dict(run)
    config["teacher"]["online_distillation"]["enabled"] = True
    for options in (config["trainer"], config["teacher"]["trainer"]):
        options.update(sequence_length=2, batch_sequences=3, ppo_epochs=1)
    config["offline"].update(
        train_episodes=2,
        validation_episodes=1,
        codec_updates=1,
        policy_updates=1,
        probe_updates=1,
        offline_batch_sequences=2,
    )
    if history == "pooled_attention":
        config["actor"]["params"]["temporal"] = {
            "type": "WindowEncoder",
            "params": {
                "input_dim": 32,
                "hidden_dim": 32,
                "window_length": 4,
                "model": {
                    "type": "SimpleResAttSeqEnc",
                    "params": {
                        "input_dim": 32,
                        "output_dim": 32,
                        "embed_dim": 32,
                        "num_heads": 4,
                        "max_seq_len": 4,
                        "dropout": 0.0,
                    },
                },
            },
        }
    device = os.environ.get("MARLITE_ACCEPTANCE_DEVICE", "cpu")
    experiment = build_experiment(config, workdir=tmp_path / "run", device=device)
    experiment.run()
    trainer = experiment.trainer
    assert trainer.phase == "online" and trainer.offline_counters == {
        "probe": 1,
        "codec": 1,
        "policy": 1,
    }
    assert trainer.learner.updates > 0
    checkpoint = tmp_path / "resume.pt"
    trainer.save_checkpoint(checkpoint)
    metrics = trainer.train_iteration(2)
    expected = deepcopy(trainer.learner.state_dict())
    trainer.load_checkpoint(checkpoint)
    assert metrics == trainer.train_iteration(2)
    assert_tree_equal(expected, trainer.learner.state_dict())
    assert all(torch.isfinite(p).all() for p in trainer.learner.actor.parameters())
