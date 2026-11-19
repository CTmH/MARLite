"""Opt-in SMAC adapter acceptance; requires a configured StarCraft installation."""

import os
import pytest

if os.environ.get("MARLITE_NATIVE_SMAC") != "1":
    pytest.skip("Enable MARLITE_NATIVE_SMAC=1 for SC2 integration", allow_module_level=True)

from marlite.configuration.catalog import default_registry
from marlite.configuration.environment import EnvironmentFactory
from marlite.util.env_util import current_action_mask
import numpy as np


def test_native_smac_observations_legal_actions_and_state():
    factory = EnvironmentFactory(
        {
            "module": "smac_pettingzoo.smacv2_pettingzoo_v1",
            "params": {"map_name": os.environ.get("MARLITE_SMAC_MAP", "10gen_terran_5_vs_5")},
            "wrappers": [{"type": "SMACWrapper"}],
        },
        default_registry(),
    )
    env = factory()
    try:
        observation, infos = env.reset(seed=42)
        assert env.state().shape[0] == len(env.possible_agents)
        for _ in range(3):
            actions = {
                a: int(
                    np.flatnonzero(
                        current_action_mask(observation[a], infos.get(a), env.action_space(a).n)
                    )[0]
                )
                for a in env.agents
            }
            observation, _, _, _, infos = env.step(actions)
    finally:
        env.close()


@pytest.mark.parametrize("recipe", ["mappo", "qplex"])
def test_smacv2_real_collection_and_sequence_update(tmp_path, recipe):
    from marlite.configuration.recipe import build_experiment
    import torch

    config = {
        "version": 3,
        "recipe": recipe,
        "seed": 42,
        "environment": {
            "module": "smac_pettingzoo.smacv2_pettingzoo_v1",
            "params": {"map_name": os.environ.get("MARLITE_SMAC_MAP", "10gen_terran_5_vs_5")},
            "wrappers": [{"type": "SMACWrapper"}],
        },
        "defaults": {"hidden_dim": 8, "mixing_dim": 4, "heads": 2},
        "collector": {"max_steps": 3},
        "trainer": {"sequence_length": 2, "batch_sequences": 2},
        "run": {"iterations": 1, "train_episodes": 1, "eval_interval": None},
    }
    if recipe == "mappo":
        config["trainer"]["ppo_epochs"] = 1
    else:
        config["trainer"]["updates_per_iteration"] = 1
    experiment = build_experiment(
        config, workdir=tmp_path / recipe, device=os.environ.get("MARLITE_ACCEPTANCE_DEVICE", "cpu")
    )
    experiment.run()
    learner = experiment.trainer.learner
    assert learner.updates > 0 and all(torch.isfinite(p).all() for p in learner.actor.parameters())
