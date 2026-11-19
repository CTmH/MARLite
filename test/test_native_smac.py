"""Opt-in SMAC adapter acceptance; requires a configured StarCraft installation."""

import os
import pytest

if os.environ.get("MARLITE_NATIVE_SMAC") != "1":
    pytest.skip("Enable MARLITE_NATIVE_SMAC=1 for SC2 integration", allow_module_level=True)

from marlite.configuration.catalog import default_registry
from marlite.configuration.environment import EnvironmentFactory
import numpy as np


def test_native_smac_observations_legal_actions_and_state():
    factory = EnvironmentFactory(
        {
            "module": "smac_pettingzoo.smacv1_pettingzoo_v1",
            "params": {"map_name": os.environ.get("MARLITE_SMAC_MAP", "2s3z")},
            "wrappers": [{"type": "SMACWrapper"}],
        },
        default_registry(),
    )
    env = factory()
    try:
        observation, _ = env.reset(seed=42)
        assert env.state().shape[0] == len(env.possible_agents)
        for _ in range(3):
            actions = {a: int(np.flatnonzero(observation[a]["action_mask"])[0]) for a in env.agents}
            observation, _, _, _, _ = env.step(actions)
    finally:
        env.close()
