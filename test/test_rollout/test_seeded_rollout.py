"""Real CPU processes and environments: task seeds survive worker scheduling."""

from copy import deepcopy

import numpy as np
import pytest

from marlite.algorithm.agents import AgentGroupConfig
from marlite.environment import EnvConfig
from marlite.rollout.rolloutmanager_config import RolloutManagerConfig
from marlite.util.serialization import serialize_to_buffer


@pytest.mark.parametrize("manager_type", ["multi-process", "persistent-env"])
def test_seeded_collection_is_repeatable_across_worker_counts(manager_type):
    from test.test_trainer.test_mappo_trainer import TestMAPPOTrainer

    fixture = TestMAPPOTrainer()
    fixture.setUp()
    config = deepcopy(fixture.config)
    agent_config = AgentGroupConfig(**config["agent_group"])
    parameters = serialize_to_buffer(agent_config.get_agent_group().state_dict())
    env_config = EnvConfig(**config["environment"])
    results = []
    for n_workers in (1, 2):
        rollout = RolloutManagerConfig(
            manager_type=manager_type, worker_type=manager_type,
            n_workers=n_workers, n_episodes=3, traj_len=3,
            episode_limit=3, device="cpu", required_attrs="mappo",
        )
        rollout.set_randomness(42)
        manager = rollout.create_manager(agent_config, parameters, env_config, epsilon=1.)
        results.append(manager.generate_episodes())
    assert len(results[0]) == len(results[1]) == 3
    for first, second in zip(*results):
        for key in ("states", "next_states"):
            np.testing.assert_array_equal(np.array(first[key]), np.array(second[key]))
        for key in ("observations", "actions", "all_log_probs"):
            for first_step, second_step in zip(first[key], second[key]):
                for agent in first_step:
                    np.testing.assert_array_equal(first_step[agent], second_step[agent])
    # Distinct episode tasks do not accidentally reset to the same seed.
    assert not np.array_equal(results[0][0]["states"][0], results[0][1]["states"][0])
