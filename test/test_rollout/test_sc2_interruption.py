from multiprocessing.shared_memory import SharedMemory

import pytest
from smac_pettingzoo.env.smacv2 import SC2EpisodeInterrupted

from marlite.algorithm.agents import AgentGroupConfig
from marlite.environment import EnvConfig
from marlite.rollout.multiprocess_rollout import multiprocess_rollout
from marlite.rollout.persistent_env_rollout import persistent_env_rollout
from marlite.util.serialization import serialize_to_buffer


@pytest.mark.parametrize("persistent", [False, True])
def test_interrupted_episode_is_recollected_from_same_seed(persistent):
    config = AgentGroupConfig(
        type="QMIX",
        agent_list={f"agent_{i}": "shared" for i in range(3)},
        models={
            "shared": {
                "encoder": {
                    "model_type": "RNN", "input_shape": 18,
                    "rnn_hidden_dim": 16, "rnn_layers": 1, "output_shape": 16,
                },
                "decoder": {
                    "model_type": "Custom",
                    "layers": [{"type": "Linear", "in_features": 16, "out_features": 5}],
                },
                "feature_extractor": {"model_type": "Identity"},
            }
        },
        optimizer={"type": "Adam", "lr": 0.001},
    )
    payload = serialize_to_buffer(config.get_agent_group().state_dict())
    shm = SharedMemory(create=True, size=len(payload))
    shm.buf[:len(payload)] = payload
    resets = []
    created = []

    class IntermittentEnv:
        def __init__(self, env, fail_step):
            self.env = env
            self.fail_step = fail_step
            self.closed = False

        def __getattr__(self, name):
            return getattr(self.env, name)

        def reset(self, seed=None):
            resets.append(seed)
            return self.env.reset(seed=seed)

        def step(self, actions):
            if self.fail_step:
                self.fail_step = False
                raise SC2EpisodeInterrupted("lost")
            return self.env.step(actions)

        def close(self):
            self.closed = True
            self.env.close()

    class EnvFactory:
        def create_env(self):
            env = IntermittentEnv(
                EnvConfig("mpe2", "simple_spread_v3").create_env(),
                fail_step=not created,
            )
            created.append(env)
            return env

    try:
        if persistent:
            episodes = persistent_env_rollout(
                EnvFactory(), config, (shm.name, len(payload)), n_episodes=2,
                rnn_traj_len=3, episode_limit=2, episode_seeds=[11, 12],
            )
        else:
            episodes = [multiprocess_rollout(
                EnvFactory(), config, (shm.name, len(payload)),
                rnn_traj_len=3, episode_limit=2, seed=11,
            )]
    finally:
        shm.close()
        shm.unlink()

    assert len(episodes) == (2 if persistent else 1)
    assert resets == ([11, 11, 12] if persistent else [11, 11])
    assert all(env.closed for env in created)
