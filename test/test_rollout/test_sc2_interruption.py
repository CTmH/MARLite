from multiprocessing.shared_memory import SharedMemory

import pytest
from smac_pettingzoo.env.smacv2 import SC2EpisodeInterrupted

from marlite.algorithm.agents import AgentGroupConfig
from marlite.environment import EnvConfig
from marlite.environment import EnvironmentInterrupted
from marlite.rollout.multiprocess_rollout import multiprocess_rollout
from marlite.rollout.persistent_env_rollout import persistent_env_rollout
from marlite.rollout.env_retry import EnvRetryPolicy
from marlite.util.serialization import serialize_to_buffer


@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize("error_type", [
    SC2EpisodeInterrupted, EnvironmentInterrupted, ConnectionError, TimeoutError, ValueError,
])
@pytest.mark.parametrize("failure_stage", ["reset", "step"])
@pytest.mark.parametrize("failures,max_retries", [(0, 2), (1, 2), (2, 2), (3, 2), (1, 0)])
def test_interrupted_episode_is_recollected_from_same_seed(
    persistent, failure_stage, failures, max_retries, error_type, monkeypatch,
):
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
    waits = []

    def sleep(delay):
        assert created[-1].closed  # Release SC2 resources before waiting.
        waits.append(delay)

    monkeypatch.setattr("marlite.rollout.env_retry.time.sleep", sleep)
    policy = EnvRetryPolicy(max_retries=max_retries, initial_delay=1, jitter=0)

    class IntermittentEnv:
        # Backend-specific errors are opt-in; generic environments need no SC2 import.
        recoverable_errors = (SC2EpisodeInterrupted,)

        def __init__(self, env, fail_step):
            self.env = env
            self.fail_step = fail_step
            self.closed = False

        def __getattr__(self, name):
            return getattr(self.env, name)

        def reset(self, seed=None):
            resets.append(seed)
            if self.fail_step and failure_stage == "reset":
                raise error_type("lost during reset")
            return self.env.reset(seed=seed)

        def step(self, actions):
            if self.fail_step:
                self.fail_step = False
                raise error_type("lost")
            return self.env.step(actions)

        def close(self):
            self.closed = True
            self.env.close()

    class EnvFactory:
        def create_env(self):
            env = IntermittentEnv(
                EnvConfig("mpe2", "simple_spread_v3").create_env(),
                fail_step=len(created) < failures,
            )
            created.append(env)
            return env

    def collect():
        if persistent:
            return persistent_env_rollout(
                EnvFactory(), config, (shm.name, len(payload)), n_episodes=2,
                rnn_traj_len=3, episode_limit=2, episode_seeds=[11, 12],
                env_retry=policy,
            )
        else:
            return [multiprocess_rollout(
                EnvFactory(), config, (shm.name, len(payload)),
                rnn_traj_len=3, episode_limit=2, seed=11,
                env_retry=policy,
            )]

    try:
        if failures and error_type is ValueError:
            with pytest.raises(RuntimeError, match=f"env.{failure_stage} failed") as exc:
                collect()
            assert isinstance(exc.value.__cause__, ValueError)
            assert resets == [11]
        elif failures > max_retries:
            with pytest.raises(RuntimeError, match="after recovery attempts") as exc:
                collect()
            assert isinstance(exc.value.__cause__, error_type)
            assert resets == [11] * (max_retries + 1)
        else:
            episodes = collect()
            assert len(episodes) == (2 if persistent else 1)
            assert resets == [11] * (failures + 1) + ([12] if persistent else [])
    finally:
        shm.close()
        shm.unlink()

    expected_retries = 0 if error_type is ValueError else min(failures, max_retries)
    assert waits == [2 ** i for i in range(expected_retries)]
    assert all(env.closed for env in created)
