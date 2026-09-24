import importlib
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from marlite.util.trajectory_dataset import TrajectoryDataset


class TinyEnv:
    possible_agents = ["a", "b"]

    def __init__(self, terminal):
        self.terminal = terminal

    def reset(self, seed):
        self.t = 0
        self.agents = self.possible_agents.copy()
        return self.observe(), self.info()

    def observe(self):
        return {a: np.array([self.t + 10.]) for a in (self.agents or ["b"])}

    def info(self):
        return {a: {"action_mask": np.array([True, False])} for a in (self.agents or ["b"])}

    def state(self):
        if self.terminal and self.t == 2:
            raise RuntimeError("terminal state unavailable")
        return np.array([self.t + 100.])

    def step(self, actions):
        self.t += 1
        self.agents = ["b"] if self.t < 2 else []
        return (self.observe(), {"a": 1., "b": 2.},
                {"a": True, "b": self.terminal and self.t == 2},
                {"a": True, "b": self.t == 2}, self.info())

    def close(self):
        pass


@pytest.mark.parametrize("persistent", [True, False])
@pytest.mark.parametrize("terminal", [True, False])
@pytest.mark.parametrize("limit", [1, 5])
def test_collect_actual_final_window_and_survivors(persistent, terminal, limit):
    name = "persistent_env_rollout" if persistent else "multiprocess_rollout"
    module = importlib.import_module("marlite.rollout." + name)
    env = TinyEnv(terminal)
    agent = Mock()
    agent.reset.return_value = agent
    agent.eval.return_value = agent
    agent.to.return_value = agent
    agent.agent_model_dict = {a: None for a in env.possible_agents}

    def act(observations, state, available, padding, alive, epsilon):
        assert all(available[a].tolist() == [True, False] for a in alive)
        return {"actions": {a: 0 for a in alive},
                "all_actions": {a: 0 for a in env.possible_agents},
                "all_group_indices": {a: env.t if a in alive else -1 for a in env.possible_agents}}

    agent.act.side_effect = act
    kwargs = dict(env_config=SimpleNamespace(create_env=lambda: env),
                  agent_group_config=SimpleNamespace(get_agent_group=lambda: agent),
                  shm_info=("fake", 0), rnn_traj_len=4, episode_limit=limit)
    if persistent:
        kwargs["n_episodes"] = 2
    with patch.object(module, "SharedMemory", return_value=SimpleNamespace(buf=b"", close=lambda: None)), \
         patch.object(module, "deserialize_from_buffer", return_value={}):
        result = getattr(module, name)(**kwargs)
    episodes = result if persistent else [result]
    assert len(episodes) == (2 if persistent else 1)
    for ep in episodes:
        last = min(limit, 2)
        true_terminal = terminal and last == 2
        assert ep["episode_length"] == last
        assert ep["next_alive_mask"][-1] == {"a": False, "b": not true_terminal}
        if not true_terminal:
            assert ep["next_states"][-1][0] == 100 + last
            assert ep["next_group_indices"][-1] == {"a": -1, "b": last}
            assert ep["truncations"][-1]["b"]
        ds = TrajectoryDataset([(0, 0)], {0: ep}, 4)
        ds.n_steps, ds.gamma = 10, .5
        sample = ds[0]
        assert sample["td_discount"] == (0 if true_terminal else .5 ** last)
        assert not sample["next_timestep_padding_mask"][-1]
    if persistent:
        assert agent.reset.call_count == 3  # Initial setup plus each episode.
