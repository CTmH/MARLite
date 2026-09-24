from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from marlite.replaybuffer.normal_replaybuffer import NormalReplayBuffer
from marlite.replaybuffer.prioritized_replaybuffer import PrioritizedReplayBuffer
from marlite.rollout.boundaries import finish_transition
from marlite.util.return_estimation import generalized_advantage_estimation, td_target, bootstrap_values
from marlite.util.trajectory_dataset import TrajectoryDataset, trajectory_collate_fn


def episode(length=4, terminal=False):
    """Two agents: a dies on transition 0; b survives until the boundary."""
    data = {k: [] for k in ("states", "next_states", "observations", "next_observations",
            "alive_mask", "next_alive_mask", "avail_actions", "next_avail_actions",
            "rewards", "terminations", "truncations", "group_indices", "next_group_indices")}
    for t in range(length):
        data["states"].append(np.array([10 + t], np.float32))
        data["next_states"].append(np.array([11 + t], np.float32))
        data["observations"].append({a: np.array([10 + t]) for a in ("a", "b")})
        data["next_observations"].append({a: np.array([11 + t]) for a in ("a", "b")})
        data["alive_mask"].append({"a": t == 0, "b": True})
        data["next_alive_mask"].append({"a": False, "b": not (terminal and t == length - 1)})
        data["rewards"].append({"a": 1. if t == 0 else 0., "b": 2.})
        data["terminations"].append({"a": True, "b": terminal and t == length - 1})
        data["truncations"].append({"a": True, "b": t == length - 1})
        data["avail_actions"].append({a: np.array([True, t % 2 == 0]) for a in ("a", "b")})
        data["next_avail_actions"].append({a: np.array([True, (t + 1) % 2 == 0]) for a in ("a", "b")})
        data["group_indices"].append({"a": -1 if t else 0, "b": t})
        data["next_group_indices"].append({"a": -1, "b": t + 1})
    data["episode_length"] = length
    return data


def dataset(data, history=4):
    return TrajectoryDataset([(0, t) for t in range(data["episode_length"])], {0: data},
                             history, required_attrs=[k for k in data if k != "episode_length"])


@pytest.mark.parametrize("history", [1, 2, 7])
@pytest.mark.parametrize("steps", [1, 3, 20])
@pytest.mark.parametrize("terminal", [False, True])
def test_n_step_windows_and_deaths(history, steps, terminal):
    ep = episode(terminal=terminal)
    original = deepcopy(ep)
    ds = dataset(ep, history)
    ds.n_steps, ds.gamma = steps, .5
    sample = ds[0]
    used = min(steps, 4)
    assert sample["td_rewards"] == sum(.5 ** k * (3 if k == 0 else 2) for k in range(used))
    assert sample["td_discount"] == (0 if terminal and used == 4 else .5 ** used)
    np.testing.assert_array_equal(sample["observations"][-1], np.stack(list(original["observations"][0].values())))
    assert sample["next_states"][-1][0] == 10 + used
    assert sample["next_group_indices"][-1][1] == used
    assert sample["next_timestep_padding_mask"] == [i < 0 for i in range(used-history+1, used+1)]
    for i, row in zip(range(used-history+1, used+1), sample["next_states"]):
        assert row[0] == (0 if i < 0 else 10 + i)
    assert ep["rewards"] == original["rewards"]  # SSL/other consumers see original rewards.
    batch = trajectory_collate_fn([sample])
    result = td_target(batch, torch.tensor([-999.]), torch.tensor([10.]), .5, torch.tensor([False]))
    assert result.item() == sample["td_rewards"] + sample["td_discount"] * 10


def test_boundary_stops_horizon_and_mean_rewards():
    ep = episode()
    ep["truncations"][1]["b"] = True
    ds = dataset(ep)
    ds.n_steps, ds.gamma, ds.reward_aggr_mode = 10, .5, "mean"
    assert ds[0]["td_rewards"] == 2.
    assert ds[0]["td_discount"] == .25
    assert ds[0]["next_states"][-1][0] == 12


def test_gae_terminal_truncated_cut_and_lambda_endpoints():
    rewards = torch.tensor([1., 2., 3., 100.])
    values = torch.tensor([10., 20., 30., 40.], requires_grad=True)
    next_values = torch.tensor([20., 30., 50., float("nan")])
    boot = torch.tensor([True, True, True, False])
    continuation = torch.tensor([True, True, False, False])
    adv, targets = generalized_advantage_estimation(rewards, values, next_values, boot, continuation, .5, 1.)
    torch.testing.assert_close(targets, torch.tensor([9., 16., 28., 100.]))
    assert not targets.requires_grad and not adv.requires_grad
    _, targets0 = generalized_advantage_estimation(rewards, values, next_values, boot, continuation, .5, 0.)
    torch.testing.assert_close(targets0, torch.tensor([11., 17., 28., 100.]))
    assert td_target({}, torch.tensor([2.]), torch.tensor([float("nan")]), .9, torch.tensor([1])).item() == 2.


@pytest.mark.parametrize("terminal", [True, False])
def test_single_transition_and_bootstrap_skips_all_dead(terminal):
    ep = episode(1, terminal)
    ds = dataset(ep, 5)
    ds.n_steps, ds.gamma = 8, .9
    assert ds[0]["td_discount"] == (0 if terminal else .9)
    batch = trajectory_collate_fn([ds[0]])
    critic = Mock(return_value={"v": torch.tensor([[42.]])})
    assert bootstrap_values(critic, batch, "cpu").item() == (0 if terminal else 42.)
    assert critic.call_count == (0 if terminal else 1)
    if not terminal:
        assert critic.call_args.args[0].shape[1] == 5
        assert critic.call_args.args[2].tolist() == [[True, True, True, False, False]]
    for buffer in (NormalReplayBuffer(2, 5), PrioritizedReplayBuffer(2, 5, "episode_length")):
        # Use an episode-level priority recognized by the prioritized buffer.
        if isinstance(buffer, PrioritizedReplayBuffer):
            buffer.priority_attr = "episode_reward"
            ep["episode_reward"] = 3.
        buffer.add_episode(ep)
        assert len(buffer.buffer) == 1


def test_rollout_timeout_survivors_final_state_and_missing_data():
    env = SimpleNamespace(agents=[], state=lambda: np.array([99.]))
    term, trunc = {"a": True, "b": False}, {"a": True, "b": True}
    state, alive, ended = finish_transition(env, {"a", "b"}, term, trunc, np.array([1.]), False, False)
    assert state[0] == 99 and alive == ["b"] and ended
    with pytest.raises(ValueError, match="final observations"):
        finish_transition(env, {"a"}, term, trunc, None, False, False)
    env.state = Mock(side_effect=RuntimeError("no state"))
    with pytest.raises(RuntimeError):
        finish_transition(env, {"b"}, term, trunc, None, False, False)
    state, alive, ended = finish_transition(env, set(), term, trunc, 7, True, False)
    assert state == 7 and alive == [] and all(term.values())


def test_individual_truncation_is_not_a_team_boundary():
    env = SimpleNamespace(agents=["b"], state=lambda: 12)
    _, alive, ended = finish_transition(env, {"b"}, {"a": False, "b": False},
                                       {"a": True, "b": False}, 11, False, False)
    assert alive == ["b"] and not ended


def test_zero_discount_does_not_use_bootstrap():
    adv, targets = generalized_advantage_estimation(torch.tensor([2.]), torch.tensor([1.]),
        torch.tensor([float("nan")]), torch.tensor([True]), torch.tensor([False]), 0., 1.)
    assert adv.item() == 1. and targets.item() == 2.


@pytest.mark.parametrize("bad", [-1., 1.1, float("nan")])
def test_invalid_lambda_is_rejected(bad):
    args = [torch.ones(1)] * 5
    with pytest.raises(ValueError):
        generalized_advantage_estimation(*args, .9, bad)
