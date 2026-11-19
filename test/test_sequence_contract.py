"""Sequence contract checks independent of simulators and GPU availability."""

import copy

import pytest
import torch

from marlite.data import Episode, SequenceSlice, collate_sequences, partition_episode
from marlite.data.recurrent import reconstruct_state
from marlite.algorithm.temporal import (
    CausalAttention,
    FrameEncoder,
    RecurrentGRU,
    detach_state,
    reset_state,
    select_state,
)


def episode(length=7):
    n = 2
    return Episode(
        agent_ids=("a", "b"),
        frames={
            "observations": torch.arange((length + 1) * n * 3).float().reshape(length + 1, n, 3),
            "alive": torch.ones(length + 1, n, dtype=torch.bool),
            "available_actions": torch.ones(length + 1, n, 4, dtype=torch.bool),
            "agent_reset": torch.zeros(length + 1, n, dtype=torch.bool),
        },
        transitions={
            "actions": torch.zeros(length, n, dtype=torch.long),
            "rewards": torch.ones(length, n),
            "agent_terminated": torch.zeros(length, n, dtype=torch.bool),
            "agent_truncated": torch.zeros(length, n, dtype=torch.bool),
            "terminated": torch.zeros(length, dtype=torch.bool),
            "truncated": torch.tensor([False] * (length - 1) + [True]),
        },
    )


def valid_episode(length=7):
    ep = episode(length)
    ep.frames["agent_reset"][0] = True
    return ep.validate()


def test_episode_masks_and_serialization(tmp_path):
    ep = valid_episode(1)
    boot, trace = ep.return_masks()
    assert boot.tolist() == [True] and trace.tolist() == [False]
    ep.transitions["terminated"][-1] = True
    assert not ep.return_masks()[0].any()
    path = tmp_path / "episode.pt"
    torch.save(ep.state_dict(), path)
    restored = Episode.from_state_dict(torch.load(path, weights_only=True, mmap=True))
    torch.testing.assert_close(ep.frames["observations"], restored.frames["observations"])


@pytest.mark.parametrize("damage", ["frames", "actions", "dtype", "boundary", "reset", "nan"])
def test_episode_rejects_invalid_records(damage):
    ep = valid_episode()
    if damage == "frames":
        ep.frames["observations"] = ep.frames["observations"][:-1]
    elif damage == "actions":
        ep.transitions["actions"][0, 0] = 99
    elif damage == "dtype":
        ep.frames["alive"] = ep.frames["alive"].float()
    elif damage == "boundary":
        ep.transitions["truncated"][0] = True
    elif damage == "reset":
        ep.frames["agent_reset"][0] = False
    else:
        ep.transitions["rewards"][0, 0] = float("nan")
    with pytest.raises((TypeError, ValueError)):
        ep.validate()


def test_partition_and_collate_keep_all_learning_positions():
    ep = valid_episode()
    slices = partition_episode(0, ep, length=3, burn_in=2, future=1)
    batch = collate_sequences({0: ep}, slices)
    assert batch.loss_mask.sum() == ep.length
    assert batch.actor_mask.sum() == ep.length * 2
    assert batch.frame_start.tolist() == [0, 1, 4]
    assert batch.learning_start.tolist() == [0, 2, 2]
    assert torch.all(batch.loss_mask <= batch.filled)
    for row, source in enumerate(slices):
        start = int(batch.frame_start[row])
        length = int(batch.filled[row].sum())
        torch.testing.assert_close(
            batch.frames["observations"][row, : length + 1],
            ep.frames["observations"][start : start + length + 1],
        )
        assert not batch.trace_mask[row, length - 1]
    assert batch.to("cpu").sources == tuple(slices)
    other = copy.deepcopy(ep)
    other.agent_ids = ("b", "a")
    with pytest.raises(ValueError, match="ordered"):
        collate_sequences({0: ep, 1: other}, [slices[0], SequenceSlice(1, 0, 3)])


@pytest.mark.parametrize("factory", [FrameEncoder, RecurrentGRU, CausalAttention])
@pytest.mark.parametrize("boundaries", [False, True])
def test_step_sequence_and_chunk_agree(factory, boundaries):
    torch.manual_seed(12)
    model = factory(3, 8)
    x = torch.randn(2, 7, 2, 3)
    valid = torch.ones(2, 7, 2, dtype=torch.bool)
    reset = torch.zeros_like(valid)
    reset[:, 0] = True
    if boundaries:
        valid[0, 2:4, 0] = False
        reset[0, 4, 0] = True
        reset[1, 3, 1] = True
    state = model.initial_state(2, 2, x.device, x.dtype)
    whole = model.forward_sequence(x, state, valid, reset)
    parts, current = [], state
    for t in range(7):
        out = model.step(x[:, t], current, valid[:, t], reset[:, t])
        parts.append(out.features)
        current = out.state
    torch.testing.assert_close(whole.features, torch.stack(parts, 1))
    for key in current:
        torch.testing.assert_close(whole.state[key], current[key])
    left = model.forward_sequence(x[:, :3], state, valid[:, :3], reset[:, :3])
    right = model.forward_sequence(x[:, 3:], left.state, valid[:, 3:], reset[:, 3:])
    torch.testing.assert_close(whole.features, torch.cat((left.features, right.features), 1))
    assert (whole.features[~valid] == 0).all()
    changed = x.clone()
    changed[:, 5:] += 100
    future = model.forward_sequence(changed, state, valid, reset)
    torch.testing.assert_close(whole.features[:, :5], future.features[:, :5])
    if boundaries:
        changed = x.clone()
        changed[0, :4, 0] += 100
        restarted = model.forward_sequence(changed, state, valid, reset)
        torch.testing.assert_close(whole.features[0, 4:, 0], restarted.features[0, 4:, 0])


def test_gru_uses_single_fused_call_for_ordinary_chunk():
    model = RecurrentGRU(3, 8, layers=2)
    x = torch.randn(2, 9, 2, 3)
    calls = []
    handle = model.gru.register_forward_hook(lambda *_: calls.append(1))
    model.forward_sequence(x, model.initial_state(2, 2, "cpu", x.dtype))
    handle.remove()
    assert len(calls) == 1


def test_reconstruction_detaches_warmup_and_reaches_all_learning_steps():
    torch.manual_seed(3)
    model = RecurrentGRU(3, 8)
    x = torch.randn(1, 9, 2, 3, requires_grad=True)
    valid = torch.ones(1, 9, 2, dtype=torch.bool)
    reset = torch.zeros_like(valid)
    initial = model.initial_state(1, 2, "cpu", x.dtype)
    cached = model.forward_sequence(x[:, :2], initial).state
    reconstructed = reconstruct_state(model, x, valid, reset, 5, burn_in=3, cached_state=cached)
    exact = reconstruct_state(model, x, valid, reset, 5, strategy="prefix_recompute")
    torch.testing.assert_close(reconstructed["h"], exact["h"])
    assert not reconstructed["h"].requires_grad
    output = model.forward_sequence(x[:, 5:], reconstructed)
    output.features.square().sum().backward()
    assert not x.grad[:, :5].any()
    assert (x.grad[:, 5:].abs().sum((0, 2, 3)) > 0).all()
    with pytest.raises(ValueError, match="requires"):
        reconstruct_state(model, x, valid, reset, 5)
    zero = reconstruct_state(model, x, valid, reset, 5, strategy="zero_burnin", burn_in=0)
    assert not zero["h"].any()


def test_state_helpers_do_not_mutate_inputs():
    state = {"h": torch.ones(2, 3, 1, 4, requires_grad=True)}
    mask = torch.tensor([[True, False, False], [False, True, False]])
    cleared = reset_state(state, mask)
    assert not cleared["h"][mask].any() and state["h"].all()
    assert select_state(state, [1])["h"].shape[0] == 1
    assert not detach_state(state)["h"].requires_grad


def test_window_encoder_preserves_finite_history():
    from marlite.algorithm.temporal import WindowEncoder

    class SumWindow(torch.nn.Module):
        def forward(self, inputs, padding_mask):
            return inputs.masked_fill(padding_mask[..., None], 0).sum(1)

    model = WindowEncoder(SumWindow(), 3, 3, 2)
    x = torch.randn(1, 5, 2, 3)
    state = model.initial_state(1, 2, "cpu", x.dtype)
    result = model.forward_sequence(x, state)
    torch.testing.assert_close(result.features[:, -1], x[:, -2:].sum(1))
    changed = x.clone()
    changed[:, :3] += 100
    torch.testing.assert_close(
        model.forward_sequence(changed, state).features[:, -1], result.features[:, -1]
    )


@pytest.mark.parametrize("kind", ["logits", "q_values"])
def test_composed_agent_group_sequence_matches_steps_and_has_gradients(kind):
    from marlite.algorithm.sequence_agent_group import DecisionInput, SequenceAgentGroup

    model = SequenceAgentGroup(
        torch.nn.Linear(3, 8), RecurrentGRU(8, 8), torch.nn.Linear(8, 4), kind
    )
    x = torch.randn(2, 5, 2, 3)
    alive = torch.ones(2, 5, 2, dtype=torch.bool)
    reset = torch.zeros_like(alive)
    state = model.initial_state(2, 2, "cpu", x.dtype)
    output = model(DecisionInput(x, alive, reset), state)
    assert output.values.shape == (2, 5, 2, 4) and output.output_kind == kind
    pieces = []
    for t in range(5):
        result = model.step(DecisionInput(x[:, t], alive[:, t], reset[:, t]), state)
        state = result.state
        pieces.append(result.values)
    torch.testing.assert_close(output.values, torch.stack(pieces, 1))
    output.values.square().sum().backward()
    assert all(p.grad is not None for p in model.parameters())
    with pytest.raises(ValueError, match="context"):
        model(DecisionInput(x, alive, reset, {"messages": x}), state)


def test_sequence_policy_rejects_batch_dependent_or_random_layers():
    from marlite.algorithm.sequence_agent_group import SequenceAgentGroup

    for frame in (torch.nn.BatchNorm1d(3), torch.nn.Dropout(0.1)):
        with pytest.raises(ValueError):
            SequenceAgentGroup(frame, RecurrentGRU(3, 8), torch.nn.Linear(8, 4))


def test_recurrent_snapshots_are_not_duplicated_in_training_windows():
    ep = valid_episode()
    ep.state_snapshots = {
        "steps": torch.tensor([0]),
        "actor_state.cache": torch.randn(1, 2, 128, 64),
    }
    ep.validate()
    batch = collate_sequences({0: ep}, [SequenceSlice(0, 2, 3)])
    assert "actor_state.cache" not in batch.frames
    assert "actor_state.cache" in ep.state_snapshots
