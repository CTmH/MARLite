"""Sparse CPU behavior snapshots bound storage without changing frozen-policy replay."""

from copy import deepcopy

import pytest
import torch

from marlite.algorithm.temporal import CausalAttention, RecurrentGRU
from marlite.collection.parallel import ParallelCollector
from marlite.data import collate_sequences, partition_episode
from marlite.learning.mappo import prepare_ppo_targets
from test.test_sequence_mappo import TinyParallel, make_trainer


@pytest.mark.parametrize("attention", [False, True])
def test_sparse_and_dense_snapshots_give_identical_ppo_labels_and_replay(attention):
    trainer = make_trainer(max_steps=12)
    if attention:
        trainer.learner.actor.temporal = CausalAttention(8, 8, heads=2, context_length=4)
    actor, critic = trainer.learner.actor, trainer.learner.critic
    factory = lambda: TinyParallel(limit=20, death=True)
    dense = ParallelCollector(factory, max_steps=12, state_snapshot_interval=1).collect_episode(
        actor, critic, seed=91
    )
    sparse = ParallelCollector(factory, max_steps=12, state_snapshot_interval=4).collect_episode(
        actor, critic, seed=91
    )
    assert sparse.state_snapshots["steps"].tolist() == [0, 4, 8, 12]
    dense_bytes = sum(v.nbytes for v in dense.state_snapshots.values())
    sparse_bytes = sum(v.nbytes for v in sparse.state_snapshots.values())
    assert sparse_bytes < dense_bytes / 3
    dense_episodes, sparse_episodes = map(prepare_ppo_targets, ({0: dense}, {0: sparse}))
    sources = partition_episode(0, sparse, 3)
    dense_losses = trainer.learner.losses(dense_episodes, sources)
    sparse_losses = trainer.learner.losses(sparse_episodes, sources)
    for expected, actual in zip(dense_losses[:2], sparse_losses[:2]):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert abs(sparse_losses[2]["approx_kl"]) < 1e-6
    batch = collate_sequences(sparse_episodes, sources)
    assert not any("state." in name for name in batch.frames)
    restored = type(sparse).from_state_dict(sparse.state_dict())
    assert torch.equal(restored.state_snapshots["steps"], sparse.state_snapshots["steps"])


@pytest.mark.parametrize("kind", ["gru", "attention"])
def test_fused_sequence_matches_step_values_states_and_gradients_with_life_boundaries(kind):
    torch.manual_seed(37)
    model = RecurrentGRU(3, 8, layers=2) if kind == "gru" else CausalAttention(3, 8, 2, 4)
    reference = deepcopy(model)
    x = torch.randn(2, 9, 3, 3, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    valid = torch.ones(2, 9, 3, dtype=torch.bool)
    valid[0, 3:6, 1] = False
    valid[1, :, 2] = False
    reset = torch.zeros_like(valid)
    reset[:, 0] = True
    reset[0, 6, 1] = True
    # Nonempty prefix also exercises cache truncation inside a training chunk.
    prefix = torch.randn(2, 2, 3, 3)
    state = model.forward_sequence(prefix, model.initial_state(2, 3, "cpu", x.dtype)).state
    state = {k: v.detach() for k, v in state.items()}
    result = model.forward_sequence(x, state, valid, reset)
    rows, other_state = [], state
    for time in range(x.shape[1]):
        other = reference.step(y[:, time], other_state, valid[:, time], reset[:, time])
        rows.append(other.features)
        other_state = other.state
    expected = torch.stack(rows, 1)
    torch.testing.assert_close(result.features, expected)
    for key in other_state:
        torch.testing.assert_close(result.state[key], other_state[key])
    result.features.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, y.grad, atol=1e-5, rtol=1e-4)
    for (name, parameter), (_, other) in zip(
        model.named_parameters(), reference.named_parameters()
    ):
        torch.testing.assert_close(parameter.grad, other.grad, atol=2e-5, rtol=5e-4, msg=name)
