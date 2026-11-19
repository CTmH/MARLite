import pytest
import torch
from marlite.util.value_learning import masked_action_values, qtran_constraint_terms


def constraints(*args):
    opt, nopt, nonopt, nnonopt = qtran_constraint_terms(*args)
    return opt / nopt.clamp_min(1), nonopt / nnonopt.clamp_min(1)


def test_qtran_uses_complete_greedy_joint_action_and_mean():
    q = torch.tensor([[[5.0, 0.0, 100.0], [5.0, 0.0, 100.0]]], requires_grad=True)
    available = torch.tensor([[[True, True, False], [True, True, False]]])
    alive = torch.ones(1, 2, dtype=torch.bool)
    actions = torch.ones(1, 2, dtype=torch.long)
    value = torch.zeros(1, 1, requires_grad=True)

    # Both heads describe the SAME joint value 10, not additive contributions.
    def critic(enc, joint_actions, alive_mask):
        assert (joint_actions == 0).all()
        return {"q_per_action": torch.full_like(q, 10.0)}

    counterfactual = torch.tensor([[[5.0, 0.0, 10000.0], [5.0, 0.0, 10000.0]]])
    optimal, nonoptimal = constraints(
        q,
        torch.zeros(1, 2, 3),
        actions,
        counterfactual,
        value,
        critic,
        alive,
        available,
    )
    torch.testing.assert_close(optimal, torch.tensor(0.0))
    torch.testing.assert_close(nonoptimal, torch.tensor(0.0))
    # Optimality is enforced even when the replayed actions are not greedy.
    shifted = torch.ones(1, 1, requires_grad=True)
    optimal, _ = constraints(
        q, torch.zeros(1, 2, 3), actions, counterfactual, shifted, critic, alive, available
    )
    optimal.backward()
    torch.testing.assert_close(shifted.grad, torch.tensor([[2.0]]))
    assert (q.grad[..., 2] == 0).all()


def test_action_masks_dead_and_invalid_rows():
    q = torch.randn(2, 3, 4)
    dead = torch.zeros(2, 3, dtype=torch.bool)
    masked = masked_action_values(q, torch.zeros_like(q, dtype=torch.bool), dead)
    torch.testing.assert_close(masked.max(-1).values, torch.zeros(2, 3))
    with pytest.raises(ValueError):
        masked_action_values(q, torch.zeros_like(q, dtype=torch.bool), ~dead)
