import pytest
import torch
from marlite.util.return_estimation import generalized_advantage_estimation


def test_gae_terminal_truncated_cut_and_lambda_endpoints():
    rewards = torch.tensor([1.0, 2.0, 3.0, 100.0])
    values = torch.tensor([10.0, 20.0, 30.0, 40.0], requires_grad=True)
    next_values = torch.tensor([20.0, 30.0, 50.0, float("nan")])
    boot = torch.tensor([True, True, True, False])
    continuation = torch.tensor([True, True, False, False])
    adv, targets = generalized_advantage_estimation(
        rewards, values, next_values, boot, continuation, 0.5, 1.0
    )
    torch.testing.assert_close(targets, torch.tensor([9.0, 16.0, 28.0, 100.0]))
    assert not targets.requires_grad and not adv.requires_grad
    _, targets0 = generalized_advantage_estimation(
        rewards, values, next_values, boot, continuation, 0.5, 0.0
    )
    torch.testing.assert_close(targets0, torch.tensor([11.0, 17.0, 28.0, 100.0]))


def test_zero_discount_does_not_use_bootstrap():
    adv, targets = generalized_advantage_estimation(
        torch.tensor([2.0]),
        torch.tensor([1.0]),
        torch.tensor([float("nan")]),
        torch.tensor([True]),
        torch.tensor([False]),
        0.0,
        1.0,
    )
    assert adv.item() == 1.0 and targets.item() == 2.0


@pytest.mark.parametrize("bad", [-1.0, 1.1, float("nan")])
def test_invalid_lambda_is_rejected(bad):
    args = [torch.ones(1)] * 5
    with pytest.raises(ValueError):
        generalized_advantage_estimation(*args, 0.9, bad)
