import itertools

import pytest
import torch

from marlite.util.value_learning import masked_action_values, qtran_constraints
from marlite.algorithm.critic import CriticConfig
from test.test_algorithm.test_critic.test_qplex_mixer import _make_qplex_mixer_cfg, _make_inputs


def test_qtran_uses_complete_greedy_joint_action_and_mean():
    q = torch.tensor([[[5., 0., 100.], [5., 0., 100.]]], requires_grad=True)
    available = torch.tensor([[[True, True, False], [True, True, False]]])
    alive = torch.ones(1, 2, dtype=torch.bool)
    actions = torch.ones(1, 2, dtype=torch.long)
    value = torch.zeros(1, 1, requires_grad=True)
    # Both heads describe the SAME joint value 10, not additive contributions.
    def critic(enc, joint_actions, alive_mask):
        assert (joint_actions == 0).all()
        return {"q_per_action": torch.full_like(q, 10.)}
    counterfactual = torch.tensor([[[5., 0., 10000.], [5., 0., 10000.]]])
    optimal, nonoptimal = qtran_constraints(
        q, torch.zeros(1, 2, 3), actions, counterfactual, value,
        critic, alive, available,
    )
    torch.testing.assert_close(optimal, torch.tensor(0.))
    torch.testing.assert_close(nonoptimal, torch.tensor(0.))
    # Optimality is enforced even when the replayed actions are not greedy.
    shifted = torch.ones(1, 1, requires_grad=True)
    optimal, _ = qtran_constraints(q, torch.zeros(1, 2, 3), actions,
        counterfactual, shifted, critic, alive, available)
    optimal.backward()
    torch.testing.assert_close(shifted.grad, torch.tensor([[2.]]))
    assert (q.grad[..., 2] == 0).all()


def test_action_masks_dead_and_invalid_rows():
    q = torch.randn(2, 3, 4)
    dead = torch.zeros(2, 3, dtype=torch.bool)
    masked = masked_action_values(q, torch.zeros_like(q, dtype=torch.bool), dead)
    torch.testing.assert_close(masked.max(-1).values, torch.zeros(2, 3))
    with pytest.raises(ValueError):
        masked_action_values(q, torch.zeros_like(q, dtype=torch.bool), ~dead)


def test_qplex_feasible_igm_and_transformation_gradients():
    torch.manual_seed(7)
    mixer = CriticConfig(**_make_qplex_mixer_cfg()).get_critic()
    inputs = _make_inputs(batch_size=1)
    inputs["alive_mask"].fill_(True)
    inputs["q_value_from_agents"] = torch.tensor([[[1., 0., 100., -1., -2.]] * 3], requires_grad=True)
    inputs["avail_actions"] = torch.tensor([[[True, True, False, False, False]] * 3])
    inputs["actions"] = torch.zeros(1, 3, dtype=torch.long)
    greedy = mixer(**inputs)["q_tot"]
    for joint_action in itertools.product([0, 1], repeat=3):
        inputs["actions"] = torch.tensor([joint_action])
        assert (mixer(**inputs)["q_tot"] <= greedy + 1e-6).all()
    greedy.sum().backward()
    for name, parameter in mixer.transformation.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert any(p.grad.abs().sum() > 0 for p in mixer.transformation.parameters())
    inputs["alive_mask"].zero_()
    inputs["avail_actions"].zero_()
    inputs["actions"].fill_(-1)
    torch.testing.assert_close(mixer(**inputs)["q_tot"], torch.zeros(1))
