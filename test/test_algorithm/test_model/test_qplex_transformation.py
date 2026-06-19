"""Tests for the QplexTransformation module.

Verifies:
- ModelConfig registration and instantiation.
- Forward output shapes and positivity of w_final.
- Attention regulariser (disabled / enabled).
- Gradient flow through states (non-nonlinear case) and agent Q-values
  (nonlinear case).
- Optional head-weight network.
- Input validation for misconfigured config lengths.
"""

import unittest
import torch
from marlite.algorithm.model import ModelConfig, QplexTransformation


def _make_transformation(
    state_dim=16,
    n_agents=3,
    n_head=2,
    embed_dim=8,
    attend_reg_coef=0.0,
    nonlinear=False,
    head_weight_config=None,
):
    head_weight = head_weight_config
    if nonlinear:
        key_in = state_dim + 1
        key_out = embed_dim
    else:
        key_in = state_dim
        key_out = n_agents * embed_dim
    cfg = {
        "model_type": "QplexTransformation",
        "state_dim": state_dim,
        "n_agents": n_agents,
        "n_head": n_head,
        "embed_dim": embed_dim,
        "attend_reg_coef": attend_reg_coef,
        "nonlinear": nonlinear,
        # Selectors always take state_dim (state -> embed_dim)
        "selector_configs": [
            {
                "model_type": "Custom",
                "layers": [{"type": "Linear", "in_features": state_dim, "out_features": embed_dim}],
            }
            for _ in range(n_head)
        ],
        # Keys produce per-agent keys.
        "key_configs": [
            {
                "model_type": "Custom",
                "layers": [
                    {
                        "type": "Linear",
                        "in_features": key_in,
                        "out_features": key_out,
                    }
                ],
            }
            for _ in range(n_head)
        ],
        "v_config": {
            "model_type": "Custom",
            "layers": [{"type": "Linear", "in_features": state_dim, "out_features": 1}],
        },
    }
    if head_weight is not None:
        cfg["head_weight_config"] = head_weight
    return ModelConfig(**cfg).get_model()


class TestQplexTransformation(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.n_agents = 3
        self.state_dim = 16
        self.embed_dim = 8
        self.t = _make_transformation(
            state_dim=self.state_dim,
            n_agents=self.n_agents,
            n_head=2,
            embed_dim=self.embed_dim,
        )

    def test_registered_as_model_type(self):
        cfg = ModelConfig(
            model_type="QplexTransformation",
            state_dim=8,
            n_agents=2,
            n_head=1,
            embed_dim=4,
            selector_configs=[
                {
                    "model_type": "Custom",
                    "layers": [{"type": "Linear", "in_features": 8, "out_features": 4}],
                }
            ],
            # non-nonlinear: out_features = n_agents * embed_dim = 2*4 = 8
            key_configs=[
                {
                    "model_type": "Custom",
                    "layers": [{"type": "Linear", "in_features": 8, "out_features": 8}],
                }
            ],
            v_config={
                "model_type": "Custom",
                "layers": [{"type": "Linear", "in_features": 8, "out_features": 1}],
            },
        )
        model = cfg.get_model()
        self.assertIsInstance(model, QplexTransformation)

    def test_forward_shape(self):
        agent_qs = torch.randn(self.batch_size, self.n_agents)
        states = torch.randn(self.batch_size, self.state_dim)
        w_final, v, att_reg = self.t(agent_qs, states)
        self.assertEqual(w_final.shape, (self.batch_size, self.n_agents))
        self.assertEqual(v.shape, (self.batch_size, self.n_agents))
        self.assertEqual(att_reg.shape, ())

    def test_w_final_is_positive(self):
        agent_qs = torch.randn(self.batch_size, self.n_agents)
        states = torch.randn(self.batch_size, self.state_dim)
        w_final, _, _ = self.t(agent_qs, states)
        self.assertTrue(torch.all(w_final > 0))

    def test_att_reg_zero_when_disabled(self):
        agent_qs = torch.randn(self.batch_size, self.n_agents)
        states = torch.randn(self.batch_size, self.state_dim)
        _, _, att_reg = self.t(agent_qs, states)
        self.assertEqual(att_reg.item(), 0.0)

    def test_att_reg_nonzero_when_enabled(self):
        t = _make_transformation(
            state_dim=self.state_dim,
            n_agents=self.n_agents,
            n_head=2,
            embed_dim=self.embed_dim,
            attend_reg_coef=0.5,
        )
        agent_qs = torch.randn(self.batch_size, self.n_agents)
        states = torch.randn(self.batch_size, self.state_dim)
        _, _, att_reg = t(agent_qs, states)
        self.assertGreater(att_reg.item(), 0.0)

    def test_gradient_flow(self):
        # Note: ``loss = w_final.sum()`` gives zero gradient because
        # w_final is the softmax output and Σ softmax(x) = 1 (identity,
        # independent of x).  We use a per-agent weighted loss so the
        # gradient is non-trivial.  The ``v.sum()`` term additionally
        # ensures ``states`` receives gradient via V(s).  In the
        # non-nonlinear case agent_qs is not used.
        torch.manual_seed(0)
        self.t = _make_transformation(
            state_dim=self.state_dim,
            n_agents=self.n_agents,
            n_head=2,
            embed_dim=self.embed_dim,
        )
        self.t.zero_grad()
        states = torch.randn(self.batch_size, self.state_dim)
        states.requires_grad_(True)
        agent_qs = torch.randn(self.batch_size, self.n_agents)
        w_final, v, att_reg = self.t(agent_qs, states)
        # Weight first agent differently so ∇(softmax) ≠ 0.
        loss = w_final[:, 0].sum() + v.sum() + att_reg
        loss.backward()
        self.assertIsNotNone(states.grad)
        if states.grad is not None:
            self.assertGreater(states.grad.abs().sum().item(), 0.0)
        n_with_grad = sum(
            1 for p in self.t.parameters()
            if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
        )
        n_total = sum(1 for p in self.t.parameters() if p.requires_grad)
        self.assertEqual(n_with_grad, n_total)

    def test_gradient_flow_through_agent_qs(self):
        # In the nonlinear case, agent_qs is used in the key computation.
        torch.manual_seed(42)
        t = _make_transformation(
            state_dim=self.state_dim,
            n_agents=self.n_agents,
            n_head=2,
            embed_dim=self.embed_dim,
            nonlinear=True,
        )
        agent_qs = torch.randn(self.batch_size, self.n_agents)
        agent_qs.requires_grad_(True)
        states = torch.randn(self.batch_size, self.state_dim)
        states.requires_grad_(True)
        w_final, v, att_reg = t(agent_qs, states)
        # Weight the first agent differently: loss = Σ w_final[:,0] + Σ v.
        # This avoids the zero-gradient identity Σ softmax(x) = 1.
        loss = w_final[:, 0].sum() + v.sum()
        loss.backward()
        self.assertIsNotNone(agent_qs.grad)
        self.assertIsNotNone(states.grad)
        if agent_qs.grad is not None:
            self.assertGreater(agent_qs.grad.abs().sum().item(), 0.0)
        if states.grad is not None:
            self.assertGreater(states.grad.abs().sum().item(), 0.0)

    def test_optional_head_weight(self):
        t = _make_transformation(
            state_dim=self.state_dim,
            n_agents=self.n_agents,
            n_head=2,
            embed_dim=self.embed_dim,
            head_weight_config={
                "model_type": "Custom",
                "layers": [{"type": "Linear", "in_features": self.state_dim, "out_features": 2}],
            },
        )
        agent_qs = torch.randn(self.batch_size, self.n_agents)
        states = torch.randn(self.batch_size, self.state_dim)
        w_final, v, _ = t(agent_qs, states)
        self.assertEqual(w_final.shape, (self.batch_size, self.n_agents))
        self.assertTrue(torch.all(w_final > 0))

    def test_nonlinear_flag(self):
        torch.manual_seed(0)
        t = _make_transformation(
            state_dim=self.state_dim,
            n_agents=self.n_agents,
            n_head=2,
            embed_dim=self.embed_dim,
            nonlinear=True,
        )
        agent_qs = torch.randn(self.batch_size, self.n_agents)
        agent_qs.requires_grad_(True)
        states = torch.randn(self.batch_size, self.state_dim)
        w_final, v, _ = t(agent_qs, states)
        self.assertEqual(w_final.shape, (self.batch_size, self.n_agents))
        # weight first agent differently (avoid zero-gradient identity).
        loss = w_final[:, 0].sum() + v.sum()
        loss.backward()
        self.assertIsNotNone(agent_qs.grad)
        if agent_qs.grad is not None:
            self.assertGreater(agent_qs.grad.abs().sum().item(), 0.0)

    def test_single_head(self):
        t = _make_transformation(
            state_dim=self.state_dim,
            n_agents=self.n_agents,
            n_head=1,
            embed_dim=self.embed_dim,
        )
        agent_qs = torch.randn(self.batch_size, self.n_agents)
        states = torch.randn(self.batch_size, self.state_dim)
        w_final, v, _ = t(agent_qs, states)
        self.assertEqual(w_final.shape, (self.batch_size, self.n_agents))
        self.assertTrue(torch.all(w_final > 0))

    def test_invalid_config_lengths(self):
        with self.assertRaises(ValueError):
            ModelConfig(
                model_type="QplexTransformation",
                state_dim=8,
                n_agents=2,
                n_head=3,
                embed_dim=4,
                selector_configs=[],  # wrong length
                key_configs=[
                    {
                        "model_type": "Custom",
                        "layers": [{"type": "Linear", "in_features": 8, "out_features": 8}],
                    }
                    for _ in range(3)
                ],
                v_config={
                    "model_type": "Custom",
                    "layers": [{"type": "Linear", "in_features": 8, "out_features": 1}],
                },
            ).get_model()


if __name__ == "__main__":
    unittest.main()
